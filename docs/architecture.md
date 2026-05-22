# Architecture

This document explains why this system is built the way it is. Read it after the [README](../README.md), which covers the *what* and the *how to run it*.

## Problem framing

Given a stream of conversations, surface PM-actionable emerging topics — phrased in PM voice ("20% of users are requesting refunds because of slow shipping"), with sentiment, and with stable identity across re-runs so dashboards don't shuffle.

The job decomposes into:

1. Ingest conversations of arbitrary size (two-message DM or 10K-message channel — same endpoint)
2. Segment them into analyzable units the embedding model can handle
3. Embed those units
4. Cluster embeddings into topics
5. Label clusters with a human-readable name, sentiment, and a PM insight
6. Track topics over time so re-runs preserve identity (centroid matching)
7. Surface emerging topics — but emergence is **per conversation**, not global: a topic emerging in a software conversation has nothing to do with the same words appearing in an unrelated chat

## Approach decision: chunk-and-cluster (no per-LLM-extraction)

Three families of approach were on the table:

| | A. Pure BERTopic | B-original. LLM extracts concerns first | **B-final (this build). Chunk raw text directly** | C. LLM-first structured |
|---|---|---|---|---|
| LLM cost | None | 1 call per conversation | 1 call per cluster | 1 call per conversation |
| Topic readability | Bag of keywords | Good | Good | Excellent |
| Surfaces novel topics | Yes | Yes | Yes | Only if prompted |
| Per-conversation scaling cost | None | O(N) LLM calls | O(N) embedding only | O(N) LLM calls |

We picked the chunk-and-cluster variant. The crucial property: **LLM cost is bounded by the number of clusters, not the number of conversations.** A million conversations and a hundred topics costs ~100 LLM calls. The expensive operations (chunking, embedding, clustering) are all open-source and CPU-friendly.

The original Approach B distilled concerns via a per-conversation LLM call, then embedded those. We dropped that step in favor of **direct chunking with LangChain's `SemanticChunker`**. Reasoning:
- The per-conversation LLM step doubles per-message cost without proportional quality gain when the input is already structured (turns with roles)
- A modern semantic chunker, given role-prefixed turns, breaks naturally on topic shifts within a conversation — recovering most of the benefit of LLM extraction
- The sliding-window recursive fallback ensures no chunk exceeds the embedder's 512-token context

## Pipeline

```
                          ┌────────────────────────────────┐
POST /conversations ──────▶  raw_store.write/append (fs)  │
                          │  conversations row upsert      │
                          │  enqueue ingest job (arq)      │
                          └──────────────┬─────────────────┘
                                         │
                                         ▼
                          ┌────────────────────────────────┐
                          │  ingest_pipeline               │
                          │  ─ slice new turns from raw    │
                          │  ─ pull overlap_prefix tail    │
                          │  ─ SemanticChunker primary     │
                          │  ─ RecursiveSplitter fallback  │
                          │  ─ BGE-small embedding         │
                          │  ─ k-NN assign to existing     │
                          │    topics (cosine >= 0.75)     │
                          │  ─ recompute mention_count     │
                          └──────────────┬─────────────────┘
                                         │
                          (periodic OR manual via API)
                                         │
                                         ▼
POST /clustering-runs ───▶┌────────────────────────────────┐
                          │  clusterer                     │
                          │  ─ UMAP(384→5, cosine)         │
                          │  ─ HDBSCAN(eom)                │
                          │  ─ Centroid match (≥0.85)      │
                          │    ↳ inherit topic_id          │
                          │    ↳ or create new topic       │
                          │  ─ retire unmatched topics     │
                          │  ─ snapshot to                 │
                          │    clustering_run_topics       │
                          │  ─ enqueue label jobs          │
                          └──────────────┬─────────────────┘
                                         │
                                         ▼
                          ┌────────────────────────────────┐
                          │  labeler (Bedrock Sonnet 4.6)  │
                          │  ─ top-N chunks near centroid  │
                          │  ─ one InvokeModel per topic   │
                          │  ─ writes label, sentiment,    │
                          │    pm_insight to topics row    │
                          └────────────────────────────────┘

                          ┌────────────────────────────────┐
GET /conversations/{id}/topics ◀── emergence calculator   │
                          │  compares batch_seq=N vs <N   │
                          │  per topic in this conversation│
                          └────────────────────────────────┘

GET /clustering-runs/latest/topics ◀── snapshot read
```

## Component-by-component

### Chunker (`app/services/chunker.py`)

LangChain `SemanticChunker` (percentile-based breakpoints over embedding distances) → `RecursiveCharacterTextSplitter` fallback for any chunk that overflows the 400-token budget.

Why semantic chunking and not fixed-size: a conversation about "refund + slow shipping + bad UI" should produce three semantic chunks that each map to one of those topics. Fixed-size windows would blur the boundaries.

Why recursive as fallback: `SemanticChunker` doesn't guarantee a token-count ceiling, but BGE-small truncates anything past 512 tokens silently. The recursive splitter enforces the ceiling using HuggingFace tokenizer length measurement.

**Separators** are chosen so the splitter prefers to break on turn boundaries:
```python
["\n\n", "\nuser:", "\nagent:", "\n", ". ", " ", ""]
```

**Append efficiency:** on appending new turns to an existing conversation, the chunker only processes the new turns. The last existing chunk's tail (~50 tokens) is prepended as `overlap_prefix` for semantic continuity; any output chunk that lies entirely in the overlap region is dropped so the previous batch isn't re-embedded.

### Embedder (`app/services/embedder.py`)

`BAAI/bge-small-en-v1.5` — 384-dim, ~120MB, CPU-friendly, strong MTEB scores. Embeddings are L2-normalized at the source so cosine similarity == dot product and pgvector's `vector_cosine_ops` semantics line up directly with the centroid thresholds (0.85 for matching, 0.75 for k-NN assign).

Single global instance shared by chunker (for `SemanticChunker`'s breakpoint detection) and ingest pipeline (for chunk embedding) — one model load per worker.

### Topic assigner (`app/services/topic_assigner.py`)

Cheap online classifier between full clustering runs: for each new chunk, run a pgvector HNSW nearest-neighbor query against active topic centroids; if the top result's cosine similarity ≥ 0.75, assign the chunk; otherwise leave `topic_id = NULL` until the next full clustering pass picks it up.

The 0.75 threshold is lower than the 0.85 centroid-match threshold (used for run-to-run topic stability) because k-NN assignment of a single chunk is a more permissive operation than declaring two cluster centroids equivalent — false positives are cheaper here and easily corrected by the next full re-run.

### Clusterer (`app/services/clusterer.py`)

UMAP (`n_neighbors=20, min_dist=0.05, n_components=5, metric=cosine`) reduces 384-dim BGE vectors to 5 dimensions for HDBSCAN. Reducing first is critical — HDBSCAN doesn't work well in 384 dimensions because of the curse of dimensionality.

HDBSCAN (`min_cluster_size=25, min_samples=10, cluster_selection_method=eom`):
- Density-based, so we don't have to specify `k`
- Surfaces an outlier bucket (`label == -1`) — these chunks become `topic_id = NULL` and represent genuinely novel content not yet a cluster
- Handles uneven cluster sizes well (typical: few huge topics, many small niche ones)

**Centroid matching** for topic stability: after HDBSCAN identifies fresh clusters, compute each cluster's mean embedding (normalized), and greedily match against existing active topics' centroids by cosine. Matches ≥ 0.85 inherit the existing `topic_id` (and its label, history, etc.). Unmatched clusters create new topics. Active topics that no new cluster matched are marked `retired`.

This solves the "PM dashboard breaks on every re-run" problem — IDs are stable across runs as long as the underlying topic still exists.

**Per-run snapshot** (`clustering_run_topics`): captures the topic IDs and mention counts that participated in each clustering run. `/clustering-runs/{id}/topics` queries this snapshot so historical runs return historically-correct topic lists, not whatever is active right now.

### Labeler (`app/services/labeler.py`)

One Bedrock `InvokeModel` call per affected topic, with `anthropic.claude-sonnet-4-6` model id and prefilled-`{` JSON forcing. Inputs are the top 20 chunks nearest to the topic's centroid (a proxy for HDBSCAN exemplars that survives the storage round-trip).

Output schema is **constrained on sentiment** (`positive | negative | neutral | mixed`) but **free-text on topic label and PM insight**. LLMs are good at producing natural-language summaries; they are bad at confidence scoring without rubrics, so we deliberately omit any score/confidence field.

`MOCK_LLM=true` short-circuits the Bedrock call with a deterministic stub, used by tests and local dev without AWS credentials.

### Emergence (`app/services/emergence.py`)

Per-conversation, never global. For conversation C with latest `batch_seq = N`:

| Case | Formula | is_emerging |
|---|---|---|
| `new_count == 0` (topic only in old batches) | 0 | false |
| Fresh conversation (`N == 1`) | `new_count` | `new_count ≥ EMERGENCE_FRESH_MIN_COUNT` (default 2) |
| Subsequent batch | `new_count / max(old_count, 1)` | `score ≥ EMERGENCE_RATIO_THRESHOLD` (default 2.0) |

Computed on-the-fly per `GET /conversations/{id}/topics` — the query is a single GROUP BY on `chunks` filtered by `conversation_id`, well-indexed and fast. No caching layer.

**Why per-conversation, not global:** "emerging topic" only makes sense within a single thread of communication. A topic with 50 mentions across 30 software conversations is meaningfully different from a topic with 50 mentions across one rambling Slack channel. Comparing the *change* within one conversation between batches is the only emergence signal that's both well-defined and PM-useful.

## Database schema rationale

| Table | Purpose | Key decisions |
|---|---|---|
| `conversations` | Lightweight metadata | NO `ended_at` (conversations are open-ended), NO `user_id`/`channel` (those belong to connector metadata if relevant). `latest_batch_seq` tracked here so append doesn't need a scan. |
| `chunks` | Embedded analyzable units | `batch_seq` enables the emergence formula. `position` (turn index) and `mention_at` (earliest contributing turn timestamp) preserved for downstream temporal analysis. HNSW index on `embedding`. |
| `topics` | Globally persistent clusters | `centroid` (vector(384)) drives both k-NN assignment and run-to-run matching. `status` (`active`/`merged`/`retired`) plus HNSW index on centroid for fast lookup. |
| `clustering_runs` | Per-pass history | `params` JSONB captures the UMAP+HDBSCAN+threshold config of each run for reproducibility. |
| `clustering_run_topics` | Snapshot of topics per run | Lets `/clustering-runs/{id}/topics` answer historical queries instead of returning whatever is active right now. |
| `jobs` | Generic background-task tracking | Bridges arq job state to API consumers via `GET /jobs/{id}` polling. |

Raw conversation turns live in `data/raw/{conversation_id}.json` — NOT in Postgres. Reasoning: turns are append-only, frequently re-read when re-chunking after a config change, and there's no SQL query benefit to having them in a relational table.

## Tradeoffs and what we deliberately *didn't* build

- **No global emerging-topic concept.** A "what's hot across the entire system" view is technically interesting but conceptually noisy across heterogeneous conversation sources. Skipped on purpose.
- **No sentiment score.** Just the categorical label. LLMs hallucinate confidence numbers without a rubric.
- **No fixed enum on topic labels.** Topics are discovered, not chosen from a fixed list. Aggressive enum-ing would defeat the point of clustering.
- **No mypy/CI/pre-commit.** Assignment-scoped — adding tooling beyond `ruff` + `pytest` was deliberate restraint.
- **No frontend.** Backend-only per assignment scope. The OpenAPI page at `/docs` is the GUI.
- **No incremental cluster splitting (Davies-Bouldin triggered refinement).** That's the 2026 research frontier (Approach D in the design notes). Out of scope; can grow into this without throwing anything away.

## What this becomes at production scale

| Component | Current | Production upgrade |
|---|---|---|
| Background worker | arq + Redis | Celery + Redis (more durable, larger ecosystem, mature monitoring) |
| Raw storage | Local filesystem (`data/raw/`) | S3 / R2 / MinIO with the same JSON-per-conversation layout |
| Vector store | Postgres + pgvector | Qdrant for ≥5M chunks (HNSW + payload filters perform better) |
| Analytics | Postgres materialized views | ClickHouse for OLAP at scale |
| Embeddings | Self-hosted BGE-small (CPU) | Same model on dedicated inference servers, or upgrade to `BGE-M3` for multilingual |
| Clustering trigger | arq cron every 24h | Lifecycle-aware (Davies-Bouldin + silhouette monitoring) targeted re-runs |
| Topic labeling | Single Bedrock call per cluster | Batched Bedrock calls; cache labels keyed by centroid hash |

The architectural shape doesn't change — only the implementations of individual modules do.

# Architecture

One-page snapshot. See [REASONING.md](REASONING.md) for *why* each piece is what it is and what was rejected.

## What the system does

Ingest conversations of any size, chunk them, embed the chunks, cluster the embeddings, label each cluster with a PM-style insight, and surface emerging topics *per conversation* by comparing the latest ingest batch against prior batches of the same conversation.

## Pipeline

```
POST /conversations ─► raw_store ──► arq job ──► chunker ──► embedder ──┐
                       (filesystem)                                     │
                                                                        ▼
                                                              ┌──────────────────┐
                                                              │ chunks (pgvector)│
                                                              └────────┬─────────┘
                                                                       │
                                          ┌─ k-NN assign (cos ≥ 0.75) ─┤
                                          │   at ingest time           │
                                          ▼                            │
                                     ┌────────┐                        │
                                     │ topics │◄─── centroid match ────┤
                                     │  (DB)  │     (cos ≥ 0.85)       │
                                     └───┬────┘                        │
                                         │              ┌──────────────┴────────┐
   POST /clustering-runs ──► arq job ────┴──► UMAP 384→5 ──► HDBSCAN(eom) ──────┘
                                                          (every 24h cron + manual)
                                         │
                                         ▼
                              one Bedrock call per affected topic
                                  (Claude Sonnet 4.6)
                              → {label, sentiment, pm_insight}

GET /conversations/{id}/topics  ◄── per-conversation emergence
                                    (new_count vs old_count for same conversation)
GET /clustering-runs/latest/topics ◄── global snapshot from latest run
```

See [data-flow.md](data-flow.md) for mermaid sequence diagrams of each path.

## Components

| Module | Job | Key tradeoff (rationale in [REASONING.md](REASONING.md)) |
|---|---|---|
| `services/chunker.py` | LangChain SemanticChunker + Recursive fallback, append-aware sliding overlap | Semantic over fixed-size; appends never re-embed old chunks |
| `services/embedder.py` | `BAAI/bge-small-en-v1.5`, self-hosted, CPU | Free + fast over API embeddings |
| `services/topic_assigner.py` | k-NN assigns each new chunk to existing topic if cosine ≥ 0.75 | Cheap online classifier between full reruns |
| `services/clusterer.py` | UMAP 384→5 then HDBSCAN(eom), centroid-match new clusters to existing topics at cosine ≥ 0.85 | Stable IDs across reruns without LLM cost |
| `services/labeler.py` | One Bedrock InvokeModel per affected topic with top-20 nearest-centroid chunks | LLM cost bounded by cluster count, not conversation count |
| `services/emergence.py` | Per-conversation: `new_count / max(old_count, 1)` vs latest `batch_seq` | Fresh-conversation special case: `new_count ≥ 2` ⇒ emerging |
| `workers/arq_worker.py` | arq + Redis, three task functions + 24h cron | Async-native, ~1/3 of Celery boilerplate |

## Database schema

| Table | What it stores | Notable columns |
|---|---|---|
| `conversations` | Lightweight metadata. NO `ended_at`, `user_id`, `channel`. Raw turns live on filesystem. | `latest_batch_seq`, `turn_count`, `raw_path` |
| `chunks` | Embedded analyzable units | `embedding vector(384)`, `batch_seq`, `position`, `mention_at` |
| `topics` | Persistent clusters across runs | `centroid vector(384)`, `label`, `sentiment ∈ {positive,negative,neutral,mixed}`, `pm_insight`, `status ∈ {active,merged,retired}` |
| `clustering_runs` | Per-pass history with `params` JSONB | `n_topics_found/new/retired` |
| `clustering_run_topics` | Per-run snapshot of topics that participated | Lets historical `/clustering-runs/{id}/topics` answer correctly after later runs mutate state |
| `jobs` | Background-task lifecycle bridge | `type`, `status ∈ {queued,running,completed,failed}`, `result jsonb`, `error` |

HNSW indexes on `chunks.embedding` and `topics.centroid` (`vector_cosine_ops`). Composite `(conversation_id, batch_seq)` index for emergence queries.

## API surface

All topic listings are nested under a parent context — no orphan `/topics` listing.

```
POST /conversations                        create-or-append, async 202+job
GET  /conversations(/{id})                 metadata
GET  /conversations/{id}/topics            per-conversation, default sort=emergence
POST /clustering-runs                      manual trigger
GET  /clustering-runs(/{id})               status / history
GET  /clustering-runs/{id}/topics          snapshot for a specific run
GET  /clustering-runs/latest/topics        current global state
GET  /topics/{id}                          flat lookup by PK only
GET  /jobs/{id}                            background-job poll
GET  /healthz                              liveness
```

See [api.md](api.md) for endpoint contracts.

## Non-goals (deliberate)

- **No global "emerging topics" endpoint.** Emergence is computed *within a conversation* only. Cross-conversation aggregation lives in `mention_count` but isn't synthesized into "X% of users want Y" sentences. See REASONING.md §Emergence.
- **No sentiment score, only the categorical label.** LLMs are bad at calibrated confidence without rubrics.
- **No fixed enum on topic labels.** Topics are discovered, not chosen from a list.
- **No mypy / pre-commit / CI / frontend.** `ruff` + `pytest` + OpenAPI at `/docs` are the surface.

## Production scale-up path

Each component swaps in place without changing the architectural shape:

| Component | Current | Production |
|---|---|---|
| Background runner | arq + Redis | Celery + Redis (or Temporal) |
| Raw storage | Local filesystem (`data/raw/`) | S3 / R2 / MinIO (same JSON-per-conversation layout) |
| Vector store | Postgres + pgvector | Qdrant (only meaningfully better at >5M vectors) |
| Analytics queries | Postgres + materialized views | ClickHouse |
| Embedding model | BGE-small (CPU) | Same model on dedicated inference, or BGE-M3 for multilingual |
| Clustering cadence | arq cron every 24h | Davies-Bouldin + silhouette monitoring → targeted re-runs |
| Topic labeling | One Bedrock call per cluster | Batched Bedrock + cache by centroid hash |

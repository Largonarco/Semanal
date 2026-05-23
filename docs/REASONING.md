# Reasoning

Why each major component is what it is, and what was rejected.

---

## Overall approach: chunk-and-cluster (not per-conversation LLM extraction)

**Considered:** (A) Pure BERTopic, (B-original) LLM extracts pain points/concerns per conversation then embed those, (B-final, **chosen**) chunk raw conversations and embed chunks directly, (C) LLM-first structured extraction with rigid schema, (D) lifecycle-aware research-grade clustering with quality-triggered refinement.

**Chosen because:** LLM cost is bounded by the *number of clusters*, not the number of conversations. With Approach C, 1M conversations = 1M LLM calls. With our approach, 1M conversations = ~100 LLM calls (one per discovered topic). A modern semantic chunker, given role-prefixed turns, breaks naturally on topic shifts within a conversation — recovering most of the benefit of a per-conversation LLM extraction step without the cost.

**Rejected:**
- **A (BERTopic only):** output is bag-of-keywords per cluster (`refund, slow, shipping, late`), not human-readable. A PM needs sentences.
- **B-original (LLM per conversation):** doubles per-message cost without proportional quality gain — semantic chunkers already capture topic shifts in dialogue.
- **C (LLM-first structured):** schema drift (LLM invents new tag names), brittle for novel-topic discovery, O(N) LLM calls.
- **D (research-grade):** overkill for assignment scope. Our architecture can grow into D without rewrites.

---

## Database: Postgres 16 + pgvector

**Considered:** Postgres + pgvector, Qdrant, Pinecone, Weaviate, Milvus.

**Chosen because:**
- One database for everything — relational rows, JSONB metadata, and 384-dim vectors all in the same Postgres. No sync pipeline.
- pgvector's HNSW index handles k-NN over millions of vectors at acceptable latency.
- Cross-table joins (chunks → topics, conversations → chunks) stay as plain SQL.
- Free, ubiquitous ops knowledge, easy to back up.

**Rejected:**
- **Qdrant:** external service to operate; sync pipeline to keep in step with Postgres; HNSW + payload-filter performance only meaningfully pulls ahead at >5M vectors. Documented as the production upgrade path.
- **Pinecone:** paid, vendor lock-in, opaque internals.
- **Weaviate:** richer schema model than we need; adds a service for no demo value.
- **Milvus:** heaviest of the three to operate; gRPC-first.

---

## Clustering: UMAP → HDBSCAN

**Considered:** K-means, DBSCAN, HDBSCAN, Agglomerative, full BERTopic stack (UMAP + HDBSCAN + c-TF-IDF).

**Chosen because:**
- **HDBSCAN auto-determines k.** PMs don't know "how many topics" in advance.
- **Density-based** → handles uneven cluster sizes (typical conversation data: a few huge themes + a long tail of niche ones).
- **Built-in `-1` noise label** = the "what's genuinely novel and doesn't fit any cluster yet" bucket. Free emerging-topic discovery channel.
- **`probabilities_`** gives per-member confidence — used directly for exemplar selection when sampling chunks for the LLM labeler.

**Rejected:**
- **K-means:** requires pre-specified k. No notion of outliers — every chunk gets forced into some cluster.
- **DBSCAN:** not hierarchical; `eps` tuning is brittle across density regions.
- **Agglomerative:** O(n²) memory; doesn't scale past ~50K vectors.
- **BERTopic full stack:** uses c-TF-IDF for labels → keyword bags. We want sentences from an LLM, not keywords.

### Why UMAP before HDBSCAN

HDBSCAN underperforms in 384 dimensions because of the curse of dimensionality — distances collapse and density estimates become unreliable. UMAP (`n_components=5`) compresses to a manifold-aware 5-dim space that preserves local + global structure.

**Rejected:** PCA (linear, loses local structure on sentence embeddings); t-SNE (no `transform` for new data, slower, less stable).

---

## Embedding model: `BAAI/bge-small-en-v1.5`

**Considered:** OpenAI `text-embedding-3-small`, Cohere `embed-v3`, Voyage `voyage-3`, BGE-M3, BGE-large, BGE-small.

**Chosen because:**
- Free, self-hosted, CPU-friendly (~120MB, ~50ms per encode on a laptop CPU).
- Strong MTEB benchmarks for its size class.
- 384 dims keeps pgvector indexes compact and HNSW queries fast.
- 512-token context window aligns naturally with our chunk budget.

**Rejected:**
- **OpenAI / Cohere / Voyage:** API cost per chunk, rate limits, vendor lock-in. At 10K chunks/day, all three are non-trivially expensive.
- **BGE-M3:** 1024 dims, slower. Multilingual not needed for an English-only assignment. Documented as the swap-in for multilingual.
- **BGE-large:** ~1.3GB, slower, marginal MTEB gain over BGE-small for our domain.

---

## Chunking: SemanticChunker + RecursiveCharacterTextSplitter fallback

**Considered:** Fixed-size sliding window, RecursiveCharacterTextSplitter alone, SemanticChunker alone, LLM-extracted concerns.

**Chosen because:**
- **SemanticChunker** (LangChain experimental) splits on embedding-distance breakpoints, so a conversation about "refund + shipping + UI" naturally lands in three chunks that each map to one topic.
- **Recursive fallback** enforces the 400-token ceiling on any oversized semantic chunk so the embedder never silently truncates.
- **Conversation-aware separators** (`["\n\n", "\nuser:", "\nagent:", ". ", " ", ""]`) bias splits toward turn boundaries when the recursive splitter has to break a chunk.
- **Append-aware:** on a new ingest batch, only new turns are chunked. The last ~50 tokens of the previous chunk are prepended as `overlap_prefix` so the first new chunk has context across the boundary, but already-embedded chunks are never re-processed.

**Rejected:**
- **Fixed-size sliding window:** cuts sentences mid-thought; ignores semantic structure.
- **RecursiveCharacterTextSplitter alone:** respects token budget but breaks blindly on character separators — a conversation about three different topics gets one chunk if it fits.
- **SemanticChunker alone:** no token-count ceiling. Would let chunks overflow the 512-token embedder context, where they get silently truncated.
- **LLM-extracted concerns (B-original):** 1 LLM call per conversation. Unbounded cost.

---

## LLM provider: AWS Bedrock + Claude Sonnet 4.6 (via inference profile)

**Considered:** Anthropic direct API, AWS Bedrock, OpenAI, Google Vertex AI, self-hosted vLLM with Qwen2.5-32B.

**Chosen because:**
- Bedrock keeps everything inside one AWS account — no separate API key procurement, easier to apply enterprise compliance / private VPC controls in production.
- Sonnet 4.6 produces high-quality, well-structured prose for labels + PM insights.
- Volume is tiny (≈1 call per discovered topic, not per conversation), so per-call price isn't a primary factor.

**Rejected:**
- **Anthropic direct API:** identical model quality, but a parallel credential pipeline. Bedrock is operationally cleaner when the rest of infra is in AWS.
- **OpenAI GPT-4o:** comparable for prose; we'd lose the existing AWS posture. JSON adherence is a wash.
- **Vertex AI:** would couple infra to GCP for one component.
- **Self-hosted vLLM (Qwen2.5-32B):** GPU host to operate, model serving ops, autoscaling — not justified for ~10s of calls per clustering run.

### Bedrock quirks the implementation respects

1. **No on-demand foundation model invocations for Sonnet 4.x.** Must call via a cross-region inference profile (`us.anthropic.claude-sonnet-4-6`, not the raw foundation model ID). The `.env.example` and `docker-compose.yml` override both encode this.
2. **No assistant-message prefill.** The "force JSON by prefilling `{`" pattern that works on Anthropic's direct API returns `ValidationException` on Bedrock. We rely on system-prompt instruction and tolerant first-`{`/last-`}` bracket extraction instead.

---

## Background runner: `arq`

**Considered:** Celery, arq, RQ, Dramatiq, FastAPI `BackgroundTasks`, Temporal.

**Chosen because:**
- Async-native — same event loop as FastAPI / httpx / SQLAlchemy async.
- Redis-backed, durable across worker restarts.
- ~1/3 the boilerplate of Celery; one `WorkerSettings` class declares functions + cron + lifecycle.
- Built-in cron support handles the scheduled clustering pass.

**Rejected:**
- **Celery:** heavyweight, sync-first internals, more complex deployment (broker + result backend + beat). Documented as the production upgrade once ops maturity matters more than dev velocity.
- **RQ / Dramatiq:** not async-native — would force `asyncio.run` wrappers around all task functions.
- **FastAPI `BackgroundTasks`:** in-process, lost on restart, no retry. Demo-only.
- **Temporal:** workflow engine — massive over-engineering for ~3 task types.

---

## Topic stability across reruns: centroid matching at cosine ≥ 0.85

**Considered:** Per-run fresh topic IDs (no stability), centroid matching, LLM-based merge ("are these two clusters the same topic?"), manual curation queue.

**Chosen because:**
- ~50 LOC: greedy 1-to-1 cosine match between new cluster centroids and existing active topic centroids.
- Empirically robust on the MultiWOZ test — 2 of 12 topics in the larger run inherited IDs from the prior run.
- Costs nothing — no LLM call to verify.

**Rejected:**
- **No stability:** every nightly clustering run renumbers topics → dashboards break, PM trust evaporates.
- **LLM-based merge:** N² LLM calls per run on edge cases; ambiguous when clusters are genuinely overlapping.
- **Manual curation:** doesn't scale, defeats automation goal.

The 0.85 threshold is the only knob. Lower it for more aggressive identity-keeping (risk of conflating two genuinely different topics that drifted apart); raise it for stricter (risk of churning IDs on minor centroid shifts).

---

## Emergence: per-conversation only (not global)

**This is the one place where the implementation diverges from a literal reading of the assignment** ("20% of users requesting refunds due to X" implies cross-conversation aggregation).

**Chosen because:** different conversations have different contexts. A topic mentioned in 50 software-support conversations is not meaningfully "the same emergence event" as the same words showing up in 50 random Slack chats. Comparing the change *within a single conversation* between its latest ingest batch and its prior batches is the only emergence signal that's both well-defined and PM-useful.

**What this means for the assignment example:** the data is there (`mention_count` on each topic = distinct conversations across the corpus), but the system doesn't currently synthesize the "X% of users requesting refunds because of Y" sentence at corpus level. Closing that gap would be ~30 LOC of additional insight synthesis — adding a percentage calculation and a Bedrock call that combines per-topic stats into a corpus-level narrative. Documented as future work.

---

## Storage split: filesystem for raw, Postgres for structure

**Considered:** All in Postgres (turns in a JSONB or `turns` table), all on S3-compatible object store (MinIO/R2/S3), hybrid (chosen).

**Chosen because:**
- Raw conversation turns are append-only and re-read whenever extraction logic changes — local filesystem is the simplest durable home.
- Structured derived data (chunks, embeddings, topics, runs) benefits from SQL joins and pgvector indexes — Postgres.
- Conversation row carries only the `raw_path`; everything else queryable lives in the relational schema.

**Rejected:**
- **All in Postgres:** at scale, raw turn JSONB bloats the heap and slows row reads of `conversations`. Re-chunking after a config change is a `SELECT raw_text_pointer FROM ...` against external storage in production; embedding that in Postgres trades cheap object storage for expensive database storage.
- **All on S3:** adds a service to docker-compose for no demo value. Documented as the production upgrade — swap filesystem path for an S3 client in `raw_store.py` and nothing else changes.

---

## What's explicitly *not* in the system

- **Sentiment confidence score.** LLMs are bad at calibrated confidence without rubrics. Only the categorical label.
- **Fixed enum on topic labels.** Topics are discovered, not chosen from a list. Aggressive enum-ing defeats the point of clustering.
- **Global "emerging topics" endpoint.** See above — deliberate design call.
- **mypy / pre-commit / CI.** Assignment-scoped; `ruff` + `pytest` are the lint+test surface.
- **Frontend.** Backend-only per assignment. OpenAPI at `/docs` is the GUI.

---

## What would you do differently with a month instead of a weekend?

The current build is correct and runnable, but several pieces were taken at the simplest defensible point. Given a month of focused work, the priorities below are roughly ordered by production-readiness impact.

### 1. Replace arq with a battle-tested queue for high-stress workloads

`arq` is async-native and lightweight, which is great for the assignment. At scale it has limits that production needs to push past:

- **Celery + Redis or Celery + RabbitMQ** for richer routing, priority queues, retry-with-backoff policies, distributed task tracing, and a 10+ year-old operational ecosystem (Flower, Prometheus exporters, alerting integrations).
- **Kafka** as the ingest log if conversations arrive faster than the worker can drain them — durable, replayable, lets us reprocess everything after a chunker config change without losing the source order.
- **Hybrid:** Kafka as the durable ingest tier, Celery/Redis for the ephemeral worker fan-out behind it. Lets ingestion absorb burst traffic (think: a single Slack connector pushing 100K backfilled messages) without coupling burst load to worker capacity.

Two specific shortcomings of the current arq setup that production needs:
- No per-task retry policy — a transient Bedrock timeout currently fails the job rather than retrying with exponential backoff.
- No dead-letter queue — failed labeling jobs sit forever in `jobs` with `status='failed'`; there's no automated recovery loop.

### 2. Per-topic mentions list — actual conversation snippets in the API response

Today, `GET /topics/{id}` returns 5 exemplar snippets across all conversations, and `GET /conversations/{id}/topics` returns 3 snippets per topic. That's enough to verify "the system thinks this is the topic," but not enough to **build user trust**. A PM needs to drill into a topic and read the full set of mentions to decide whether the insight is real.

**Build:**
- `GET /topics/{id}/mentions?limit=N&offset=M` — paginated list of every chunk in the topic with its source `conversation_id`, the surrounding turns from `data/raw/{conversation_id}.json`, the `mention_at` timestamp, and a permalink to the conversation.
- `GET /topics/{id}/mentions?conversation_id=X` — filter to a single conversation's mentions for that topic.
- Inline highlighting: the chunk-level slice the LLM saw, plus 1-2 turns of surrounding context for natural reading.
- Sort options: chronological, by cluster-membership confidence (HDBSCAN `probabilities_` — currently used only for exemplar selection, but worth persisting on the `chunks` row), by sentiment if we add per-chunk sentiment.

This single feature transforms the system from "a black-box clustering output" into "a debuggable evidence ledger." It's the difference between "trust me" and "look for yourself."

### 3. Better embedding models without proportionally higher cost

BGE-small is the right choice for free + CPU + fast. With budget for one inference GPU:

- **`BAAI/bge-large-en-v1.5`** (1024 dim) — meaningfully better MTEB scores, still self-hosted, ~10x latency on CPU but trivial on a small GPU.
- **`nomic-embed-text-v1.5`** (768 dim) — context window up to 8K, lets us embed entire short conversations as single chunks where the topic is unambiguous.
- **`mxbai-embed-large-v1`** — strong MTEB performance, Apache-2.0, often top of leaderboard for English.
- **`BAAI/bge-m3`** — multilingual + 8K context + multi-granularity (dense + sparse + colbert). Pays for itself the moment a non-English conversation arrives.

Then a structured A/B harness: same MultiWOZ + a held-out customer-support corpus, swap the embedder, score clustering coherence (NPMI / topic-coherence-v measure on the labeled clusters) plus retrieval quality (recall@10 against a curated query set). Codify the winner; never pick by vibes.

Side benefit: a proper offline eval lets us tune `min_cluster_size`, `umap_n_neighbors`, `knn_assign_threshold`, and `centroid_match_threshold` as a Pareto problem instead of by hand.

### 4. Cross-conversation topic deduplication + LLM-driven inter-link synthesis

The current system surfaces topics per conversation, and the global view at `/clustering-runs/latest/topics` is just a flat list. In the 500-dialogue run, we saw two pairs of near-duplicate topics (two "conversation closing" clusters, two "contact info" clusters). HDBSCAN found genuinely distinct sub-clusters, but a PM would (correctly) say "these are the same theme."

**Build a new endpoint:** `POST /topics/synthesize` with body `{conversation_ids: [...]}` (or `?since=...` / `?from_runs=[...]` variants).

Pipeline:
1. **Gather** the union of topics touched by chunks in the selected conversations.
2. **Dedup pass:** compute pairwise cosine similarity between topic centroids; cluster topics with cosine ≥ ~0.90 into "topic groups." (Same centroid-matching primitive as run-to-run stability, run on the topic level instead of the cluster level.)
3. **Inter-link pass:** for any topic groups whose centroids are NOT near-duplicate but whose chunks frequently co-occur in the same conversations (Jaccard on conversation_id sets), send the cluster labels + representative snippets to Sonnet 4.6 with a prompt like: *"Are these N topics facets of one broader theme? If yes, produce one cohesive parent topic name + the relationships between the sub-topics."*
4. **Output:** a hierarchical tree — parent themes containing the original topic IDs as children, with explicit relationship notes ("X is the cause; Y is the symptom" or "X and Y both manifest as Z"). Each parent gets its own LLM-generated label, sentiment, and PM insight, but the children remain navigable.

This turns the system into a true **insight synthesis** layer rather than a flat topic list. Crucially, the user selects the scope (single team's conversations, a date range, conversations from one connector) so the dedup + inter-link is meaningful in context, not noise-prone like global cross-everything dedup.

### 5. Observability & operational maturity

- **OpenTelemetry tracing** across the request → enqueue → worker → DB → Bedrock chain. Span on each chunker call, each embedding batch, each Bedrock invoke. Trace IDs flow through `jobs.payload` for end-to-end correlation.
- **Prometheus metrics:** ingest job latency p50/p95/p99, chunks per conversation distribution, k-NN assignment rate, clustering run duration, Bedrock token usage per topic, topic churn (new vs retired) per run.
- **Structured logging** (JSON, with correlation IDs) instead of the current plain-text logs.
- **Health endpoints beyond `/healthz`:** `/readyz` (DB + Redis + Bedrock auth all reachable), `/livez` (process alive but maybe degraded).
- **Worker autoscaling:** scale arq/Celery worker replicas horizontally on queue depth.

### 6. Cluster lifecycle monitoring (Approach D territory)

Today we recluster on a 24h cron or manual trigger. Production needs:
- **Davies-Bouldin + silhouette scoring** of each clustering pass. If quality drops below a threshold, page someone or auto-trigger a refinement run with adjusted params.
- **Drift detection:** monitor the rate of `topic_id = NULL` chunks (the k-NN-unassigned bucket). If that climbs above a threshold, current centroids no longer cover the incoming data — auto-trigger a re-clustering.
- **Targeted re-clustering** instead of full passes: if only one cluster is bloated or drift-detecting, recluster that subspace only, not the entire chunk set. The 2026 research direction in approach D.

### 7. Multi-tenancy + auth

- API keys per organization, scoped to their own `conversations` / `topics` namespace via row-level security.
- Per-tenant rate limits.
- Hierarchical topic namespaces (so two tenants' "refund requests" topics don't collide).
- Audit log of who triggered what (`POST /clustering-runs`, `POST /topics/synthesize`).

### 8. Sentiment trajectory and trend signals

- **Per-chunk sentiment** from a small classifier (DistilBERT-SST2 or similar) — cheap, fast — so we can compute trajectory *within* a conversation (started positive, ended negative).
- **Topic sentiment drift over time:** a topic that was 80% positive last month and 30% positive this month is its own insight worth surfacing automatically.
- **`GET /topics/{id}/sentiment-timeline`** — bucketed daily sentiment for the topic across all its mentions.

### 9. Bedrock cost optimization

- **Bedrock batch API** for cluster labeling when re-running over many topics (e.g., after a chunker config change touches everything). Up to 50% cheaper than synchronous InvokeModel.
- **Label cache** keyed by a hash of the top-N exemplar chunks — if the same exemplars are sent to Bedrock again (often happens when a topic's centroid barely moves), serve the cached label.
- **Prompt caching** if we ever standardize the system prompt to be reusable across many topic-labeling calls (Anthropic's prompt caching: 90% discount on cached prefix tokens). Restructure to put the long instruction block in the system prompt and the variable snippets at the end.

### 10. Storage & data lifecycle

- **S3 / R2 for `data/raw/`** with the same JSON-per-conversation layout — drop-in swap in `app/services/raw_store.py`.
- **Postgres → ClickHouse** for analytical queries (topic counts by day, sentiment distributions). Postgres stays the source of truth; CDC into ClickHouse for OLAP.
- **Hot/cold tier:** active topics in pgvector, retired topics in cheaper storage with a slower query path. A topic retired six months ago doesn't need an HNSW slot.
- **Conversation TTL / archival** so the operational store stays small while history is preserved.

### 11. Testing rigor

- **Integration tests with `testcontainers-python`** spinning real Postgres + pgvector + Redis per test session.
- **Golden tests for clustering output:** lock the cluster IDs and labels for a fixed seed dataset; flag any drift in CI.
- **Property-based tests** (Hypothesis) for the chunker — invariants like "no chunk exceeds 400 tokens" or "every input turn is covered by at least one chunk."
- **Synthetic load tests:** 100K conversations, 5M chunks, measure ingest p99 + clustering wall-clock + memory ceiling. Set a SLO and gate releases on it.
- **Bedrock mock with realistic latency / failure injection** for E2E tests in CI without burning real LLM cost.

### 12. Frontend / dashboard

The OpenAPI `/docs` page is great for engineers but not for PMs. A minimal Next.js dashboard with:
- Topic list view sorted by emergence / mention count / sentiment
- Click into a topic → mentions list (item 2) → click into a mention → full conversation
- Inline LLM-generated "investigate this" follow-ups
- Saved views for recurring queries (e.g., "all negative-sentiment topics with >100 mentions in the last 14 days")

This is where the "PM trust" promise is actually delivered.

# Sentiment Analytics Engine

A backend service that ingests multi-party conversations, discovers emerging topics via embedding-based clustering, and surfaces PM-actionable insights.

## What it does

- Accepts conversations of any size — from a two-message exchange to a 10K-message Slack channel — through a single ingest endpoint, with append semantics for ongoing conversations.
- Chunks conversations using semantic + recursive sliding-window strategies tuned to the embedding model's context window.
- Embeds chunks with `BAAI/bge-small-en-v1.5`, stored in Postgres + pgvector.
- Clusters chunks via UMAP + HDBSCAN, maintaining stable topic IDs across re-runs via centroid matching.
- Labels each cluster with Claude Sonnet 4.6 (via AWS Bedrock) — topic name, sentiment, PM insight, supporting snippets.
- Computes per-conversation emerging topics by comparing topic distribution in the latest ingest batch against prior batches of the same conversation.

## Architecture overview

```
                                                                    ┌─────────────┐
POST /conversations ──► (raw → fs) ──► arq job ──► chunk ──► embed ─┤             │
                                                                    │  pgvector   │
                                                                    │             │
                          POST /clustering-runs ──► arq job ────────┤  UMAP +     │
                                                                    │  HDBSCAN +  │
                                                                    │  centroid   │
                                                                    │  matching   │
                                                                    └──────┬──────┘
                                                                           │
                                                            Bedrock        │
                                                            (Claude        ▼
                                                            Sonnet 4.6) ◄─ label, sentiment, insight
                                                                           │
GET /conversations/{id}/topics       ◄────────────────────────────────────┤
GET /clustering-runs/latest/topics   ◄────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full design rationale and [docs/data-flow.md](docs/data-flow.md) for sequence diagrams.

## Quickstart

```bash
cp .env.example .env
# fill in AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY for Bedrock,
# or set MOCK_LLM=true to skip labeling during dev.

docker compose up --build
```

Once healthy, ingest the MultiWOZ seed dataset:

```bash
docker compose exec api python -m scripts.seed_multiwoz --limit 500
```

Trigger a clustering run and inspect topics:

```bash
curl -X POST http://localhost:8000/clustering-runs
curl http://localhost:8000/clustering-runs/latest/topics
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/conversations` | Create new conversation or append to existing (async; returns 202 + job) |
| `GET` | `/conversations` | Paginated list |
| `GET` | `/conversations/{id}` | Conversation metadata |
| `GET` | `/conversations/{id}/topics` | Topics this conversation contributes to, sorted by emergence |
| `POST` | `/clustering-runs` | Trigger full re-clustering |
| `GET` | `/clustering-runs` | Run history |
| `GET` | `/clustering-runs/{id}` | Single run status |
| `GET` | `/clustering-runs/{id}/topics` | Topics in this run, sorted by mention count |
| `GET` | `/clustering-runs/latest/topics` | Current global topic snapshot |
| `GET` | `/topics/{id}` | Single topic detail |
| `GET` | `/jobs/{id}` | Background job poll |
| `GET` | `/healthz` | Liveness |

Full OpenAPI schema at `http://localhost:8000/docs` once running.

## Running tests

```bash
docker compose run --rm -e MOCK_LLM=true api pytest
```

## Configuration

All tunables exposed via environment variables — see [`.env.example`](.env.example). Notable thresholds:

- `KNN_ASSIGN_THRESHOLD` (default `0.75`) — cosine cutoff for assigning a new chunk to an existing topic at ingest time
- `CENTROID_MATCH_THRESHOLD` (default `0.85`) — cosine cutoff for inheriting a topic ID across clustering runs
- `EMERGENCE_RATIO_THRESHOLD` (default `2.0`) — `new_count / old_count` cutoff for the per-conversation emerging-topic flag
- HDBSCAN / UMAP parameters tuned for chunked-conversation embeddings (~6K–10K chunks for a 2K-conversation dataset)

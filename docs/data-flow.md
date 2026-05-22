# Data flow

Sequence diagrams for the three primary flows. Rendered via Mermaid — paste any block into [mermaid.live](https://mermaid.live) for a visual view, or view this file in a Mermaid-aware Markdown renderer.

## 1. Ingest (new or append)

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as FastAPI (/conversations)
    participant FS as data/raw/{id}.json
    participant DB as Postgres
    participant Q as arq queue
    participant W as Worker
    participant Emb as BGE-small (in-process)

    Client->>API: POST /conversations<br/>{conversation_id?, turns[], metadata?}
    alt conversation_id absent
        API->>API: generate UUID, mode = "created"<br/>batch_seq = 1<br/>start_turn_index = 0
    else conversation_id present + exists
        API->>DB: SELECT FOR UPDATE conversations WHERE id = ?
        API->>API: mode = "appended"<br/>batch_seq = latest_batch_seq + 1<br/>start_turn_index = turn_count
    else conversation_id present + missing
        API-->>Client: 404 Conversation not found
    end
    API->>FS: write (new) or append turns
    API->>DB: INSERT/UPDATE conversations<br/>INSERT jobs (status=queued)
    API->>Q: enqueue process_ingest(job_id, ...)
    API-->>Client: 202 + {conversation_id, job_id, mode, batch_seq, turn_count}<br/>Location: /jobs/{job_id}

    W->>DB: mark_started(job)
    W->>FS: read turns[start : start+N]
    opt batch_seq > 1
        W->>DB: SELECT latest chunk content for overlap_prefix
    end
    W->>W: SemanticChunker.split_text → list[chunk_text]<br/>RecursiveSplitter fallback for oversized<br/>map chunks back to (position, mention_at)
    W->>Emb: encode(chunk_texts) → embeddings
    W->>DB: INSERT chunks rows (batch_seq, position, embedding, ...)
    loop for each new chunk
        W->>DB: SELECT topic ORDER BY centroid <=> ? LIMIT 1
        alt similarity >= 0.75
            W->>DB: UPDATE chunks SET topic_id = ?
        else below threshold
            W->>W: leave topic_id NULL (awaits next clustering run)
        end
    end
    W->>DB: recompute mention_count for affected topics
    W->>DB: mark_completed(job, result)

    Client->>API: GET /jobs/{job_id}
    API-->>Client: {status: completed, result: {chunks_created, chunks_assigned, ...}}
```

## 2. Full clustering run

```mermaid
sequenceDiagram
    autonumber
    participant Trigger as POST /clustering-runs<br/>or cron
    participant API as FastAPI
    participant DB as Postgres
    participant Q as arq queue
    participant W as Worker
    participant BR as AWS Bedrock<br/>(Claude Sonnet 4.6)

    Trigger->>API: trigger
    API->>DB: INSERT clustering_runs (status=pending)<br/>INSERT jobs (type=run_clustering, status=queued)
    API->>Q: enqueue run_clustering(job_id, run_id)
    API-->>Trigger: 202 + {clustering_run_id, job_id}

    W->>DB: clustering_runs.status = running<br/>params = current thresholds
    W->>DB: SELECT id, embedding FROM chunks ORDER BY created_at
    Note over W: in-memory: chunk_ids[], X (N×384)

    W->>W: UMAP fit_transform → X_reduced (N×5)
    W->>W: HDBSCAN.fit_predict → labels[], probs[]
    W->>W: for each cluster:<br/>centroid = mean(X[mask]) normalized<br/>exemplars = top-N by probability

    W->>DB: SELECT * FROM topics WHERE status=active
    W->>W: for each new cluster:<br/>greedy match against active topics<br/>by cosine ≥ 0.85

    loop for each new cluster
        alt matched existing topic
            W->>DB: UPDATE topics SET centroid, last_updated_at WHERE id = matched_id
        else unmatched
            W->>DB: INSERT topics (centroid, status=active)
        end
        W->>DB: UPDATE chunks SET topic_id = ?<br/>WHERE id IN (member_chunk_ids)
    end
    W->>DB: UPDATE chunks SET topic_id = NULL WHERE id IN (noise_chunk_ids)
    W->>DB: UPDATE topics SET status='retired'<br/>WHERE id IN (unmatched_active_topics)
    W->>DB: recompute mention_count for all affected topics
    W->>DB: INSERT clustering_run_topics (run_id, topic_id, mention_count_at_run, was_new)
    W->>DB: clustering_runs.status = completed<br/>n_topics_found, n_topics_new, n_topics_retired
    W->>Q: enqueue label_topic(job_id, topic_id) for each affected topic

    par per topic
        W->>DB: load topic + top-20 chunks nearest centroid
        W->>BR: InvokeModel(claude-sonnet-4-6, JSON-prefilled "{")
        BR-->>W: {topic, sentiment, pm_insight, supporting_snippets}
        W->>DB: UPDATE topics SET label, sentiment, pm_insight, last_labeled_at
    end
```

## 3. Per-conversation emergence query

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as FastAPI<br/>/conversations/{id}/topics
    participant DB as Postgres

    Client->>API: GET /conversations/{id}/topics?sort=emergence

    API->>DB: SELECT * FROM conversations WHERE id = ?
    alt missing
        API-->>Client: 404
    end

    API->>DB: SELECT MAX(batch_seq) FROM chunks WHERE conversation_id = ?
    Note over API: latest_batch = N<br/>is_fresh = (N == 1)

    API->>DB: SELECT topic_id,<br/>SUM(CASE WHEN batch_seq=N THEN 1 ELSE 0) AS new_count,<br/>SUM(CASE WHEN batch_seq<N THEN 1 ELSE 0) AS old_count<br/>FROM chunks WHERE conversation_id=? AND topic_id IS NOT NULL<br/>GROUP BY topic_id

    loop per topic row
        API->>API: score_topic(new_count, old_count, is_fresh, thresholds)<br/>→ (emergence_score, is_emerging)
    end

    API->>DB: SELECT * FROM topics WHERE id IN (?)
    loop per topic
        API->>DB: SELECT content FROM chunks<br/>WHERE conversation_id=? AND topic_id=?<br/>ORDER BY embedding <=> topic.centroid LIMIT 3
    end

    API->>API: sort: emergence (default) or mentions
    API-->>Client: 200 + TopicWithEmergence[]
```

-- Sentiment Analytics Engine — initial schema
-- Loaded once by the postgres container on first boot.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- conversations: lightweight metadata. Raw turns live in data/raw/{id}.json.
CREATE TABLE IF NOT EXISTS conversations (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_path         TEXT        NOT NULL,
    metadata         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    latest_batch_seq INT         NOT NULL DEFAULT 0,
    turn_count       INT         NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_started_at ON conversations (started_at DESC);

DROP TRIGGER IF EXISTS conversations_set_updated_at ON conversations;
CREATE TRIGGER conversations_set_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- topics: globally persistent clusters. cluster_id stays stable across runs via centroid matching.
CREATE TABLE IF NOT EXISTS topics (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label           TEXT,
    sentiment       TEXT CHECK (sentiment IS NULL OR sentiment IN ('positive', 'negative', 'neutral', 'mixed')),
    pm_insight      TEXT,
    centroid        vector(384),
    mention_count   INT         NOT NULL DEFAULT 0,
    status          TEXT        NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'merged', 'retired')),
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_labeled_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_topics_status         ON topics (status);
CREATE INDEX IF NOT EXISTS idx_topics_mention_count  ON topics (mention_count DESC);
CREATE INDEX IF NOT EXISTS idx_topics_centroid_hnsw  ON topics USING hnsw (centroid vector_cosine_ops);


-- chunks: embedded analyzable units. One conversation → many chunks.
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    content         TEXT        NOT NULL,
    token_count     INT         NOT NULL,
    embedding       vector(384) NOT NULL,
    topic_id        UUID        REFERENCES topics(id) ON DELETE SET NULL,
    batch_seq       INT         NOT NULL,
    position        INT         NOT NULL,
    mention_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_conversation_id     ON chunks (conversation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_topic_id            ON chunks (topic_id);
CREATE INDEX IF NOT EXISTS idx_chunks_convo_batch         ON chunks (conversation_id, batch_seq);
CREATE INDEX IF NOT EXISTS idx_chunks_convo_position      ON chunks (conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_chunks_mention_at          ON chunks (mention_at);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw      ON chunks USING hnsw (embedding vector_cosine_ops);


-- clustering_runs: history of full HDBSCAN passes.
CREATE TABLE IF NOT EXISTS clustering_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status              TEXT        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    n_topics_found      INT,
    n_topics_new        INT,
    n_topics_retired    INT,
    n_chunks_processed  INT,
    params              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_clustering_runs_started_at ON clustering_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_clustering_runs_status     ON clustering_runs (status);


-- clustering_run_topics: per-run snapshot of topics that participated in a clustering pass.
-- Lets /clustering-runs/{id}/topics return historically-accurate results even after
-- later runs retire or update those topics.
CREATE TABLE IF NOT EXISTS clustering_run_topics (
    clustering_run_id     UUID    NOT NULL REFERENCES clustering_runs(id) ON DELETE CASCADE,
    topic_id              UUID    NOT NULL REFERENCES topics(id)          ON DELETE CASCADE,
    mention_count_at_run  INT     NOT NULL DEFAULT 0,
    was_new               BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (clustering_run_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_crt_run_id    ON clustering_run_topics (clustering_run_id);
CREATE INDEX IF NOT EXISTS idx_crt_topic_id  ON clustering_run_topics (topic_id);


-- jobs: generic background task tracking. Bridges arq job state to API consumers.
CREATE TABLE IF NOT EXISTS jobs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type         TEXT        NOT NULL,
    status       TEXT        NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    payload      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    result       JSONB,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_jobs_status      ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_type        ON jobs (type);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at  ON jobs (created_at DESC);

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from arq import cron
from arq.connections import RedisSettings

from app.config import get_settings
from app.db.session import session_scope
from app.workers import job_tracker

log = logging.getLogger(__name__)
_settings = get_settings()


# ---------- task functions ----------

async def process_ingest(
    ctx: dict[str, Any],
    job_id: str,
    conversation_id: str,
    batch_seq: int,
    start_turn_index: int,
    n_new_turns: int,
) -> dict[str, Any]:
    """Chunk + embed new turns, assign topics via k-NN, compute emergence."""
    from app.services.ingest_pipeline import run_ingest_pipeline

    async with session_scope() as session:
        await job_tracker.mark_started(session, job_id)

    try:
        result = await run_ingest_pipeline(
            conversation_id=uuid.UUID(conversation_id),
            batch_seq=batch_seq,
            start_turn_index=start_turn_index,
            n_new_turns=n_new_turns,
            embedder=ctx.get("embedder"),
        )
    except Exception as e:
        log.exception("process_ingest failed")
        async with session_scope() as session:
            await job_tracker.mark_failed(session, job_id, str(e))
        raise

    async with session_scope() as session:
        await job_tracker.mark_completed(session, job_id, result)
    return result


async def run_clustering(
    ctx: dict[str, Any],
    job_id: str,
    clustering_run_id: str,
) -> dict[str, Any]:
    """Run UMAP + HDBSCAN over all chunks, match centroids, label new topics."""
    from app.services.clusterer import run_clustering_pipeline

    async with session_scope() as session:
        await job_tracker.mark_started(session, job_id)

    try:
        result = await run_clustering_pipeline(
            clustering_run_id=uuid.UUID(clustering_run_id),
            embedder=ctx.get("embedder"),
        )
    except Exception as e:
        log.exception("run_clustering failed")
        async with session_scope() as session:
            await job_tracker.mark_failed(session, job_id, str(e))
        raise

    async with session_scope() as session:
        await job_tracker.mark_completed(session, job_id, result)
    return result


async def label_topic(ctx: dict[str, Any], job_id: str, topic_id: str) -> dict[str, Any]:
    """Call Bedrock to label a single topic."""
    from app.services.labeler import label_topic_via_bedrock

    async with session_scope() as session:
        await job_tracker.mark_started(session, job_id)

    try:
        result = await label_topic_via_bedrock(topic_id=uuid.UUID(topic_id))
    except Exception as e:
        log.exception("label_topic failed")
        async with session_scope() as session:
            await job_tracker.mark_failed(session, job_id, str(e))
        raise

    async with session_scope() as session:
        await job_tracker.mark_completed(session, job_id, result)
    return result


# ---------- scheduled clustering cron ----------

async def scheduled_clustering(ctx: dict[str, Any]) -> None:
    """Cron-triggered clustering: creates a clustering_run row and enqueues run_clustering."""
    from app.db.models import ClusteringRun, Job
    from app.services.queue import enqueue_clustering, make_payload

    async with session_scope() as session:
        run = ClusteringRun(status="pending", params={"trigger": "cron"})
        session.add(run)
        await session.flush()

        job = Job(
            type="run_clustering",
            status="queued",
            payload=make_payload(clustering_run_id=run.id, trigger="cron"),
        )
        session.add(job)
        await session.flush()
        job_id = job.id
        run_id = run.id

    pool = ctx["redis"]
    await enqueue_clustering(pool=pool, job_id=job_id, clustering_run_id=run_id)
    log.info("Scheduled clustering run %s enqueued (job %s)", run_id, job_id)


# ---------- worker lifecycle ----------

async def on_startup(ctx: dict[str, Any]) -> None:
    log.info("Worker starting — loading embedding model %s", _settings.embedding_model)
    from app.services.embedder import get_embedder

    ctx["embedder"] = get_embedder()
    ctx["started_at"] = datetime.now(timezone.utc).isoformat()
    log.info("Worker ready")


async def on_shutdown(ctx: dict[str, Any]) -> None:
    log.info("Worker shutting down")


class WorkerSettings:
    """arq Worker entry point.

    Run with: arq app.workers.arq_worker.WorkerSettings
    """

    functions = [process_ingest, run_clustering, label_topic]
    on_startup = on_startup
    on_shutdown = on_shutdown
    redis_settings = RedisSettings.from_dsn(_settings.redis_url)
    max_jobs = 4
    job_timeout = 60 * 30  # 30 min cap

    cron_jobs = [
        cron(
            scheduled_clustering,
            hour=set(range(0, 24, max(_settings.clustering_cron_hours, 1))),
            minute=0,
            run_at_startup=False,
        ),
    ]

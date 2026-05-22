from __future__ import annotations

import uuid
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config import get_settings


def _redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


async def get_arq_pool() -> ArqRedis:
    return await create_pool(_redis_settings())


async def enqueue_ingest(
    pool: ArqRedis,
    job_id: uuid.UUID,
    conversation_id: uuid.UUID,
    batch_seq: int,
    start_turn_index: int,
    n_new_turns: int,
) -> None:
    await pool.enqueue_job(
        "process_ingest",
        str(job_id),
        str(conversation_id),
        batch_seq,
        start_turn_index,
        n_new_turns,
        _job_id=str(job_id),
    )


async def enqueue_clustering(
    pool: ArqRedis,
    job_id: uuid.UUID,
    clustering_run_id: uuid.UUID,
) -> None:
    await pool.enqueue_job(
        "run_clustering",
        str(job_id),
        str(clustering_run_id),
        _job_id=str(job_id),
    )


async def enqueue_label_topic(
    pool: ArqRedis,
    job_id: uuid.UUID,
    topic_id: uuid.UUID,
) -> None:
    await pool.enqueue_job(
        "label_topic",
        str(job_id),
        str(topic_id),
        _job_id=str(job_id),
    )


def make_payload(**kwargs: Any) -> dict[str, Any]:
    return {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in kwargs.items()}

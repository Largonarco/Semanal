from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Job

log = logging.getLogger(__name__)


async def mark_started(session: AsyncSession, job_id: uuid.UUID | str) -> None:
    job_uuid = uuid.UUID(str(job_id))
    job = await session.get(Job, job_uuid)
    if job is None:
        log.warning("Job row not found at mark_started: %s", job_id)
        return
    job.status = "running"
    job.started_at = datetime.now(timezone.utc)


async def mark_completed(
    session: AsyncSession, job_id: uuid.UUID | str, result: dict[str, Any] | None = None
) -> None:
    job_uuid = uuid.UUID(str(job_id))
    job = await session.get(Job, job_uuid)
    if job is None:
        log.warning("Job row not found at mark_completed: %s", job_id)
        return
    job.status = "completed"
    job.finished_at = datetime.now(timezone.utc)
    if result is not None:
        job.result = result


async def mark_failed(
    session: AsyncSession, job_id: uuid.UUID | str, error: str
) -> None:
    job_uuid = uuid.UUID(str(job_id))
    job = await session.get(Job, job_uuid)
    if job is None:
        log.warning("Job row not found at mark_failed: %s", job_id)
        return
    job.status = "failed"
    job.finished_at = datetime.now(timezone.utc)
    job.error = error

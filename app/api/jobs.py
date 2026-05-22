from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import APIError
from app.db.models import Job
from app.db.session import get_session
from app.schemas.job import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    job = await session.get(Job, job_id)
    if job is None:
        raise APIError(
            status_code=404,
            title="Job not found",
            detail=f"No job with id {job_id}",
        )
    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        payload=job.payload,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )

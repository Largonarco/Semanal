from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Path, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import APIError
from app.api.topics import _topic_summary_fields
from app.db.models import ClusteringRun, Job, Topic
from app.db.session import get_session
from app.schemas.clustering_run import (
    ClusteringRunList,
    ClusteringRunResponse,
    ClusteringRunTriggerResponse,
)
from app.schemas.topic import TopicInRun, TopicInRunList
from app.services.queue import enqueue_clustering, get_arq_pool, make_payload
from app.services.topic_queries import fetch_latest_completed_run, fetch_run_topic_snapshots

router = APIRouter(prefix="/clustering-runs", tags=["clustering-runs"])


def _to_run_response(run: ClusteringRun) -> ClusteringRunResponse:
    return ClusteringRunResponse(
        id=run.id,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        n_topics_found=run.n_topics_found,
        n_topics_new=run.n_topics_new,
        n_topics_retired=run.n_topics_retired,
        n_chunks_processed=run.n_chunks_processed,
        params=run.params,
        error=run.error,
    )


@router.post(
    "",
    response_model=ClusteringRunTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_clustering_run(
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> ClusteringRunTriggerResponse:
    run = ClusteringRun(status="pending", params={"trigger": "manual"})
    session.add(run)
    await session.flush()

    job = Job(
        type="run_clustering",
        status="queued",
        payload=make_payload(clustering_run_id=run.id, trigger="manual"),
    )
    session.add(job)
    await session.commit()

    pool = await get_arq_pool()
    try:
        await enqueue_clustering(pool=pool, job_id=job.id, clustering_run_id=run.id)
    finally:
        await pool.close(close_connection_pool=True)

    response.headers["Location"] = f"/clustering-runs/{run.id}"
    return ClusteringRunTriggerResponse(clustering_run_id=run.id, job_id=job.id)


@router.get("", response_model=ClusteringRunList)
async def list_clustering_runs(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> ClusteringRunList:
    items_stmt = (
        select(ClusteringRun)
        .order_by(ClusteringRun.started_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = list((await session.execute(items_stmt)).scalars().all())
    total = (await session.execute(select(func.count(ClusteringRun.id)))).scalar_one()
    return ClusteringRunList(
        items=[_to_run_response(r) for r in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/latest", response_model=ClusteringRunResponse)
async def get_latest_clustering_run(
    session: AsyncSession = Depends(get_session),
) -> ClusteringRunResponse:
    run = await fetch_latest_completed_run(session)
    if run is None:
        raise APIError(
            status_code=404,
            title="No clustering runs",
            detail="No completed clustering run yet — POST /clustering-runs to trigger one.",
        )
    return _to_run_response(run)


@router.get("/latest/topics", response_model=TopicInRunList)
async def get_latest_run_topics(
    sort: Literal["emergence", "mentions"] = Query(default="mentions"),
    session: AsyncSession = Depends(get_session),
) -> TopicInRunList:
    run = await fetch_latest_completed_run(session)
    if run is None:
        raise APIError(
            status_code=404,
            title="No completed clustering runs",
            detail="No completed clustering run yet — POST /clustering-runs to trigger one.",
        )
    return await _build_run_topics_response(session, run, sort)


@router.get("/{clustering_run_id}", response_model=ClusteringRunResponse)
async def get_clustering_run(
    clustering_run_id: uuid.UUID = Path(...),
    session: AsyncSession = Depends(get_session),
) -> ClusteringRunResponse:
    run = await session.get(ClusteringRun, clustering_run_id)
    if run is None:
        raise APIError(
            status_code=404,
            title="Clustering run not found",
            detail=f"No clustering run with id {clustering_run_id}",
        )
    return _to_run_response(run)


@router.get("/{clustering_run_id}/topics", response_model=TopicInRunList)
async def get_run_topics(
    clustering_run_id: uuid.UUID = Path(...),
    sort: Literal["emergence", "mentions"] = Query(default="mentions"),
    session: AsyncSession = Depends(get_session),
) -> TopicInRunList:
    run = await session.get(ClusteringRun, clustering_run_id)
    if run is None:
        raise APIError(
            status_code=404,
            title="Clustering run not found",
            detail=f"No clustering run with id {clustering_run_id}",
        )
    return await _build_run_topics_response(session, run, sort)


async def _build_run_topics_response(
    session: AsyncSession, run: ClusteringRun, sort: str
) -> TopicInRunList:
    snapshots = await fetch_run_topic_snapshots(session, run.id)
    if not snapshots:
        return TopicInRunList(
            clustering_run_id=run.id, run_finished_at=run.finished_at, items=[]
        )

    topic_ids = [s.topic_id for s in snapshots]
    topics_stmt = select(Topic).where(Topic.id.in_(topic_ids))
    topics = {t.id: t for t in (await session.execute(topics_stmt)).scalars().all()}

    items: list[TopicInRun] = []
    for snap in snapshots:
        t = topics.get(snap.topic_id)
        if t is None:
            continue
        items.append(
            TopicInRun(
                **_topic_summary_fields(t),
                mention_count_at_run=snap.mention_count_at_run,
                was_new=snap.was_new,
            )
        )

    items.sort(key=lambda x: x.mention_count_at_run, reverse=True)
    return TopicInRunList(
        clustering_run_id=run.id, run_finished_at=run.finished_at, items=items
    )

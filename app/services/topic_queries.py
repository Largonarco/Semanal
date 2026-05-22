from __future__ import annotations

import uuid

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, ClusteringRun, ClusteringRunTopic, Topic


async def fetch_topics_by_ids(
    session: AsyncSession, topic_ids: list[uuid.UUID]
) -> dict[uuid.UUID, Topic]:
    if not topic_ids:
        return {}
    stmt = select(Topic).where(Topic.id.in_(topic_ids))
    topics = (await session.execute(stmt)).scalars().all()
    return {t.id: t for t in topics}


async def fetch_snippets_for_conversation_topic(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    topic_id: uuid.UUID,
    n: int = 3,
) -> list[str]:
    topic = await session.get(Topic, topic_id)
    if topic is None or topic.centroid is None:
        return []
    stmt = (
        select(Chunk.content)
        .where(Chunk.conversation_id == conversation_id)
        .where(Chunk.topic_id == topic_id)
        .order_by(Chunk.embedding.cosine_distance(topic.centroid))
        .limit(n)
    )
    return [r.content for r in (await session.execute(stmt)).all()]


async def fetch_exemplar_snippets_global(
    session: AsyncSession, topic_id: uuid.UUID, n: int = 5
) -> list[str]:
    topic = await session.get(Topic, topic_id)
    if topic is None or topic.centroid is None:
        return []
    stmt = (
        select(Chunk.content)
        .where(Chunk.topic_id == topic_id)
        .order_by(Chunk.embedding.cosine_distance(topic.centroid))
        .limit(n)
    )
    return [r.content for r in (await session.execute(stmt)).all()]


async def fetch_latest_completed_run(session: AsyncSession) -> ClusteringRun | None:
    stmt = (
        select(ClusteringRun)
        .where(ClusteringRun.status == "completed")
        .order_by(desc(ClusteringRun.finished_at))
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def fetch_run_topic_snapshots(
    session: AsyncSession, clustering_run_id: uuid.UUID
) -> list[ClusteringRunTopic]:
    stmt = (
        select(ClusteringRunTopic)
        .where(ClusteringRunTopic.clustering_run_id == clustering_run_id)
    )
    return list((await session.execute(stmt)).scalars().all())

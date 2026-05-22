from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, Topic

log = logging.getLogger(__name__)


async def assign_chunks_to_topics(
    session: AsyncSession,
    chunks: list[Chunk],
    threshold: float | None = None,
) -> dict[uuid.UUID, int]:
    """For each chunk without a topic, find the nearest active topic by cosine.

    Assigns chunk.topic_id when similarity >= threshold. Returns the set of
    topics that gained at least one new chunk in this batch (caller is
    expected to recompute mention_count for those topics afterward).
    """
    settings = get_settings()
    threshold = settings.knn_assign_threshold if threshold is None else threshold
    assigned: dict[uuid.UUID, int] = {}

    for chunk in chunks:
        if chunk.topic_id is not None:
            continue

        stmt = (
            select(
                Topic.id.label("topic_id"),
                Topic.centroid.cosine_distance(chunk.embedding).label("distance"),
            )
            .where(Topic.status == "active")
            .where(Topic.centroid.is_not(None))
            .order_by(Topic.centroid.cosine_distance(chunk.embedding))
            .limit(1)
        )
        row = (await session.execute(stmt)).first()
        if row is None:
            continue
        similarity = 1.0 - float(row.distance)
        if similarity >= threshold:
            chunk.topic_id = row.topic_id
            assigned[row.topic_id] = assigned.get(row.topic_id, 0) + 1

    return assigned


async def recompute_mention_count(
    session: AsyncSession, topic_id: uuid.UUID
) -> int:
    """Set topics.mention_count = COUNT(DISTINCT conversation_id) for this topic."""
    count_stmt = (
        select(func.count(func.distinct(Chunk.conversation_id)))
        .where(Chunk.topic_id == topic_id)
    )
    count = (await session.execute(count_stmt)).scalar_one()
    await session.execute(
        update(Topic)
        .where(Topic.id == topic_id)
        .values(mention_count=count, last_updated_at=datetime.now(timezone.utc))
    )
    return count


async def recompute_mention_counts(
    session: AsyncSession, topic_ids: set[uuid.UUID] | list[uuid.UUID]
) -> None:
    for tid in set(topic_ids):
        await recompute_mention_count(session, tid)

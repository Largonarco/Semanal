from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk


@dataclass
class TopicEmergence:
    topic_id: uuid.UUID
    new_count: int
    old_count: int
    mention_count_in_conversation: int
    emergence_score: float
    is_emerging: bool


def score_topic(
    new_count: int,
    old_count: int,
    is_fresh_conversation: bool,
    fresh_min_count: int,
    ratio_threshold: float,
) -> tuple[float, bool]:
    """Pure helper for emergence formula (testable without DB).

    Returns (emergence_score, is_emerging).
    """
    if new_count == 0:
        return 0.0, False
    if is_fresh_conversation:
        return float(new_count), new_count >= fresh_min_count
    score = new_count / max(old_count, 1)
    return score, score >= ratio_threshold


async def compute_emergence(
    session: AsyncSession, conversation_id: uuid.UUID
) -> list[TopicEmergence]:
    """Compute per-topic emergence stats for a single conversation.

    Compares chunks in the latest batch_seq against all prior batches of the
    same conversation. Returns one entry per distinct topic touched by this
    conversation (assigned chunks only). Topics that only appear in prior
    batches are included with `is_emerging=False`, `emergence_score=0`.
    """
    settings = get_settings()

    latest_stmt = (
        select(func.max(Chunk.batch_seq))
        .where(Chunk.conversation_id == conversation_id)
    )
    latest_batch = (await session.execute(latest_stmt)).scalar()
    if latest_batch is None:
        return []

    is_fresh = latest_batch == 1

    new_count_expr = func.sum(
        case((Chunk.batch_seq == latest_batch, 1), else_=0)
    ).label("new_count")
    old_count_expr = func.sum(
        case((Chunk.batch_seq < latest_batch, 1), else_=0)
    ).label("old_count")

    stmt = (
        select(Chunk.topic_id, new_count_expr, old_count_expr)
        .where(Chunk.conversation_id == conversation_id)
        .where(Chunk.topic_id.is_not(None))
        .group_by(Chunk.topic_id)
    )
    rows = (await session.execute(stmt)).all()

    out: list[TopicEmergence] = []
    for row in rows:
        new_count = int(row.new_count or 0)
        old_count = int(row.old_count or 0)

        score, emerging = score_topic(
            new_count=new_count,
            old_count=old_count,
            is_fresh_conversation=is_fresh,
            fresh_min_count=settings.emergence_fresh_min_count,
            ratio_threshold=settings.emergence_ratio_threshold,
        )

        out.append(
            TopicEmergence(
                topic_id=row.topic_id,
                new_count=new_count,
                old_count=old_count,
                mention_count_in_conversation=new_count + old_count,
                emergence_score=score,
                is_emerging=emerging,
            )
        )

    return out

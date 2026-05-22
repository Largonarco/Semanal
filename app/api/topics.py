from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import APIError
from app.db.models import Topic
from app.db.session import get_session
from app.schemas.topic import TopicDetail
from app.services.topic_queries import fetch_exemplar_snippets_global

router = APIRouter(prefix="/topics", tags=["topics"])


def _topic_summary_fields(t: Topic) -> dict:
    return {
        "id": t.id,
        "label": t.label,
        "sentiment": t.sentiment,
        "pm_insight": t.pm_insight,
        "status": t.status,
        "mention_count": t.mention_count,
        "first_seen_at": t.first_seen_at,
        "last_updated_at": t.last_updated_at,
        "last_labeled_at": t.last_labeled_at,
    }


@router.get("/{topic_id}", response_model=TopicDetail)
async def get_topic(
    topic_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> TopicDetail:
    topic = await session.get(Topic, topic_id)
    if topic is None:
        raise APIError(
            status_code=404,
            title="Topic not found",
            detail=f"No topic with id {topic_id}",
        )
    snippets = await fetch_exemplar_snippets_global(session, topic_id, n=5)
    return TopicDetail(**_topic_summary_fields(topic), exemplar_snippets=snippets)

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import APIError
from app.api.topics import _topic_summary_fields
from app.db.models import Conversation
from app.db.session import get_session
from app.schemas.conversation import (
    ConversationDetail,
    ConversationIngestRequest,
    ConversationIngestResponse,
    ConversationList,
)
from app.schemas.topic import TopicWithEmergence, TopicWithEmergenceList
from app.services import ingest as ingest_service
from app.services.emergence import compute_emergence
from app.services.queue import enqueue_ingest, get_arq_pool
from app.services.topic_queries import (
    fetch_snippets_for_conversation_topic,
    fetch_topics_by_ids,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _to_detail(c: Conversation) -> ConversationDetail:
    return ConversationDetail(
        id=c.id,
        started_at=c.started_at,
        latest_batch_seq=c.latest_batch_seq,
        turn_count=c.turn_count,
        metadata=c.extra_metadata,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


@router.post(
    "",
    response_model=ConversationIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_conversation(
    payload: ConversationIngestRequest,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> ConversationIngestResponse:
    try:
        convo, job, mode, _, _ = await ingest_service.ingest_conversation(
            session=session,
            conversation_id=payload.conversation_id,
            turns=payload.turns,
            metadata=payload.metadata,
        )
        await session.commit()
    except APIError:
        await session.rollback()
        raise
    except Exception:
        await session.rollback()
        raise

    pool = await get_arq_pool()
    try:
        await enqueue_ingest(
            pool=pool,
            job_id=job.id,
            conversation_id=convo.id,
            batch_seq=convo.latest_batch_seq,
            start_turn_index=job.payload["start_turn_index"],
            n_new_turns=job.payload["n_new_turns"],
        )
    finally:
        await pool.close(close_connection_pool=True)

    response.headers["Location"] = f"/jobs/{job.id}"
    return ConversationIngestResponse(
        conversation_id=convo.id,
        job_id=job.id,
        mode=mode,  # type: ignore[arg-type]
        batch_seq=convo.latest_batch_seq,
        turn_count=convo.turn_count,
    )


@router.get("", response_model=ConversationList)
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> ConversationList:
    items, total = await ingest_service.list_conversations(session, limit, offset)
    return ConversationList(
        items=[_to_detail(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> ConversationDetail:
    convo = await session.get(Conversation, conversation_id)
    if convo is None:
        raise APIError(
            status_code=404,
            title="Conversation not found",
            detail=f"No conversation with id {conversation_id}",
        )
    return _to_detail(convo)


@router.get("/{conversation_id}/topics", response_model=TopicWithEmergenceList)
async def get_conversation_topics(
    conversation_id: uuid.UUID,
    sort: Literal["emergence", "mentions"] = Query(default="emergence"),
    session: AsyncSession = Depends(get_session),
) -> TopicWithEmergenceList:
    convo = await session.get(Conversation, conversation_id)
    if convo is None:
        raise APIError(
            status_code=404,
            title="Conversation not found",
            detail=f"No conversation with id {conversation_id}",
        )

    emergence_rows = await compute_emergence(session, conversation_id)
    topic_map = await fetch_topics_by_ids(
        session, [e.topic_id for e in emergence_rows]
    )

    items: list[TopicWithEmergence] = []
    for e in emergence_rows:
        topic = topic_map.get(e.topic_id)
        if topic is None:
            continue
        snippets = await fetch_snippets_for_conversation_topic(
            session, conversation_id, e.topic_id, n=3
        )
        items.append(
            TopicWithEmergence(
                **_topic_summary_fields(topic),
                new_count=e.new_count,
                old_count=e.old_count,
                mention_count_in_conversation=e.mention_count_in_conversation,
                emergence_score=e.emergence_score,
                is_emerging=e.is_emerging,
                snippets_in_conversation=snippets,
            )
        )

    if sort == "mentions":
        items.sort(key=lambda x: x.mention_count_in_conversation, reverse=True)
    else:
        items.sort(key=lambda x: (x.emergence_score, x.new_count), reverse=True)

    return TopicWithEmergenceList(
        conversation_id=conversation_id,
        latest_batch_seq=convo.latest_batch_seq,
        items=items,
    )

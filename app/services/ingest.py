from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import APIError
from app.db.models import Conversation, Job
from app.schemas.conversation import TurnInput
from app.services import raw_store
from app.services.queue import make_payload


async def ingest_conversation(
    session: AsyncSession,
    conversation_id: uuid.UUID | None,
    turns: list[TurnInput],
    metadata: dict[str, Any],
) -> tuple[Conversation, Job, str, int, int]:
    """Create or append a conversation.

    Returns (conversation, job, mode, start_turn_index, n_new_turns).
    `mode` is 'created' or 'appended'. Caller must commit and then enqueue.
    """
    turn_dicts = [t.model_dump(mode="json") for t in turns]

    if conversation_id is None:
        return await _create_new(session, turn_dicts, metadata)

    convo = await session.get(Conversation, conversation_id, with_for_update=True)
    if convo is None:
        raise APIError(
            status_code=404,
            title="Conversation not found",
            detail=f"No conversation with id {conversation_id}",
        )
    return await _append(session, convo, turn_dicts)


async def _create_new(
    session: AsyncSession,
    turn_dicts: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Conversation, Job, str, int, int]:
    new_id = uuid.uuid4()
    started_at = datetime.now(timezone.utc)
    raw_path = raw_store.write_new(new_id, started_at, turn_dicts, metadata)

    convo = Conversation(
        id=new_id,
        started_at=started_at,
        raw_path=raw_path,
        extra_metadata=metadata,
        latest_batch_seq=1,
        turn_count=len(turn_dicts),
    )
    session.add(convo)

    job = Job(
        type="ingest",
        status="queued",
        payload=make_payload(
            conversation_id=new_id,
            batch_seq=1,
            start_turn_index=0,
            n_new_turns=len(turn_dicts),
        ),
    )
    session.add(job)
    await session.flush()
    return convo, job, "created", 0, len(turn_dicts)


async def _append(
    session: AsyncSession,
    convo: Conversation,
    turn_dicts: list[dict[str, Any]],
) -> tuple[Conversation, Job, str, int, int]:
    start_turn_index = convo.turn_count
    n_new_turns = len(turn_dicts)
    new_batch_seq = convo.latest_batch_seq + 1

    raw_store.append_turns(convo.id, turn_dicts)

    convo.latest_batch_seq = new_batch_seq
    convo.turn_count = start_turn_index + n_new_turns

    job = Job(
        type="ingest",
        status="queued",
        payload=make_payload(
            conversation_id=convo.id,
            batch_seq=new_batch_seq,
            start_turn_index=start_turn_index,
            n_new_turns=n_new_turns,
        ),
    )
    session.add(job)
    await session.flush()
    return convo, job, "appended", start_turn_index, n_new_turns


async def list_conversations(
    session: AsyncSession, limit: int, offset: int
) -> tuple[list[Conversation], int]:
    stmt = (
        select(Conversation)
        .order_by(Conversation.started_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    items = list(result.scalars().all())

    count_stmt = select(Conversation.id)
    count_result = await session.execute(count_stmt)
    total = len(list(count_result.scalars().all()))

    return items, total

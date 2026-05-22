from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk
from app.db.session import session_scope
from app.services import raw_store
from app.services.chunker import chunk_turns, overlap_prefix_from
from app.services.embedder import Embedder, get_embedder
from app.services.topic_assigner import assign_chunks_to_topics, recompute_mention_counts

log = logging.getLogger(__name__)


async def _fetch_overlap_prefix(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    batch_seq: int,
    embedder: Embedder,
) -> str | None:
    """Pull the most recent chunk of this conversation from a prior batch and
    return its tail as an overlap prefix for the new batch's chunking pass.
    """
    if batch_seq <= 1:
        return None
    settings = get_settings()

    stmt = (
        select(Chunk.content)
        .where(Chunk.conversation_id == conversation_id)
        .where(Chunk.batch_seq < batch_seq)
        .order_by(Chunk.position.desc(), Chunk.created_at.desc())
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row is None or not row.content:
        return None
    return overlap_prefix_from(row.content, settings.chunk_overlap_tokens, embedder)


async def run_ingest_pipeline(
    conversation_id: uuid.UUID,
    batch_seq: int,
    start_turn_index: int,
    n_new_turns: int,
    embedder: Embedder | None = None,
) -> dict[str, object]:
    embedder = embedder or get_embedder()

    new_turns = raw_store.slice_turns(conversation_id, start_turn_index, n_new_turns)
    if not new_turns:
        log.warning("Ingest pipeline saw 0 new turns for %s batch %s", conversation_id, batch_seq)
        return {"chunks_created": 0, "chunks_assigned": 0}

    async with session_scope() as session:
        overlap_prefix = await _fetch_overlap_prefix(
            session, conversation_id, batch_seq, embedder
        )

    chunk_results = chunk_turns(
        turns=new_turns,
        start_turn_index=start_turn_index,
        overlap_prefix=overlap_prefix,
        embedder=embedder,
    )
    if not chunk_results:
        log.info("Chunker produced 0 chunks for %s batch %s", conversation_id, batch_seq)
        return {"chunks_created": 0, "chunks_assigned": 0}

    embeddings = embedder.encode([c.content for c in chunk_results])

    async with session_scope() as session:
        chunk_rows: list[Chunk] = []
        for cr, emb in zip(chunk_results, embeddings, strict=True):
            chunk = Chunk(
                conversation_id=conversation_id,
                content=cr.content,
                token_count=cr.token_count,
                embedding=emb.tolist(),
                batch_seq=batch_seq,
                position=cr.position,
                mention_at=cr.mention_at,
            )
            session.add(chunk)
            chunk_rows.append(chunk)
        await session.flush()

        assigned = await assign_chunks_to_topics(session, chunk_rows)
        await recompute_mention_counts(session, list(assigned.keys()))

        return {
            "chunks_created": len(chunk_rows),
            "chunks_assigned": sum(assigned.values()),
            "topics_touched": [str(t) for t in assigned.keys()],
            "batch_seq": batch_seq,
        }

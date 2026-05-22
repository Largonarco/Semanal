from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.services.embedder import Embedder, LangchainEmbedderAdapter, get_embedder

log = logging.getLogger(__name__)

# Conversation-aware separators. Order matters — RecursiveCharacterTextSplitter
# tries them in order, so paragraph + turn-boundary separators win over
# sentence + word boundaries.
SEPARATORS = ["\n\n", "\nuser:", "\nagent:", "\n", ". ", " ", ""]


@dataclass
class ChunkResult:
    content: str
    token_count: int
    position: int
    mention_at: datetime | None


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _build_text_with_offsets(
    turns: list[dict[str, Any]], overlap_prefix: str | None
) -> tuple[str, list[int]]:
    """Return (text, turn_offsets).

    `turn_offsets[i]` = char offset where turns[i] starts in `text`.
    The overlap prefix (if present) occupies chars [0, turn_offsets[0]).
    """
    text = ""
    if overlap_prefix:
        text = overlap_prefix.rstrip() + "\n"
    offsets: list[int] = []
    for turn in turns:
        offsets.append(len(text))
        role = turn.get("role", "speaker")
        content = (turn.get("content") or "").strip()
        text += f"{role}: {content}\n"
    return text, offsets


def chunk_turns(
    turns: list[dict[str, Any]],
    start_turn_index: int,
    overlap_prefix: str | None = None,
    embedder: Embedder | None = None,
) -> list[ChunkResult]:
    """Chunk new turns into embeddable units.

    `start_turn_index` is the position in the overall conversation where these
    new turns begin (used for chunk.position). `overlap_prefix` is the tail of
    the most-recent existing chunk, prepended for semantic continuity across
    an append boundary — chunks that contain ONLY overlap (no new turns) are
    dropped, since the previous batch already covered them.
    """
    settings = get_settings()
    embedder = embedder or get_embedder()

    if not turns:
        return []

    text, offsets = _build_text_with_offsets(turns, overlap_prefix)

    # 1. Primary: semantic breakpoints
    semantic_chunker = SemanticChunker(
        embeddings=LangchainEmbedderAdapter(embedder),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=settings.semantic_breakpoint_percentile,
    )
    try:
        semantic_chunks = semantic_chunker.split_text(text)
    except Exception as e:
        log.warning("SemanticChunker failed (%s); falling back to single block", e)
        semantic_chunks = [text]

    # 2. Recursive fallback for chunks that overflow the token budget
    recursive_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=embedder.tokenizer,
        chunk_size=settings.chunk_max_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        separators=SEPARATORS,
    )
    final_chunk_texts: list[str] = []
    for chunk in semantic_chunks:
        if embedder.count_tokens(chunk) > settings.chunk_max_tokens:
            final_chunk_texts.extend(recursive_splitter.split_text(chunk))
        else:
            final_chunk_texts.append(chunk)

    # 3. Map each chunk back to its source turn(s) via char offsets
    results: list[ChunkResult] = []
    cursor = 0
    for chunk_text in final_chunk_texts:
        if not chunk_text.strip():
            continue
        idx = text.find(chunk_text, cursor)
        if idx == -1:
            idx = cursor
        chunk_end = idx + len(chunk_text)

        contributing: list[int] = []
        for i, turn_start in enumerate(offsets):
            turn_end = offsets[i + 1] if i + 1 < len(offsets) else len(text)
            if turn_end > idx and turn_start < chunk_end:
                contributing.append(i)

        if not contributing:
            # Chunk is entirely in the overlap-prefix region — already covered
            # by a previous chunk in a prior batch. Drop it.
            cursor = chunk_end
            continue

        timestamps = [_parse_ts(turns[i].get("timestamp")) for i in contributing]
        timestamps = [t for t in timestamps if t is not None]

        results.append(
            ChunkResult(
                content=chunk_text.strip(),
                token_count=embedder.count_tokens(chunk_text),
                position=start_turn_index + contributing[0],
                mention_at=min(timestamps) if timestamps else None,
            )
        )
        cursor = chunk_end

    return results


def overlap_prefix_from(content: str, n_tokens: int, embedder: Embedder | None = None) -> str:
    """Return the tail of an existing chunk's content (~n_tokens worth)."""
    embedder = embedder or get_embedder()
    tokenizer = embedder.tokenizer
    token_ids = tokenizer.encode(content, add_special_tokens=False)
    if len(token_ids) <= n_tokens:
        return content
    return tokenizer.decode(token_ids[-n_tokens:], skip_special_tokens=True)

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings


def _normalize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _path_for(conversation_id: uuid.UUID) -> Path:
    base = Path(get_settings().raw_storage_path)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{conversation_id}.json"


def write_new(
    conversation_id: uuid.UUID,
    started_at: datetime,
    turns: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    """Create the raw conversation file. Returns the absolute path."""
    payload = {
        "conversation_id": str(conversation_id),
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "turns": _normalize(turns),
        "metadata": _normalize(metadata),
    }
    path = _path_for(conversation_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return str(path)


def append_turns(conversation_id: uuid.UUID, new_turns: list[dict[str, Any]]) -> str:
    """Append turns to an existing raw file. Returns the absolute path."""
    path = _path_for(conversation_id)
    data = json.loads(path.read_text())
    data["turns"].extend(_normalize(new_turns))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return str(path)


def read(conversation_id: uuid.UUID) -> dict[str, Any]:
    return json.loads(_path_for(conversation_id).read_text())


def slice_turns(
    conversation_id: uuid.UUID, start_index: int, count: int
) -> list[dict[str, Any]]:
    """Return turns[start_index : start_index + count] from the raw file."""
    data = read(conversation_id)
    return data["turns"][start_index : start_index + count]


def format_turns_for_chunking(turns: list[dict[str, Any]]) -> str:
    """Render a list of turns into a single string the chunker can split.

    Uses explicit `role:` prefixes so the recursive splitter's separators
    (`\\nuser:`, `\\nagent:`) can fall on turn boundaries.
    """
    lines = []
    for turn in turns:
        role = turn.get("role", "speaker")
        content = (turn.get("content") or "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

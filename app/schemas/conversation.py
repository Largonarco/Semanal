from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TurnInput(BaseModel):
    """A single message from one participant in a conversation.

    `timestamp` is optional and connector-supplied (e.g., Slack provides it).
    The server does not generate or validate turn timestamps — they are stored
    as data only and used downstream by the emergence calculator.
    """

    role: str = Field(..., min_length=1, max_length=64)
    content: str = Field(..., min_length=1)
    timestamp: datetime | None = None


class ConversationIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Omit for a new conversation. Provide to append to an existing one. "
            "A non-existent id returns 404."
        ),
    )
    turns: list[TurnInput] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationIngestResponse(BaseModel):
    conversation_id: uuid.UUID
    job_id: uuid.UUID
    mode: Literal["created", "appended"]
    batch_seq: int
    turn_count: int


class ConversationDetail(BaseModel):
    id: uuid.UUID
    started_at: datetime
    latest_batch_seq: int
    turn_count: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ConversationList(BaseModel):
    items: list[ConversationDetail]
    total: int
    limit: int
    offset: int

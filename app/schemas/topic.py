from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class TopicSummary(BaseModel):
    """Lightweight topic shape — used inside conversation and run listings."""

    id: uuid.UUID
    label: str | None
    sentiment: str | None
    pm_insight: str | None
    status: str
    mention_count: int
    first_seen_at: datetime
    last_updated_at: datetime
    last_labeled_at: datetime | None


class TopicWithEmergence(TopicSummary):
    """A topic in the context of a specific conversation."""

    new_count: int
    old_count: int
    mention_count_in_conversation: int
    emergence_score: float
    is_emerging: bool
    snippets_in_conversation: list[str]


class TopicInRun(TopicSummary):
    """A topic as it appeared in a specific clustering run snapshot."""

    mention_count_at_run: int
    was_new: bool


class TopicDetail(TopicSummary):
    """Single topic detail with exemplar snippets (global, all conversations)."""

    exemplar_snippets: list[str]


class TopicWithEmergenceList(BaseModel):
    conversation_id: uuid.UUID
    latest_batch_seq: int
    items: list[TopicWithEmergence]


class TopicInRunList(BaseModel):
    clustering_run_id: uuid.UUID
    run_finished_at: datetime | None
    items: list[TopicInRun]

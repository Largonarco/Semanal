from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.schemas.conversation import (
    ConversationIngestRequest,
    TurnInput,
)


def test_minimal_new_conversation_payload():
    payload = ConversationIngestRequest(
        turns=[TurnInput(role="user", content="hi")]
    )
    assert payload.conversation_id is None
    assert len(payload.turns) == 1


def test_append_with_existing_id():
    cid = uuid.uuid4()
    payload = ConversationIngestRequest(
        conversation_id=cid,
        turns=[TurnInput(role="user", content="hello again")],
    )
    assert payload.conversation_id == cid


def test_rejects_empty_turns_list():
    with pytest.raises(ValidationError):
        ConversationIngestRequest(turns=[])


def test_rejects_empty_turn_content():
    with pytest.raises(ValidationError):
        TurnInput(role="user", content="")


def test_rejects_empty_turn_role():
    with pytest.raises(ValidationError):
        TurnInput(role="", content="hi")


def test_rejects_unknown_extra_fields():
    with pytest.raises(ValidationError):
        ConversationIngestRequest(
            turns=[TurnInput(role="user", content="hi")],
            user_id="this-field-was-removed-by-design",  # type: ignore[call-arg]
        )


def test_accepts_metadata_jsonb():
    payload = ConversationIngestRequest(
        turns=[TurnInput(role="user", content="hi")],
        metadata={"source": "slack", "thread_ts": "1234.5678"},
    )
    assert payload.metadata["source"] == "slack"

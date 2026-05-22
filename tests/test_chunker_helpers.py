from __future__ import annotations

from datetime import datetime, timezone

from app.services.chunker import _build_text_with_offsets, _parse_ts


def test_build_text_no_overlap():
    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "agent", "content": "Hi there"},
    ]
    text, offsets = _build_text_with_offsets(turns, overlap_prefix=None)
    assert "user: Hello" in text
    assert "agent: Hi there" in text
    assert len(offsets) == 2
    assert offsets[0] == 0  # First turn starts at the very beginning


def test_build_text_with_overlap_pushes_first_offset():
    turns = [{"role": "user", "content": "Hello"}]
    text, offsets = _build_text_with_offsets(turns, overlap_prefix="prev context")
    # The overlap_prefix occupies chars before offsets[0]
    assert text.startswith("prev context")
    assert offsets[0] > 0
    assert text[offsets[0] :].startswith("user: Hello")


def test_offsets_are_strictly_increasing():
    turns = [
        {"role": "user", "content": "a"},
        {"role": "agent", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    _, offsets = _build_text_with_offsets(turns, overlap_prefix=None)
    assert all(offsets[i] < offsets[i + 1] for i in range(len(offsets) - 1))


def test_parse_ts_iso_with_z():
    dt = _parse_ts("2026-01-01T12:00:00Z")
    assert dt is not None
    assert dt.year == 2026


def test_parse_ts_iso_with_offset():
    dt = _parse_ts("2026-01-01T12:00:00+00:00")
    assert dt is not None


def test_parse_ts_none():
    assert _parse_ts(None) is None


def test_parse_ts_invalid():
    assert _parse_ts("not-a-date") is None


def test_parse_ts_datetime_passthrough():
    original = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _parse_ts(original) is original

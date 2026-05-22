from __future__ import annotations

from app.services.labeler import _coerce_label, _mock_label


def test_mock_label_shape():
    snippets = ["user: I want a refund", "agent: I can help with that"]
    result = _mock_label(snippets)
    assert "topic" in result
    assert result["sentiment"] == "neutral"
    assert isinstance(result["supporting_snippets"], list)


def test_coerce_label_accepts_valid_sentiment():
    out = _coerce_label({
        "topic": "Refund requests",
        "sentiment": "negative",
        "pm_insight": "Lots of refund pain — investigate fulfillment.",
        "supporting_snippets": ["s1", "s2"],
    })
    assert out["topic"] == "Refund requests"
    assert out["sentiment"] == "negative"


def test_coerce_label_coerces_unknown_sentiment_to_mixed():
    out = _coerce_label({
        "topic": "X",
        "sentiment": "frustrated",  # not in the allowed enum
        "pm_insight": "",
    })
    assert out["sentiment"] == "mixed"


def test_coerce_label_lowercases_sentiment():
    out = _coerce_label({"topic": "X", "sentiment": "POSITIVE"})
    assert out["sentiment"] == "positive"


def test_coerce_label_trims_long_label():
    long = "x" * 1000
    out = _coerce_label({"topic": long})
    assert len(out["topic"]) == 500


def test_coerce_label_caps_supporting_snippets_at_5():
    snippets = [f"snippet-{i}" for i in range(20)]
    out = _coerce_label({"topic": "X", "supporting_snippets": snippets})
    assert len(out["supporting_snippets"]) == 5


def test_coerce_label_defaults_when_missing():
    out = _coerce_label({})
    assert out["topic"] == "(unlabeled)"
    assert out["sentiment"] == "neutral"
    assert out["pm_insight"] == ""
    assert out["supporting_snippets"] == []

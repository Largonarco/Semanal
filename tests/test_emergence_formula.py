from __future__ import annotations

from app.services.emergence import score_topic


def test_fresh_conversation_single_mention_not_emerging():
    score, emerging = score_topic(
        new_count=1, old_count=0, is_fresh_conversation=True,
        fresh_min_count=2, ratio_threshold=2.0,
    )
    assert score == 1.0
    assert emerging is False


def test_fresh_conversation_multiple_mentions_emerging():
    score, emerging = score_topic(
        new_count=3, old_count=0, is_fresh_conversation=True,
        fresh_min_count=2, ratio_threshold=2.0,
    )
    assert score == 3.0
    assert emerging is True


def test_append_topic_growing_above_threshold():
    # 6 new vs 2 old → ratio = 3.0 ≥ 2.0
    score, emerging = score_topic(
        new_count=6, old_count=2, is_fresh_conversation=False,
        fresh_min_count=2, ratio_threshold=2.0,
    )
    assert score == 3.0
    assert emerging is True


def test_append_topic_growing_below_threshold():
    # 3 new vs 2 old → ratio = 1.5 < 2.0
    score, emerging = score_topic(
        new_count=3, old_count=2, is_fresh_conversation=False,
        fresh_min_count=2, ratio_threshold=2.0,
    )
    assert score == 1.5
    assert emerging is False


def test_append_topic_only_in_old_batches_not_emerging():
    # new_count=0 means topic only appeared in prior batches; not emerging
    score, emerging = score_topic(
        new_count=0, old_count=10, is_fresh_conversation=False,
        fresh_min_count=2, ratio_threshold=2.0,
    )
    assert score == 0.0
    assert emerging is False


def test_append_brand_new_topic_no_baseline():
    # First appearance in batch_seq=2 — old_count=0 means infinite ratio (max(0,1)=1)
    score, emerging = score_topic(
        new_count=5, old_count=0, is_fresh_conversation=False,
        fresh_min_count=2, ratio_threshold=2.0,
    )
    assert score == 5.0
    assert emerging is True


def test_threshold_tuning_via_args():
    score, emerging_strict = score_topic(
        new_count=3, old_count=1, is_fresh_conversation=False,
        fresh_min_count=2, ratio_threshold=5.0,
    )
    assert score == 3.0
    assert emerging_strict is False

    _, emerging_loose = score_topic(
        new_count=3, old_count=1, is_fresh_conversation=False,
        fresh_min_count=2, ratio_threshold=1.5,
    )
    assert emerging_loose is True

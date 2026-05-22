from __future__ import annotations

import uuid

import numpy as np

from app.db.models import Topic
from app.services.clusterer import _cosine, _match_clusters_to_topics


def _unit(vec: list[float]) -> np.ndarray:
    a = np.array(vec, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)


def _topic_with_centroid(vec: list[float]) -> Topic:
    t = Topic(
        id=uuid.uuid4(),
        centroid=_unit(vec).tolist(),
        status="active",
    )
    return t


def test_cosine_identical_vectors():
    a = _unit([1, 0, 0, 0])
    b = _unit([1, 0, 0, 0])
    assert _cosine(a, b) > 0.999


def test_cosine_orthogonal_vectors():
    a = _unit([1, 0, 0, 0])
    b = _unit([0, 1, 0, 0])
    assert abs(_cosine(a, b)) < 1e-5


def test_match_inherits_topic_id_when_above_threshold():
    existing = _topic_with_centroid([1.0, 0.0, 0.0, 0.0])
    # New centroid almost identical
    new_centroid = _unit([0.99, 0.02, 0.01, 0.0])
    new_clusters = {
        0: {"centroid": new_centroid, "members": np.array([0, 1]), "exemplars": np.array([0])}
    }
    matched = _match_clusters_to_topics(new_clusters, [existing], threshold=0.85)
    assert matched[0] == existing.id


def test_match_returns_none_when_below_threshold():
    existing = _topic_with_centroid([1.0, 0.0, 0.0, 0.0])
    # Roughly orthogonal new centroid
    new_centroid = _unit([0.1, 1.0, 0.0, 0.0])
    new_clusters = {
        0: {"centroid": new_centroid, "members": np.array([0]), "exemplars": np.array([0])}
    }
    matched = _match_clusters_to_topics(new_clusters, [existing], threshold=0.85)
    assert matched[0] is None


def test_greedy_one_to_one_matching():
    # Two existing topics, two new clusters; each new cluster should match a different topic.
    t_refund = _topic_with_centroid([1.0, 0.0, 0.0, 0.0])
    t_shipping = _topic_with_centroid([0.0, 1.0, 0.0, 0.0])

    new_clusters = {
        0: {
            "centroid": _unit([0.95, 0.05, 0.0, 0.0]),
            "members": np.array([0]),
            "exemplars": np.array([0]),
        },
        1: {
            "centroid": _unit([0.05, 0.95, 0.0, 0.0]),
            "members": np.array([1]),
            "exemplars": np.array([1]),
        },
    }
    matched = _match_clusters_to_topics(new_clusters, [t_refund, t_shipping], threshold=0.85)
    # Each cluster picked the right existing topic
    assert matched[0] == t_refund.id
    assert matched[1] == t_shipping.id


def test_unmatched_clusters_create_new_topics_signal():
    t_only = _topic_with_centroid([1.0, 0.0, 0.0, 0.0])
    new_clusters = {
        0: {
            "centroid": _unit([0.0, 0.0, 1.0, 0.0]),  # orthogonal — no match
            "members": np.array([0]),
            "exemplars": np.array([0]),
        }
    }
    matched = _match_clusters_to_topics(new_clusters, [t_only], threshold=0.85)
    assert matched[0] is None

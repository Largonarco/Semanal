from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import hdbscan
import numpy as np
import umap
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, ClusteringRun, ClusteringRunTopic, Job, Topic
from app.db.session import session_scope
from app.services.embedder import Embedder, get_embedder
from app.services.queue import enqueue_label_topic, get_arq_pool, make_payload
from app.services.topic_assigner import recompute_mention_count

log = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


async def _load_chunks(session: AsyncSession) -> tuple[list[uuid.UUID], np.ndarray]:
    stmt = select(Chunk.id, Chunk.embedding).order_by(Chunk.created_at)
    rows = (await session.execute(stmt)).all()
    if not rows:
        return [], np.zeros((0, get_settings().embedding_dim), dtype=np.float32)
    ids = [r.id for r in rows]
    vecs = np.array([r.embedding for r in rows], dtype=np.float32)
    return ids, vecs


async def _load_active_topics(session: AsyncSession) -> list[Topic]:
    stmt = (
        select(Topic)
        .where(Topic.status == "active")
        .where(Topic.centroid.is_not(None))
    )
    return list((await session.execute(stmt)).scalars().all())


def _build_cluster_summary(
    labels: np.ndarray,
    probs: np.ndarray,
    embeddings: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    """For each non-noise cluster, compute its centroid and exemplar indices."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for cluster_id in sorted(set(int(c) for c in labels)):
        if cluster_id < 0:
            continue
        mask = labels == cluster_id
        member_indices = np.where(mask)[0]
        cluster_vecs = embeddings[mask]
        centroid = cluster_vecs.mean(axis=0)
        norm = float(np.linalg.norm(centroid)) + 1e-9
        centroid = centroid / norm
        member_probs = probs[mask]
        top_n = min(20, len(member_indices))
        top_local = np.argsort(-member_probs)[:top_n]
        exemplar_indices = member_indices[top_local]
        out[cluster_id] = {
            "centroid": centroid,
            "members": member_indices,
            "exemplars": exemplar_indices,
        }
    return out


def _match_clusters_to_topics(
    new_clusters: dict[int, dict[str, np.ndarray]],
    active_topics: list[Topic],
    threshold: float,
) -> dict[int, uuid.UUID | None]:
    """Greedy 1-1 match by cosine similarity. Returns {cluster_id: topic_id or None}."""
    matched: dict[int, uuid.UUID | None] = {}
    used: set[uuid.UUID] = set()
    for cluster_id, info in new_clusters.items():
        best_topic: Topic | None = None
        best_sim = -1.0
        for topic in active_topics:
            if topic.id in used:
                continue
            sim = _cosine(info["centroid"], np.asarray(topic.centroid, dtype=np.float32))
            if sim > best_sim:
                best_sim = sim
                best_topic = topic
        if best_topic is not None and best_sim >= threshold:
            matched[cluster_id] = best_topic.id
            used.add(best_topic.id)
        else:
            matched[cluster_id] = None
    return matched


async def run_clustering_pipeline(
    clustering_run_id: uuid.UUID,
    embedder: Embedder | None = None,
) -> dict[str, object]:
    settings = get_settings()
    embedder = embedder or get_embedder()  # noqa: F841 — kept for future param hooks

    async with session_scope() as session:
        run = await session.get(ClusteringRun, clustering_run_id)
        if run is None:
            return {"status": "missing", "clustering_run_id": str(clustering_run_id)}
        run.status = "running"
        run.params = {
            "umap_n_neighbors": settings.umap_n_neighbors,
            "umap_min_dist": settings.umap_min_dist,
            "umap_n_components": settings.umap_n_components,
            "umap_metric": settings.umap_metric,
            "hdbscan_min_cluster_size": settings.hdbscan_min_cluster_size,
            "hdbscan_min_samples": settings.hdbscan_min_samples,
            "centroid_match_threshold": settings.centroid_match_threshold,
        }

    try:
        return await _execute(clustering_run_id, settings)
    except Exception as e:
        log.exception("Clustering pipeline failed")
        async with session_scope() as session:
            run = await session.get(ClusteringRun, clustering_run_id)
            if run is not None:
                run.status = "failed"
                run.finished_at = datetime.now(timezone.utc)
                run.error = str(e)
        raise


async def _execute(clustering_run_id: uuid.UUID, settings) -> dict[str, object]:
    async with session_scope() as session:
        chunk_ids, embeddings = await _load_chunks(session)

    n_chunks = len(chunk_ids)
    min_needed = max(settings.hdbscan_min_cluster_size, settings.umap_n_neighbors + 1)
    if n_chunks < min_needed:
        async with session_scope() as session:
            run = await session.get(ClusteringRun, clustering_run_id)
            if run is not None:
                run.status = "completed"
                run.finished_at = datetime.now(timezone.utc)
                run.n_chunks_processed = n_chunks
                run.n_topics_found = 0
                run.n_topics_new = 0
                run.n_topics_retired = 0
                run.error = (
                    f"Insufficient chunks ({n_chunks} < {min_needed}); skipping clustering"
                )
        return {"status": "skipped", "reason": "insufficient_data", "n_chunks": n_chunks}

    log.info("UMAP fit_transform over %d chunks", n_chunks)
    reducer = umap.UMAP(
        n_neighbors=settings.umap_n_neighbors,
        min_dist=settings.umap_min_dist,
        n_components=settings.umap_n_components,
        metric=settings.umap_metric,
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    log.info("HDBSCAN clustering")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=settings.hdbscan_min_samples,
        cluster_selection_method=settings.hdbscan_cluster_selection,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    probs = clusterer.probabilities_

    new_clusters = _build_cluster_summary(labels, probs, embeddings)

    if not new_clusters:
        async with session_scope() as session:
            run = await session.get(ClusteringRun, clustering_run_id)
            if run is not None:
                run.status = "completed"
                run.finished_at = datetime.now(timezone.utc)
                run.n_chunks_processed = n_chunks
                run.n_topics_found = 0
                run.n_topics_new = 0
                run.n_topics_retired = 0
            # Null out all chunks' topic_id since nothing clustered
            await session.execute(update(Chunk).values(topic_id=None))
            # Retire all active topics
            await session.execute(
                update(Topic).where(Topic.status == "active").values(status="retired")
            )
        return {"status": "completed", "n_topics_found": 0, "all_noise": True}

    affected_topic_ids: set[uuid.UUID] = set()
    newly_created: list[uuid.UUID] = []

    async with session_scope() as session:
        active_topics = await _load_active_topics(session)
        matched = _match_clusters_to_topics(
            new_clusters, active_topics, settings.centroid_match_threshold
        )

        for cluster_id, info in new_clusters.items():
            existing_topic_id = matched[cluster_id]
            centroid_list = info["centroid"].tolist()
            if existing_topic_id is None:
                topic = Topic(
                    centroid=centroid_list,
                    status="active",
                    last_updated_at=datetime.now(timezone.utc),
                )
                session.add(topic)
                await session.flush()
                topic_id = topic.id
                newly_created.append(topic_id)
            else:
                topic_id = existing_topic_id
                await session.execute(
                    update(Topic)
                    .where(Topic.id == topic_id)
                    .values(
                        centroid=centroid_list,
                        status="active",
                        last_updated_at=datetime.now(timezone.utc),
                    )
                )
            affected_topic_ids.add(topic_id)

            member_chunk_ids = [chunk_ids[int(i)] for i in info["members"]]
            await session.execute(
                update(Chunk)
                .where(Chunk.id.in_(member_chunk_ids))
                .values(topic_id=topic_id)
            )

        # Null out noise chunks
        noise_indices = np.where(labels == -1)[0]
        if len(noise_indices) > 0:
            noise_chunk_ids = [chunk_ids[int(i)] for i in noise_indices]
            await session.execute(
                update(Chunk).where(Chunk.id.in_(noise_chunk_ids)).values(topic_id=None)
            )

        used_topic_ids = {tid for tid in matched.values() if tid is not None}
        retired_ids: list[uuid.UUID] = []
        for topic in active_topics:
            if topic.id not in used_topic_ids:
                await session.execute(
                    update(Topic).where(Topic.id == topic.id).values(status="retired")
                )
                retired_ids.append(topic.id)

        new_topic_set = set(newly_created)
        for tid in affected_topic_ids:
            count = await recompute_mention_count(session, tid)
            session.add(
                ClusteringRunTopic(
                    clustering_run_id=clustering_run_id,
                    topic_id=tid,
                    mention_count_at_run=count,
                    was_new=tid in new_topic_set,
                )
            )

        run = await session.get(ClusteringRun, clustering_run_id)
        if run is not None:
            run.status = "completed"
            run.finished_at = datetime.now(timezone.utc)
            run.n_topics_found = len(new_clusters)
            run.n_topics_new = len(newly_created)
            run.n_topics_retired = len(retired_ids)
            run.n_chunks_processed = n_chunks

    pool = await get_arq_pool()
    enqueued_count = 0
    try:
        async with session_scope() as session:
            for tid in affected_topic_ids:
                job = Job(
                    type="label_topic",
                    status="queued",
                    payload=make_payload(topic_id=tid),
                )
                session.add(job)
                await session.flush()
                await enqueue_label_topic(pool, job.id, tid)
                enqueued_count += 1
    finally:
        await pool.close(close_connection_pool=True)

    return {
        "status": "completed",
        "n_topics_found": len(new_clusters),
        "n_topics_new": len(newly_created),
        "n_topics_retired": len(retired_ids),
        "n_chunks_processed": n_chunks,
        "n_label_jobs_enqueued": enqueued_count,
    }

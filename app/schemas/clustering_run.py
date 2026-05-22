from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ClusteringRunResponse(BaseModel):
    id: uuid.UUID
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    n_topics_found: int | None = None
    n_topics_new: int | None = None
    n_topics_retired: int | None = None
    n_chunks_processed: int | None = None
    params: dict[str, Any]
    error: str | None = None


class ClusteringRunList(BaseModel):
    items: list[ClusteringRunResponse]
    total: int
    limit: int
    offset: int


class ClusteringRunTriggerResponse(BaseModel):
    clustering_run_id: uuid.UUID
    job_id: uuid.UUID

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/sentiment"
    database_url_sync: str = "postgresql://postgres:postgres@postgres:5432/sentiment"

    # Redis (arq)
    redis_url: str = "redis://redis:6379/0"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # Chunking
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 50
    semantic_breakpoint_percentile: int = 85

    # UMAP
    umap_n_neighbors: int = 20
    umap_min_dist: float = 0.05
    umap_n_components: int = 5
    umap_metric: str = "cosine"

    # HDBSCAN
    hdbscan_min_cluster_size: int = 25
    hdbscan_min_samples: int = 10
    hdbscan_cluster_selection: str = "eom"

    # Topic lifecycle
    knn_assign_threshold: float = 0.75
    centroid_match_threshold: float = 0.85

    # Per-conversation emergence
    emergence_fresh_min_count: int = 2
    emergence_ratio_threshold: float = 2.0

    # Cron
    clustering_cron_hours: int = 24

    # Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_model_id: str = "anthropic.claude-sonnet-4-6-20251022-v1:0"

    mock_llm: bool = False

    # Raw storage
    raw_storage_path: str = "/app/data/raw"

    @property
    def has_aws_credentials(self) -> bool:
        return bool(self.aws_access_key_id and self.aws_secret_access_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()

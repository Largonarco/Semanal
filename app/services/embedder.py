from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.config import get_settings

log = logging.getLogger(__name__)


class Embedder:
    """Singleton wrapper around a sentence-transformers model.

    All embeddings are L2-normalized so cosine similarity == dot product, which
    matches pgvector's `vector_cosine_ops` semantics and the centroid-matching
    threshold (0.85) used downstream.
    """

    _instance: "Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        settings = get_settings()
        log.info("Loading sentence-transformers model %s", settings.embedding_model)
        self._model = SentenceTransformer(settings.embedding_model, device="cpu")
        self._dim = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size
        # bge-small has a 512-token context window — sentence-transformers
        # truncates by default; surface the max for the chunker to respect.
        self._max_seq_length = self._model.max_seq_length or 512

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @property
    def tokenizer(self):
        return self._model.tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


class LangchainEmbedderAdapter(Embeddings):
    """Adapter so the same Embedder instance can power LangChain's SemanticChunker."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or get_embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode_one(text).tolist()

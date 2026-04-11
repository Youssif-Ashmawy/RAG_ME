"""
Custom in-memory vector store with disk persistence.

Uses numpy for fast cosine similarity. Stores chunks + embeddings
as a .pkl file under .rag-cache/{repo_id}/ for reuse across sessions.

Design goals:
  • No external vector DB required — fully self-contained
  • Cosine similarity with numpy broadcasting
  • Diversity-aware retrieval: max 2 chunks per source file
  • Simple pickle-based persistence
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .chunker import Chunk

CACHE_ROOT = ".rag-cache"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    # ── Properties ──────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._chunks)

    # ── Indexing ─────────────────────────────────────────────

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    # ── Retrieval ─────────────────────────────────────────────

    def search(self, query_embedding: list[float], top_k: int = 6) -> list[SearchResult]:
        if not self._chunks:
            return []

        mat = np.array(self._embeddings, dtype=np.float32)   # (N, D)
        q   = np.array(query_embedding,  dtype=np.float32)   # (D,)

        # Cosine similarity: dot(mat, q) / (||mat|| * ||q||)
        mat_norms = np.linalg.norm(mat, axis=1)              # (N,)
        q_norm    = np.linalg.norm(q)
        denom = mat_norms * q_norm
        denom = np.where(denom == 0, 1e-10, denom)

        scores = (mat @ q) / denom                           # (N,)
        ranked = np.argsort(scores)[::-1]

        # Diversity: limit to 2 chunks per file
        file_count: dict[str, int] = {}
        results: list[SearchResult] = []

        for idx in ranked:
            if len(results) >= top_k:
                break
            chunk = self._chunks[int(idx)]
            count = file_count.get(chunk.path, 0)
            if count < 2:
                results.append(SearchResult(chunk=chunk, score=float(scores[idx])))
                file_count[chunk.path] = count + 1

        return results

    # ── Persistence ───────────────────────────────────────────

    def save(self, repo_id: str) -> None:
        cache_dir = _cache_dir(repo_id)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "store.pkl", "wb") as f:
            pickle.dump(
                {"chunks": self._chunks, "embeddings": self._embeddings}, f
            )

    @classmethod
    def load(cls, repo_id: str) -> Optional["VectorStore"]:
        path = _cache_dir(repo_id) / "store.pkl"
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            store = cls()
            store._chunks    = data["chunks"]
            store._embeddings = data["embeddings"]
            return store
        except Exception as exc:
            print(f"[warning] Failed to load cached store for {repo_id}: {exc}")
            return None

    @staticmethod
    def clear(repo_id: str) -> None:
        cache = _cache_dir(repo_id)
        for name in ("store.pkl", "graph.json"):
            p = cache / name
            if p.exists():
                p.unlink()

    # ── Summary ───────────────────────────────────────────────

    def indexed_files(self) -> list[dict]:
        """Return per-file summary: path, file_type, chunk count."""
        counts: dict[str, dict] = {}
        for chunk in self._chunks:
            if chunk.path not in counts:
                counts[chunk.path] = {"path": chunk.path, "file_type": chunk.file_type, "chunks": 0}
            counts[chunk.path]["chunks"] += 1
        return sorted(counts.values(), key=lambda x: x["path"])


# ─────────────────────────────────────────────────────────────
# Module-level in-process cache
# (avoids re-loading from disk on every query within a session)
# ─────────────────────────────────────────────────────────────

_store_cache: dict[str, VectorStore] = {}


def get_store(repo_id: str) -> Optional[VectorStore]:
    if repo_id in _store_cache:
        return _store_cache[repo_id]
    store = VectorStore.load(repo_id)
    if store:
        _store_cache[repo_id] = store
    return store


def set_store(repo_id: str, store: VectorStore) -> None:
    _store_cache[repo_id] = store


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _cache_dir(repo_id: str) -> Path:
    return Path(CACHE_ROOT) / repo_id

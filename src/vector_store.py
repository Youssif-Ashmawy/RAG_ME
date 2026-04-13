"""
Hybrid vector store — cosine similarity + BM25 with Reciprocal Rank Fusion.

Design:
  • numpy cosine similarity for dense semantic search
  • BM25Okapi for exact keyword / symbol matching
  • RRF (k=60) fuses the two ranked lists → final score
  • Diversity cap: max 2 chunks per source file
  • Disk persistence: .rag-cache/{repo_id}/store.pkl
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk

CACHE_ROOT   = ".rag-cache"
RRF_K        = 60    # RRF constant — higher = smoother blend
MAX_PER_FILE = 3     # diversity cap per source file (raised from 2)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk]           = []
        self._embeddings: list[list[float]] = []
        self._bm25: Optional[BM25Okapi]     = None

    # ── Properties ──────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._chunks)

    # ── Indexing ─────────────────────────────────────────────────

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        tokenized    = [_tokenize(c.text) for c in self._chunks]
        self._bm25   = BM25Okapi(tokenized)

    # ── Retrieval ─────────────────────────────────────────────────

    def search(self, query_embedding: list[float], top_k: int = 6) -> list[SearchResult]:
        """Pure cosine-similarity search (used internally)."""
        if not self._chunks:
            return []
        mat    = np.array(self._embeddings, dtype=np.float32)
        q      = np.array(query_embedding,  dtype=np.float32)
        norms  = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
        norms  = np.where(norms == 0, 1e-10, norms)
        scores = (mat @ q) / norms
        ranked = np.argsort(scores)[::-1]

        file_count: dict[str, int] = {}
        results: list[SearchResult] = []
        for idx in ranked:
            if len(results) >= top_k:
                break
            chunk = self._chunks[int(idx)]
            if file_count.get(chunk.path, 0) < MAX_PER_FILE:
                results.append(SearchResult(chunk=chunk, score=float(scores[idx])))
                file_count[chunk.path] = file_count.get(chunk.path, 0) + 1
        return results

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 6,
    ) -> list[SearchResult]:
        """
        Hybrid search: BM25 + cosine similarity fused with Reciprocal Rank Fusion.

        RRF score = 1/(k + rank_vector) + 1/(k + rank_bm25)
        BM25 catches exact symbol/keyword matches; vector catches semantic intent.
        """
        if not self._chunks:
            return []

        n = len(self._chunks)

        # ── Vector ranking ──────────────────────────────────────
        mat        = np.array(self._embeddings, dtype=np.float32)
        q          = np.array(query_embedding,  dtype=np.float32)
        norms      = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
        norms      = np.where(norms == 0, 1e-10, norms)
        vec_scores = (mat @ q) / norms
        vec_order  = np.argsort(vec_scores)[::-1].tolist()

        # ── BM25 ranking ────────────────────────────────────────
        tokens     = _tokenize(query_text)
        bm25_raw   = self._bm25.get_scores(tokens) if self._bm25 else np.zeros(n)
        bm25_order = np.argsort(bm25_raw)[::-1].tolist()

        vec_rank  = {idx: rank for rank, idx in enumerate(vec_order)}
        bm25_rank = {idx: rank for rank, idx in enumerate(bm25_order)}

        # ── RRF fusion ──────────────────────────────────────────
        rrf: dict[int, float] = {
            idx: (
                1.0 / (RRF_K + vec_rank[idx] + 1)
                + 1.0 / (RRF_K + bm25_rank[idx] + 1)
            )
            for idx in range(n)
        }
        ranked = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)

        # ── Diversity cap ───────────────────────────────────────
        file_count: dict[str, int] = {}
        results: list[SearchResult] = []
        for idx in ranked:
            if len(results) >= top_k:
                break
            chunk = self._chunks[idx]
            if file_count.get(chunk.path, 0) < MAX_PER_FILE:
                results.append(SearchResult(chunk=chunk, score=rrf[idx]))
                file_count[chunk.path] = file_count.get(chunk.path, 0) + 1
        return results

    def chunks_for_file(self, path: str) -> list[Chunk]:
        """Return all chunks belonging to a specific file path."""
        return [c for c in self._chunks if c.path == path]

    # ── Persistence ───────────────────────────────────────────────

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
            store             = cls()
            store._chunks     = data["chunks"]
            store._embeddings = data["embeddings"]
            store._rebuild_bm25()
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

    # ── Summary ───────────────────────────────────────────────────

    def indexed_files(self) -> list[dict]:
        counts: dict[str, dict] = {}
        for chunk in self._chunks:
            if chunk.path not in counts:
                counts[chunk.path] = {
                    "path": chunk.path,
                    "file_type": chunk.file_type,
                    "chunks": 0,
                }
            counts[chunk.path]["chunks"] += 1
        return sorted(counts.values(), key=lambda x: x["path"])


# ─────────────────────────────────────────────────────────────
# Module-level in-process cache
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

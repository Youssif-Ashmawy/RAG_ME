"""
Embeddings — fastembed (ONNX runtime, no Ollama server required).

Uses BAAI/bge-small-en-v1.5 with asymmetric retrieval:
  - Documents: embed as-is  (embed_texts / embed_text)
  - Queries:   query_embed adds the asymmetric prefix internally

The ONNX model (~130 MB) is downloaded once to ~/.cache/fastembed/ on first use.
Set FASTEMBED_CACHE_PATH env var to override the cache location.
"""

from __future__ import annotations

from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE      = 50
MAX_EMBED_CHARS = 800   # 512-token ctx; dense code ~2 chars/token → 400 tokens

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], **_) -> list[list[float]]:
    """Embed a list of document strings (no query prefix)."""
    model = _get_model()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t[:MAX_EMBED_CHARS] for t in texts[i : i + BATCH_SIZE]]
        all_embeddings.extend(v.tolist() for v in model.embed(batch))
    return all_embeddings


def embed_text(text: str, **_) -> list[float]:
    """Embed a single document string (no query prefix)."""
    return next(_get_model().embed([text[:MAX_EMBED_CHARS]])).tolist()


def embed_query(text: str) -> list[float]:
    """
    Embed a search query with the asymmetric retrieval prefix.
    Use this whenever embedding a user question, not a document.
    """
    return next(_get_model().query_embed([text[:MAX_EMBED_CHARS]])).tolist()

"""
Embeddings — Ollama local mxbai-embed-large (lightweight REST client).

mxbai-embed-large uses asymmetric retrieval:
  - Documents: embed as-is
  - Queries:   prepend QUERY_PREFIX so the model optimises for retrieval

Pull once with: ollama pull mxbai-embed-large
"""

from __future__ import annotations

import ollama

EMBEDDING_MODEL = "mxbai-embed-large"
QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "
BATCH_SIZE      = 50
MAX_EMBED_CHARS = 800   # mxbai-embed-large: 512-token ctx; dense code ~2 chars/token → 400 tokens


def embed_texts(texts: list[str], **_) -> list[list[float]]:
    """Embed a list of document strings (no query prefix)."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t[:MAX_EMBED_CHARS] for t in texts[i : i + BATCH_SIZE]]
        response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(response.embeddings)
    return all_embeddings


def embed_text(text: str, **_) -> list[float]:
    """Embed a single document string (no query prefix)."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=[text[:MAX_EMBED_CHARS]])
    return response.embeddings[0]


def embed_query(text: str) -> list[float]:
    """
    Embed a search query with the asymmetric retrieval prefix.
    Use this whenever embedding a user question, not a document.
    """
    prefixed = QUERY_PREFIX + text[:MAX_EMBED_CHARS]
    response = ollama.embed(model=EMBEDDING_MODEL, input=[prefixed])
    return response.embeddings[0]

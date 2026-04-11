"""
Embeddings — uses Ollama's local nomic-embed-text model.

No API key, no internet, no rate limits.
Requires Ollama running locally: https://ollama.com
Pull the model once with: ollama pull nomic-embed-text
"""

from __future__ import annotations

import ollama

EMBEDDING_MODEL = "nomic-embed-text"
BATCH_SIZE      = 50
# nomic-embed-text has a 2048-token context window (~4 chars/token on average,
# but code is denser). Cap at 1500 chars (~375 tokens) to stay safely within
# the limit even for dense code content.
MAX_EMBED_CHARS = 1_500


def _prepare(texts: list[str]) -> list[str]:
    return [t[:MAX_EMBED_CHARS] for t in texts]


def embed_texts(texts: list[str], **_) -> list[list[float]]:
    """
    Embed a list of strings. Returns one float vector per input.
    Batches requests; truncates at MAX_EMBED_CHARS and sets num_ctx explicitly.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = _prepare(texts[i : i + BATCH_SIZE])
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend(response.embeddings)

    return all_embeddings


def embed_text(text: str, **_) -> list[float]:
    """Embed a single string."""
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=[text[:MAX_EMBED_CHARS]],
    )
    return response.embeddings[0]

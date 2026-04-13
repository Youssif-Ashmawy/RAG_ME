"""
RAG pipeline — fully local via Ollama + LangChain agent.

Ingestion:
  1. Fetch ALL source files from GitHub (code + docs)
  2. Parse with language-aware AST/regex chunker
  3. Build import graph (file dependency edges)
  4. Embed all chunks locally (mxbai-embed-large)
  5. Persist hybrid vector store + import graph to .rag-cache/

Querying (agent mode):
  1. Hybrid retrieval (BM25 + cosine) for initial sources
  2. Import-graph walk adds context from related files
  3. LangChain tool-calling agent can search multiple times
  4. Stream final answer token-by-token
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Generator, Optional

from .agent import run_agent_stream
from .chunker import Chunk, chunk_files
from .code_parser import parse_file
from .embeddings import embed_query, embed_texts
from .github_client import fetch_repo_files
from .import_graph import ImportGraph
from .pdf_parser import parse_pdf
from .vector_store import (
    SearchResult, VectorStore,
    get_store, set_store, _cache_dir,
)

EMBED_BATCH    = 50
TOP_K_SEMANTIC = 8    # initial hybrid retrieval
TOP_K_GRAPH    = 3    # extra chunks from graph neighbours
GRAPH_DEPTH    = 1

PDF_ID_PREFIX  = "pdf__"   # repo_id prefix for PDF-mode stores


# ─────────────────────────────────────────────────────────────
# In-process graph cache
# ─────────────────────────────────────────────────────────────

_graph_cache: dict[str, ImportGraph] = {}


def _get_graph(repo_id: str) -> Optional[ImportGraph]:
    if repo_id in _graph_cache:
        return _graph_cache[repo_id]
    g = ImportGraph.load(str(_cache_dir(repo_id)))
    if g:
        _graph_cache[repo_id] = g
    return g


# ─────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    repo_id: str
    files_count: int
    chunks_count: int
    graph_edges: int
    files: list[dict]


@dataclass
class Source:
    path: str
    raw_url: str
    file_type: str
    language: str
    heading: Optional[str]
    unit_name: Optional[str]
    unit_type: Optional[str]
    diagram_type: Optional[str]
    score: float
    via_graph: bool
    preview: str


# ─────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────

def ingest_repo(
    repo_url: str,
    on_progress: Optional[Callable[[str, int], None]] = None,
) -> IngestResult:
    def progress(msg: str, pct: int) -> None:
        if on_progress:
            on_progress(msg, pct)

    progress("Connecting to GitHub…", 5)
    token = os.getenv("GITHUB_TOKEN")
    files, meta = fetch_repo_files(
        repo_url, token=token,
        on_progress=lambda msg: progress(msg, 10),
    )

    progress(f"Parsing and chunking {len(files)} files…", 25)
    chunks = chunk_files(files)

    progress("Building import dependency graph…", 35)
    all_paths    = {f.path for f in files}
    parsed_files = [parse_file(f.content, f.path) for f in files]
    graph        = ImportGraph.build(parsed_files, all_paths)

    progress(f"Embedding {len(chunks)} chunks locally…", 40)
    texts: list[str] = [c.text for c in chunks]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        all_embeddings.extend(embed_texts(batch))
        done = min(i + EMBED_BATCH, len(texts))
        pct  = 40 + int((done / len(texts)) * 50)
        progress(f"Embedded {done}/{len(texts)} chunks…", pct)

    progress("Saving index and graph to disk…", 95)
    store = VectorStore()
    store.add(chunks, all_embeddings)
    cache_dir = str(_cache_dir(meta.repo_id))
    store.save(meta.repo_id)
    graph.save(cache_dir)
    set_store(meta.repo_id, store)
    _graph_cache[meta.repo_id] = graph

    progress("Done! Repository fully indexed.", 100)
    return IngestResult(
        repo_id=meta.repo_id,
        files_count=len(files),
        chunks_count=len(chunks),
        graph_edges=graph.edge_count,
        files=store.indexed_files(),
    )


# ─────────────────────────────────────────────────────────────
# PDF ingestion
# ─────────────────────────────────────────────────────────────

def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    on_progress: Optional[Callable[[str, int], None]] = None,
) -> IngestResult:
    """
    Parse, chunk, embed and index a PDF file.
    Uses the same VectorStore / agent infrastructure as GitHub repos.
    The repo_id is derived from the filename so each PDF gets its own cache.
    """
    def progress(msg: str, pct: int) -> None:
        if on_progress:
            on_progress(msg, pct)

    # Sanitise filename → stable repo_id
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", filename)
    repo_id   = f"{PDF_ID_PREFIX}{safe_name}"

    progress("Parsing PDF…", 10)
    chunks = parse_pdf(file_bytes, filename)

    progress(f"Embedding {len(chunks)} chunks…", 30)
    texts: list[str] = [c.text for c in chunks]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        all_embeddings.extend(embed_texts(batch))
        done = min(i + EMBED_BATCH, len(texts))
        pct  = 30 + int((done / len(texts)) * 60)
        progress(f"Embedded {done}/{len(texts)} chunks…", pct)

    progress("Saving index…", 95)
    store = VectorStore()
    store.add(chunks, all_embeddings)
    store.save(repo_id)
    set_store(repo_id, store)

    progress("Done!", 100)
    return IngestResult(
        repo_id=repo_id,
        files_count=1,
        chunks_count=len(chunks),
        graph_edges=0,
        files=[{"path": filename, "file_type": "pdf", "chunks": len(chunks)}],
    )


# ─────────────────────────────────────────────────────────────
# Initial retrieval (for source display)
# ─────────────────────────────────────────────────────────────

def retrieve(repo_id: str, question: str) -> tuple[list[SearchResult], list[SearchResult]]:
    """
    Hybrid retrieval (BM25 + cosine) + import-graph augmentation.
    Returns (semantic_results, graph_results).
    """
    store = get_store(repo_id)
    if store is None:
        raise ValueError(f"Repository '{repo_id}' is not indexed yet.")

    query_vec = embed_query(question)
    semantic  = store.hybrid_search(query_vec, question, top_k=TOP_K_SEMANTIC)

    graph = _get_graph(repo_id)
    graph_results: list[SearchResult] = []

    if graph and graph.edge_count > 0:
        matched_files   = {r.chunk.path for r in semantic}
        neighbour_files: set[str] = set()
        for fp in matched_files:
            neighbour_files.update(graph.neighbours(fp, depth=GRAPH_DEPTH))
        neighbour_files -= matched_files

        if neighbour_files:
            all_candidates = store.hybrid_search(
                query_vec, question, top_k=store.size
            )
            seen_files: set[str] = set()
            for r in all_candidates:
                if r.chunk.path in neighbour_files and r.chunk.path not in seen_files:
                    graph_results.append(r)
                    seen_files.add(r.chunk.path)
                    if len(graph_results) >= TOP_K_GRAPH:
                        break

    return semantic, graph_results


def build_sources(
    semantic: list[SearchResult],
    graph_results: list[SearchResult],
) -> list[Source]:
    sources: list[Source] = []
    for via_graph, results in [(False, semantic), (True, graph_results)]:
        for r in results:
            c = r.chunk
            sources.append(Source(
                path=c.path, raw_url=c.raw_url,
                file_type=c.file_type, language=c.language,
                heading=c.heading, unit_name=c.unit_name,
                unit_type=c.unit_type, diagram_type=c.diagram_type,
                score=r.score, via_graph=via_graph,
                preview=c.text[:200].replace("\n", " ").strip() + "…",
            ))
    return sources


# ─────────────────────────────────────────────────────────────
# Streaming generation (agent-powered)
# ─────────────────────────────────────────────────────────────

def stream_answer(
    repo_id: str,
    question: str,
) -> Generator[list[Source] | dict | str, None, None]:
    """
    Yields:
      list[Source]                          — initial retrieved sources
      {"type": "tool_call", "name": str, "query": str}  — agent tool calls
      str                                   — final answer text tokens
    """
    store = get_store(repo_id)
    if store is None:
        raise ValueError(f"Repository '{repo_id}' is not indexed yet.")

    # Initial retrieval — displayed as source cards immediately
    semantic, graph_results = retrieve(repo_id, question)
    sources = build_sources(semantic, graph_results)
    yield sources  # type: ignore[misc]

    # Agent reasoning + streaming answer
    yield from run_agent_stream(store, question)

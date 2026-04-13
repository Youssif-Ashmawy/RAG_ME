"""
RAG pipeline — fully local via Ollama + LangChain agent.

Ingestion:
  1. Fetch ALL source files from GitHub (code + docs)
  2. Parse with language-aware AST/regex chunker
  3. Build import graph (file dependency edges)
  4. Embed all chunks locally (mxbai-embed-large)
  5. Persist hybrid vector store + import graph to .rag-cache/

Querying (agent mode):
  1. Multi-query expansion — LLM generates 2 alternative phrasings
  2. Hybrid retrieval (BM25 + cosine) for each query, results merged + boosted
  3. Import-graph walk adds context from related files
  4. Initial context passed directly to agent (no cold start)
  5. Agent searches further if needed, then streams final answer
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Generator, Optional

import ollama

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

GENERATION_MODEL = "llama3.2"
EMBED_BATCH      = 50
TOP_K_SEMANTIC   = 14   # chunks retrieved per query pass
TOP_K_FINAL      = 10   # chunks kept after multi-query merge
TOP_K_GRAPH      = 4    # extra chunks from graph neighbours
GRAPH_DEPTH      = 1
PDF_ID_PREFIX    = "pdf__"


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
    def progress(msg: str, pct: int) -> None:
        if on_progress:
            on_progress(msg, pct)

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
# Multi-query expansion
# ─────────────────────────────────────────────────────────────

def _expand_queries(question: str) -> list[str]:
    """
    Ask the LLM to generate 2 alternative search queries for the question.
    Returns [original] + up to 2 alternatives.
    Fails gracefully — returns [original] on any error.
    """
    try:
        resp = ollama.chat(
            model=GENERATION_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Write 2 short search queries to find information relevant to this question. "
                    "Output only the queries, one per line, no numbering or explanation.\n\n"
                    f"Question: {question}"
                ),
            }],
            stream=False,
            options={"num_predict": 60},
        )
        lines = [
            ln.strip() for ln in resp.message.content.strip().split("\n")
            if ln.strip() and ln.strip() != question
        ]
        return [question] + lines[:2]
    except Exception:
        return [question]


def _multi_retrieve(
    store: VectorStore,
    question: str,
    top_k_per_query: int,
    top_k_final: int,
) -> list[SearchResult]:
    """
    Run hybrid search for each expanded query, merge results, and re-rank.

    Scoring:
      - Base score = highest RRF score seen for the chunk across all queries
      - Multi-query bonus: +15 % per additional query that also retrieved the chunk
        (chunks relevant to multiple phrasings are more likely to be truly relevant)
    """
    queries = _expand_queries(question)

    score_map: dict[str, float] = {}   # chunk_id → best base score
    chunk_map: dict[str, SearchResult] = {}
    hit_count: dict[str, int] = {}

    for q in queries:
        q_vec   = embed_query(q)
        results = store.hybrid_search(q_vec, q, top_k=top_k_per_query)
        for r in results:
            cid = r.chunk.id
            if cid not in score_map or r.score > score_map[cid]:
                score_map[cid] = r.score
                chunk_map[cid] = r
            hit_count[cid] = hit_count.get(cid, 0) + 1

    # Apply multi-query agreement bonus then sort
    boosted: list[SearchResult] = []
    for cid, r in chunk_map.items():
        bonus = 1.0 + 0.15 * (hit_count[cid] - 1)
        boosted.append(SearchResult(chunk=r.chunk, score=r.score * bonus))

    boosted.sort(key=lambda r: r.score, reverse=True)

    # Re-apply file diversity cap (3 per file) and cap final count
    file_count: dict[str, int] = {}
    results: list[SearchResult] = []
    for r in boosted:
        fc = file_count.get(r.chunk.path, 0)
        if fc < 3 and len(results) < top_k_final:
            results.append(r)
            file_count[r.chunk.path] = fc + 1

    return results


# ─────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────

def retrieve(repo_id: str, question: str) -> tuple[list[SearchResult], list[SearchResult]]:
    """
    Multi-query hybrid retrieval + import-graph augmentation.
    Returns (semantic_results, graph_results).
    """
    store = get_store(repo_id)
    if store is None:
        raise ValueError(f"Repository '{repo_id}' is not indexed yet.")

    semantic = _multi_retrieve(store, question, TOP_K_SEMANTIC, TOP_K_FINAL)

    graph = _get_graph(repo_id)
    graph_results: list[SearchResult] = []

    if graph and graph.edge_count > 0:
        matched_files   = {r.chunk.path for r in semantic}
        neighbour_files: set[str] = set()
        for fp in matched_files:
            neighbour_files.update(graph.neighbours(fp, depth=GRAPH_DEPTH))
        neighbour_files -= matched_files

        if neighbour_files:
            # Cap graph search at 100 candidates for large repos
            q_vec          = embed_query(question)
            all_candidates = store.hybrid_search(q_vec, question, top_k=min(store.size, 100))
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


_CONTEXT_SNIPPET_CHARS = 400   # chars per chunk in the agent's initial context
_CONTEXT_MAX_CHUNKS    = 4     # how many chunks to include (full text available via tools)


def _format_context(
    semantic: list[SearchResult],
    graph_results: list[SearchResult],
) -> str:
    """
    Build a short initial-context string for the agent prompt.

    We pass only the top few chunks at ~400 chars each (~100 tokens each).
    This keeps the initial prompt under ~600 tokens so the model's 8192-token
    context window is mostly free for the ReAct loop and tool observations.
    Full chunk bodies are always retrievable via the search/get tools.
    """
    parts: list[str] = []
    all_results = semantic[:_CONTEXT_MAX_CHUNKS] + graph_results[:1]

    for r in all_results:
        c = r.chunk
        label = c.path
        if c.unit_name:
            label += f" :: {c.unit_name}"
        elif c.heading:
            label += f" › {c.heading}"
        snippet = c.text[:_CONTEXT_SNIPPET_CHARS].replace("\n", " ").strip()
        if len(c.text) > _CONTEXT_SNIPPET_CHARS:
            snippet += "…"
        parts.append(f"[{label}]\n{snippet}")

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Streaming generation (agent-powered)
# ─────────────────────────────────────────────────────────────

def stream_answer(
    repo_id: str,
    question: str,
) -> Generator[list[Source] | dict | str, None, None]:
    """
    Yields:
      list[Source]                                          — retrieved source cards
      {"type": "status",   "msg": str}                     — pipeline status updates
      {"type": "tool_call","name": str, "query": str}      — agent tool invocations
      str                                                   — final answer characters
    """
    store = get_store(repo_id)
    if store is None:
        raise ValueError(f"Repository '{repo_id}' is not indexed yet.")

    source_type = "document" if repo_id.startswith(PDF_ID_PREFIX) else "code"

    # Multi-query retrieval
    yield {"type": "status", "msg": "Expanding queries and retrieving context…"}
    semantic, graph_results = retrieve(repo_id, question)
    sources = build_sources(semantic, graph_results)
    yield sources  # type: ignore[misc]

    # Pass the full retrieved context to the agent so it doesn't start cold
    initial_context = _format_context(semantic, graph_results)

    yield {"type": "status", "msg": "Agent reasoning…"}
    yield from run_agent_stream(
        store, question,
        source_type=source_type,
        initial_context=initial_context,
    )

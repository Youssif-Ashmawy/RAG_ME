"""
RAG pipeline — fully local via Ollama.

Ingestion:
  1. Fetch ALL source files from GitHub (code + docs)
  2. Parse with language-aware AST/regex chunker
  3. Build import graph (file dependency edges)
  4. Embed all chunks locally (nomic-embed-text)
  5. Persist vector store + import graph to .rag-cache/

Querying:
  1. Embed the question
  2. Semantic retrieval (cosine similarity, top-K)
  3. Graph walk — add chunks from import-related files
  4. Build enriched context prompt
  5. Stream answer from llama3.2
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Generator, Optional

import ollama

from .chunker import Chunk, chunk_files
from .code_parser import parse_file
from .embeddings import embed_text, embed_texts
from .github_client import fetch_repo_files
from .import_graph import ImportGraph
from .vector_store import (
    SearchResult, VectorStore,
    get_store, set_store, _cache_dir,
)

GENERATION_MODEL = "llama3.2"
EMBED_BATCH      = 50
TOP_K_SEMANTIC   = 6    # semantic retrieval
TOP_K_GRAPH      = 3    # extra chunks from graph neighbours
GRAPH_DEPTH      = 1    # hops in the import graph


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
    via_graph: bool       # True if added via import graph, not semantic search
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

    # Build import graph from parsed files
    progress("Building import dependency graph…", 35)
    all_paths = {f.path for f in files}
    parsed_files = [parse_file(f.content, f.path) for f in files]
    graph = ImportGraph.build(parsed_files, all_paths)

    progress(f"Embedding {len(chunks)} chunks locally…", 40)
    texts = [c.text for c in chunks]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i: i + EMBED_BATCH]
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
# Retrieval (semantic + graph-augmented)
# ─────────────────────────────────────────────────────────────

def retrieve(repo_id: str, question: str) -> tuple[list[SearchResult], list[SearchResult]]:
    """
    Returns (semantic_results, graph_results).
    semantic_results: top-K by cosine similarity
    graph_results:    top chunks from import-adjacent files
    """
    store = get_store(repo_id)
    if store is None:
        raise ValueError(f"Repository '{repo_id}' is not indexed yet.")

    query_vec = embed_text(question)
    semantic  = store.search(query_vec, top_k=TOP_K_SEMANTIC)

    # Graph augmentation
    graph = _get_graph(repo_id)
    graph_results: list[SearchResult] = []

    if graph and graph.edge_count > 0:
        matched_files = {r.chunk.path for r in semantic}
        neighbour_files: set[str] = set()
        for fp in matched_files:
            neighbour_files.update(graph.neighbours(fp, depth=GRAPH_DEPTH))
        neighbour_files -= matched_files  # exclude already retrieved

        if neighbour_files:
            # Get the single best chunk from each neighbour file
            all_candidates = store.search(query_vec, top_k=len(store._chunks))
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
    for r in semantic:
        c = r.chunk
        sources.append(Source(
            path=c.path, raw_url=c.raw_url,
            file_type=c.file_type, language=c.language,
            heading=c.heading, unit_name=c.unit_name,
            unit_type=c.unit_type, diagram_type=c.diagram_type,
            score=r.score, via_graph=False,
            preview=c.text[:200].replace("\n", " ").strip() + "…",
        ))
    for r in graph_results:
        c = r.chunk
        sources.append(Source(
            path=c.path, raw_url=c.raw_url,
            file_type=c.file_type, language=c.language,
            heading=c.heading, unit_name=c.unit_name,
            unit_type=c.unit_type, diagram_type=c.diagram_type,
            score=r.score, via_graph=True,
            preview=c.text[:200].replace("\n", " ").strip() + "…",
        ))
    return sources


# ─────────────────────────────────────────────────────────────
# Prompt building
# ─────────────────────────────────────────────────────────────

def _build_system_prompt(
    semantic: list[SearchResult],
    graph_results: list[SearchResult],
    repo_id: str,
) -> str:
    graph = _get_graph(repo_id)

    # ── Semantic context ──────────────────────────────────────
    sem_blocks: list[str] = []
    for i, r in enumerate(semantic, 1):
        c = r.chunk
        label = f"[Source {i}]"
        if c.unit_name:
            label += f" {c.language.title()} {c.unit_type} `{c.unit_name}` in {c.path}"
        elif c.file_type == "mmd":
            label += f" Mermaid {c.diagram_type} — {c.path}"
        else:
            label += f" {c.path}" + (f" › {c.heading}" if c.heading else "")
        label += f"  (similarity: {r.score * 100:.0f}%)"

        # Add graph context for this file
        graph_info = ""
        if graph:
            summary = graph.summary(c.path)
            if summary != "no resolved dependencies":
                graph_info = f"\nFile relationships: {summary}"

        sem_blocks.append(f"{label}{graph_info}\n\n{c.text}")

    # ── Graph-augmented context ───────────────────────────────
    graph_blocks: list[str] = []
    for i, r in enumerate(graph_results, len(semantic) + 1):
        c = r.chunk
        label = f"[Source {i}] (via import graph)"
        if c.unit_name:
            label += f" {c.language.title()} {c.unit_type} `{c.unit_name}` in {c.path}"
        else:
            label += f" {c.path}"
        graph_blocks.append(f"{label}\n\n{c.text}")

    semantic_ctx = "\n\n---\n\n".join(sem_blocks)
    graph_ctx    = "\n\n---\n\n".join(graph_blocks) if graph_blocks else ""

    graph_section = ""
    if graph_ctx:
        graph_section = f"""

RELATED FILES (retrieved via import dependency graph):
{graph_ctx}"""

    return f"""You are an expert software engineer assistant helping users understand a codebase.

You have been given code and documentation excerpts retrieved from the repository, \
plus related files discovered through the import dependency graph.

DIRECTLY RELEVANT SOURCES:
{semantic_ctx}{graph_section}

INSTRUCTIONS FOR YOUR ANSWER:
- Explain clearly what the code does, not just what it says.
- Trace data flow and control flow across functions and files when relevant.
- When multiple files interact, describe the chain: "A calls B, which uses C to..."
- For Mermaid diagrams, explain what the diagram represents in plain language.
- Point out design patterns, potential issues, or notable decisions if evident.
- Reference sources with [Source N] labels.
- If the context is insufficient to answer fully, say so and suggest what to look for."""


# ─────────────────────────────────────────────────────────────
# Streaming generation
# ─────────────────────────────────────────────────────────────

def stream_answer(
    repo_id: str,
    question: str,
) -> Generator[str | list[Source], None, None]:
    """
    Yields list[Source] first, then str text chunks.
    """
    semantic, graph_results = retrieve(repo_id, question)
    sources = build_sources(semantic, graph_results)

    yield sources  # type: ignore[misc]

    system_prompt = _build_system_prompt(semantic, graph_results, repo_id)

    stream = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        stream=True,
    )
    for chunk in stream:
        text = chunk.message.content
        if text:
            yield text

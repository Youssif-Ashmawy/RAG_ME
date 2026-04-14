"""
Two-phase RAG agent for repository / document Q&A.

Phase 1 — Context gathering (ReAct agent, tool calls only):
  The agent runs up to MAX_GATHER_ITERS steps using search/get tools to
  collect relevant context.  Its own "Final Answer" text is discarded.

Phase 2 — Direct synthesis (streaming Groq chat):
  All gathered context (initial retrieval + tool observations) is bundled
  into a system prompt and streamed back token-by-token.

Why two phases?
  Separating tool-call reasoning from synthesis makes each task simple
  enough for the model to handle reliably without context overflow.
"""

from __future__ import annotations

import os
from typing import Generator, Literal

import groq as groq_sdk
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_groq import ChatGroq

from .embeddings import embed_query
from .vector_store import VectorStore

AGENT_MODEL      = "llama-3.3-70b-versatile"
TOP_K_AGENT      = 6
MAX_GATHER_ITERS = 4

SourceType = Literal["code", "document"]


# ─────────────────────────────────────────────────────────────
# Phase-1 ReAct prompt  (gather context, don't write final answer)
# ─────────────────────────────────────────────────────────────

_GATHER_TMPL = """\
You are a research assistant gathering information to answer a question.
Your ONLY job is to search for relevant content using the tools below.
Do NOT try to answer the question yourself — just search.

Available tools:
{tools}

Format EXACTLY:
Thought: what to search for next
Action: {tool_names}
Action Input: search query as plain text
Observation: <tool result>
... repeat 1-3 times ...
Thought: I have gathered enough context
Final Answer: done

Rules:
- Action must be one of: {tool_names}
- Action Input is a plain string, never JSON
- Stop after 1-3 searches
- If unsure what sections/files exist, use list_sections or list_files first

Begin!

Question: {input}
{agent_scratchpad}"""


# ─────────────────────────────────────────────────────────────
# Phase-2 synthesis prompts
# ─────────────────────────────────────────────────────────────

_CODE_SYNTHESIS = """\
You are an expert software engineer. Answer the question using ONLY the retrieved context below.
- Explain what the code does, trace data flow across files, highlight design patterns.
- Reference sources by file path and function/class name.
- Be thorough and specific. If context is insufficient, say so."""

_DOCUMENT_SYNTHESIS = """\
You are a precise document analyst. Answer the question using ONLY the retrieved context below.
- Cite page numbers and section headings for every claim you make.
- Quote directly when exact wording matters.
- Be thorough. If the document lacks the information, say so explicitly."""


# ─────────────────────────────────────────────────────────────
# Tool implementations (shared between modes)
# ─────────────────────────────────────────────────────────────

def _make_tools(store: VectorStore, is_doc: bool) -> list[Tool]:
    def _search(query: str) -> str:
        q_vec   = embed_query(query)
        results = store.hybrid_search(q_vec, query, top_k=TOP_K_AGENT)
        if not results:
            return "No results found."
        parts: list[str] = []
        for r in results:
            c = r.chunk
            label = c.path
            if c.unit_name:
                label += f" :: {c.unit_name} ({c.unit_type})"
            elif c.heading:
                label += f" › {c.heading}"
            parts.append(f"[{label}]\n{c.text}")
        return "\n---\n".join(parts)

    def _fetch(ref: str) -> str:
        chunks = store.chunks_for_file(ref)
        if not chunks:
            ref_lower = ref.lower()
            chunks = [
                c for c in store._chunks
                if ref_lower in (c.heading or "").lower()
                or ref_lower in c.path.lower()
            ]
        if not chunks:
            return f"'{ref}' not found in the index."
        return "\n\n---\n\n".join(c.text for c in chunks)

    def _list_sections(_input: str) -> str:
        headings: list[str] = []
        seen: set[str] = set()
        for c in store._chunks:
            h = c.heading or c.unit_name or c.path
            if h and h not in seen:
                seen.add(h)
                headings.append(h)
        if not headings:
            return "No sections found in index."
        return "Indexed sections:\n" + "\n".join(f"- {h}" for h in headings)

    def _list_files(_input: str) -> str:
        paths = sorted({c.path for c in store._chunks})
        if not paths:
            return "No files found in index."
        return "Indexed files:\n" + "\n".join(f"- {p}" for p in paths)

    if is_doc:
        return [
            Tool(name="search_document", func=_search,
                 description="Search the PDF for passages. Input: plain text query."),
            Tool(name="get_section",     func=_fetch,
                 description="Get full text of a section by heading. Input: heading string."),
            Tool(name="list_sections",   func=_list_sections,
                 description="List all sections/headings indexed in the document. Input: any string (ignored)."),
        ]
    else:
        return [
            Tool(name="search_codebase", func=_search,
                 description="Search the codebase for code or docs. Input: plain text query."),
            Tool(name="get_file",        func=_fetch,
                 description="Get full content of a file by path. Input: file path string."),
            Tool(name="list_files",      func=_list_files,
                 description="List all files indexed in the codebase. Input: any string (ignored)."),
        ]


# ─────────────────────────────────────────────────────────────
# Phase 1 — context-gathering agent
# ─────────────────────────────────────────────────────────────

def _gather_context(
    store: VectorStore,
    question: str,
    source_type: SourceType,
    api_key: str,
) -> tuple[list[dict], list[str]]:
    is_doc = source_type == "document"
    tools  = _make_tools(store, is_doc)

    prompt = PromptTemplate.from_template(_GATHER_TMPL)
    llm    = ChatGroq(model=AGENT_MODEL, temperature=0, groq_api_key=api_key)

    agent    = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=MAX_GATHER_ITERS,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        verbose=False,
    )

    try:
        result = executor.invoke({"input": question})
    except Exception:
        result = {"intermediate_steps": []}

    tool_events:  list[dict] = []
    observations: list[str]  = []

    for action, obs in result.get("intermediate_steps", []):
        tool_name  = getattr(action, "tool", "")
        tool_input = getattr(action, "tool_input", "")
        query = tool_input if isinstance(tool_input, str) else str(tool_input)
        tool_events.append({"type": "tool_call", "name": tool_name, "query": query})
        if isinstance(obs, str) and obs.strip():
            observations.append(obs)

    return tool_events, observations


# ─────────────────────────────────────────────────────────────
# Phase 2 — direct streaming synthesis
# ─────────────────────────────────────────────────────────────

def _synthesise(
    question: str,
    source_type: SourceType,
    initial_context: str,
    tool_observations: list[str],
    api_key: str,
) -> Generator[str, None, None]:
    base_prompt = _DOCUMENT_SYNTHESIS if source_type == "document" else _CODE_SYNTHESIS

    context_parts: list[str] = []
    if initial_context:
        context_parts.append(initial_context)
    context_parts.extend(tool_observations)

    context_block = "\n\n---\n\n".join(context_parts)
    system = f"{base_prompt}\n\n{context_block}" if context_block else base_prompt

    client = groq_sdk.Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ],
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_agent_stream(
    store: VectorStore,
    question: str,
    source_type: SourceType = "code",
    initial_context: str = "",
    api_key: str = "",
) -> Generator[dict | str, None, None]:
    """
    Yields:
      {"type": "tool_call", "name": str, "query": str}  — each tool invocation
      str                                                 — final answer tokens
    """
    resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")

    tool_events, observations = _gather_context(store, question, source_type, resolved_key)
    yield from tool_events

    yield from _synthesise(question, source_type, initial_context, observations, resolved_key)

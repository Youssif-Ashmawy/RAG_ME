"""
LangChain tool-calling agent for repository Q&A.

The agent has two tools:
  search_codebase  — hybrid search (BM25 + vector) over indexed chunks
  get_file         — retrieve all chunks from a specific file path

Flow:
  1. Agent decides which tools to call (and with what queries)
  2. Tools return formatted context
  3. Agent synthesises a final answer
  4. Final answer is streamed token-by-token back to the caller

Requires: langchain, langchain-ollama
"""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Any, Generator, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama

from .embeddings import embed_query
from .vector_store import VectorStore

AGENT_MODEL  = "llama3.2"
TOP_K_AGENT  = 5     # results per search_codebase call
MAX_ITERS    = 6     # max agent reasoning steps

SYSTEM_PROMPT = """\
You are an expert software engineer assistant. You have access to a fully indexed \
code repository and can search it using the tools provided.

Guidelines:
- Use search_codebase to find relevant functions, classes, or documentation.
- Use get_file when you know the exact file path and need its full content.
- Call tools multiple times with different queries if the first result is insufficient.
- After gathering enough context, write a thorough answer:
    • Explain what the code does, not just what it says.
    • Trace data flow and control flow across files when relevant.
    • Describe design patterns or notable architectural decisions.
    • Reference sources by file path and function/class name.
- If the context is insufficient, say so and suggest what to look for."""


# ─────────────────────────────────────────────────────────────
# Streaming callback
# ─────────────────────────────────────────────────────────────

class _TokenQueue(BaseCallbackHandler):
    """Pushes LLM tokens into a thread-safe queue for the generator to read."""

    DONE = object()  # sentinel

    def __init__(self) -> None:
        self.q: Queue = Queue()

    def on_llm_new_token(self, token: str, **_: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        self.q.put(self.DONE)

    def on_llm_error(self, error: Exception, **_: Any) -> None:
        self.q.put(self.DONE)


# ─────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────

def build_agent(store: VectorStore) -> AgentExecutor:
    """
    Build a tool-calling AgentExecutor bound to the given VectorStore.
    The LLM used for tool-calling decisions is non-streaming; the final
    synthesis step is handled separately with a streaming LLM.
    """

    # ── Tools ────────────────────────────────────────────────────

    @tool
    def search_codebase(query: str) -> str:
        """
        Search the indexed repository for code, documentation, or concepts
        matching the query. Returns the most relevant snippets with file paths.
        Use different queries to find different aspects.
        """
        q_vec   = embed_query(query)
        results = store.hybrid_search(q_vec, query, top_k=TOP_K_AGENT)
        if not results:
            return "No results found."
        lines: list[str] = []
        for r in results:
            c = r.chunk
            label = c.path
            if c.unit_name:
                label += f" :: {c.unit_name} ({c.unit_type})"
            elif c.heading:
                label += f" › {c.heading}"
            lines.append(f"[{label}]\n{c.text}\n")
        return "\n---\n".join(lines)

    @tool
    def get_file(path: str) -> str:
        """
        Retrieve the full indexed content of a specific file by its exact path
        (e.g. 'src/auth.py'). Use this when search_codebase points to a file
        but you need more complete context.
        """
        chunks = store.chunks_for_file(path)
        if not chunks:
            # Try case-insensitive partial match
            path_lower = path.lower()
            chunks = [
                c for c in store._chunks
                if path_lower in c.path.lower()
            ]
        if not chunks:
            return f"File '{path}' not found in the index."
        return "\n\n---\n\n".join(c.text for c in chunks)

    tools = [search_codebase, get_file]

    # ── LLM (non-streaming for tool-call decisions) ───────────────
    llm = ChatOllama(model=AGENT_MODEL, temperature=0)

    # ── Prompt ────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent    = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=MAX_ITERS,
        return_intermediate_steps=True,
        verbose=False,
    )
    return executor


# ─────────────────────────────────────────────────────────────
# Streaming runner
# ─────────────────────────────────────────────────────────────

def run_agent_stream(
    store: VectorStore,
    question: str,
) -> Generator[dict | str, None, None]:
    """
    Run the agent and stream results back.

    Yields:
      {"type": "tool_call", "name": str, "query": str}  — when a tool is invoked
      str                                                 — final answer tokens
    """
    executor = build_agent(store)

    # Run agent (non-streaming) to collect tool calls + final output
    result = executor.invoke({"input": question})

    # Emit tool call notifications so the UI can show what was searched
    for action, _observation in result.get("intermediate_steps", []):
        tool_input = action.tool_input
        query = (
            tool_input.get("query") or tool_input.get("path") or str(tool_input)
            if isinstance(tool_input, dict) else str(tool_input)
        )
        yield {"type": "tool_call", "name": action.tool, "query": query}

    # Stream the final answer token-by-token
    final_answer: str = result.get("output", "")
    if not final_answer:
        return

    token_cb  = _TokenQueue()
    streaming_llm = ChatOllama(
        model=AGENT_MODEL,
        temperature=0,
        streaming=True,
        callbacks=[token_cb],
    )

    def _stream_in_thread() -> None:
        streaming_llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
                {"role": "assistant", "content": final_answer},
            ]
        )

    # We already have the full answer from the agent run.
    # Stream it character-by-character to simulate token streaming
    # without a second LLM call — gives smooth UI without extra latency.
    chunk_size = 4
    for i in range(0, len(final_answer), chunk_size):
        yield final_answer[i : i + chunk_size]

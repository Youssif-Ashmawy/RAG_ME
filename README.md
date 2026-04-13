# Local RAG Assistant

A fully local Retrieval-Augmented Generation (RAG) system that lets you chat with **GitHub repositories** or **PDF documents** using models running on your machine via Ollama. No API keys, no cloud calls.

---

## What it does

| Mode | What gets indexed | Ask questions about |
|---|---|---|
| **GitHub** | All source files (code, docs, Mermaid diagrams) from any public repo | Architecture, functions, classes, how things work |
| **PDF** | Any PDF document, page by page | Content, summaries, specific sections |

The system retrieves the most relevant context using hybrid search, then a two-phase LangChain agent reasons over it and streams back a thorough answer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                             │
│                                                                         │
│  GitHub URL ──► github_client.py ──► Raw file content                  │
│                                            │                            │
│  PDF bytes  ──► pdf_parser.py    ──►       │                            │
│                                      chunker.py / code_parser.py       │
│                                            │                            │
│                                     List[Chunk]                         │
│                                            │                            │
│                                     embeddings.py                       │
│                                    (mxbai-embed-large via Ollama)       │
│                                            │                            │
│                              ┌─────────────┴──────────────┐            │
│                              │       vector_store.py       │            │
│                              │  • numpy cosine vectors     │            │
│                              │  • BM25Okapi keyword index  │            │
│                              │  • Persisted to .rag-cache/ │            │
│                              └─────────────────────────────┘            │
│                                                                         │
│  (GitHub only) import_graph.py — file dependency graph saved alongside  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                                 │
│                                                                         │
│  User question                                                          │
│       │                                                                 │
│       ▼                                                                 │
│  rag_pipeline.py: _expand_queries()                                     │
│  LLM generates 2 alternative phrasings → [q1, q2, q3]                  │
│       │                                                                 │
│       ▼                                                                 │
│  _multi_retrieve() — hybrid search for each query                      │
│  • Vector: cosine similarity (mxbai-embed-large)                        │
│  • Keyword: BM25Okapi                                                   │
│  • Fusion: Reciprocal Rank Fusion (RRF, k=60)                          │
│  • Multi-query bonus: +15% per extra query that also returns the chunk  │
│  • Diversity cap: max 3 chunks per file                                 │
│       │                                                                 │
│  (GitHub only) import_graph walk — adds context from related files     │
│       │                                                                 │
│       ▼                                                                 │
│  agent.py: run_agent_stream()                                           │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  Phase 1 — ReAct Agent (LangChain + llama3.2)            │          │
│  │  Tools available:                                         │          │
│  │   • search_document / search_codebase (hybrid search)    │          │
│  │   • get_section / get_file (fetch full content by ref)   │          │
│  │   • list_sections / list_files (discover what's indexed) │          │
│  │  Agent gathers context via tool calls only               │          │
│  └──────────────────────────────────────────────────────────┘          │
│       │                                                                 │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  Phase 2 — Direct synthesis (ollama.chat, streaming)     │          │
│  │  All retrieved context bundled into system prompt        │          │
│  │  Answer streamed token-by-token to the UI                │          │
│  └──────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tech stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| LLM & embeddings | [Ollama](https://ollama.com) — `llama3.2`, `mxbai-embed-large` |
| Agent framework | [LangChain](https://langchain.com) (`create_react_agent`) |
| Vector search | NumPy cosine similarity |
| Keyword search | `rank-bm25` (BM25Okapi) |
| PDF parsing | `pdfplumber` |
| Code parsing | AST (Python), regex (JS/TS/Go/Rust/…) |

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally

Pull the required models once:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <this-repo>
cd Test_RAG

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set a GitHub token to avoid rate limits on large repos
echo "GITHUB_TOKEN=ghp_your_token_here" > .env
```

---

## Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Chat with a GitHub repository

1. Select **GitHub Repository** in the sidebar
2. Paste a public repo URL (e.g. `https://github.com/owner/repo`)
3. Click **Index Repository** — progress is shown live
4. Ask questions in the chat

Supported file types: `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.kt`, `.cs`, `.rb`, `.md`, `.mmd`, `.txt`, and more.

### Chat with a PDF

1. Select **PDF Document** in the sidebar
2. Upload a PDF file
3. Click **Index PDF**
4. Ask questions in the chat

The index is cached to `.rag-cache/` so re-opening the app does not require re-indexing.

---

## Project structure

```
app.py                  — Streamlit UI
src/
  rag_pipeline.py       — Ingestion + query orchestration
  agent.py              — Two-phase LangChain ReAct agent
  vector_store.py       — Hybrid BM25 + vector store with RRF
  embeddings.py         — Ollama embed wrapper (mxbai-embed-large)
  chunker.py            — Route files to the right chunking strategy
  code_parser.py        — AST/regex semantic unit extractor
  pdf_parser.py         — pdfplumber-based PDF → sections → chunks
  github_client.py      — GitHub REST API file fetcher
  import_graph.py       — File-level dependency graph
.rag-cache/             — Persisted indexes (auto-created, gitignored)
```

---

## How retrieval works

1. **Query expansion** — the LLM rephrases the question into 2 additional queries, casting a wider net
2. **Hybrid search** — each query runs both BM25 (keyword) and cosine similarity (semantic); the two ranked lists are merged with Reciprocal Rank Fusion
3. **Multi-query bonus** — chunks that appear in results for more than one query phrasing get a score boost, rewarding genuine relevance
4. **Graph augmentation** *(GitHub mode only)* — files that import or are imported by the top results are also retrieved, adding related context automatically
5. **Two-phase agent** — a ReAct agent first uses search/fetch tools to gather more targeted context, then a second direct LLM call synthesises and streams the final answer

---

## Notes

- All processing is local — no data leaves your machine
- The index is stored in `.rag-cache/` and loaded on startup; delete this folder to force a clean re-index
- For large repositories, a `GITHUB_TOKEN` is recommended to avoid the 60 req/h unauthenticated API rate limit
- Scanned/image-only PDFs are not supported (no OCR); the PDF must contain selectable text

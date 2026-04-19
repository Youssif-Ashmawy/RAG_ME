# RAG ME! - GitHub Repository Assistant

A cloud-ready Retrieval-Augmented Generation (RAG) system that lets you chat with any **public GitHub repository**. Powered by [Groq](https://console.groq.com) for fast LLM inference and [fastembed](https://github.com/qdrant/fastembed) for fully local embeddings. No local GPU required.

🌐 Live site: https://rag-me.streamlit.app/
---

## Architecture

![Architecture](architecture.svg)

### Tech stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| LLM inference | [Groq](https://console.groq.com) (`llama-3.3-70b-versatile`) |
| Embeddings | [fastembed](https://github.com/qdrant/fastembed) (`mxbai-embed-large`, local ONNX) |
| Agent framework | [LangChain](https://langchain.com) (`create_react_agent`) |
| Vector search | NumPy cosine similarity |
| Keyword search | `rank-bm25` (BM25Okapi) |
| Code parsing | AST (Python), regex (JS/TS/Go/Rust and more) |

---

## Requirements

- Python 3.10+
- A free [Groq API key](https://console.groq.com) - entered in the app on first launch
- (Optional) A [GitHub token](https://github.com/settings/tokens) - recommended for large repos

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
```

---

## Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. You will be prompted for your Groq API key on first launch.

---

## Usage

1. Enter your Groq API key (and optionally a GitHub token) in the sidebar
2. Paste a public GitHub repo URL (e.g. `https://github.com/owner/repo`)
3. Click **Index Repository** - progress is shown live
4. Ask questions in the chat

Supported file types: `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.kt`, `.cs`, `.rb`, `.md`, `.mmd`, `.txt`, and more.

The index is cached to `.rag-cache/` so re-opening the app does not require re-indexing.

---

## Project structure

```
app.py               - Streamlit UI
src/
  rag_pipeline.py    - Ingestion + query orchestration
  agent.py           - Two-phase LangChain ReAct agent
  vector_store.py    - Hybrid BM25 + vector store with RRF
  embeddings.py      - fastembed wrapper (mxbai-embed-large)
  chunker.py         - Route files to the right chunking strategy
  code_parser.py     - AST/regex semantic unit extractor
  github_client.py   - GitHub REST API file fetcher
  import_graph.py    - File-level dependency graph
  diagram.py         - Mermaid diagram + graph summary builder
.rag-cache/          - Persisted indexes (auto-created, gitignored)
```

---

## How retrieval works

1. **Query expansion** - the LLM rephrases the question into 2 additional queries, casting a wider net
2. **Hybrid search** - each query runs both BM25 (keyword) and cosine similarity (semantic); the two ranked lists are merged with Reciprocal Rank Fusion
3. **Multi-query bonus** - chunks that appear in results for more than one query phrasing get a score boost, rewarding genuine relevance
4. **Graph augmentation** - files that import or are imported by the top results are also retrieved, adding related context automatically
5. **Two-phase agent** - a ReAct agent first uses search/fetch tools to gather more targeted context, then a second direct LLM call synthesises and streams the final answer

---

## Notes

- Embeddings run fully locally via ONNX (no GPU needed, no Ollama required)
- The index is stored in `.rag-cache/` and loaded on startup; delete this folder to force a clean re-index
- For large repositories, a GitHub token is recommended to avoid the 60 req/h unauthenticated API rate limit
- Each user supplies their own API keys - no shared credentials are stored server-side

---

## License

Youssif Ashmawy © 2026

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

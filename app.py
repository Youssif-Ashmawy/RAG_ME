"""
GitHub RAG — Streamlit interface.

Paste a public GitHub repository URL, index it once, then ask any question
about its documentation (.md, .mdx, .txt) and diagrams (.mmd).
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# ── Env + path ────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.rag_pipeline import IngestResult, Source, ingest_repo, ingest_pdf, stream_answer  # noqa: E402

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage { padding: 0.5rem 0; }
    .source-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.82rem;
        line-height: 1.4;
    }
    .source-card a { color: #89b4fa; text-decoration: none; }
    .source-card a:hover { text-decoration: underline; }
    .source-badge {
        display: inline-block;
        background: #313244;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.72rem;
        margin-left: 6px;
        vertical-align: middle;
    }
    .score-bar {
        height: 3px;
        background: linear-gradient(90deg, #89b4fa, #cba6f7);
        border-radius: 2px;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "github"          # "github" | "pdf"
if "repo_id" not in st.session_state:
    st.session_state.repo_id = None
if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, sources?}]


# ═════════════════════════════════════════════════════════════
# Helper — source rendering (defined early, used throughout)
# ═════════════════════════════════════════════════════════════

def render_sources(sources: list) -> None:
    """Render a collapsible source list below an assistant message."""
    if not sources:
        return

    with st.expander(f"📎 {len(sources)} source(s) retrieved", expanded=False):
        for src in sources:
            # Accept both Source dataclass and plain dict (from session history)
            if isinstance(src, dict):
                path         = src["path"]
                raw_url      = src["raw_url"]
                file_type    = src["file_type"]
                heading      = src.get("heading")
                score        = src["score"]
                preview      = src["preview"]
            else:
                path         = src.path
                raw_url      = src.raw_url
                file_type    = src.file_type
                heading      = src.heading
                score        = src.score
                preview      = src.preview

            via_graph = src.get("via_graph", False) if isinstance(src, dict) else src.via_graph
            unit_name = src.get("unit_name") if isinstance(src, dict) else src.unit_name
            unit_type = src.get("unit_type") if isinstance(src, dict) else src.unit_type
            language  = src.get("language", "") if isinstance(src, dict) else src.language

            if file_type == "mmd":
                badge = "📊 MMD"
            elif unit_type in ("function", "method"):
                badge = f"⚙️ {language.title()} fn"
            elif unit_type == "class":
                badge = f"🔷 {language.title()} class"
            else:
                badge = f"📄 {file_type.upper()}"

            if via_graph:
                badge += " 🔗graph"

            label = path
            if unit_name:
                label += f" :: {unit_name}"
            elif heading:
                label += f" › {heading}"

            bar_width = int(score * 100)
            link_html = (
                f'<a href="{raw_url}" target="_blank">View on GitHub ↗</a>'
                if raw_url else ""
            )

            st.markdown(
                f"""<div class="source-card">
                    <strong>{label}</strong>
                    <span class="source-badge">{badge}</span>
                    <span class="source-badge" style="color:#a6e3a1">{score*100:.0f}% match</span>
                    <div style="margin-top:6px;color:#cdd6f4;font-size:0.8rem">{preview}</div>
                    <div class="score-bar" style="width:{bar_width}%"></div>
                    {"<div style='margin-top:6px'>" + link_html + "</div>" if link_html else ""}
                </div>""",
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════
# Sidebar — mode toggle + source management
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📚 RAG Assistant")
    st.divider()

    # ── Mode toggle ──────────────────────────────────────────
    mode = st.radio(
        "Source",
        options=["github", "pdf"],
        format_func=lambda m: "🐙 GitHub Repository" if m == "github" else "📄 PDF Document",
        index=0 if st.session_state.mode == "github" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode != st.session_state.mode:
        # Reset state when switching modes
        st.session_state.mode          = mode
        st.session_state.repo_id       = None
        st.session_state.ingest_result = None
        st.session_state.messages      = []
        st.rerun()

    st.divider()

    # ══════════════════════════════════════════════════════════
    # GitHub mode
    # ══════════════════════════════════════════════════════════
    if st.session_state.mode == "github":
        st.caption("Index a public repo, then chat with its code and docs.")

        repo_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/owner/repo",
            help="Paste any public GitHub repository URL.",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            index_btn = st.button(
                "Index Repository",
                type="primary",
                use_container_width=True,
                disabled=not bool(repo_url.strip()),
            )
        with col2:
            reindex_btn = st.button(
                "Re-index",
                use_container_width=True,
                disabled=not bool(repo_url.strip()),
                help="Force re-index even if already cached.",
            )

        if (index_btn or reindex_btn) and repo_url.strip():
            st.divider()
            progress_text = st.empty()
            progress_bar  = st.progress(0)

            def on_progress(msg: str, pct: int) -> None:
                progress_text.markdown(f"*{msg}*")
                progress_bar.progress(pct / 100)

            try:
                result: IngestResult = ingest_repo(repo_url.strip(), on_progress)
                st.session_state.repo_id       = result.repo_id
                st.session_state.ingest_result = result
                st.session_state.messages      = []

                progress_bar.progress(1.0)
                progress_text.markdown("")
                st.success(
                    f"Indexed **{result.files_count}** files · "
                    f"**{result.chunks_count}** chunks · "
                    f"**{result.graph_edges}** import edges",
                    icon="✅",
                )
            except Exception as exc:
                progress_bar.empty()
                progress_text.empty()
                st.error(f"**Error:** {exc}", icon="🚨")

        # Indexed files list
        ingest_result: IngestResult | None = st.session_state.ingest_result
        if ingest_result:
            st.divider()
            st.markdown(f"**Indexed files** ({len(ingest_result.files)})")

            mmd_files = [f for f in ingest_result.files if f["file_type"] == "mmd"]
            doc_files = [f for f in ingest_result.files if f["file_type"] not in ("mmd",)]

            if mmd_files:
                with st.expander(f"📊 Mermaid diagrams ({len(mmd_files)})", expanded=False):
                    for f in mmd_files:
                        chunks_label = f"{f['chunks']} chunk{'s' if f['chunks'] != 1 else ''}"
                        st.markdown(
                            f"<small><code>{f['path']}</code> "
                            f"<span style='color:#666'>· {chunks_label}</span></small>",
                            unsafe_allow_html=True,
                        )

            if doc_files:
                with st.expander(f"📄 Docs & Code ({len(doc_files)})", expanded=False):
                    for f in doc_files:
                        chunks_label = f"{f['chunks']} chunk{'s' if f['chunks'] != 1 else ''}"
                        st.markdown(
                            f"<small><code>{f['path']}</code> "
                            f"<span style='color:#666'>· {chunks_label}</span></small>",
                            unsafe_allow_html=True,
                        )

    # ══════════════════════════════════════════════════════════
    # PDF mode
    # ══════════════════════════════════════════════════════════
    else:
        st.caption("Upload a PDF, then ask questions about it.")

        uploaded = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload any PDF document to index and query.",
        )

        if uploaded is not None:
            file_bytes = uploaded.read()
            # Only re-index if the file changed
            expected_id = f"pdf__{uploaded.name.replace('.', '_')}"
            already_indexed = st.session_state.repo_id == expected_id

            col1, col2 = st.columns([2, 1])
            with col1:
                index_pdf_btn = st.button(
                    "Index PDF",
                    type="primary",
                    use_container_width=True,
                    disabled=already_indexed,
                )
            with col2:
                reindex_pdf_btn = st.button(
                    "Re-index",
                    use_container_width=True,
                    help="Force re-embed this PDF.",
                )

            if index_pdf_btn or reindex_pdf_btn:
                st.divider()
                progress_text = st.empty()
                progress_bar  = st.progress(0)

                def on_pdf_progress(msg: str, pct: int) -> None:
                    progress_text.markdown(f"*{msg}*")
                    progress_bar.progress(pct / 100)

                try:
                    result: IngestResult = ingest_pdf(
                        file_bytes, uploaded.name, on_pdf_progress
                    )
                    st.session_state.repo_id       = result.repo_id
                    st.session_state.ingest_result = result
                    st.session_state.messages      = []

                    progress_bar.progress(1.0)
                    progress_text.markdown("")
                    st.success(
                        f"Indexed **{result.chunks_count}** chunks from "
                        f"**{uploaded.name}**",
                        icon="✅",
                    )
                except Exception as exc:
                    progress_bar.empty()
                    progress_text.empty()
                    st.error(f"**Error:** {exc}", icon="🚨")

            # Show chunk summary if already indexed
            ingest_result = st.session_state.ingest_result
            if ingest_result and st.session_state.repo_id and \
                    st.session_state.repo_id.startswith("pdf__"):
                st.divider()
                st.markdown(
                    f"**{uploaded.name}** · {ingest_result.chunks_count} chunks indexed"
                )

    # ── Environment status (shared) ──────────────────────────
    st.divider()
    try:
        import ollama as _ollama
        _ollama.list()
        ollama_status = "✅ Running"
    except Exception:
        ollama_status = "❌ Not running"
    st.markdown("**Environment**")
    st.markdown(
        f"<small>Ollama: {ollama_status}<br>"
        f"Generation: <code>llama3.2</code><br>"
        f"Embeddings: <code>mxbai-embed-large</code><br>"
        f"GITHUB_TOKEN: {'✅ Set' if os.getenv('GITHUB_TOKEN') else '⚪ Optional'}</small>",
        unsafe_allow_html=True,
    )

    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ═════════════════════════════════════════════════════════════
# Main area — chat interface
# ═════════════════════════════════════════════════════════════

if not st.session_state.repo_id:
    # ── Welcome screen ──
    is_pdf_mode = st.session_state.mode == "pdf"
    st.markdown(
        f"""
        <div style="text-align:center; padding: 3rem 1rem;">
            <div style="font-size:3.5rem">{"📄" if is_pdf_mode else "📚"}</div>
            <h2 style="margin-top:0.5rem">
                {"Chat with any PDF document" if is_pdf_mode else "Chat with any GitHub repository"}
            </h2>
            <p style="color:#888; max-width:520px; margin:0 auto 2rem">
                {"Upload a PDF in the sidebar and click <strong>Index PDF</strong>. The pipeline extracts text page-by-page, chunks it, embeds with <code>mxbai-embed-large</code>, and stores a searchable hybrid index." if is_pdf_mode else
                "Paste a public repository URL in the sidebar and click <strong>Index Repository</strong>. The pipeline fetches all source files, parses them with AST-aware chunking, builds an import graph, and stores a hybrid vector index."}
                Then ask anything!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    if is_pdf_mode:
        with c1:
            st.info("**📄 PDF upload**\n\nDrop any PDF — research papers, manuals, reports. Text is extracted page-by-page.")
        with c2:
            st.info("**🔍 Hybrid search**\n\nBM25 keyword matching + vector similarity fused with Reciprocal Rank Fusion.")
        with c3:
            st.info("**🤖 Agent reasoning**\n\nLangChain agent searches multiple times and synthesises a thorough answer.")
    else:
        with c1:
            st.info("**🐙 All file types**\n\nCode, docs, Mermaid diagrams. AST-aware chunking for functions and classes.")
        with c2:
            st.info("**🔗 Import graph**\n\nFile dependency graph adds related-file context automatically.")
        with c3:
            st.info("**🤖 Agent reasoning**\n\nLangChain agent searches multiple times and synthesises a thorough answer.")

    st.markdown(
        "<div style='text-align:center;color:#555;font-size:0.8rem;margin-top:2rem'>"
        "Runs fully locally via <strong>Ollama</strong> — no API key needed"
        "</div>",
        unsafe_allow_html=True,
    )

else:
    # ── Chat interface ──
    repo_id = st.session_state.repo_id

    # Context badge
    if repo_id.startswith("pdf__"):
        ingest_res = st.session_state.ingest_result
        badge_label = ingest_res.files[0]["path"] if ingest_res else repo_id
        badge_icon, badge_color = "📄", "#3a1e5f"
    else:
        badge_label = repo_id.replace("__", "/")
        badge_icon, badge_color = "📂", "#1e3a5f"

    st.markdown(
        f"<div style='margin-bottom:1rem'>"
        f"<span style='background:{badge_color};border-radius:6px;padding:4px 10px;"
        f"font-size:0.8rem;color:#89b4fa'>{badge_icon} {badge_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    placeholder = (
        "Ask anything about this document…"
        if repo_id.startswith("pdf__")
        else "Ask anything about this repository…"
    )
    if question := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            status_slot = st.empty()
            answer_slot = st.empty()

            full_answer = ""
            collected_sources: list[Source] = []

            try:
                for item in stream_answer(repo_id, question):
                    if isinstance(item, list):
                        # Initial source retrieval
                        collected_sources = item
                        status_slot.markdown(
                            f"*Found {len(collected_sources)} relevant source(s)…*"
                        )
                    elif isinstance(item, dict) and item.get("type") == "tool_call":
                        # Agent is doing a follow-up search
                        icon = "🔍" if item["name"] == "search_codebase" else "📄"
                        status_slot.markdown(
                            f"*{icon} Agent searching: `{item['query']}`…*"
                        )
                    else:
                        full_answer += item
                        answer_slot.markdown(full_answer + "▌")

                # Final render — remove cursor
                answer_slot.markdown(full_answer)
                status_slot.empty()

                if collected_sources:
                    render_sources(collected_sources)

            except Exception as exc:
                answer_slot.error(f"**Error:** {exc}")
                collected_sources = []

        # Persist to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_answer,
                "sources": [
                    {
                        "path":         s.path,
                        "raw_url":      s.raw_url,
                        "file_type":    s.file_type,
                        "language":     s.language,
                        "heading":      s.heading,
                        "unit_name":    s.unit_name,
                        "unit_type":    s.unit_type,
                        "diagram_type": s.diagram_type,
                        "score":        s.score,
                        "via_graph":    s.via_graph,
                        "preview":      s.preview,
                    }
                    for s in collected_sources
                ],
            }
        )

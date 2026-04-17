"""
GitHub RAG — Streamlit interface.

Paste a public GitHub repository URL, index it once, then ask any question
about its code and documentation.
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# ── Env + path ────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

import groq as _groq_sdk  # noqa: E402

from src.diagram import build_mermaid, build_graph_summary  # noqa: E402
from src.import_graph import ImportGraph  # noqa: E402
from src.rag_pipeline import IngestResult, Source, ingest_repo, stream_answer, stream_diagram_explanation  # noqa: E402
from src.vector_store import get_store, _cache_dir  # noqa: E402


def _validate_groq_key(key: str) -> tuple[bool, str]:
    """Return (is_valid, error_message). Makes a minimal API call to verify the key."""
    try:
        client = _groq_sdk.Groq(api_key=key)
        client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True, ""
    except _groq_sdk.AuthenticationError:
        return False, "Invalid API key — check your key at console.groq.com"
    except _groq_sdk.APIConnectionError:
        return False, "Could not reach Groq — check your internet connection"
    except Exception as exc:
        return False, f"Unexpected error: {exc}"


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG ME!",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="InputInstructions"] { display: none; }
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
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "github_token" not in st.session_state:
    st.session_state.github_token = ""
if "repo_id" not in st.session_state:
    st.session_state.repo_id = None
if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ═════════════════════════════════════════════════════════════
# Diagram dialog
# ═════════════════════════════════════════════════════════════

@st.dialog("Dependency Diagram", width="large")
def _show_diagram(repo_id: str) -> None:
    store = get_store(repo_id)
    if store is None:
        st.error("Index not found.")
        return

    graph = ImportGraph.load(str(_cache_dir(repo_id)))
    mermaid = build_mermaid(store, graph)

    if not mermaid:
        st.info("No dependency data found for this repository.")
        return

    edge_count = graph.edge_count if graph else 0
    file_count = len({c.path for c in store._chunks})
    st.caption(f"{file_count} files · {edge_count} import edges")

    import base64
    _html = (
        f'<!DOCTYPE html><html><head>'
        f'<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>'
        f'<script>mermaid.initialize({{startOnLoad:true,theme:"dark",flowchart:{{useMaxWidth:true,htmlLabels:true}}}});</script>'
        f'<style>body{{margin:0;background:#0e1117}}.mermaid{{width:100%}}</style>'
        f'</head><body><div class="mermaid">{mermaid}</div></body></html>'
    )
    _src = "data:text/html;base64," + base64.b64encode(_html.encode()).decode()
    st.iframe(src=_src, height=620)

    st.divider()
    if st.button("🤖 Explain this diagram", use_container_width=True, type="primary"):
        summary = build_graph_summary(store, graph)
        st.write_stream(
            stream_diagram_explanation(summary, api_key=st.session_state.groq_api_key)
        )


# ═════════════════════════════════════════════════════════════
# Helper — source rendering
# ═════════════════════════════════════════════════════════════

def render_sources(sources: list) -> None:
    if not sources:
        return

    with st.expander(f"📎 {len(sources)} source(s) retrieved", expanded=False):
        for src in sources:
            if isinstance(src, dict):
                path      = src["path"]
                raw_url   = src["raw_url"]
                file_type = src["file_type"]
                heading   = src.get("heading")
                score     = src["score"]
                preview   = src["preview"]
                via_graph = src.get("via_graph", False)
                unit_name = src.get("unit_name")
                unit_type = src.get("unit_type")
                language  = src.get("language", "")
            else:
                path      = src.path
                raw_url   = src.raw_url
                file_type = src.file_type
                heading   = src.heading
                score     = src.score
                preview   = src.preview
                via_graph = src.via_graph
                unit_name = src.unit_name
                unit_type = src.unit_type
                language  = src.language

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
# Sidebar
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📚 RAG ME Assistant")
    st.divider()

    # ── API keys ──────────────────────────────────────────────
    if st.session_state.groq_api_key:
        st.markdown(
            "<small>"
            f"🔑 Groq key {'✓' if st.session_state.groq_api_key else '✗'} <br>"
            f"{'🐙 GitHub token ✓' if st.session_state.github_token else '🐙 GitHub token <span style=\"color:#888\">optional</span>'} <br>"
            "</small>",
            unsafe_allow_html=True,
        )
        if st.button("Change keys", use_container_width=True):
            st.session_state.groq_api_key = ""
            st.session_state.github_token = ""
            st.rerun()
    else:
        st.markdown("**Enter your API keys to get started**")
        with st.form("groq_key_form", border=False):
            key_input = st.text_input(
                "Groq API key",
                type="password",
                placeholder="gsk_…",
            )
            gh_input = st.text_input(
                "GitHub token (optional)",
                type="password",
                placeholder="ghp_…",
            )
            st.caption("Recommended for large repos as it raises GitHub API rate limit from 60 to 5,000 requests/hour.")
            submitted = st.form_submit_button(
                "Connect", use_container_width=True, type="primary"
            )
        if submitted:
            if not key_input.strip():
                st.error("Please enter a Groq API key.")
            else:
                with st.spinner("Validating Groq key…"):
                    valid, err = _validate_groq_key(key_input.strip())
                if valid:
                    st.session_state.groq_api_key = key_input.strip()
                    st.session_state.github_token = gh_input.strip()
                    st.rerun()
                else:
                    st.error(err)
        st.caption(
            "Groq key: [console.groq.com](https://console.groq.com)  \n"
            "GitHub token *(optional)*: [github.com/settings/tokens](https://github.com/settings/tokens) "
        )
        st.stop()

    st.divider()

    # ── GitHub repo input ─────────────────────────────────────
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
            result: IngestResult = ingest_repo(
                repo_url.strip(), on_progress,
                github_token=st.session_state.github_token,
            )
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

    # ── Indexed files list ────────────────────────────────────
    ingest_result: IngestResult | None = st.session_state.ingest_result
    if ingest_result:
        st.divider()
        st.markdown(f"**Indexed files** ({len(ingest_result.files)})")

        mmd_files = [f for f in ingest_result.files if f["file_type"] == "mmd"]
        doc_files = [f for f in ingest_result.files if f["file_type"] != "mmd"]

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

    if st.session_state.repo_id:
        st.divider()
        if st.button("📊 Dependency Diagram", use_container_width=True):
            _show_diagram(st.session_state.repo_id)

    # ── Environment status ────────────────────────────────────
    st.divider()
    st.markdown("**Environment**")
    st.markdown(
        f"<small>Generation: <code>llama-3.3-70b-versatile</code> (Groq)<br>"
        f"Embeddings: <code>mxbai-embed-large</code> (fastembed/local)<br>"
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
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 1rem;">
            <div style="font-size:3.5rem">📚</div>
            <h2 style="margin-top:0.5rem">Chat with any GitHub repository</h2>
            <p style="color:#888; max-width:520px; margin:0 auto 2rem">
                Paste a public repository URL in the sidebar and click
                <strong>Index Repository</strong>. The pipeline fetches all source files,
                parses them with AST-aware chunking, builds an import graph, and stores
                a hybrid vector index. Then ask anything!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🐙 All file types**\n\nCode, docs, Mermaid diagrams. AST-aware chunking for functions and classes.")
    with c2:
        st.info("**🔗 Import graph**\n\nFile dependency graph adds related-file context automatically.")
    with c3:
        st.info("**🤖 Agent reasoning**\n\nLangChain agent searches multiple times and synthesises a thorough answer.")

    st.markdown(
        "<div style='text-align:center;color:#555;font-size:0.8rem;margin-top:2rem'>"
        "Made by Youssif Ashmawy 🚀 - Hope that helps!"
        "</div>",
        unsafe_allow_html=True,
    )

else:
    # ── Chat interface ──
    repo_id = st.session_state.repo_id

    badge_label = repo_id.replace("__", "/")
    st.markdown(
        f"<div style='margin-bottom:1rem'>"
        f"<span style='background:#1e3a5f;border-radius:6px;padding:4px 10px;"
        f"font-size:0.8rem;color:#89b4fa'>📂 {badge_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    if question := st.chat_input("Ask anything about this repository…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            status_slot = st.empty()
            answer_slot = st.empty()

            full_answer = ""
            collected_sources: list[Source] = []

            try:
                for item in stream_answer(repo_id, question, api_key=st.session_state.groq_api_key):
                    if isinstance(item, list):
                        collected_sources = item
                        status_slot.markdown(
                            f"*Found {len(collected_sources)} relevant source(s)…*"
                        )
                    elif isinstance(item, dict) and item.get("type") == "status":
                        status_slot.markdown(f"*{item['msg']}*")
                    elif isinstance(item, dict) and item.get("type") == "tool_call":
                        icon = "🔍" if "search" in item["name"] else "📄"
                        status_slot.markdown(f"*{icon} Agent: `{item['query']}`…*")
                    else:
                        full_answer += item
                        answer_slot.markdown(full_answer + "▌")

                answer_slot.markdown(full_answer)
                status_slot.empty()

                if collected_sources:
                    render_sources(collected_sources)

            except Exception as exc:
                answer_slot.error(f"**Error:** {exc}")
                collected_sources = []

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

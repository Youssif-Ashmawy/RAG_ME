"""
Document / code chunker.

Routing:
  .mmd               → one chunk per Mermaid diagram block
  .md / .mdx / .txt  → split by heading then paragraph (existing logic)
  everything else    → code_parser.py semantic units (function / class / block)

Each chunk carries rich metadata so the LLM knows exactly where it comes
from and how it relates to the rest of the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .code_parser import ParsedFile, ParsedUnit, detect_language, parse_file
from .github_client import RepoFile

MAX_CHUNK_CHARS = 2_400   # ~600 tokens — for doc files; embedding truncates at 1500 chars
OVERLAP_CHARS   = 300     # carry-over between consecutive chunks


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str               # full text sent to the embedder / LLM
    path: str
    raw_url: str
    file_type: str          # extension without dot, e.g. "py", "md", "mmd"
    language: str           # detected language string
    chunk_index: int
    total_chunks: int
    # Optional enrichment fields
    heading: Optional[str]       = None  # markdown section heading
    unit_name: Optional[str]     = None  # function / class name
    unit_type: Optional[str]     = None  # "function" | "class" | "module"
    diagram_type: Optional[str]  = None  # Mermaid diagram type
    imports: Optional[list[str]] = None  # file-level raw imports (for context)
    signature: Optional[str]     = None  # function/class signature


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_files(files: list[RepoFile]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for f in files:
        if f.file_type == "mmd":
            all_chunks.extend(_chunk_mermaid(f))
        elif f.file_type in ("md", "mdx", "txt", "rst"):
            all_chunks.extend(_chunk_markdown(f))
        else:
            all_chunks.extend(_chunk_code(f))
    return all_chunks


# ---------------------------------------------------------------------------
# Mermaid (.mmd)
# ---------------------------------------------------------------------------

_MERMAID_TYPES = {
    "flowchart": "Flowchart", "graph": "Flowchart",
    "sequencediagram": "Sequence Diagram",
    "classd": "Class Diagram",
    "statediagram": "State Diagram",
    "erdiagram": "ER Diagram",
    "gantt": "Gantt Chart",
    "pie": "Pie Chart",
    "gitgraph": "Git Graph",
    "mindmap": "Mind Map",
    "timeline": "Timeline",
    "c4": "C4 Diagram",
    "quadrantchart": "Quadrant Chart",
}


def _detect_mermaid_type(block: str) -> str:
    key = block.strip().split("\n")[0].lower().replace("-", "").replace(" ", "")
    for k, v in _MERMAID_TYPES.items():
        if key.startswith(k):
            return v
    return "Diagram"


def _chunk_mermaid(f: RepoFile) -> list[Chunk]:
    blocks = [b.strip() for b in re.split(r"\n{2,}", f.content) if len(b.strip()) > 10]
    if not blocks:
        blocks = [f.content.strip()]
    chunks: list[Chunk] = []
    for i, block in enumerate(blocks):
        dtype = _detect_mermaid_type(block)
        text = f"[Mermaid {dtype}] File: {f.path}\n\n{block}"
        chunks.append(Chunk(
            id=f"{f.path}::{i}", text=text,
            path=f.path, raw_url=f.raw_url,
            file_type=f.file_type, language="mermaid",
            chunk_index=i, total_chunks=len(blocks),
            diagram_type=dtype, heading=dtype,
        ))
    return chunks


# ---------------------------------------------------------------------------
# Markdown / plain text
# ---------------------------------------------------------------------------

def _chunk_markdown(f: RepoFile) -> list[Chunk]:
    sections = _split_by_headings(f.content)
    raw: list[tuple[str, str]] = []  # (heading, text)

    for heading, text in sections:
        if len(text) <= MAX_CHUNK_CHARS:
            raw.append((heading, text))
        else:
            raw.extend(_split_by_paragraphs(text, heading))

    filtered = [(h, t) for h, t in raw if len(t.strip()) > 50]
    if not filtered:
        filtered = [("", f.content)]

    chunks: list[Chunk] = []
    for i, (heading, text) in enumerate(filtered):
        header = f"File: {f.path}"
        if heading:
            header += f" | Section: {heading}"
        full = f"{header}\n\n{text.strip()}"
        chunks.append(Chunk(
            id=f"{f.path}::{i}", text=full,
            path=f.path, raw_url=f.raw_url,
            file_type=f.file_type, language="markdown",
            chunk_index=i, total_chunks=len(filtered),
            heading=heading or None,
        ))
    return chunks


def _split_by_headings(content: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading, current_lines = "", []
    for line in content.split("\n"):
        m = re.match(r"^(#{1,4})\s+(.+)", line)
        if m:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading, current_lines = m.group(2).strip(), [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))
    return sections or [("", content)]


def _split_by_paragraphs(text: str, heading: str) -> list[tuple[str, str]]:
    paragraphs = re.split(r"\n{2,}", text)
    result: list[tuple[str, str]] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > MAX_CHUNK_CHARS:
            result.append((heading, current.strip()))
            current = current[-OVERLAP_CHARS:] + "\n\n" + para
        else:
            current = (current + "\n\n" + para).lstrip() if current else para
    if current.strip():
        result.append((heading, current.strip()))
    return result or [(heading, text)]


# ---------------------------------------------------------------------------
# Code files — via code_parser
# ---------------------------------------------------------------------------

def _chunk_code(f: RepoFile) -> list[Chunk]:
    language = detect_language(f.path)
    parsed: ParsedFile = parse_file(f.content, f.path)
    units = parsed.units

    if not units:
        # Empty or unparseable file — produce one minimal chunk
        return [Chunk(
            id=f"{f.path}::0", text=f"File: {f.path}\n\n(empty or binary)",
            path=f.path, raw_url=f.raw_url,
            file_type=f.file_type, language=language,
            chunk_index=0, total_chunks=1,
        )]

    chunks: list[Chunk] = []
    imports_summary = _imports_summary(parsed.raw_imports)

    for i, unit in enumerate(units):
        text = _build_code_chunk_text(
            unit=unit,
            path=f.path,
            language=language,
            imports_summary=imports_summary,
            module_doc=parsed.module_docstring,
        )
        chunks.append(Chunk(
            id=f"{f.path}::{i}", text=text,
            path=f.path, raw_url=f.raw_url,
            file_type=f.file_type, language=language,
            chunk_index=i, total_chunks=len(units),
            unit_name=unit.name,
            unit_type=unit.unit_type,
            signature=unit.signature,
            imports=parsed.raw_imports[:20],  # keep first 20 for metadata
        ))

    return chunks


def _build_code_chunk_text(
    unit: ParsedUnit,
    path: str,
    language: str,
    imports_summary: str,
    module_doc: str,
) -> str:
    """
    Build the full text sent to the embedder and LLM for one code unit.
    Format:
        [<Language> <UnitType>] <path> :: <name>
        Signature: <sig>
        Docstring: <doc>
        File imports: <imports>
        ─────────────────────
        <body>
    """
    label = f"[{language.title()} {unit.unit_type.title()}] {path} :: {unit.name}"
    lines: list[str] = [label]

    if unit.signature:
        lines.append(f"Signature: {unit.signature}")
    if unit.docstring:
        lines.append(f"Docstring: {unit.docstring[:300]}")
    if imports_summary:
        lines.append(f"File imports: {imports_summary}")
    if module_doc and unit.unit_type == "module":
        lines.append(f"File description: {module_doc[:200]}")

    lines.append("─" * 40)
    lines.append(unit.body)

    return "\n".join(lines)


def _imports_summary(raw_imports: list[str]) -> str:
    if not raw_imports:
        return ""
    shown = raw_imports[:12]
    suffix = f" (+{len(raw_imports) - 12} more)" if len(raw_imports) > 12 else ""
    return ", ".join(shown) + suffix

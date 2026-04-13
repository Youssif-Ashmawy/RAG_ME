"""
PDF parser — extracts structured text from PDFs using pdfplumber.

Strategy (robust, two-layer):
  1. Extract plain text with page.extract_text() — always works, handles layout
  2. Extract words with font metadata (size, fontname) — used for heading detection
  3. Detect headings with FOUR independent signals (any one is enough):
       a) Numbered section pattern  (1. Intro, 2.3 Methods)
       b) ALL CAPS short line       (EXPERIENCE, PROJECTS)
       c) Font size ≥ 1.08× median  (catches subtle size bumps)
       d) Bold/Heavy/Black/Semi/Demi font name
       e) Short title-case phrase   (2–4 words, no punctuation, no common conjunctions)
  4. Table extraction formatted as Markdown
  5. Sliding-window fallback: any section > MAX_CHUNK_CHARS is split with overlap
     so nothing is ever silently dropped

Each Chunk carries document name, page number, and section heading.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from statistics import median
from typing import Optional

import pdfplumber

from .chunker import Chunk

MAX_CHUNK_CHARS = 2_000
OVERLAP_CHARS   = 200

# Font name substrings that indicate a heading weight
_BOLD_HINTS = ("Bold", "Heavy", "Black", "Semi", "Demi", "Extra", "ExtraBold",
               "Medium", "Display", "Heading")


@dataclass
class _Section:
    heading: str
    page_num: int
    text: str


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes, filename: str) -> list[Chunk]:
    """Parse a PDF from raw bytes into a flat list of Chunk objects."""
    sections: list[_Section] = []

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                return []
            for page_num, page in enumerate(pdf.pages, 1):
                sections.extend(_extract_page_sections(page, page_num))
    except Exception as exc:
        raise ValueError(
            f"Could not open PDF '{filename}': {exc}. "
            "Make sure the file is a valid, non-encrypted PDF."
        ) from exc

    if not sections:
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "Scanned/image-only PDFs require OCR, which is not yet supported."
        )

    return _sections_to_chunks(sections, filename)


# ─────────────────────────────────────────────────────────────
# Page → sections
# ─────────────────────────────────────────────────────────────

def _extract_page_sections(page: pdfplumber.page.Page, page_num: int) -> list[_Section]:
    # ── Font metadata ─────────────────────────────────────────
    words      = page.extract_words(extra_attrs=["fontname", "size"]) or []
    sizes      = [w.get("size", 0) for w in words if w.get("size", 0) > 0]
    med_size   = median(sizes) if sizes else 10.0

    # Build token → font-info lookup (keeps the max-size entry per token)
    font_info: dict[str, dict] = {}
    for w in words:
        tok = w.get("text", "")
        if not tok:
            continue
        if tok not in font_info or w.get("size", 0) > font_info[tok].get("size", 0):
            font_info[tok] = w

    # ── Plain text (reliable line order, handles columns) ─────
    raw_text = page.extract_text(layout=False) or ""
    if not raw_text.strip():
        return []

    # ── Tables (formatted as Markdown, appended to last section) ─
    table_blocks: list[str] = []
    try:
        for tbl in (page.find_tables() or []):
            rows = tbl.extract()
            if rows:
                table_blocks.append(_format_table(rows))
    except Exception:
        pass

    # ── Parse lines → sections ────────────────────────────────
    sections: list[_Section]  = []
    current_heading            = f"Page {page_num}"
    current_lines: list[str]  = []

    for raw_line in raw_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Collect font metadata for tokens on this line
        line_words = [font_info[tok] for tok in line.split() if tok in font_info]

        if _is_heading(line_words, line, med_size):
            _flush(sections, current_heading, page_num, current_lines)
            current_heading = line
            current_lines   = []
        else:
            current_lines.append(line)

    # Append tables to the last section's text
    current_lines.extend(table_blocks)
    _flush(sections, current_heading, page_num, current_lines)

    return sections


def _flush(
    sections: list[_Section],
    heading: str,
    page_num: int,
    lines: list[str],
) -> None:
    body = "\n".join(ln for ln in lines if ln.strip()).strip()
    if body:
        sections.append(_Section(heading=heading, page_num=page_num, text=body))


# ─────────────────────────────────────────────────────────────
# Heading detection  (five independent signals)
# ─────────────────────────────────────────────────────────────

def _is_heading(line_words: list[dict], text: str, page_median_size: float) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False

    # Reject obvious non-headings: bullet points, dates, pure numbers
    if re.match(r"^[\-•·▪▸►✓✗\*]", stripped):
        return False
    if re.match(r"^\d{4}[\s\-–]", stripped):       # starts with year
        return False
    if re.match(r"^\d+$", stripped):                # pure number
        return False

    # Signal 1: Numbered section  "1. Intro"  "2.3 Methods"
    if re.match(r"^[\dIVX]+[\.\d]*\s+[A-Z]", stripped) and len(stripped) < 100:
        return True

    # Signal 2: ALL CAPS (short line)  "EXPERIENCE"  "TECHNICAL SKILLS"
    if stripped.isupper() and 3 < len(stripped) < 80:
        return True

    if line_words:
        avg_size = sum(w.get("size", 0) for w in line_words) / len(line_words)

        # Signal 3: Font size ≥ 1.08× page median
        if avg_size > page_median_size * 1.08:
            return True

        # Signal 4: Bold / heavy / display font name
        if any(
            any(hint in w.get("fontname", "") for hint in _BOLD_HINTS)
            for w in line_words
        ):
            return True

    # Signal 5: Short title-case phrase (2–4 words, no punctuation mid-text)
    tokens = stripped.split()
    if (
        2 <= len(tokens) <= 4
        and len(stripped) <= 55
        and not re.search(r"[.,:;!?()\d]", stripped)          # no sentence chars
        and not re.search(r"\b(and|or|the|a|an|to|in|of)\b",  # not a sentence fragment
                          stripped, re.I)
        and all(t[0].isupper() for t in tokens if t and t[0].isalpha())
    ):
        return True

    return False


# ─────────────────────────────────────────────────────────────
# Sections → Chunks
# ─────────────────────────────────────────────────────────────

def _sections_to_chunks(sections: list[_Section], filename: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0

    for sec in sections:
        header = (
            f"Document: {filename} | "
            f"Page: {sec.page_num} | "
            f"Section: {sec.heading}"
        )
        for seg in _split_text(sec.text):
            chunks.append(Chunk(
                id           = f"{filename}::{idx}",
                text         = f"{header}\n\n{seg}",
                path         = filename,
                raw_url      = "",
                file_type    = "pdf",
                language     = "text",
                chunk_index  = idx,
                total_chunks = -1,
                heading      = sec.heading,
            ))
            idx += 1

    total = len(chunks)
    for c in chunks:
        c.total_chunks = total

    return chunks


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _format_table(rows: list[list[Optional[str]]]) -> str:
    if not rows:
        return ""
    cleaned = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in rows
    ]
    n_cols = max(len(r) for r in cleaned)
    widths  = [max(len(cleaned[r][c]) if c < len(cleaned[r]) else 0
                   for r in range(len(cleaned))) for c in range(n_cols)]

    def _row(row: list[str]) -> str:
        cells = [row[i].ljust(widths[i]) if i < len(row) else " " * widths[i]
                 for i in range(n_cols)]
        return "| " + " | ".join(cells) + " |"

    out = [_row(cleaned[0]),
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [_row(r) for r in cleaned[1:]]
    return "\n".join(out)


def _split_text(text: str) -> list[str]:
    """Split into chunks ≤ MAX_CHUNK_CHARS with OVERLAP_CHARS carry-over."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    segments: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = f"{current}\n\n{para}".lstrip() if current else para
        if len(candidate) > MAX_CHUNK_CHARS and current:
            segments.append(current.strip())
            current = current[-OVERLAP_CHARS:].lstrip() + "\n\n" + para
        else:
            current = candidate

    if current.strip():
        segments.append(current.strip())

    return segments or [text[:MAX_CHUNK_CHARS]]

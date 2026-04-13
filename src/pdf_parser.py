"""
PDF parser — extracts text page-by-page and produces Chunk objects
compatible with the existing hybrid vector store and agent pipeline.

Each page is split into paragraph-sized chunks (≤ MAX_PDF_CHUNK_CHARS).
Metadata includes page number and inferred section heading (first non-empty line).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from pypdf import PdfReader

from .chunker import Chunk

MAX_PDF_CHUNK_CHARS = 1_200
OVERLAP_CHARS       = 150


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes, filename: str) -> list[Chunk]:
    """
    Parse a PDF from raw bytes.  Returns a flat list of Chunk objects.
    filename is used as the chunk's `path` field (e.g. 'report.pdf').
    """
    reader    = PdfReader(io.BytesIO(file_bytes))
    all_chunks: list[Chunk] = []
    chunk_idx = 0

    for page_num, page in enumerate(reader.pages, 1):
        raw = page.extract_text() or ""
        text = _clean(raw)
        if not text:
            continue

        heading = _infer_heading(text, page_num)
        segments = _split_page(text, heading, page_num, filename)

        for seg_text in segments:
            all_chunks.append(Chunk(
                id          = f"{filename}::{chunk_idx}",
                text        = seg_text,
                path        = filename,
                raw_url     = "",                 # local file — no remote URL
                file_type   = "pdf",
                language    = "text",
                chunk_index = chunk_idx,
                total_chunks= -1,                 # filled in after all pages parsed
                heading     = heading,
            ))
            chunk_idx += 1

    # Back-fill total_chunks now that we know the count
    total = len(all_chunks)
    for c in all_chunks:
        c.total_chunks = total

    return all_chunks


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalise whitespace from PDF extraction."""
    # Collapse runs of whitespace/newlines produced by PDF layout artefacts
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_heading(text: str, page_num: int) -> str:
    """
    Use the first short non-empty line as the section heading,
    falling back to 'Page N'.
    """
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line) < 120:
            return line
    return f"Page {page_num}"


def _split_page(
    text: str,
    heading: str,
    page_num: int,
    filename: str,
) -> list[str]:
    """
    Split a page's text into chunks ≤ MAX_PDF_CHUNK_CHARS.
    Each chunk gets a header with filename, page, and section heading.
    """
    header = f"File: {filename} | Page: {page_num} | Section: {heading}"

    paragraphs = re.split(r"\n{2,}", text)
    segments: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (current + "\n\n" + para).lstrip() if current else para
        if len(candidate) > MAX_PDF_CHUNK_CHARS and current:
            segments.append(f"{header}\n\n{current.strip()}")
            # Overlap: carry the tail of the previous chunk into the next
            current = current[-OVERLAP_CHARS:].lstrip() + "\n\n" + para
        else:
            current = candidate

    if current.strip():
        segments.append(f"{header}\n\n{current.strip()}")

    return segments if segments else [f"{header}\n\n{text[:MAX_PDF_CHUNK_CHARS]}"]

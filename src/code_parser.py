"""
Language-aware code parser.

For each source file it produces:
  • A list of ParsedUnit (function / class / method / module-level block)
  • A list of raw import strings (used by import_graph.py)
  • A module-level docstring / file comment

Supported strategies:
  Python      — stdlib `ast` module (exact)
  JS / TS     — brace-counting + regex (good for ~95 % of real code)
  Go          — regex + brace-counting
  Rust        — regex + brace-counting
  Java/Kotlin — regex + brace-counting
  Generic     — blank-line / indentation heuristics (fallback)
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Maximum characters kept per unit before truncation
MAX_UNIT_CHARS = 1_200

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    # Python
    ".py": "python", ".pyi": "python",
    # JavaScript
    ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript", ".tsx": "typescript",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Java / JVM
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".groovy": "groovy",
    # C / C++
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    # C#
    ".cs": "csharp",
    # Ruby / PHP
    ".rb": "ruby", ".php": "php",
    # Swift / Dart
    ".swift": "swift", ".dart": "dart",
    # Shell
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".fish": "shell", ".ps1": "shell",
    # Config / data (treated as generic text)
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".ini": "ini", ".cfg": "cfg",
    # SQL
    ".sql": "sql",
    # Web
    ".html": "html", ".htm": "html",
    ".svelte": "svelte", ".vue": "vue",
    ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    # Other
    ".lua": "lua", ".r": "r",
    ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell",
    # Docs (handled by chunker.py but language-tagged here)
    ".md": "markdown", ".mdx": "markdown",
    ".mmd": "mermaid", ".txt": "text", ".rst": "rst",
}

SPECIAL_FILENAMES: dict[str, str] = {
    "Makefile": "makefile",
    "Dockerfile": "dockerfile",
    "Jenkinsfile": "groovy",
    "Procfile": "procfile",
    "Rakefile": "ruby",
    "Gemfile": "ruby",
}


def detect_language(path: str) -> str:
    name = Path(path).name
    if name in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[name]
    ext = Path(path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "generic")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedUnit:
    name: str
    unit_type: str          # "function" | "class" | "method" | "module"
    signature: str          # concise declaration line
    body: str               # full source text (possibly truncated)
    docstring: str          # extracted doc/comment
    line_start: int
    line_end: int


@dataclass
class ParsedFile:
    path: str
    language: str
    raw_imports: list[str]  # raw import strings for graph building
    units: list[ParsedUnit]
    module_docstring: str   # file-level doc string or leading comment


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_file(content: str, path: str) -> ParsedFile:
    language = detect_language(path)
    try:
        if language == "python":
            return _parse_python(content, path)
        if language in ("javascript", "typescript"):
            return _parse_js_ts(content, path, language)
        if language == "go":
            return _parse_go(content, path)
        if language in ("java", "kotlin", "scala", "csharp"):
            return _parse_java_like(content, path, language)
        if language == "rust":
            return _parse_rust(content, path)
    except Exception:
        pass
    return _parse_generic(content, path, language)


# ---------------------------------------------------------------------------
# Python parser (ast module — exact)
# ---------------------------------------------------------------------------

def _parse_python(content: str, path: str) -> ParsedFile:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _parse_generic(content, path, "python")

    lines = content.splitlines()

    # ── Imports ─────────────────────────────────────────────
    raw_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                raw_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            dots = "." * (node.level or 0)
            raw_imports.append(dots + module)

    module_doc = ast.get_docstring(tree) or ""

    # ── Top-level units ──────────────────────────────────────
    units: list[ParsedUnit] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            u = _py_function(node, lines)
            if u:
                units.append(u)
        elif isinstance(node, ast.ClassDef):
            u = _py_class(node, lines)
            if u:
                units.append(u)

    # Fallback: module-level block if no units extracted
    if not units and content.strip():
        units.append(ParsedUnit(
            name=Path(path).stem,
            unit_type="module",
            signature=f"# {path}",
            body=_truncate(content),
            docstring=module_doc,
            line_start=1,
            line_end=len(lines),
        ))

    return ParsedFile(path=path, language="python",
                      raw_imports=raw_imports, units=units,
                      module_docstring=module_doc)


def _py_function(node: ast.FunctionDef | ast.AsyncFunctionDef,
                 lines: list[str]) -> Optional[ParsedUnit]:
    body = _lines_slice(lines, node.lineno - 1, node.end_lineno)
    async_pfx = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    args = _py_args(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    sig = f"{async_pfx}def {node.name}({args}){ret}"
    return ParsedUnit(
        name=node.name, unit_type="function",
        signature=sig, body=_truncate(body),
        docstring=ast.get_docstring(node) or "",
        line_start=node.lineno, line_end=node.end_lineno,
    )


def _py_class(node: ast.ClassDef, lines: list[str]) -> Optional[ParsedUnit]:
    body = _lines_slice(lines, node.lineno - 1, node.end_lineno)
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
    return ParsedUnit(
        name=node.name, unit_type="class",
        signature=sig, body=_truncate(body),
        docstring=ast.get_docstring(node) or "",
        line_start=node.lineno, line_end=node.end_lineno,
    )


def _py_args(args: ast.arguments) -> str:
    parts: list[str] = []
    for a in args.args:
        s = a.arg
        if a.annotation:
            s += f": {ast.unparse(a.annotation)}"
        parts.append(s)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for a in args.kwonlyargs:
        parts.append(a.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript parser (regex + brace counting)
# ---------------------------------------------------------------------------

_JS_IMPORT = re.compile(
    r"""import\s+.+?\s+from\s+['"](.+?)['"]"""
    r"""|require\s*\(\s*['"](.+?)['"]\s*\)""",
    re.S,
)
_JS_FUNC = re.compile(
    r"^[ \t]*(export\s+)?(export\s+default\s+)?(async\s+)?function\s*\*?\s*(\w+)\s*[<(]",
    re.M,
)
_JS_ARROW = re.compile(
    r"^[ \t]*(export\s+)?(const|let|var)\s+(\w+)\s*(?::[^=\n]+)?\s*=\s*(async\s+)?(?:\([^)]*\)|[^\s=])\s*=>",
    re.M,
)
_JS_CLASS = re.compile(
    r"^[ \t]*(export\s+)?(export\s+default\s+)?class\s+(\w+)(?:\s+extends\s+[\w.]+)?",
    re.M,
)
_JS_LEADING_COMMENT = re.compile(r"^\s*(//[^\n]*\n|/\*[\s\S]*?\*/)\s*", re.M)


def _parse_js_ts(content: str, path: str, language: str) -> ParsedFile:
    raw_imports = [m.group(1) or m.group(2)
                   for m in _JS_IMPORT.finditer(content)
                   if m.group(1) or m.group(2)]

    units: list[ParsedUnit] = []

    for pattern, unit_type, name_group in [
        (_JS_CLASS, "class", 3),
        (_JS_FUNC,  "function", 4),
        (_JS_ARROW, "function", 3),
    ]:
        for m in pattern.finditer(content):
            name = m.group(name_group)
            if not name:
                continue
            brace_start = content.find("{", m.end())
            if brace_start == -1:
                continue
            block_end = _find_block_end(content, brace_start)
            body = content[m.start(): block_end].strip()
            comment = _extract_preceding_comment(content, m.start())
            line_start = content[:m.start()].count("\n") + 1
            line_end = content[:block_end].count("\n") + 1
            units.append(ParsedUnit(
                name=name, unit_type=unit_type,
                signature=content[m.start(): m.end()].strip(),
                body=_truncate(body), docstring=comment,
                line_start=line_start, line_end=line_end,
            ))

    if not units:
        return _parse_generic(content, path, language)

    # Deduplicate overlapping units (arrows inside classes etc.)
    units = _deduplicate_units(units)
    module_doc = _extract_file_comment(content)
    return ParsedFile(path=path, language=language,
                      raw_imports=raw_imports, units=units,
                      module_docstring=module_doc)


# ---------------------------------------------------------------------------
# Go parser
# ---------------------------------------------------------------------------

_GO_IMPORT_BLOCK = re.compile(r'import\s*\(([^)]+)\)', re.S)
_GO_IMPORT_SINGLE = re.compile(r'import\s+"([^"]+)"')
_GO_FUNC = re.compile(
    r'^func\s+(?:\(\s*\w+\s+\*?[\w.]+\s*\)\s+)?(\w+)\s*\(',
    re.M,
)
_GO_STRUCT = re.compile(r'^type\s+(\w+)\s+struct\s*\{', re.M)
_GO_INTERFACE = re.compile(r'^type\s+(\w+)\s+interface\s*\{', re.M)


def _parse_go(content: str, path: str) -> ParsedFile:
    raw_imports: list[str] = []
    for m in _GO_IMPORT_BLOCK.finditer(content):
        for line in m.group(1).splitlines():
            line = line.strip().strip('"')
            if line and not line.startswith("//"):
                raw_imports.append(line)
    for m in _GO_IMPORT_SINGLE.finditer(content):
        raw_imports.append(m.group(1))

    units: list[ParsedUnit] = []
    for pattern, utype in [(_GO_FUNC, "function"),
                            (_GO_STRUCT, "class"),
                            (_GO_INTERFACE, "class")]:
        for m in pattern.finditer(content):
            brace_start = content.find("{", m.end() - 1)
            if brace_start == -1:
                continue
            block_end = _find_block_end(content, brace_start)
            body = content[m.start(): block_end].strip()
            comment = _extract_preceding_comment(content, m.start())
            line_start = content[:m.start()].count("\n") + 1
            line_end = content[:block_end].count("\n") + 1
            units.append(ParsedUnit(
                name=m.group(1), unit_type=utype,
                signature=m.group(0).strip(),
                body=_truncate(body), docstring=comment,
                line_start=line_start, line_end=line_end,
            ))

    if not units:
        return _parse_generic(content, path, "go")

    units = _deduplicate_units(units)
    return ParsedFile(path=path, language="go",
                      raw_imports=raw_imports, units=units,
                      module_docstring=_extract_file_comment(content))


# ---------------------------------------------------------------------------
# Java / Kotlin / Scala / C# (similar syntax)
# ---------------------------------------------------------------------------

_JAVA_IMPORT = re.compile(r'^import\s+([\w.]+)\s*;', re.M)
_JAVA_CLASS = re.compile(
    r'^(?:public\s+|private\s+|protected\s+|abstract\s+|final\s+)*'
    r'(?:class|interface|enum|record)\s+(\w+)',
    re.M,
)
_JAVA_METHOD = re.compile(
    r'^[ \t]+(?:(?:public|private|protected|static|final|abstract|synchronized|override|suspend)\s+)*'
    r'(?:[\w<>\[\]?]+\s+)+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{',
    re.M,
)


def _parse_java_like(content: str, path: str, language: str) -> ParsedFile:
    raw_imports = [m.group(1) for m in _JAVA_IMPORT.finditer(content)]
    units: list[ParsedUnit] = []

    for pattern, utype in [(_JAVA_CLASS, "class"), (_JAVA_METHOD, "function")]:
        for m in pattern.finditer(content):
            brace_start = content.find("{", m.end() - 1)
            if brace_start == -1:
                continue
            block_end = _find_block_end(content, brace_start)
            body = content[m.start(): block_end].strip()
            comment = _extract_preceding_comment(content, m.start())
            line_start = content[:m.start()].count("\n") + 1
            line_end = content[:block_end].count("\n") + 1
            units.append(ParsedUnit(
                name=m.group(1), unit_type=utype,
                signature=m.group(0).rstrip("{").strip(),
                body=_truncate(body), docstring=comment,
                line_start=line_start, line_end=line_end,
            ))

    if not units:
        return _parse_generic(content, path, language)
    return ParsedFile(path=path, language=language,
                      raw_imports=raw_imports,
                      units=_deduplicate_units(units),
                      module_docstring=_extract_file_comment(content))


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

_RUST_USE = re.compile(r'^use\s+([\w::{}, ]+)\s*;', re.M)
_RUST_FN = re.compile(
    r'^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)',
    re.M,
)
_RUST_STRUCT = re.compile(r'^(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)', re.M)
_RUST_IMPL = re.compile(r'^(?:pub(?:\([^)]*\))?\s+)?impl(?:\s+[\w<>]+)?\s+(\w+)', re.M)
_RUST_TRAIT = re.compile(r'^(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)', re.M)
_RUST_ENUM = re.compile(r'^(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)', re.M)


def _parse_rust(content: str, path: str) -> ParsedFile:
    raw_imports = [m.group(1).strip() for m in _RUST_USE.finditer(content)]
    units: list[ParsedUnit] = []

    for pattern, utype in [
        (_RUST_FN, "function"), (_RUST_IMPL, "class"),
        (_RUST_STRUCT, "class"), (_RUST_TRAIT, "class"),
        (_RUST_ENUM, "class"),
    ]:
        for m in pattern.finditer(content):
            brace_start = content.find("{", m.end())
            if brace_start == -1:
                continue
            block_end = _find_block_end(content, brace_start)
            body = content[m.start(): block_end].strip()
            comment = _extract_preceding_comment(content, m.start())
            line_start = content[:m.start()].count("\n") + 1
            line_end = content[:block_end].count("\n") + 1
            units.append(ParsedUnit(
                name=m.group(1), unit_type=utype,
                signature=m.group(0).strip(),
                body=_truncate(body), docstring=comment,
                line_start=line_start, line_end=line_end,
            ))

    if not units:
        return _parse_generic(content, path, "rust")
    return ParsedFile(path=path, language="rust",
                      raw_imports=raw_imports,
                      units=_deduplicate_units(units),
                      module_docstring=_extract_file_comment(content))


# ---------------------------------------------------------------------------
# Generic / fallback parser
# ---------------------------------------------------------------------------

def _parse_generic(content: str, path: str, language: str = "generic") -> ParsedFile:
    """Split by blank lines, group into max-size chunks."""
    paragraphs = re.split(r"\n{2,}", content)
    MAX = MAX_UNIT_CHARS

    units: list[ParsedUnit] = []
    current = ""
    line = 1

    for para in paragraphs:
        if current and len(current) + len(para) > MAX:
            units.append(ParsedUnit(
                name=f"block_{len(units) + 1}",
                unit_type="module",
                signature="",
                body=current.strip(),
                docstring="",
                line_start=line,
                line_end=line + current.count("\n"),
            ))
            line += current.count("\n") + 2
            current = para
        else:
            current = (current + "\n\n" + para).lstrip()

    if current.strip():
        units.append(ParsedUnit(
            name=f"block_{len(units) + 1}",
            unit_type="module",
            signature="",
            body=current.strip(),
            docstring="",
            line_start=line,
            line_end=line + current.count("\n"),
        ))

    return ParsedFile(path=path, language=language,
                      raw_imports=[], units=units,
                      module_docstring=_extract_file_comment(content))


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _find_block_end(content: str, start: int) -> int:
    """Return the index after the closing brace of the block starting at `start`."""
    depth = 0
    in_str: Optional[str] = None
    i = start
    n = len(content)

    while i < n:
        c = content[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == in_str:
                in_str = None
        else:
            if c in ('"', "'", "`"):
                in_str = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return n


def _extract_preceding_comment(content: str, pos: int) -> str:
    """Extract the doc-comment immediately before `pos`."""
    before = content[:pos].rstrip()
    # Block comment: /** ... */ or /* ... */
    m = re.search(r"/\*\*?([\s\S]*?)\*/\s*$", before)
    if m:
        return re.sub(r"^\s*\*\s?", "", m.group(1), flags=re.M).strip()
    # Line comments: // or #
    lines = before.split("\n")
    comment_lines: list[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("#"):
            comment_lines.insert(0, stripped.lstrip("/#").strip())
        else:
            break
    return "\n".join(comment_lines)


def _extract_file_comment(content: str) -> str:
    """Extract the leading file-level comment."""
    m = re.match(r"\s*(/\*[\s\S]*?\*/|(?:#[^\n]*\n)+|(?://[^\n]*\n)+)", content)
    if m:
        raw = m.group(1)
        return re.sub(r"^[/*#\s]+|[/*#\s]+$", "", raw).strip()
    return ""


def _lines_slice(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start:end])


def _truncate(text: str) -> str:
    if len(text) <= MAX_UNIT_CHARS:
        return text
    return text[:MAX_UNIT_CHARS] + "\n\n... [truncated — file too large]"


def _deduplicate_units(units: list[ParsedUnit]) -> list[ParsedUnit]:
    """Remove units whose line ranges are fully contained in a larger unit."""
    units_sorted = sorted(units, key=lambda u: (u.line_start, -(u.line_end - u.line_start)))
    result: list[ParsedUnit] = []
    for u in units_sorted:
        if not any(
            p.line_start <= u.line_start and p.line_end >= u.line_end and p is not u
            for p in result
        ):
            result.append(u)
    return result

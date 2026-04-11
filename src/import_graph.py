"""
Import graph — builds and queries a file-level dependency graph.

For each repository we resolve raw import strings (from code_parser.py)
to actual file paths inside the repo, then build a bidirectional graph:

    file A imports file B  →  edge A → B

During retrieval, graph neighbours of semantically matched files are
used to pull in additional related context.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Import resolution helpers
# ---------------------------------------------------------------------------

_JS_EXTS     = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_INDEX_FILES = ("index.ts", "index.tsx", "index.js", "index.jsx",
                "__init__.py", "mod.rs", "lib.rs")


def _resolve_python(imp: str, source: str, all_paths: set[str]) -> Optional[str]:
    src_dir = os.path.dirname(source)
    dots = len(imp) - len(imp.lstrip("."))
    module = imp[dots:].replace(".", "/")

    if dots:
        base = src_dir
        for _ in range(dots - 1):
            base = os.path.dirname(base)
        candidate = os.path.join(base, module).lstrip("/")
    else:
        # Absolute — try a few common root prefixes
        candidates_abs = [module, f"src/{module}", f"lib/{module}"]
        for c in candidates_abs:
            for suffix in (".py", "/__init__.py"):
                full = (c + suffix).lstrip("/")
                if full in all_paths:
                    return full
        return None

    for suffix in (".py", "/__init__.py"):
        full = (candidate + suffix).lstrip("/")
        if full in all_paths:
            return full
    return None


def _resolve_js(imp: str, source: str, all_paths: set[str]) -> Optional[str]:
    if not imp.startswith("."):
        return None  # external package
    base = os.path.normpath(os.path.join(os.path.dirname(source), imp)).lstrip("/")
    for ext in _JS_EXTS + ("",):
        if (base + ext).lstrip("/") in all_paths:
            return (base + ext).lstrip("/")
    for idx in _INDEX_FILES:
        full = os.path.join(base, idx).lstrip("/")
        if full in all_paths:
            return full
    return None


def _resolve_go(imp: str, all_paths: set[str]) -> Optional[str]:
    # Go imports use module paths; match by path suffix
    suffix = imp.split("/")[-1]
    for p in all_paths:
        if p.endswith(f"/{suffix}.go") or p.endswith(f"/{suffix}/"):
            return p
    return None


def _resolve_import(
    raw: str,
    source: str,
    language: str,
    all_paths: set[str],
) -> Optional[str]:
    if language == "python":
        return _resolve_python(raw, source, all_paths)
    if language in ("javascript", "typescript"):
        return _resolve_js(raw, source, all_paths)
    if language == "go":
        return _resolve_go(raw, all_paths)
    return None


# ---------------------------------------------------------------------------
# ImportGraph
# ---------------------------------------------------------------------------

@dataclass
class ImportGraph:
    # file → set of files it imports
    imports: dict[str, set[str]]     = field(default_factory=dict)
    # file → set of files that import it
    imported_by: dict[str, set[str]] = field(default_factory=dict)

    # ── Building ─────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        parsed_files: list,           # list[ParsedFile] — avoid circular import
        all_paths: set[str],
    ) -> "ImportGraph":
        graph = cls()
        for pf in parsed_files:
            for raw in pf.raw_imports:
                target = _resolve_import(raw, pf.path, pf.language, all_paths)
                if target and target != pf.path:
                    graph.imports.setdefault(pf.path, set()).add(target)
                    graph.imported_by.setdefault(target, set()).add(pf.path)
        return graph

    # ── Querying ─────────────────────────────────────────────

    def neighbours(self, path: str, depth: int = 1) -> set[str]:
        """
        Return all files related to `path` up to `depth` hops.
        Includes both directions (imports + imported_by).
        """
        visited = {path}
        frontier = {path}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for p in frontier:
                next_frontier.update(self.imports.get(p, set()))
                next_frontier.update(self.imported_by.get(p, set()))
            frontier = next_frontier - visited
            visited.update(frontier)
        visited.discard(path)
        return visited

    def summary(self, path: str) -> str:
        """Human-readable one-liner about a file's connections."""
        deps = self.imports.get(path, set())
        rdeps = self.imported_by.get(path, set())
        parts: list[str] = []
        if deps:
            parts.append(f"imports: {', '.join(sorted(deps))}")
        if rdeps:
            parts.append(f"imported by: {', '.join(sorted(rdeps))}")
        return " | ".join(parts) if parts else "no resolved dependencies"

    # ── Persistence ───────────────────────────────────────────

    def save(self, cache_dir: str) -> None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        data = {
            "imports":     {k: list(v) for k, v in self.imports.items()},
            "imported_by": {k: list(v) for k, v in self.imported_by.items()},
        }
        with open(os.path.join(cache_dir, "graph.json"), "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, cache_dir: str) -> Optional["ImportGraph"]:
        path = os.path.join(cache_dir, "graph.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            g = cls()
            g.imports     = {k: set(v) for k, v in data["imports"].items()}
            g.imported_by = {k: set(v) for k, v in data["imported_by"].items()}
            return g
        except Exception:
            return None

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.imports.values())

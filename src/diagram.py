"""
Dependency diagram generator.

Builds a Mermaid diagram from the import graph and chunk metadata:
  - Nodes  = source files, labelled with their top functions/classes
  - Edges  = resolved import relationships (A imports B → A --> B)

Capped at MAX_NODES files; when more exist the most-connected files are kept.
Files with no resolved edges are included only if they have named units.
"""

from __future__ import annotations

import re
from typing import Optional

from .import_graph import ImportGraph
from .vector_store import VectorStore

MAX_NODES         = 60   # hard cap to keep the diagram readable
MAX_UNITS_SHOWN   = 4    # max function/class names shown per node
MAX_SUMMARY_FILES = 40   # files included in the LLM explanation text

# Mermaid theme init block
_INIT = '%%{init: {"theme": "dark", "themeVariables": {"fontSize": "13px"}}}%%'


def build_mermaid(store: VectorStore, graph: Optional[ImportGraph]) -> str:
    """
    Return a Mermaid `graph LR` string for the indexed repository.
    Returns an empty string if there is nothing to visualise.
    """
    # ── Collect top units per file ─────────────────────────────
    file_units: dict[str, list[str]] = {}
    for chunk in store._chunks:
        if chunk.unit_name and chunk.unit_type in ("function", "class", "method"):
            file_units.setdefault(chunk.path, [])
            if chunk.unit_name not in file_units[chunk.path]:
                file_units[chunk.path].append(chunk.unit_name)

    # ── Determine which files to include ──────────────────────
    all_files: set[str] = set(file_units.keys())
    if graph:
        for src, targets in graph.imports.items():
            all_files.add(src)
            all_files.update(targets)

    if not all_files:
        return ""

    # Sort by connection count descending, then alphabetically
    def _conn(f: str) -> int:
        if not graph:
            return 0
        return len(graph.imports.get(f, set())) + len(graph.imported_by.get(f, set()))

    ordered = sorted(all_files, key=lambda f: (-_conn(f), f))
    selected: set[str] = set(ordered[:MAX_NODES])

    # ── Helpers ───────────────────────────────────────────────
    def _nid(path: str) -> str:
        """Sanitised Mermaid node ID."""
        return re.sub(r"[^a-zA-Z0-9]", "_", path)

    def _label(path: str) -> str:
        filename = path.split("/")[-1]
        units = file_units.get(path, [])[:MAX_UNITS_SHOWN]
        if units:
            body = "\\n".join(f"+ {u}" for u in units)
            return f'"{filename}\\n{body}"'
        return f'"{filename}"'

    def _ext(path: str) -> str:
        return path.rsplit(".", 1)[-1] if "." in path else "other"

    # ── Group files by top-level directory ────────────────────
    dir_files: dict[str, list[str]] = {}
    for f in sorted(selected):
        top = f.split("/")[0] if "/" in f else "root"
        dir_files.setdefault(top, []).append(f)

    # ── Build Mermaid source ──────────────────────────────────
    lines: list[str] = [_INIT, "graph LR"]

    for dirname, files in sorted(dir_files.items()):
        safe_dir = re.sub(r"[^a-zA-Z0-9]", "_", dirname)
        lines.append(f"    subgraph {safe_dir}[\"{dirname}/\"]")
        for f in files:
            ext = _ext(f)
            nid   = _nid(f)
            label = _label(f)
            # Style class by language
            if ext == "py":
                lines.append(f"        {nid}[{label}]:::py")
            elif ext in ("ts", "tsx"):
                lines.append(f"        {nid}[{label}]:::ts")
            elif ext in ("js", "jsx", "mjs"):
                lines.append(f"        {nid}[{label}]:::js")
            elif ext in ("go",):
                lines.append(f"        {nid}[{label}]:::go")
            elif ext in ("rs",):
                lines.append(f"        {nid}[{label}]:::rs")
            else:
                lines.append(f"        {nid}[{label}]")
        lines.append("    end")

    # ── Edges ─────────────────────────────────────────────────
    if graph:
        seen_edges: set[tuple[str, str]] = set()
        for src, targets in graph.imports.items():
            if src not in selected:
                continue
            for tgt in targets:
                if tgt not in selected:
                    continue
                edge = (_nid(src), _nid(tgt))
                if edge not in seen_edges:
                    lines.append(f"    {edge[0]} --> {edge[1]}")
                    seen_edges.add(edge)

    # ── Style classes ─────────────────────────────────────────
    lines += [
        "    classDef py  fill:#3572a5,stroke:#264f78,color:#fff",
        "    classDef ts  fill:#2b7489,stroke:#1a5c6b,color:#fff",
        "    classDef js  fill:#f1e05a,stroke:#c9b700,color:#333",
        "    classDef go  fill:#00acd7,stroke:#007fa3,color:#fff",
        "    classDef rs  fill:#dea584,stroke:#b5724d,color:#333",
    ]

    return "\n".join(lines)


def build_graph_summary(store: VectorStore, graph: Optional[ImportGraph]) -> str:
    """
    Build a plain-text description of the dependency graph for the LLM to explain.
    Includes file names, their key units, and import relationships.
    """
    # Collect units per file
    file_units: dict[str, list[str]] = {}
    for chunk in store._chunks:
        if chunk.unit_name and chunk.unit_type in ("function", "class", "method"):
            file_units.setdefault(chunk.path, [])
            if chunk.unit_name not in file_units[chunk.path]:
                file_units[chunk.path].append(chunk.unit_name)

    all_files: set[str] = set(file_units.keys())
    if graph:
        for src, targets in graph.imports.items():
            all_files.add(src)
            all_files.update(targets)

    def _conn(f: str) -> int:
        if not graph:
            return 0
        return len(graph.imports.get(f, set())) + len(graph.imported_by.get(f, set()))

    ordered = sorted(all_files, key=lambda f: (-_conn(f), f))[:MAX_SUMMARY_FILES]

    lines: list[str] = [
        f"Repository summary: {len(all_files)} files total, "
        f"{graph.edge_count if graph else 0} resolved import relationships.\n"
    ]

    for path in ordered:
        lines.append(f"File: {path}")
        units = file_units.get(path, [])[:6]
        if units:
            lines.append(f"  Defines: {', '.join(units)}")
        if graph:
            imports = sorted(graph.imports.get(path, set()))
            imported_by = sorted(graph.imported_by.get(path, set()))
            if imports:
                lines.append(f"  Imports: {', '.join(imports)}")
            if imported_by:
                lines.append(f"  Imported by: {', '.join(imported_by)}")
        lines.append("")

    return "\n".join(lines)

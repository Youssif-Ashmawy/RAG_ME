"""
GitHub client — fetches documentation files from public repositories.

Supports: .md, .mmd, .mdx, .txt files
Uses:
  - GitHub REST API (tree endpoint) for file discovery — 1 API call
  - Raw GitHub URLs for file content — no API rate limit
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse

import requests

SUPPORTED_EXTENSIONS = {
    # Documentation
    ".md", ".mdx", ".mmd", ".txt", ".rst", ".adoc",
    # Python
    ".py", ".pyi",
    # JavaScript / TypeScript
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    # Go
    ".go",
    # Rust
    ".rs",
    # Java / JVM
    ".java", ".kt", ".kts", ".scala", ".groovy",
    # C / C++
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    # C#
    ".cs",
    # Ruby / PHP
    ".rb", ".php",
    # Swift / Dart
    ".swift", ".dart",
    # Shell
    ".sh", ".bash", ".zsh", ".ps1",
    # Config / data
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    # SQL / web
    ".sql", ".html", ".svelte", ".vue",
    ".css", ".scss", ".sass", ".less",
}

# Paths/patterns to skip even if the extension matches
SKIP_PATTERNS = (
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "Gemfile.lock", "poetry.lock", "composer.lock",
    "/node_modules/", "/vendor/", "/dist/", "/build/", "/.next/",
    "/target/", "/__pycache__/", "/.cache/", "/venv/", "/.venv/",
    ".min.js", ".min.css",
    ".pyc", ".pyo",
)

# Special filenames with no extension
SPECIAL_FILENAMES = {
    "Makefile", "Dockerfile", "Jenkinsfile",
    "Procfile", "Rakefile", "Gemfile",
}

MAX_FILE_SIZE_BYTES = 200_000   # 200 KB per file (skip minified/generated blobs)
MAX_FILES = 400                  # cap on number of files to ingest
FETCH_WORKERS = 15               # parallel file downloads


@dataclass
class RepoFile:
    path: str
    content: str
    raw_url: str
    file_type: str   # "md", "mmd", "mdx", "txt"


@dataclass
class RepoMeta:
    owner: str
    repo: str
    ref: str
    repo_id: str   # "{owner}__{repo}"


# ─────────────────────────────────────────────────────────────
# URL parsing
# ─────────────────────────────────────────────────────────────

def parse_repo_url(url: str) -> tuple[str, str, Optional[str]]:
    """Return (owner, repo, branch_or_None) from a GitHub URL."""
    url = url.strip().rstrip("/")
    match = re.search(
        r"github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+?)(?:\.git)?(?:/tree/([^/\s]+))?(?:/|$)",
        url,
    )
    if not match:
        raise ValueError(
            "Invalid GitHub URL. Expected: https://github.com/owner/repo"
        )
    return match.group(1), match.group(2), match.group(3)


def repo_id_from(owner: str, repo: str) -> str:
    return f"{owner}__{repo}"


# ─────────────────────────────────────────────────────────────
# GitHub API helpers
# ─────────────────────────────────────────────────────────────

def _headers(token: Optional[str] = None) -> dict:
    h = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHub-RAG-App",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_default_branch(owner: str, repo: str, token: Optional[str]) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = requests.get(url, headers=_headers(token), timeout=15)
    resp.raise_for_status()
    return resp.json()["default_branch"]


def _get_tree(
    owner: str, repo: str, ref: str, token: Optional[str]
) -> list[dict]:
    url = (
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    )
    resp = requests.get(url, headers=_headers(token), timeout=30)
    if resp.status_code == 409:
        raise ValueError("Repository is empty.")
    resp.raise_for_status()
    data = resp.json()
    if data.get("truncated"):
        print("[warning] GitHub tree was truncated (>100k files). Only partial results.")
    return data.get("tree", [])


def _fetch_raw(raw_url: str) -> str:
    resp = requests.get(
        raw_url,
        headers={"User-Agent": "GitHub-RAG-App"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.text


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def fetch_repo_files(
    repo_url: str,
    token: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[list[RepoFile], RepoMeta]:
    """
    Fetch all documentation files from a public GitHub repository.

    Returns (files, meta).
    """
    token = token or os.getenv("GITHUB_TOKEN")

    def progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    owner, repo, branch = parse_repo_url(repo_url)
    progress(f"Connecting to github.com/{owner}/{repo}…")

    ref = branch or _get_default_branch(owner, repo, token)
    progress(f"Reading repository tree (branch: {ref})…")

    tree = _get_tree(owner, repo, ref, token)

    def _should_include(entry: dict) -> bool:
        path = entry.get("path", "")
        size = int(entry.get("size", 0))
        if entry.get("type") != "blob":
            return False
        if size > MAX_FILE_SIZE_BYTES or size < 10:
            return False
        if any(pat in path for pat in SKIP_PATTERNS):
            return False
        name = path.split("/")[-1]
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in name else ""
        return ext in SUPPORTED_EXTENSIONS or name in SPECIAL_FILENAMES

    doc_entries = [e for e in tree if _should_include(e)][:MAX_FILES]

    if not doc_entries:
        raise ValueError(
            "No documentation files (.md, .mmd, .mdx, .txt) found in this repository."
        )

    progress(
        f"Found {len(doc_entries)} documentation file(s). Downloading content…"
    )

    files: list[RepoFile] = []

    def fetch_one(entry: dict) -> Optional[RepoFile]:
        path = entry["path"]
        raw_url = (
            f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        )
        try:
            content = _fetch_raw(raw_url)
            ext = path.rsplit(".", 1)[-1].lower()
            return RepoFile(path=path, content=content, raw_url=raw_url, file_type=ext)
        except Exception as exc:
            print(f"[warning] Could not fetch {path}: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_one, e): e for e in doc_entries}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                files.append(result)
            progress(f"Downloaded {done}/{len(doc_entries)} files…")

    meta = RepoMeta(
        owner=owner,
        repo=repo,
        ref=ref,
        repo_id=repo_id_from(owner, repo),
    )
    return files, meta

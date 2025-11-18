#!/usr/bin/env python3
"""Fail if README*.md or CLEANUP*.md exist outside project root or docs/.

Usage: invoked via pre-commit.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Allow only the canonical root README and anything under docs/
ALLOWED = {
    ROOT / "README.md",
}

# Directories to skip while scanning to avoid false positives
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "site",  # mkdocs build output
    "htmlcov",  # coverage HTML output
    "node_modules",
    "webapp-linux",
}


def _iter_markdown_files(base: Path) -> list[Path]:
    files: list[Path] = []
    for p in base.rglob("*.md"):
        # Skip excluded directories
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        files.append(p)
    return files


def main() -> int:
    bad: list[Path] = []
    for p in _iter_markdown_files(ROOT):
        name_upper = p.name.upper()
        if name_upper.startswith("README") or name_upper.startswith("CLEANUP"):
            # Allowed only at root README.md or anywhere under docs/
            if p in ALLOWED:
                continue
            if str(p).startswith(str(ROOT / "docs")):
                continue
            bad.append(p)

    if bad:
        print("Stray README/CLEANUP files detected outside root or docs:")
        for p in bad:
            print(f" - {p.relative_to(ROOT)}")
        print("Remove or migrate these files into docs/ or root README.md")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

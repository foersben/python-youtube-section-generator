#!/usr/bin/env python3
"""Fail if README*.md or CLEANUP*.md exist outside project root or docs/.

Usage: invoked via pre-commit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ALLOWED = {
    ROOT / "README.md",
}


def main() -> int:
    bad: list[Path] = []
    for p in ROOT.rglob("*.md"):
        if p.name.upper().startswith("README") or p.name.upper().startswith("CLEANUP"):
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
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.5
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  - repo: local
    hooks:
      - id: forbid-stray-readmes
        name: Forbid stray README/CLEANUP outside root or docs
        entry: python scripts/tools/check_stray_docs.py
        language: system
        pass_filenames: false


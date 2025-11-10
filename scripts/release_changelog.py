#!/usr/bin/env python3
"""Changelog release helper.

This script creates a dated release file under `docs/changelog/releases/` and
moves entries from `docs/changelog/unreleased.md` into the new release file.

Usage:
    python scripts/release_changelog.py --version 1.0.1 --date 2025-11-10

If --date is omitted, today's date is used. By default the script will create the
release file and leave `unreleased.md` untouched; use --cut to remove the moved
entries from `unreleased.md`.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_DIR = ROOT / "docs" / "changelog"
UNRELEASED = CHANGELOG_DIR / "unreleased.md"
RELEASES_DIR = CHANGELOG_DIR / "releases"

TEMPLATE = """# Release v{version} — {date}

{content}

---

*This release file was generated from `docs/changelog/unreleased.md`.*
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a changelog release file")
    parser.add_argument("--version", required=True, help="Release version e.g. 1.0.1")
    parser.add_argument("--date", help="Release date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--cut",
        action="store_true",
        help="Remove the moved sections from unreleased.md (cut mode)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the generated release file to stdout without writing",
    )

    args = parser.parse_args(argv)

    date_str = args.date or datetime.date.today().isoformat()
    version = args.version

    if not UNRELEASED.exists():
        print(f"Error: unreleased file not found: {UNRELEASED}")
        return 2

    content = UNRELEASED.read_text(encoding="utf-8")

    # Basic heuristic: include the whole unreleased.md as the release content.
    # For finer-grained control, users can edit unreleased.md before running this.
    release_content = content.strip()

    release_file = RELEASES_DIR / f"{date_str}-v{version}.md"
    release_text = TEMPLATE.format(version=version, date=date_str, content=release_content)

    if args.preview:
        print(release_text)
        return 0

    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    if release_file.exists():
        print(f"Error: release file already exists: {release_file}")
        return 3

    release_file.write_text(release_text, encoding="utf-8")
    print(f"Wrote release file: {release_file}")

    if args.cut:
        # remove the entire unreleased content (replace with an empty Unreleased header)
        UNRELEASED.write_text("# Changelog — Unreleased\n\n", encoding="utf-8")
        print("Cleared unreleased.md (cut mode enabled)")

    print("Done. Consider committing the new release file and updated unreleased.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Convert TODO entries from docs/prospects/todos.md into GitHub issues (dry-run).

This script parses the `docs/prospects/todos.md` master list and prints a
summary. With `--create` it will attempt to create issues using the GitHub API.

By default it performs a dry-run and prints proposed issues. Set GITHUB_TOKEN in
env to enable creation.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import requests

TODO_FILE = "docs/prospects/todos.md"
GITHUB_API = "https://api.github.com"


@dataclass
class TodoItem:
    title: str
    body: str
    priority: str | None = None


def parse_todos(path: str = TODO_FILE) -> list[TodoItem]:
    """Parse the compact master todos file and return TodoItem list."""
    items: list[TodoItem] = []
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    # Simple parsing: lines starting with '- [ ]' or '- [x]'
    pattern = re.compile(
        r"^- \[.\] (.+?)(?: — see `(.*?)`)?(?: \(priority: (.*?)\))?$", re.MULTILINE
    )
    for m in pattern.finditer(text):
        title = m.group(1).strip()
        link = m.group(2) or ""
        priority = m.group(3)
        body = f"Auto-imported todo. See {link}" if link else "Auto-imported todo"
        items.append(TodoItem(title=title, body=body, priority=priority))

    return items


def create_issue(repo: str, title: str, body: str, token: str) -> dict:
    url = f"{GITHUB_API}/repos/{repo}/issues"
    headers = {"Authorization": f"token {token}"}
    data = {"title": title, "body": body}
    r = requests.post(url, json=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Create issues on GitHub")
    parser.add_argument(
        "--repo", type=str, default=None, help="owner/repo (defaults to git remote origin)"
    )
    args = parser.parse_args()

    items = parse_todos()
    print(f"Found {len(items)} todo items in {TODO_FILE}")
    for i, it in enumerate(items, 1):
        print(f"{i}. {it.title} (priority={it.priority})")

    if args.create:
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise SystemExit("GITHUB_TOKEN not set in environment")
        repo = args.repo
        if not repo:
            # Try to infer from git remote
            import subprocess

            out = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])  # type: ignore
            url = out.decode().strip()
            # parse git@github.com:owner/repo.git or https://github.com/owner/repo.git
            m = re.search(r"[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)(?:\.git)?$", url)
            if not m:
                raise SystemExit("Could not infer repo from git remote; use --repo owner/repo")
            repo = f"{m.group('owner')}/{m.group('repo')}"

        print(f"Creating issues in {repo} (dry-run=false)")
        for it in items:
            resp = create_issue(repo, it.title, it.body, token)
            print(f"Created issue #{resp['number']}: {resp['html_url']}")


if __name__ == "__main__":
    main()

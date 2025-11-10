# Changelog

This directory contains the project's changelog broken into per-release files.

Structure

- `docs/changelog/unreleased.md` — in-progress changes (working area)
- `docs/changelog/releases/` — dated release files (immutable)

Recommended release workflow

1. When preparing a release, create a new file under
   `docs/changelog/releases/` named `YYYY-MM-DD-vX.Y.Z.md`.
2. Move the relevant entries from `unreleased.md` into the new release file.
3. Update `unreleased.md` to remove moved entries.
4. Commit the release file and the updated `unreleased.md`.
5. Optionally update release notes on GitHub.

Example

- `docs/changelog/releases/2025-11-09-v1.0.0.md`

Why split the changelog?

- Large changelogs grow over time and are easier to maintain as per-release
  files.
- Per-release files are immutable and easy to reference in release notes.
- `unreleased.md` remains the editable working area for the next release.

Viewing the changelog

- The documentation site links to this index. Use the index to find the
  release you want.



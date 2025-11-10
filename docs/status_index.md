# Status & Updates (moved)

To reduce noise in the main documentation we moved short status/update files into `.github/status/`.

Files moved there:

- `.github/status/IMPROVEMENTS_SUMMARY.md`
- `.github/status/OPTIMIZATION_SUMMARY.md`
- `.github/status/REFINEMENT_IMPLEMENTATION_SUMMARY.md`
- `.github/status/REFINEMENT_PERFORMANCE.md`
- `.github/status/REFINEMENT_QUICK_REF.md`
- `.github/status/REFINEMENT_STATUS_CHECK.md`
- `.github/status/QUALITY_IMPROVEMENTS.md`
- `.github/status/ISSUE_RESOLUTION_REFINEMENT.md`

Why:
- These are operational status documents and change logs intended for maintainers.
- Keeping them out of the main docs nav keeps the user-facing docs focused and easier to navigate.

How to view them locally:

```bash
# From the project root
less .github/status/OPTIMIZATION_SUMMARY.md
# or open in your editor
code .github/status/OPTIMIZATION_SUMMARY.md
```

If you'd like these pages included in the public docs site instead, I can move them back under `docs/` or selectively import key pages into `docs/`.


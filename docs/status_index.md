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

Where to write new status updates

- New status/operational updates MUST be created under `.github/status/`.
- Do NOT create new `docs/` files for short operational updates or agent run logs.

Status file conventions (required)

- Location: `.github/status/`
- Filename: lowercase, kebab-case, with a date prefix `YYYY-MM-DD` recommended.
  Examples:
  - `.github/status/2025-11-10-agent-run.md`
  - `.github/status/status-2025-11-10-copilot-refactor.md`
- Header (required): include a short front-matter like block at the top with the
  following fields (YAML-like or plain text):

  Title: Short title
  Date: 2025-11-10
  Author: copilot-agent
  Tags: status, maintenance
  Summary: One-line summary

- Body: Keep it concise. Explain what changed, why, and any follow-ups.
- Keep status files small (recommended < 1,000 words).

Examples (valid)

- `.github/status/2025-11-10-refinement-optimization.md`

  Header:
  ```text
  Title: Refinement optimization applied
  Date: 2025-11-10
  Author: copilot-agent
  Tags: status, performance
  Summary: Increased batch size and shortened prompts to speed up refinement.
  ```

- `.github/status/status-2025-11-11-ci-docs-update.md`

Docs file naming and conventions

To keep `docs/` consistent and machine-friendly, follow these rules for
user-facing documentation:

- Filenames: lowercase, kebab-case (e.g. `transcript-refinement.md`, `copilot.md`).
- Headings: Use title case for top-level headings (e.g. `# Transcript Refinement`).
- Keep documentation pages focused: one topic per file.
- Avoid large, frequently-changing maintenance notes in `docs/`.

How to view status files locally

```bash
# From project root
less .github/status/2025-11-10-refinement-optimization.md
code .github/status/2025-11-10-refinement-optimization.md
```

If you need public-facing copies

If a status update should be publicly visible and stable, create a curated
`docs/` page instead (follow the `docs/` naming rules). For ephemeral or
operation-focused updates, keep them under `.github/status/`.

If you want me to migrate selected status files back into `docs/` (curated and
renamed to kebab-case), tell me which ones and I'll import them and update the
`mkdocs.yml` nav accordingly.

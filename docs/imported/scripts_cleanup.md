# Scripts Cleanup Log (imported)

This page was imported from `scripts/CLEANUP.md` and documents the cleanup actions performed on the `scripts/` folder.

Moved to `scripts/obsolete/` (developer/debug artifacts & generated files):

- `verify_migration.py` (moved)
- `verify_refactoring.py` (moved)
- `run_pytest_programmatic.py` (moved)
- `capture_llm_output.py` (moved)
- `debug_llm_output.py` (moved)
- `inspect_config.py` (moved)
- `gen_run.log` (moved) — structured run log
- `check_local_model_paths.log` (moved)
- `pytest_output.log` (moved)
- `downloaded_a24109bf2b5e_sections.txt` (moved)
- `test_generate_response.json` (moved)
- `webapp_response.json` (moved)

Rationale:
- These files are useful for historical debugging but are not required for normal project setup or CI. They were moved to `scripts/obsolete/` to keep the `scripts/` directory focused on executable helpers and tests.

If any of the moved scripts should be restored, they remain in `scripts/obsolete/` and can be reintroduced as needed.

# Current layout (post-cleanup)

The `scripts/` directory now follows this structure:

```
/scripts
  /cli
  /tools
  /verify
  /dev
  /obsolete
```

# How to reintroduce a script from `scripts/obsolete/`

If a script that was archived turns out to be required, restore it with:

```bash
mv scripts/obsolete/<script.py> scripts/
# If needed, move it to the appropriate subfolder
mv scripts/<script.py> scripts/tools/
chmod +x scripts/tools/<script.py>
```

# Notes

- Logs and generated files were archived to reduce clutter. Keep large files out of git.
- Tests moved to `tests/scripts/` to comply with pytest discovery and CI conventions.
- If you'd like a different layout (e.g., `scripts/tests/` instead of `tests/scripts/`), I can adjust it — but the current layout aligns with common CI expectations.


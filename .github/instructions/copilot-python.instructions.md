---
applyTo:
  - "**/*.py"
  - "tests/**/*.py"
exclude:
  - "docs/**"
  - ".venv/**"
---

# Copilot — Python project guidance (optimal)

## ⚠️ File Location Rules

**NEVER create files outside proper directories:**
- ❌ NEVER use `/tmp/` or system directories
- ✅ Source code → `src/` ONLY
- ✅ Tests → `tests/` ONLY
- ✅ Docs → `docs/` ONLY
- ✅ Scripts → `scripts/` ONLY (if needed)

## Project context (single-source hints)

* Repository layout:
    * Source code: `src/` - ALL source code here
    * Tests: `tests/` - ALL tests here (pytest only)
    * Docs: `docs/` - ALL documentation here
    * Scripts: `scripts/` - Helper scripts (if needed)
* Dependency manager: **Poetry**.
    * Add runtime deps: `poetry add <package>`
    * Add dev deps: `poetry add --group dev <package>`
    * Run commands: `poetry run <cmd>`
* Supported Python versions: `>=3.11,<3.14`
* Test runner: `pytest` (run with `poetry run pytest`)

---

## Code style (strong preferences — non-forcing where noted)
These are **preferences** to improve consistency and maintainability. They are guidance, not rigid rules — do not invent arbitrary numeric limits.

### Docstrings & typing
* Prefer **Google-style** docstrings for public functions/classes (short summary, Args, Returns, Raises, Examples when useful).
* Always annotate public function/method arguments and returns using Python typing (`str`, `int`, `list[str]`, `Optional[...]`, `collections.abc` types, etc.). Use `from __future__ import annotations` where helpful.
* Keep docstrings concise but comprehensive: communicate behavior, edge cases, units, and side effects.

### Modules, functions & package structure (non-forcing guidance)
* Preference (non-forcing): **shorter modules** are generally better. Aim to keep a module focused on a single area of responsibility.
* Avoid hard line-count limits. Instead prefer: "If a module feels like it implements multiple responsibilities or is becoming hard to navigate, split it."
* Prefer **smaller functions** with a single responsibility — but avoid splitting into tiny functions that hurt readability. When the split improves testability, readability, or reuse, prefer the split.
* Prefer more small, well-named functions rather than one big function doing many things.
* Use clear package separation: domain logic, I/O (adapters), tests, CLI/tools, and config should be separated into logical packages/modules.

### Design patterns (use when appropriate)
* Encourage using established design patterns (Factory, Strategy, Adapter, Observer, Dependency Injection, Singleton only when necessary) **when they clarify intent or reduce duplication**.
* Do **not** over-engineer — prefer simple, explicit solutions unless a pattern measurably improves clarity or extensibility.
* When a pattern is used, include a short comment explaining *why* that pattern was chosen.

### Readability & maintainability
* Prefer descriptive names over clever brevity.
* Keep public API stable; prefer deprecation with clear notes rather than abrupt breaking changes.
* Add short examples in docstrings for nontrivial APIs.

---

## Testing & refactoring workflow (required practice)
These rules protect correctness during change:

1. **Run tests before any change** (document the baseline test state).
2. For **small, incremental refactors**:
    * Make a single focused change.
    * Run the test suite (or affected tests) immediately.
    * If tests fail and they were already failing before the change in the same way, this is acceptable — note that the failure predates the change.
    * If new tests fail that were passing previously, fix them or revert the change before proceeding.
    * Only continue with further refactors after validating that the refactor's impact is understood and either tests pass or failures are unchanged from the baseline.
3. Use `tests/conftest.py` fixtures for shared setup and `@pytest.mark.parametrize` for table-driven cases.
4. Prefer fast, isolated unit tests and keep integration tests explicit and separated (e.g., `tests/integration/`).
5. All tests created are pytests and all tests are stored in `tests` or in packages within `tests`.

---

## Examples (short)
**Google docstring + typing**
```py
def compute_area(width: float, height: float) -> float:
    """Compute rectangle area.

    Optional longer explanation of the behaviour that can stretch over multiple sentences.

    Args:
        width: Width in metres.
        height: Height in metres.

    Returns:
        Area in square metres.

    Raises:
      SpecificError: Some reason.
      AssertionError: Another reason.
    """

    return width * height
```

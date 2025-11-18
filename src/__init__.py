"""Top-level src package.

Expose a lightweight `core` attribute by importing the minimal
`src.core` package to make string-based patch targets work in tests
(e.g. "src.core.local_llm_client.AutoTokenizer...").
"""

from __future__ import annotations

# Import minimal core package so that attribute access like `src.core` works
# during unittest.mock.resolve_name without triggering heavy imports.
import importlib
from types import ModuleType

core: ModuleType | None
try:
    core = importlib.import_module("src.core")
except Exception:
    # Keep import-time failures silent; tests will report more specific errors
    core = None

__all__: list[str] = ["core"]

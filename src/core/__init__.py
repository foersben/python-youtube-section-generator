"""src.core package (lightweight).

This file intentionally avoids importing submodules at package import time
so that unit tests and tools can import `src.core` without triggering
heavy imports. Import submodules explicitly where needed.
"""

from __future__ import annotations

__all__ = []

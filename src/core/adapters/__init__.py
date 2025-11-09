"""Compatibility adapters package.

Contains thin backward-compatible adapters that forward to the new
refactored implementations in the `llm` and `services` packages.

This module intentionally avoids importing submodules at package import
time to keep startup lightweight and to avoid circular import issues.
"""

# Export submodule names – import submodules directly when needed
__all__ = ["local_llm_client", "gemini_client", "deepl_client"]

"""Deprecated DeepL adapter shim.

The old `DeepLClient` adapter has been pruned. This module keeps a
compatibility shim which raises a clear error if instantiated.

Please use `src.core.services.translation.DeepLAdapter` directly.
"""

from __future__ import annotations

import warnings


class DeepLClient:  # pragma: no cover - compatibility shim
    """Compatibility shim for removed DeepLClient adapter.

    Instantiating this class raises a RuntimeError with guidance on the
    new API. We keep a shim to avoid ImportError on import but prevent use.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DeepLClient has been removed. Use src.core.services.translation.DeepLAdapter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise RuntimeError(
            "DeepLClient adapter removed: please use src.core.services.translation.DeepLAdapter(api_key) "
            "or the adapters in src.core.adapters for supported clients."
        )


__all__ = ["DeepLClient"]

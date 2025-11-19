# Adapter Migration Guide

This document explains the removal of legacy adapter shims and how to migrate
existing code to the modern providers and factory.

## Background

Legacy adapter shims such as `DeepLClient` and `GeminiService` were once small
compatibility wrappers around older provider APIs. They have been pruned in
favor of direct use of the refactored provider classes and `LLMFactory`.

To make migration explicit, the repository now provides compatibility shims that
raise a `RuntimeError` when instantiated. This prevents accidental continued
use of the old adapters while allowing code that simply imports these symbols
without instantiation to continue to load.

## Migration steps

### DeepL

Old:
```python
from src.core.adapters.deepl_client import DeepLClient
client = DeepLClient(api_key="...")
translated = client.translate(text, "EN")
```

New:
```python
from src.core.services.translation import DeepLAdapter
adapter = DeepLAdapter(api_key=os.getenv("DEEPL_API_KEY"))
translated = adapter.translate(text, "EN")
```

### Gemini (Google GenAI)

Old:
```python
from src.core.adapters.gemini_client import GeminiService
svc = GeminiService(api_key_env="GOOGLE_API_KEY")
result = svc.generate(prompt, [])
```

New:
```python
from src.core.llm.gemini_provider import GeminiProvider
provider = GeminiProvider(api_key=os.getenv("GOOGLE_API_KEY"))
result = provider.generate_text(prompt, max_tokens=1024)
```

Or use the factory shorthand:
```python
from src.core.llm.factory import LLMFactory
provider = LLMFactory.create_gemini_provider(api_key=os.getenv("GOOGLE_API_KEY"))
```

## Tests & Validation

- Replace old adapter usage in unit tests with the new provider class or mock the provider with `unittest.mock`.
- Run `poetry run pytest` after migration.

## Troubleshooting

If you hit the compatibility shim `RuntimeError`, search the codebase for
`DeepLClient` or `GeminiService` and update each instantiation to the new API.

If you maintain third-party scripts that rely on the old adapters, update them
or provide a compatibility layer in your script that adapts to the new API.

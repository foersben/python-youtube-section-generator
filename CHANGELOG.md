# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Removed / Deprecated

- Deprecated adapter shims: `DeepLClient` and `GeminiService` are removed and replaced by compatibility shims that raise a clear runtime error. Users should migrate to the new APIs:
  - `src.core.services.translation.DeepLAdapter` for DeepL translations
  - `src.core.llm.GeminiProvider` or `LLMFactory.create_provider()` for Gemini/Google genAI usage

Migration note: The compatibility shims intentionally raise an error at instantiation to make migration visible at runtime; search your codebase for `DeepLClient` and `GeminiService` and update those call sites to the new APIs.

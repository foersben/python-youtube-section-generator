# Processing pipeline architecture (concept)

Goal

Define a modular, configurable processing pipeline for transcript -> sections -> titles
that supports multiple strategies (RAG-based, staged LLM, heuristic) and is easy
to extend and test.

Key ideas

- Pipeline stages are small, pure functions or services with well-defined I/O:
  - Input/Output contract: list[dict[str,Any]] for transcript/segments and
    list[Section] for sections, where Section is a small dataclass.
- Use a 'Pipeline' orchestrator that sequences stages and handles retries,
  parallelism, and fallback strategies.
- Implement stages as plugins to allow swapping strategies without code changes.

Suggested stages

1. Preprocessing: language detection, basic cleaning, normalization
2. Section detection: identify anchors/anchors candidates (no titles)
3. Section refinement: expand/merge anchors, compute time windows
4. Title + description generation: per-section LLM job (flexible length)
5. Title polishing: deterministic LLM/local heuristics for title length and style
6. Validation & formatting: ensure timestamps, length constraints, uniqueness

Design patterns

- Pipes & Filters: each stage transforms an input stream to an output stream.
- Chain of Responsibility / Orchestrator: The pipeline orchestrator can apply
  fallbacks and alternative stage implementations.
- Factory/Plugin loader: to select stage implementations based on config
  (environment variables or a JSON/YAML config file).

Configuration

- `PROCESSING_PIPELINE` env var selects `single` (legacy), `staged`, or `rag`.
- `PIPELINE_CONFIG` provides per-stage overrides (JSON/YAML file path).
- All LLM calls routed through an abstraction (LLMProvider) allowing local or
  cloud switches.

Observability & testing

- Emit structured logs for each stage including timing and sample inputs/outputs.
- Provide unit tests for each stage and integration tests for pipeline variants.
- Add a benchmark harness to compare quality/latency for each pipeline option.

Next steps for implementation

- Implement the `Section` dataclass and stage interface in `src/core/models`.
- Implement a simple pipeline orchestrator class in `src/core/pipeline.py`.
- Start with a `staged` implementation that mirrors the 3 phases described in the prospects todos.

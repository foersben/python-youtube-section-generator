# Processing Pipeline Architecture

## Motivation

The previous section generation workflow relied on a single complex prompt that forced the LLM to simultaneously detect sections, craft titles, and enforce formatting. This design resulted in:

- **Long, error-prone prompts** exceeding context limits on longer transcripts
- **Coupled responsibilities** that made experimentation and tuning difficult
- **Rigid control flow** where minor changes required touching multiple modules

Splitting the workflow into discrete stages reduces prompt size, isolates responsibilities, and enables selective experimentation.

## Stage-Based Design

The new pipeline follows a pipes-and-filters style architecture:

1. **DetermineSectionsStage**
   - Goal: identify candidate section boundaries without enforcing naming rules
   - Behavior: call the configured LLM provider with relaxed constraints and optional overshoot ratio
   - Output: list of `Section` objects stored in pipeline context

2. **EnrichSectionsStage**
   - Goal: produce contextual titles and descriptions for each section
   - Behavior: iterate each Section, pull transcript slices around the timestamp, prompt LLM for structured metadata, store optional descriptions for downstream consumers

3. **FinalizeTitlesStage**
   - Goal: apply fuzzy title length limits and polish wording, ensuring final output suits YouTube formatting
   - Behavior: rewrite titles to stay within configured word ranges while preserving meaning

The stages share a `PipelineContext` structure (transcript, sections, metadata) allowing future stages like `DescriptionSummariesStage` or `LocalizationStage` to plug in easily.

## Strategy Layer

A `PipelineStrategy` interface allows composing stage sequences dynamically. The `.env` variable `PIPELINE_STRATEGY` selects which strategy to run:

- `legacy`: maintains single-stage behavior for backward compatibility
- `split`: runs Determine → Enrich → Finalize stages

The `STRATEGY_REGISTRY` in `src/core/services/pipeline.py` provides a central place to register new strategies, enabling future variants such as:

- **Lightweight Mode** (Determine + Finalize) for rapid drafts
- **Rich Narrative Mode** (Determine + Enrich + Summaries + Finalize)
- **Adaptive RAG Mode** that inserts retrieval stages conditionally based on transcript length or topic

## Configuration and Overshoot

The `.env` options control pipeline behavior:

- `PIPELINE_STRATEGY`: selects which strategy to run (`legacy` or `split` today)
- `PIPELINE_SECTION_OVERSHOOT`: decimal multiplier applied to requested section count during the determination stage. Example: with `max_sections=15` and overshoot `0.2`, the detection stage will request 18 sections, giving later stages more material to prune.

CLI users can override the strategy via `--pipeline-strategy`. The Flask endpoint accepts a `pipeline_strategy` form field, making it straightforward to expose a dropdown in the UI.

## Extension Points

- **Stage Registry**: future work can formalize a registry of stage classes and support declarative YAML/JSON stage sequences for A/B experimentation.
- **Context Contracts**: currently, metadata keys (`raw_sections`, `descriptions`) are string-based. Moving to typed dataclasses per stage would reinforce contracts.
- **Parallel Branching**: implement fan-out (e.g., parallel description and translation stages) followed by merge.
- **Retry Policies**: each stage can adopt stage-specific retry/backoff policies when calling external services.
- **Observation Hooks**: add instrumentation callbacks before/after stages for metrics, tracing, and debugging.
- **UI Integration**: the Flask UI now exposes a `pipeline_strategy` selector and shares the active strategy back to the operator for quick experimentation.

## Testing Strategy

- Unit tests per stage verifying behavior with mocked LLM providers
- Integration tests confirming stage ordering and context sharing
- Regression suite comparing legacy vs split outputs on reference transcripts
- Browser smoke tests to ensure the pipeline selector wiring between templates and backend remains functional

This architecture makes pipeline changes incremental and safer, supports experimentation through environment toggles, reduces prompt sizes, and keeps the UI and backend aligned on the selected processing pipeline.

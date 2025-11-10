# Local LLM Configuration & Licensing

This page documents how the project configures local LLM providers, model paths,
quantization options and licensing considerations for including local models in
CI or development.

## Local model path

- Default env var: `LOCAL_MODEL_PATH`
- Typical default used in the project: `models/Phi-3-mini-4k-instruct-q4.gguf`
- Set this in your `.env` or in the environment where the process runs.

Example `.env` entry:

```env
LOCAL_MODEL_PATH=models/Phi-3-mini-4k-instruct-q4.gguf
USE_LOCAL_LLM=true
```

## Quantization and memory

- 4-bit quantized models (`q4`) greatly reduce memory usage and are preferred
  for CPU-only inference.
- Typical RAM guidance:
  - 4-bit quantized: ~5–8 GB required (depends on model)
  - Full precision: 16+ GB recommended
- See `scripts/download_model.sh` for automated download helpers.

## Model providers supported

- Local GGUF models (llama.cpp via `llama-cpp-python`) — default when `USE_LOCAL_LLM=true`.
- Google Gemini (cloud) — used when `USE_LOCAL_LLM=false` and appropriate API keys are set.
- The project uses an LLM factory pattern; configuration is read from env vars and `pyproject.toml` scripts.

## Licensing & redistribution

- Many modern LLM weights are distributed under non-permissive licenses. Before
  bundling or redistributing a model, confirm the license and whether usage is
  allowed. For example, Meta Llama weights require acceptance of specific terms.
- Do NOT commit model weights into Git.
- Keep license acknowledgements in documentation and the project `LICENSE` file.

## CI considerations

- Self-hosted runners must have models present at `LOCAL_MODEL_PATH` or set the
  env var in runner configuration.
- The heavy CI job expects the runner to have sufficient resources and to be
  trusted. Keep these runners restricted to authorized builds.

## Troubleshooting

- If a local model fails to load:
  - Confirm `LOCAL_MODEL_PATH` is correct and file exists.
  - Confirm `llama-cpp-python` is installed in the environment.
  - Try reduced `n_ctx` or enable quantization.

- If memory errors occur:
  - Use the quantized variant (q4) or increase system RAM / use swap cautiously.



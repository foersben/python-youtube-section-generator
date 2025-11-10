# Scripts Directory (imported)

This page was imported from `scripts/README.md` and documents the project's helper scripts.

## CPU-Only RAG Setup

### Quick Start

```bash
# Automated setup (recommended)
./setup_cpu_rag.sh
```

This installs all RAG dependencies (sentence-transformers, chromadb) with CPU-only torch.

---

## Available Scripts

### `setup_cpu_rag.sh`
**Purpose**: Install RAG dependencies with CPU-only torch (no GPU packages)

**Usage**:

```bash
./setup_cpu_rag.sh
```

**What it does**:
1. Runs `poetry install` (clean dependencies)
2. Installs CPU-only torch from PyTorch index
3. Installs sentence-transformers and chromadb
4. Verifies no NVIDIA GPU packages

**When to use**: Fresh setup or after removing GPU packages

---

### `test_rag.py`
**Purpose**: Test RAG system with simulated long video

**Usage**:

```bash
poetry run python scripts/test_rag.py
```

**What it tests**:
- RAG system initialization
- Embeddings model loading
- ChromaDB indexing
- Hierarchical section generation

---

### `test_llm_cpu.py`
**Purpose**: Test local LLM (llama.cpp) without RAG

**Usage**:

```bash
poetry run python scripts/test_llm_cpu.py
```

**What it tests**:
- Model loading
- Direct section generation
- CPU inference

---

### `test_web_integration.py`
**Purpose**: Test web app integration with local LLM

**Usage**:

```bash
poetry run python scripts/test_web_integration.py
```

**What it tests**:
- sections.py integration
- LocalLLMClient initialization
- End-to-end section generation

---

### `download_model.sh`
**Purpose**: Download Phi-3 GGUF model for llama.cpp

**Usage**:

```bash
./download_model.sh
```

**What it does**:
- Downloads Phi-3-mini-4k-instruct Q4 GGUF (~2.4GB)
- Verifies file size
- Supports resume on interruption

---

### `check_download.sh`
**Purpose**: Check model download status

**Usage**:

```bash
./check_download.sh
```

**What it shows**:
- Current file size
- Download progress
- Instructions if incomplete

---

## Typical Workflow

### 1. Fresh Setup

```bash
# Install all dependencies
./setup_cpu_rag.sh

# Download model
./download_model.sh

# Test everything
poetry run python scripts/test_llm_cpu.py
poetry run python scripts/test_rag.py
```

### 2. After Removing GPU Packages

```bash
# Clean install
./setup_cpu_rag.sh

# Verify
poetry run pip list | grep nvidia  # Should be empty
```

### 3. Testing

```bash
# Test local LLM
poetry run python scripts/test_llm_cpu.py

# Test RAG system
poetry run python scripts/test_rag.py

# Test web integration
poetry run python scripts/test_web_integration.py
```

---

## Environment Variables

Set in `.env` file:

```bash
# Local LLM
USE_LOCAL_LLM=true
LOCAL_MODEL_PATH=models/Phi-3-mini-4k-instruct-q4.gguf

# RAG Configuration
USE_RAG=auto  # auto, always, never
RAG_HIERARCHICAL=true  # Hierarchical sections
LLM_TEMPERATURE=0.3  # Creativity (0.0-1.0)
```

---

## Troubleshooting

### NVIDIA Packages Installed

```bash
# Clean them
poetry run pip uninstall $(poetry run pip list | grep nvidia | awk '{print $1}') -y

# Re-run setup
./setup_cpu_rag.sh
```

### Model Not Found

```bash
# Download it
./download_model.sh

# Check status
./check_download.sh
```

### Import Errors

```bash
# Re-install RAG dependencies
./setup_cpu_rag.sh
```

---

## Support

For issues or questions:
1. Check documentation in `docs/`
2. Verify environment with `poetry run pip list`
3. Run test scripts to identify issues

---

## Maintenance notes

During maintenance, several legacy and generated artifacts were moved to `scripts/obsolete/` to keep the primary `scripts/` folder focused and easy to navigate. Files moved include migration scripts, debug dumps, logs, and generated outputs. The `scripts/obsolete/` folder preserves these.

If you need any file back, it's preserved in `scripts/obsolete/`.


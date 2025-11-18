"""Local LLM provider using llama-cpp-python (CPU-optimized).

Implements LLMProvider interface for local GGUF models.
"""

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any

from src.core.config import config as _config
from src.core.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class LocalLLMProvider(LLMProvider):
    """Local LLM using llama.cpp for CPU-only inference."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        temperature: float = 0.2,
    ) -> None:
        """Initialize local LLM provider.

        Args:
            model_path: Path to GGUF model file (can be relative).
            n_ctx: Context window size.
            n_threads: CPU threads (None = auto-detect).
            temperature: Default sampling temperature.
        """
        # Normalize model_path and try multiple candidate locations to be robust
        supplied = Path(model_path)
        candidates: list[Path] = []

        # If absolute or exists as given, check directly
        candidates.append(supplied)
        # Resolved path
        try:
            candidates.append(supplied.resolve())
        except Exception:
            pass

        # Project-root based candidate
        try:
            project_root = Path(_config.project_root)
            candidates.append(project_root / supplied)
            candidates.append(project_root / "models" / supplied.name)
        except Exception:
            pass

        # CWD candidate
        candidates.append(Path.cwd() / supplied)
        # Avoid src-based relative candidates to prevent resolving to src/models
        # (we prefer project_root and explicit models/ directory)

        # Home cache candidate
        candidates.append(Path.home() / ".cache" / "models" / supplied.name)

        found: Path | None = None
        tried = []
        for c in candidates:
            try:
                p = c if c.is_absolute() else c.resolve()
            except Exception:
                p = c
            tried.append(str(p))
            if p.exists():
                found = p
                break

        if found is None:
            raise FileNotFoundError(
                f"Model file not found: {model_path}\nTried these locations:\n - "
                + "\n - ".join(tried)
                + "\nDownload with: ./scripts/download_model.sh"
            )

        self.model_path = found

        # Allow overriding via environment for experiments / CI
        try:
            env_n_ctx = os.getenv("LLM_N_CTX")
            if env_n_ctx:
                n_ctx = int(env_n_ctx)
        except Exception:
            logger.debug("Invalid LLM_N_CTX, falling back to provided n_ctx=%s", n_ctx)

        self.n_ctx = n_ctx
        self.n_threads = n_threads or multiprocessing.cpu_count()
        self.default_temperature = temperature

        logger.info(f"Loading local LLM: {self.model_path.name} (from {self.model_path})")
        logger.info(f"Context: {n_ctx}, Threads: {self.n_threads}")

        # Import llama-cpp only after we've confirmed the model file exists to avoid
        # failing early on missing models when llama-cpp isn't installed in test envs.
        try:
            from llama_cpp import Llama
        except Exception as err:
            raise RuntimeError(
                "llama-cpp-python not installed. Install with: poetry add llama-cpp-python"
            ) from err

        # Check for GPU support and use it if available for faster inference
        n_gpu_layers = int(
            os.getenv("LLM_GPU_LAYERS", "-1")
        )  # -1 = auto (all layers if GPU available)

        # n_batch: make configurable by env var; default to 64 (GGML_KQ_MASK_PAD safe value)
        try:
            env_n_batch = int(os.getenv("LLM_N_BATCH", "64"))
        except Exception:
            env_n_batch = 64

        # enforce a sensible minimum to avoid llama.cpp auto-adjust messages
        n_batch = max(64, env_n_batch)

        logger.debug(
            "Initializing local LLM with n_ctx=%s, n_batch=%s, n_gpu_layers=%s",
            n_ctx,
            n_batch,
            n_gpu_layers,
        )

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=self.n_threads,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,  # Use GPU if available for massive speedup
            verbose=False,
        )

        logger.info("✅ Local LLM loaded successfully")

    def generate_sections(
        self,
        transcript: list[dict[str, Any]],
        num_sections: int,
        max_retries: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate sections from transcript."""
        if not transcript:
            raise ValueError("Transcript cannot be empty")

        prompt = self._build_section_prompt(transcript, num_sections)

        for attempt in range(max_retries):
            try:
                logger.info(f"Generating sections (attempt {attempt + 1}/{max_retries})")
                response = self.generate_text(prompt, max_tokens=512)
                sections = self._extract_json(response)

                if self._validate_sections(sections, transcript):
                    logger.info(f"Successfully generated {len(sections)} sections")
                    return sections

                logger.warning(f"Validation failed on attempt {attempt + 1}")

            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Section generation failed after retries") from e

        raise RuntimeError("Section generation failed after retries")

    def generate_text(
        self, prompt: str, max_tokens: int = 512, temperature: float | None = None
    ) -> str:
        """Generate text from prompt."""
        temp = temperature if temperature is not None else self.default_temperature

        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=0.9,
            echo=False,
            stop=["</s>", "\n\n\n"],
        )

        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["text"].strip()

        return str(response).strip()

    def get_info(self) -> dict[str, Any]:
        """Get provider information."""
        return {
            "provider": "local_llm",
            "backend": "llama.cpp",
            "model": self.model_path.name,
            "device": "cpu",
            "context_size": self.n_ctx,
            "threads": self.n_threads,
        }

    def _build_section_prompt(self, transcript: list[dict[str, Any]], num_sections: int) -> str:
        """Build prompt for section generation."""
        max_segments = 50
        transcript_text = "\n".join(
            f"[{seg['start']:.1f}s] {seg['text']}" for seg in transcript[:max_segments]
        )

        return f"""You are a helpful assistant that analyzes video transcripts.

Create {num_sections} sections from this transcript:

{transcript_text}

Respond ONLY with a JSON array in this exact format:
[
  {{"title": "Section Title", "start": 0.0}},
  {{"title": "Another Section", "start": 45.5}}
]

JSON:"""

    def _extract_json(self, text: str) -> list[dict[str, Any]]:
        """Extract JSON array from LLM response."""
        # Strategy 1: Find JSON array
        start = text.find("[")
        end = text.rfind("]") + 1

        if start != -1 and end > start:
            json_text = text[start:end]
            try:
                sections = __import__("json").loads(json_text)
                if isinstance(sections, list):
                    return sections
            except Exception:
                pass

        # Strategy 2: Regex extraction
        pattern = r'"title"\s*:\s*"([^"]+)"\s*,\s*"start"\s*:\s*(\d+\.?\d*)'
        matches = __import__("re").findall(pattern, text, __import__("re").IGNORECASE)

        sections = []
        for title, start in matches:
            sections.append({"title": title.strip(), "start": float(start)})

        if sections:
            return sections

        raise ValueError("Could not extract valid JSON from response")

    def _validate_sections(
        self, sections: list[dict[str, Any]], transcript: list[dict[str, Any]]
    ) -> bool:
        """Validate generated sections."""
        if not sections:
            return False

        max_time = max(seg["start"] for seg in transcript)

        for section in sections:
            if "title" not in section or "start" not in section:
                return False
            if not isinstance(section["start"], (int, float)):
                return False
            if section["start"] < 0 or section["start"] > max_time:
                return False

        return True


__all__ = ["LocalLLMProvider"]

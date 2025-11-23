"""Local LLM provider using llama-cpp-python (CPU-optimized).

Implements LLMProvider interface for local GGUF models.
"""

import logging
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Any

from src.core.config import config as _config
from src.core.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class LocalLLMProvider(LLMProvider):
    """Local LLM using llama.cpp for CPU-only inference.

    Thread-safe implementation optimized for limited RAM/CPU environments.
    """

    def __init__(self, model_path: str | Path, n_ctx: int = 4096, n_threads: int | None = None,
            temperature: float = 0.2, ) -> None:
        supplied = Path(model_path)
        candidates: list[Path] = []
        candidates.append(supplied)
        try:
            candidates.append(supplied.resolve())
        except Exception:
            pass
        try:
            project_root = Path(_config.project_root)
            candidates.append(project_root / supplied)
            candidates.append(project_root / "models" / supplied.name)
        except Exception:
            pass
        candidates.append(Path.cwd() / supplied)
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
            raise FileNotFoundError(f"Model file not found: {model_path}\nTried these locations:\n - " + "\n - ".join(
                tried) + "\nDownload with: ./scripts/download_model.sh")

        self.model_path = found

        try:
            env_n_ctx = os.getenv("LLM_N_CTX")
            if env_n_ctx:
                n_ctx = int(env_n_ctx)
        except Exception:
            pass

        # HARDWARE OPTIMIZATION FOR i7-10750H
        # Use physical cores (usually half of logical count) for best cache locality
        physical_cores = multiprocessing.cpu_count() // 2
        self.n_threads = n_threads or max(4, physical_cores)
        self.n_ctx = n_ctx
        self.default_temperature = temperature

        # Critical for Web Apps: Ensure only one inference runs at a time
        self._lock = threading.Lock()

        try:
            from llama_cpp import Llama
        except Exception as err:
            raise RuntimeError("llama-cpp-python not installed. Install with: poetry add llama-cpp-python") from err

        # Force CPU mode explicitly to save VRAM/Overhead
        n_gpu_layers = 0

        # Lower batch size slightly to conserve RAM during prompt processing
        n_batch = 512

        logger.info("Initializing CPU Optimized LLM: %s (Threads=%d, Context=%d)", self.model_path.name, self.n_threads,
            self.n_ctx)

        self.llm = Llama(model_path=str(self.model_path), n_ctx=n_ctx, n_threads=self.n_threads, n_batch=n_batch,
            n_gpu_layers=n_gpu_layers, verbose=False, use_mmap=True,  # Keeps model on disk mapping, good for RAM usage
            use_mlock=False,  # Don't lock RAM, allow swap if absolutely needed
        )

        logger.info("✅ Local LLM loaded successfully")

    def generate_sections(self, transcript: list[dict[str, Any]], num_sections: int, max_retries: int = 3, ) -> list[
        dict[str, Any]]:
        """Generate sections from transcript."""
        if not transcript:
            raise ValueError("Transcript cannot be empty")

        prompt = self._build_section_prompt(transcript, num_sections)

        for attempt in range(max_retries):
            try:
                # Decrease tokens slightly to ensure quick returns on CPU
                response = self.generate_text(prompt, max_tokens=512)
                sections = self._extract_json(response)

                if self._validate_sections(sections, transcript):
                    return sections
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")

        raise RuntimeError("Section generation failed after retries")

    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float | None = None) -> str:
        """Generate text from prompt with thread locking."""
        temp = temperature if temperature is not None else self.default_temperature

        # CRITICAL: Lock ensures requests queue up instead of crashing the C++ backend
        with self._lock:
            try:
                response = self.llm(prompt, max_tokens=max_tokens, temperature=temp, top_p=0.9, echo=False,
                    stop=["</s>", "\n\n\n", "```"],  # Add strict stops
                )
                if isinstance(response, dict) and "choices" in response:
                    return response["choices"][0]["text"].strip()
                return str(response).strip()
            except Exception as e:
                logger.error(f"LLM Inference failed: {e}")
                return ""

    def get_info(self) -> dict[str, Any]:
        return {"provider": "local_llm", "backend": "llama.cpp", "model": self.model_path.name, "device": "cpu",
            "context_size": self.n_ctx, "threads": self.n_threads, }

    def _build_section_prompt(self, transcript: list[dict[str, Any]], num_sections: int) -> str:
        max_segments = 50
        transcript_text = "\n".join(f"[{seg['start']:.1f}s] {seg['text']}" for seg in transcript[:max_segments])
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
        pattern = r'"title"\s*:\s*"([^"]+)"\s*,\s*"start"\s*:\s*(\d+\.?\d*)'
        matches = __import__("re").findall(pattern, text, __import__("re").IGNORECASE)
        sections = []
        for title, start in matches:
            sections.append({"title": title.strip(), "start": float(start)})
        if sections:
            return sections
        raise ValueError("Could not extract valid JSON from response")

    def _validate_sections(self, sections: list[dict[str, Any]], transcript: list[dict[str, Any]]) -> bool:
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

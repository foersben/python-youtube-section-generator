"""Environment diagnostic for GPU contamination and LLM binaries.

Run this inside your project's virtualenv to capture the "before" and "after"
state when migrating from a GPU-enabled environment to CPU-only.

Usage:
    python scripts/check_environment.py > before_fix.log

The script is intentionally defensive: missing packages are reported but do not
cause the script to crash.
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata


def _version_of(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except Exception:
        return None


def check_torch() -> None:
    print("--- PyTorch Diagnostic ---")
    try:
        import torch

        print(f"PyTorch import: OK (module: {torch.__name__})")
        print(f"PyTorch version: {getattr(torch, '__version__', 'unknown')}")
        try:
            cuda_avail = torch.cuda.is_available()
            print(f"CUDA Available: {cuda_avail}")
            if cuda_avail:
                try:
                    dev = torch.cuda.current_device()
                    print(f"Current CUDA Device index: {dev}")
                    print(f"Device Name: {torch.cuda.get_device_name(dev)}")
                    print(f"Device Capability: {torch.cuda.get_device_capability(dev)}")
                except Exception as e:
                    print(f"Could not retrieve CUDA device details: {e}")
            else:
                print("No CUDA devices are visible to PyTorch.")
        except Exception as e:
            print(f"Error checking CUDA availability: {e}")
    except Exception:
        print("PyTorch: NOT INSTALLED")


def check_llama_cpp() -> None:
    print("\n--- llama-cpp-python Diagnostic ---")
    tried = []
    try:
        # try public import
        import llama_cpp as lc  # type: ignore

        tried.append("import llama_cpp")
        print(f"llama-cpp-python import: OK (module: {lc.__name__})")
        # try access wrapper Llama
        try:
            llama_binding = getattr(lc, "Llama", None)
            print("Llama binding present: ", bool(llama_binding))
        except Exception:
            print("Could not find Llama class in llama_cpp module")

        # Try to find internal _load_shared_library if available
        try:
            # internal API varies by version; try common locations
            from llama_cpp.llama_cpp import _load_shared_library  # type: ignore

            lib = _load_shared_library("llama")
            # attempt to query for GPU support if symbol exists
            supports = getattr(lib, "llama_supports_gpu_offload", None)
            if supports is not None:
                try:
                    gpu_off = bool(supports())
                    print(f"llama.cpp GPU Offload Supported (compiled): {gpu_off}")
                except Exception as e:
                    print(f"Could not invoke llama_supports_gpu_offload: {e}")
            else:
                print("llama.cpp: no 'llama_supports_gpu_offload' symbol available in this build")
        except Exception as e:
            print(f"Could not probe llama-cpp shared library: {e}")

        # Try a dummy instance creation; we will not pass a real model path
        try:
            from llama_cpp import Llama as LlamaClass  # type: ignore

            print("Attempting a dummy Llama instantiation (expected to fail) to observe logs...")
            try:
                # This is intentionally pointing to a non-existing model: logs will reveal GPU attempts
                LlamaClass(model_path="DUMMY_PATH.gguf", n_gpu_layers=1, verbose=True)
            except Exception as e:
                print(f"Dummy Llama instantiation failed (as expected): {e}")
        except Exception:
            print("Llama class not importable via expected API (llama_cpp.Llama)")

    except Exception as e:
        print("llama-cpp-python: NOT INSTALLED or import failed", str(e))


def check_other_pkgs() -> None:
    print("\n--- Other package versions ---")
    pkgs = [
        "langchain",
        "langchain-community",
        "chromadb",
        "sentence-transformers",
        "google-genai",
        "google-generativeai",
        "llama-cpp-python",
    ]
    for p in pkgs:
        v = _version_of(p)
        print(f"{p}: {v or 'NOT INSTALLED'}")


def main() -> None:
    print("Environment diagnostic")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")

    check_torch()
    check_llama_cpp()
    check_other_pkgs()

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()

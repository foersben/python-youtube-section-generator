"""Interactive project setup for PythonYoutubeTranscript.

Checks for local models, API keys, pip-only dependencies, and offers advanced configuration.
Automatically run after poetry install (via plugin hook) or manually as a script.
"""

import argparse
import importlib
import importlib.metadata as metadata
import logging
import os
import subprocess
import sys
from pathlib import Path

# Module logger; debug messages are emitted via this logger and will only appear
# if the application configures logging to DEBUG (so normal runs remain quiet).
logger = logging.getLogger(__name__)

# Global flag to make prompts non-interactive (assume yes / use defaults)
AUTO_YES = False

# PROJECT_ROOT should point to the repository root (two parents above this file):
# scripts/cli/setup_interactive.py -> parents[0]=scripts/cli, [1]=scripts, [2]=repo_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
ENV_PATH = PROJECT_ROOT / ".env"

REQUIRED_MODELS = [
    "Phi-3-mini-4k-instruct-q4.gguf",
    # Add other required models here
]
PIP_DEPENDENCIES = ["torch", "torchvision", "torchaudio", "sentence-transformers", "chromadb"]
API_KEYS = {
    "DEEPL_API_KEY": "DeepL API key (for translation)",
    "GEMINI_API_KEY": "Google Gemini API key (for AI section generation)",
}
ADVANCED_PARAMS = {
    "LOCAL_MODEL_PATH": str(MODELS_DIR / REQUIRED_MODELS[0]),
    "REFINEMENT_BATCH_SIZE": "64",
    "LOCAL_MODEL_4BIT": "true",
    "USE_LOCAL_LLM": "true",
    "USE_TRANSLATION": "true",
}


def prompt(msg, default=None, secret=False):
    if AUTO_YES:
        # In AUTO_YES mode, prefer the provided default; if none, return empty string
        return default or ""
    if secret:
        import getpass

        val = getpass.getpass(f"{msg} [{default or 'required'}]: ")
    else:
        val = input(f"{msg} [{default or 'required'}]: ")
    return val.strip() or default


def _detect_torch_build(pkg_name: str):
    """Return 'cpu', 'gpu', 'not_installed' or 'unknown' for a given package name."""
    try:
        ver = metadata.version(pkg_name)
    except Exception:
        return "not_installed"
    v = ver.lower()
    # wheels often include +cpu or +cu (cu117, cu118, etc.)
    if "+cpu" in v:
        return "cpu"
    if "+cu" in v or "+cuda" in v or "cu" in v.split("+")[-1]:
        return "gpu"
    # fallback: try import and inspect known attributes (best-effort)
    try:
        m = importlib.import_module(pkg_name)
        if pkg_name == "torch":
            cuda_ver = getattr(getattr(m, "version", None), "cuda", None)
            if cuda_ver:
                return "gpu"
            try:
                # If CUDA runtime is available, it's likely a GPU build
                if getattr(m, "cuda", None) and getattr(m.cuda, "is_available", lambda: False)():
                    return "gpu"
            except Exception:
                pass
        # If we can't determine, return unknown
        return "unknown"
    except Exception:
        return "unknown"


def _detect_installed_torch_packages():
    pkgs = ["torch", "torchvision", "torchaudio"]
    found = {}
    for p in pkgs:
        found[p] = _detect_torch_build(p)
    return found


def _prompt_yes_no(msg, default="n"):
    if AUTO_YES:
        return default
    return prompt(msg, default=default)


def _uninstall_packages(packages):
    if not packages:
        return 0
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", *packages]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Warning: failed to uninstall some packages: {packages}")
    return res.returncode


def check_models():
    # Ensure the top-level models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Diagnostic logging to help detect path issues (logged at DEBUG so they are
    # quiet during normal runs unless the app configures DEBUG logging).
    logger.debug("PROJECT_ROOT=%s", PROJECT_ROOT)
    logger.debug("MODELS_DIR=%s exists=%s", MODELS_DIR, MODELS_DIR.exists())
    logger.debug("ENV LOCAL_MODEL_PATH=%s", os.getenv("LOCAL_MODEL_PATH"))
    try:
        listing = list(MODELS_DIR.iterdir())
        logger.debug("models/ listing (%s): %s", len(listing), [p.name for p in listing])
    except Exception as _e:
        logger.debug("models/ listing error: %s", _e)

    # First, respect an explicit environment override (LOCAL_MODEL_PATH)
    env_local = os.getenv("LOCAL_MODEL_PATH")
    found_models = set()
    if env_local:
        try:
            p = Path(env_local)
            if p.exists():
                logger.debug("Found model via LOCAL_MODEL_PATH env: %s", p)
                found_models.add(p.name)
        except Exception:
            pass

    # Next, if project config exposes a local_model_path, respect it (best-effort)
    try:
        from src.core.config import config as _config

        cfg_path = Path(str(getattr(_config, "local_model_path", "")))
        if cfg_path and cfg_path.exists():
            logger.debug("Found model via config.local_model_path: %s", cfg_path)
            found_models.add(cfg_path.name)
    except Exception:
        # If config isn't importable in this environment, ignore silently
        pass

    # Finally check the repository models/ directory
    missing = []
    for m in REQUIRED_MODELS:
        if m in found_models:
            continue
        candidate = MODELS_DIR / m
        if candidate.exists():
            logger.debug("Found model in models/: %s", candidate)
            found_models.add(m)
        else:
            missing.append(m)

    if missing:
        print(f"Missing model files: {', '.join(missing)}")
        dl = prompt("Download missing models now? (y/n)", "y")
        if dl and dl.lower().startswith("y"):
            for m in missing:
                print(f"Downloading {m}...")
                # The download helper lives under the repository's scripts/tools/ directory.
                dl_script = PROJECT_ROOT / "scripts" / "tools" / "download_model.sh"
                subprocess.run(["bash", str(dl_script), m])
    else:
        print("All required models present.")


def check_api_keys():
    env = {}
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    env[k] = v
    print("API keys are optional. The DEEPL_API_KEY is recommended for translation functionality.")
    for key, desc in API_KEYS.items():
        if not env.get(key):
            note = desc + (" (recommended)" if key == "DEEPL_API_KEY" else " (optional)")
            val = prompt(f"Enter {note}", secret=True)
            if val:
                env[key] = val
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        for k, v in env.items():
            f.write(f"{k}={v}\n")
    print(f".env updated: {', '.join(env.keys())}")


def _run_pip_install(pkgs, index_url=None):
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    if index_url:
        cmd += ["--index-url", index_url]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Failed to install: {', '.join(pkgs)}. You can try installing them manually, e.g:")
        print("  poetry run python -m pip install ", " ".join(pkgs))
        if index_url:
            print(
                f"If these are PyTorch packages, try: poetry run python -m pip install {' '.join(pkgs)} --index-url {index_url}"
            )
        # return non-zero so caller may decide what to do
    return res.returncode


def check_pip_dependencies():
    missing = []
    for pkg in PIP_DEPENDENCIES:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing pip-only dependencies: {', '.join(missing)}")
        inst = _prompt_yes_no("Install missing pip dependencies now? (y/n)", "y")
        if inst.lower().startswith("y"):
            # Before installing, detect any existing torch GPU builds and offer to clean them up.
            installed = _detect_installed_torch_packages()
            gpu_pkgs = [p for p, status in installed.items() if status == "gpu"]
            if gpu_pkgs:
                print("Detected existing GPU builds for:", ", ".join(gpu_pkgs))
                clean = _prompt_yes_no(
                    "Remove those GPU builds before installing CPU-only packages? (y/n)", "n"
                )
                if clean.lower().startswith("y"):
                    _uninstall_packages(gpu_pkgs)

            # Separate torch-related packages (which need the PyTorch index) from
            # pure-PyPI packages (like sentence-transformers, chromadb).
            torch_pkgs = [p for p in missing if p in ("torch", "torchvision", "torchaudio")]
            other_pkgs = [p for p in missing if p not in torch_pkgs]
            # Install torch CPU packages first to prevent downstream packages from
            # pulling GPU builds as dependencies.
            if torch_pkgs:
                _run_pip_install(torch_pkgs, index_url="https://download.pytorch.org/whl/cpu")
            if other_pkgs:
                _run_pip_install(other_pkgs)
    else:
        print("All pip-only dependencies present.")


def advanced_config():
    print("\nAdvanced configuration (hidden, for experts):")
    print("You can manually adjust advanced parameters or use defaults.")
    use_manual = prompt("Manually adjust advanced parameters? (y/n)", "n")
    params = ADVANCED_PARAMS.copy()
    if use_manual.lower().startswith("y"):
        for k, v in ADVANCED_PARAMS.items():
            params[k] = prompt(f"Set {k}", v)
    with open(ENV_PATH, "a", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")
    print("Advanced parameters saved to .env.")


def main():
    global AUTO_YES
    # Argument parsing: supports -y/--yes to assume defaults and --help for usage
    parser = argparse.ArgumentParser(
        prog="setup_interactive",
        description="Interactive project setup for PythonYoutubeTranscript",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Assume yes/use defaults for prompts"
    )
    args, _ = parser.parse_known_args()
    AUTO_YES = bool(args.yes)
    print("\n=== PythonYoutubeTranscript Interactive Setup ===\n")
    print("This script will check for required models, API keys, and dependencies.")
    print("You can use advanced configuration if desired.\n")
    check_models()
    check_api_keys()
    check_pip_dependencies()
    adv = prompt("Show advanced configuration options? (y/n)", "n")
    if adv.lower().startswith("y"):
        advanced_config()
    print("\nSetup complete! You can now run the application.")


if __name__ == "__main__":
    main()

"""Interactive project setup for PythonYoutubeTranscript.

Checks for local models, API keys, pip-only dependencies, and offers advanced configuration.
Automatically run after poetry install (via plugin hook) or manually as a script.
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ENV_PATH = PROJECT_ROOT / ".env"

REQUIRED_MODELS = [
    "Phi-3-mini-4k-instruct-q4.gguf",
    # Add other required models here
]
PIP_DEPENDENCIES = [
    "torch", "torchvision", "torchaudio", "sentence-transformers", "chromadb"
]
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
    if secret:
        import getpass
        val = getpass.getpass(f"{msg} [{default or 'required'}]: ")
    else:
        val = input(f"{msg} [{default or 'required'}]: ")
    return val.strip() or default

def check_models():
    MODELS_DIR.mkdir(exist_ok=True)
    missing = [m for m in REQUIRED_MODELS if not (MODELS_DIR / m).exists()]
    if missing:
        print(f"Missing model files: {', '.join(missing)}")
        dl = prompt("Download missing models now? (y/n)", "y")
        if dl.lower().startswith("y"):
            for m in missing:
                print(f"Downloading {m}...")
                # Example: download script
                subprocess.run(["bash", str(PROJECT_ROOT / "scripts" / "download_model.sh"), m])
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
    for key, desc in API_KEYS.items():
        if not env.get(key):
            val = prompt(f"Enter {desc}", secret=True)
            if val:
                env[key] = val
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        for k, v in env.items():
            f.write(f"{k}={v}\n")
    print(f".env updated: {', '.join(env.keys())}")

def check_pip_dependencies():
    missing = []
    for pkg in PIP_DEPENDENCIES:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing pip-only dependencies: {', '.join(missing)}")
        inst = prompt("Install missing pip dependencies now? (y/n)", "y")
        if inst.lower().startswith("y"):
            subprocess.run([
                sys.executable, "-m", "pip", "install", *missing,
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ])
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


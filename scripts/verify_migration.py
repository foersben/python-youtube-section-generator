#!/usr/bin/env python3
"""Verify complete Phi-3 migration and show system info."""

import sys
from pathlib import Path

print("=" * 70)
print("PHI-3 MIGRATION VERIFICATION")
print("=" * 70)
print()

# Check configuration
print("📋 Configuration Check:")
print("-" * 70)

env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                if "LOCAL" in line or "USE_LOCAL" in line:
                    print(f"  ✓ {line.strip()}")
else:
    print("  ⚠️  .env file not found")

print()

# Check Python dependencies
print("🔧 Dependencies Check:")
print("-" * 70)

try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    - GPU {i}: {name} ({mem:.1f}GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"  ✓ Apple MPS available: True")
    else:
        print(f"  ✓ CPU mode: {torch.get_num_threads()} threads")
except ImportError:
    print("  ❌ PyTorch not installed")
    sys.exit(1)

try:
    import transformers
    print(f"  ✓ Transformers: {transformers.__version__}")
except ImportError:
    print("  ❌ Transformers not installed")
    sys.exit(1)

print()

# Check model cache
print("💾 Model Cache Check:")
print("-" * 70)

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
if cache_dir.exists():
    models = list(cache_dir.glob("models--*"))
    if models:
        print(f"  ✓ Cached models: {len(models)}")
        for model in models:
            model_name = model.name.replace("models--", "").replace("--", "/")
            try:
                size_mb = sum(f.stat().st_size for f in model.rglob("*") if f.is_file()) / 1024**2
                print(f"    - {model_name}: {size_mb:.0f}MB")
            except:
                print(f"    - {model_name}")
    else:
        print("  ℹ️  No models cached (will download on first use)")
else:
    print("  ℹ️  Cache directory doesn't exist yet")

print()

# Check source files
print("📁 Source Files Check:")
print("-" * 70)

src_files = [
    "src/core/local_llm_client.py",
    "src/core/sections.py",
    "scripts/test_phi3.py",
]

for file_path in src_files:
    file = Path(file_path)
    if file.exists():
        size_kb = file.stat().st_size / 1024
        print(f"  ✓ {file_path} ({size_kb:.1f}KB)")
    else:
        print(f"  ❌ {file_path} missing")

print()

# Disk space check
print("💿 Disk Space Summary:")
print("-" * 70)

project_root = Path(".")
src_size = sum(f.stat().st_size for f in project_root.rglob("*.py") if f.is_file()) / 1024**2
docs_size = sum(f.stat().st_size for f in project_root.glob("docs/*.md") if f.is_file()) / 1024**2

print(f"  Source code: {src_size:.1f}MB")
print(f"  Documentation: {docs_size:.1f}MB")

if cache_dir.exists() and list(cache_dir.glob("models--*")):
    cache_size = sum(
        f.stat().st_size 
        for model in cache_dir.glob("models--*") 
        for f in model.rglob("*") 
        if f.is_file()
    ) / 1024**3
    print(f"  Model cache: {cache_size:.2f}GB")
else:
    print(f"  Model cache: 0GB (will be ~8GB after download)")

print()

# Summary
print("=" * 70)
print("MIGRATION STATUS")
print("=" * 70)

checks = {
    "Configuration set to Phi-3": "USE_LOCAL_LLM=true" in open(".env").read() if Path(".env").exists() else False,
    "PyTorch installed": True,  # We already checked above
    "LocalLLMClient exists": Path("src/core/local_llm_client.py").exists(),
    "Test script ready": Path("scripts/test_phi3.py").exists(),
}

all_passed = True
for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check}")
    if not passed:
        all_passed = False

print()

if all_passed:
    print("🎉 Migration Complete!")
    print()
    print("Next steps:")
    print("  1. Run: poetry run python scripts/test_phi3.py")
    print("  2. First run will download ~8GB Phi-3 model")
    print("  3. Subsequent runs will use cached model")
else:
    print("⚠️  Some checks failed. Please review above.")

print("=" * 70)


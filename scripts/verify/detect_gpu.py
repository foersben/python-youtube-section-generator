#!/usr/bin/env python3
"""Comprehensive GPU and system detection for local LLM."""

import sys
import platform
from pathlib import Path

print("=" * 70)
print("SYSTEM & GPU DETECTION")
print("=" * 70)
print()

# System info
print("📋 System Information:")
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Architecture: {platform.machine()}")
print(f"  Python: {platform.python_version()}")
print()

# CPU info
import multiprocessing
cpu_count = multiprocessing.cpu_count()
print("💻 CPU Information:")
print(f"  Cores: {cpu_count}")
print(f"  Processor: {platform.processor() or 'Unknown'}")
print()

# Memory info
try:
    import psutil
    mem = psutil.virtual_memory()
    print("🧠 Memory Information:")
    print(f"  Total RAM: {mem.total / 1024**3:.1f} GB")
    print(f"  Available RAM: {mem.available / 1024**3:.1f} GB")
    print(f"  Used: {mem.percent:.1f}%")
    print()
except ImportError:
    print("🧠 Memory Information:")
    print("  ⚠️  psutil not installed (install with: poetry add psutil)")
    print()

# PyTorch detection
print("🔥 PyTorch Detection:")
try:
    import torch
    print(f"  ✅ PyTorch version: {torch.__version__}")
    print(f"  Build: {torch.version.debug if hasattr(torch.version, 'debug') else 'Release'}")
    print()
except ImportError:
    print("  ❌ PyTorch not installed")
    sys.exit(1)

# GPU Detection
print("🎮 GPU Detection:")
print()

# 1. NVIDIA CUDA
print("1️⃣  NVIDIA CUDA:")
if torch.cuda.is_available():
    print(f"  ✅ CUDA Available: YES")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    num_gpus = torch.cuda.device_count()
    print(f"  GPU Count: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {props.name}")
        print(f"      Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"      Compute Capability: {props.major}.{props.minor}")
    
    # Test CUDA
    try:
        test_tensor = torch.zeros(1).cuda()
        print(f"  ✅ CUDA Verification: PASSED")
    except Exception as e:
        print(f"  ⚠️  CUDA Verification: FAILED ({e})")
else:
    print("  ❌ CUDA Available: NO")
    print("     (No NVIDIA GPU detected or drivers not installed)")
print()

# 2. AMD ROCm
print("2️⃣  AMD ROCm:")
if hasattr(torch.version, "hip") and torch.version.hip is not None:
    print(f"  ✅ ROCm Available: YES")
    print(f"  HIP Version: {torch.version.hip}")
    print("  Note: ROCm uses CUDA-compatible API")
else:
    print("  ❌ ROCm Available: NO")
    print("     (No AMD GPU with ROCm support detected)")
print()

# 3. Apple Metal (MPS)
print("3️⃣  Apple Metal (MPS):")
if hasattr(torch.backends, "mps"):
    if torch.backends.mps.is_available():
        print(f"  ✅ MPS Available: YES")
        print(f"  MPS Built: {torch.backends.mps.is_built()}")
        
        # Test MPS
        try:
            test_tensor = torch.zeros(1).to("mps")
            print(f"  ✅ MPS Verification: PASSED")
        except Exception as e:
            print(f"  ⚠️  MPS Verification: FAILED ({e})")
    else:
        print("  ❌ MPS Available: NO")
        print("     (Not on Apple Silicon Mac or macOS < 12.3)")
else:
    print("  ❌ MPS Available: NO")
    print("     (PyTorch version too old or not on macOS)")
print()

# 4. Intel GPU (future support)
print("4️⃣  Intel GPU:")
if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available"):
    if torch.xpu.is_available():
        print(f"  ✅ Intel XPU Available: YES")
    else:
        print("  ❌ Intel XPU Available: NO")
else:
    print("  ❌ Intel XPU: Not supported in this PyTorch build")
print()

# CPU Threads
print("5️⃣  CPU Fallback:")
num_threads = torch.get_num_threads()
print(f"  ✅ Available: YES (universal)")
print(f"  Threads: {num_threads}")
print(f"  Note: Works on all systems including older Intel Macs")
print()

# Summary and Recommendation
print("=" * 70)
print("RECOMMENDED DEVICE")
print("=" * 70)
print()

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"🎯 Use: Multi-GPU CUDA ({num_gpus} GPUs)")
        print(f"   Speed: ⚡⚡⚡⚡⚡ (Excellent)")
        print(f"   Config: Device will auto-distribute across GPUs")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 Use: Single GPU CUDA")
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {mem_gb:.1f} GB")
        print(f"   Speed: ⚡⚡⚡⚡ (Very Fast)")
        print(f"   4-bit quantization: {'✅ Recommended' if mem_gb >= 6 else '⚠️ May need it'}")
        
elif hasattr(torch.version, "hip") and torch.version.hip is not None:
    print(f"🎯 Use: AMD ROCm GPU")
    print(f"   Speed: ⚡⚡⚡ (Fast)")
    print(f"   Note: Uses CUDA-compatible API")
    
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print(f"🎯 Use: Apple Metal (MPS)")
    print(f"   Speed: ⚡⚡⚡ (Fast)")
    print(f"   Note: Uses unified memory")
    print(f"   Works on: M1/M2/M3 Macs")
    
else:
    print(f"🎯 Use: CPU")
    print(f"   Speed: ⚡ (Slow but works everywhere)")
    print(f"   Cores: {cpu_count}")
    print(f"   Note: Compatible with all systems including older Intel Macs")
    print()
    print("   ⚡ Performance Tips:")
    print("     • Generation will take 30-120 seconds")
    print("     • Consider using Gemini API for development")
    print("     • Or use a smaller model (Llama-3.2-1B)")

print()
print("=" * 70)
print("COMPATIBILITY SUMMARY")
print("=" * 70)
print()

systems = [
    ("NVIDIA GPUs (Windows/Linux)", torch.cuda.is_available()),
    ("AMD GPUs with ROCm (Linux)", hasattr(torch.version, "hip") and torch.version.hip is not None),
    ("Apple Silicon (M1/M2/M3)", hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    ("Older Intel Macs", True),  # CPU always works
    ("Windows (CPU)", True),
    ("Linux (CPU)", True),
]

for system, supported in systems:
    status = "✅ Supported" if supported else "⚠️  Not Detected (but may work)"
    print(f"  {status:20} - {system}")

print()
print("🎉 Your system is compatible with local LLM!")
print()


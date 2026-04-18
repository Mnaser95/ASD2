"""
system_info.py — collect hardware and software specs for reporting in academic papers.
Saves a human-readable report to <ROOT>/outputs/system_info.txt
"""

import platform
import sys
import subprocess
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT     = CODE_DIR.parent
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "system_info.txt"

lines = []

def section(title):
    lines.append(f"\n{'='*60}")
    lines.append(f"  {title}")
    lines.append(f"{'='*60}")

def add(label, value):
    lines.append(f"  {label:<30} {value}")

# ── OS / Python ──────────────────────────────────────────────
section("OS & Python")
add("OS",            platform.platform())
add("Python",        sys.version.split()[0])
add("Architecture",  platform.machine())

# ── CPU ──────────────────────────────────────────────────────
section("CPU")
try:
    import psutil
    add("Physical cores",   str(psutil.cpu_count(logical=False)))
    add("Logical cores",    str(psutil.cpu_count(logical=True)))
    add("Max frequency",    f"{psutil.cpu_freq().max:.0f} MHz")
    add("RAM (total)",      f"{psutil.virtual_memory().total / 1e9:.1f} GB")
    add("RAM (available)",  f"{psutil.virtual_memory().available / 1e9:.1f} GB")
except ImportError:
    add("psutil", "not installed — run: pip install psutil")

try:
    import cpuinfo
    info = cpuinfo.get_cpu_info()
    add("CPU model", info.get("brand_raw", "unknown"))
except ImportError:
    # fallback via platform
    add("CPU model", platform.processor() or "unknown (install py-cpuinfo)")

# ── GPU (NVIDIA via nvidia-smi) ──────────────────────────────
section("GPU (NVIDIA)")
try:
    smi = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=name,memory.total,driver_version,compute_cap",
         "--format=csv,noheader"],
        stderr=subprocess.DEVNULL
    ).decode().strip()
    for i, row in enumerate(smi.splitlines()):
        parts = [p.strip() for p in row.split(",")]
        add(f"GPU {i} name",         parts[0] if len(parts) > 0 else "?")
        add(f"GPU {i} VRAM",         parts[1] if len(parts) > 1 else "?")
        add(f"GPU {i} driver",       parts[2] if len(parts) > 2 else "?")
        add(f"GPU {i} compute cap",  parts[3] if len(parts) > 3 else "?")
except (subprocess.CalledProcessError, FileNotFoundError):
    add("nvidia-smi", "not available (no NVIDIA GPU or driver not installed)")

# ── CUDA ─────────────────────────────────────────────────────
section("CUDA")
try:
    nvcc = subprocess.check_output(["nvcc", "--version"],
                                    stderr=subprocess.DEVNULL).decode().strip()
    for line in nvcc.splitlines():
        if "release" in line.lower():
            add("CUDA version", line.strip())
            break
except (subprocess.CalledProcessError, FileNotFoundError):
    add("nvcc", "not found")

# ── TensorFlow ───────────────────────────────────────────────
section("TensorFlow")
try:
    import tensorflow as tf
    add("TensorFlow",   tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    add("TF visible GPUs", str(len(gpus)))
    for i, g in enumerate(gpus):
        add(f"  GPU {i}", g.name)
    try:
        from tensorflow.python.framework.config import get_memory_info
        add("TF built with CUDA", str(tf.test.is_built_with_cuda()))
    except Exception:
        pass
except ImportError:
    add("TensorFlow", "not installed")

# ── XGBoost ──────────────────────────────────────────────────
section("XGBoost")
try:
    import xgboost
    add("XGBoost", xgboost.__version__)
except ImportError:
    add("XGBoost", "not installed")

# ── scikit-learn ─────────────────────────────────────────────
section("scikit-learn")
try:
    import sklearn
    add("scikit-learn", sklearn.__version__)
except ImportError:
    add("scikit-learn", "not installed")

# ── NumPy / Pandas ───────────────────────────────────────────
section("Core libraries")
try:
    import numpy as np
    add("NumPy", np.__version__)
except ImportError:
    add("NumPy", "not installed")
try:
    import pandas as pd
    add("Pandas", pd.__version__)
except ImportError:
    add("Pandas", "not installed")

# ── Write output ─────────────────────────────────────────────
report = "\n".join(lines)
print(report)
OUT_PATH.write_text(report, encoding="utf-8")
print(f"\nSaved: {OUT_PATH}")

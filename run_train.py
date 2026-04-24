#!/usr/bin/env python3
"""
run_train.py — wrapper around ns-train that activates the marsgs conda env.

Usage (exactly as before):

  # Bilinear
  python ~/run_train.py splatfacto --data ~/COLMAP --output-dir ~/COLMAP/outputs/bilinear --viewer.quit-on-train-completion True --max-num-iterations 30000 colmap --colmap-path sparse/0

  # Malvar
  python ~/run_train.py splatfacto --data ~/COLMAP --output-dir ~/COLMAP/outputs/malvar --viewer.quit-on-train-completion True --max-num-iterations 30000 colmap --colmap-path sparse/0

If marsgs environment is missing, recreate it first:
  conda create -n marsgs python=3.10 -y
  conda run -n marsgs pip install nerfstudio
"""

import os
import sys
import shutil
import subprocess

CONDA_ENV = "marsgs"

# ── Find ns-train — first check if it's already on PATH (env already active) ─
ns_train = shutil.which("ns-train")

# ── Otherwise find it inside the marsgs conda environment ────────────────────
if ns_train is None:
    conda_exe = shutil.which("conda") or os.path.expanduser("~/miniconda3/bin/conda")
    result = subprocess.run(
        [conda_exe, "run", "-n", CONDA_ENV, "which", "ns-train"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        ns_train = result.stdout.strip()

if ns_train is None:
    print(f"ERROR: ns-train not found in conda env '{CONDA_ENV}'.", file=sys.stderr)
    print("Recreate the environment with:", file=sys.stderr)
    print(f"  conda create -n {CONDA_ENV} python=3.10 -y", file=sys.stderr)
    print(f"  conda run -n {CONDA_ENV} pip install nerfstudio", file=sys.stderr)
    sys.exit(1)

# ── Build environment with correct LD_LIBRARY_PATH for CUDA ──────────────────
# Get the Python path inside the marsgs env to locate torch libs
conda_exe = shutil.which("conda") or os.path.expanduser("~/miniconda3/bin/conda")
py_result = subprocess.run(
    [conda_exe, "run", "-n", CONDA_ENV, "python", "-c",
     "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"],
    capture_output=True, text=True
)
env = dict(os.environ)
if py_result.returncode == 0:
    torch_lib = py_result.stdout.strip()
    env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")

# ── Run ns-train via conda run so the full env is active ─────────────────────
cmd = [conda_exe, "run", "--no-capture-output", "-n", CONDA_ENV, ns_train] + sys.argv[1:]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, env=env)
sys.exit(result.returncode)

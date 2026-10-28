#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python --version
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. In Colab select Runtime > Change runtime type > T4 GPU, then reconnect."
    )
print("CUDA device:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
PY

python -m pip install --upgrade pip
python -m pip install --upgrade-strategy only-if-needed -r requirements-colab.txt
python -m pip check
python scripts/verify_device.py

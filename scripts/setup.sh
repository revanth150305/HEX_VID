#!/usr/bin/env bash
set -euo pipefail
TORCH_BACKEND="cpu"
NATIVE_SORT=false
SKIP_TORCH=false
PYTHON_BIN="${PYTHON:-python3}"
usage() { cat <<'EOF'
Usage: bash scripts/setup.sh [--torch cpu|cu118|cu121|cu124|rocm6.2] [--native-sort] [--skip-torch] [--python PATH]
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch) TORCH_BACKEND="$2"; shift 2 ;;
    --native-sort) NATIVE_SORT=true; shift ;;
    --skip-torch) SKIP_TORCH=true; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$TORCH_BACKEND" in cpu|cu118|cu121|cu124|rocm6.2) ;; *) echo "Unsupported PyTorch backend: $TORCH_BACKEND" >&2; exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Python 3.10-3.12 is required"'
[[ -d .venv ]] || "$PYTHON_BIN" -m venv .venv
VENV_PYTHON="$ROOT/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Existing .venv must use Python 3.10-3.12; remove .venv and rerun setup"'
if [[ "$SKIP_TORCH" == false ]]; then
  "$VENV_PYTHON" -m pip install --upgrade --index-url "https://download.pytorch.org/whl/$TORCH_BACKEND" torch torchvision
fi
"$VENV_PYTHON" -m pip install --upgrade-strategy only-if-needed -r requirements.txt
if [[ "$NATIVE_SORT" == true ]]; then
  echo "Installing native sort-tracker (requires OpenCV/Eigen development headers and a compiler)."
  "$VENV_PYTHON" -m pip install -r requirements-tracker-native.txt
fi
mkdir -p data/raw data/processed data/subset checkpoints outputs results logs configs src tests
"$VENV_PYTHON" -c "import cv2, torch, transformers; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available()); print('opencv=', cv2.__version__); print('transformers=', transformers.__version__)"
echo "Setup complete. Activate with: source .venv/bin/activate"


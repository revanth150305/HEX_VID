# Video anomaly detection — environment setup

This starter setup supports Python **3.10–3.12** and creates a local `.venv`.
Deep-learning wheels are deliberately installed separately from `requirements.txt`:
CUDA/ROCm wheels are platform-specific, and putting one in requirements is a
common source of dependency conflicts.

Install Git before setup: it is required for the OpenAI CLIP dependency. On
Windows, install [Git for Windows](https://git-scm.com/download/win); on Linux
use your package manager; and on macOS it is included with Xcode Command Line
Tools (`xcode-select --install`).

`requirements.txt` corrects `EasyDict` to `1.13`, includes OpenAI CLIP,
`open-clip-torch`, and Transformers' built-in CLIP support, and avoids the unrelated PyPI `sort`
package. `filterpy` and `lapx` provide dependencies typically used by a
pure-Python SORT implementation. An optional `sort-tracker` native binding is
available only when explicitly requested.

## Quick start

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1 -Torch cpu
# NVIDIA example: .\scripts\setup.ps1 -Torch cu121
.\.venv\Scripts\Activate.ps1
```

### Linux

```bash
sudo apt-get update && sudo apt-get install -y python3-venv ffmpeg
bash scripts/setup.sh --torch cpu
# NVIDIA example: bash scripts/setup.sh --torch cu121
source .venv/bin/activate
```

### macOS

```bash
brew install python ffmpeg
bash scripts/setup.sh --torch cpu
source .venv/bin/activate
```

For CUDA/ROCm, choose a backend supported by your driver and hardware; see the
[official PyTorch selector](https://pytorch.org/get-started/locally/). The
scripts accept `cpu`, `cu118`, `cu121`, `cu124`, or `rocm6.2`. `rocm6.2`
is Linux-only. The current training scripts require CUDA, so use the Colab
workflow below for this project.

### Google Colab — Tesla T4 (recommended)

The active project configuration targets Colab's CUDA Tesla T4 runtime and
uses a pretrained **R3D-18 3D video model**. In Colab, select **Runtime →
Change runtime type → T4 GPU**, then clone/upload this project and run:

```bash
!bash scripts/setup_colab.sh
!python scripts/verify_device.py
```

The Colab setup intentionally leaves Colab's CUDA-matched PyTorch installation
in place. It only installs the non-PyTorch project dependencies and then
verifies that a tensor and the R3D-18 model run on CUDA.

Training defaults are optimized for the T4: CUDA mixed precision, pinned host
memory, and batch size 8. Reduce `--batch-size` to 4 if the runtime reports an
out-of-memory error; pass `--no-amp` only when debugging numerical issues.

For a dataset stored in Google Drive, mount the drive and expose it at the
project's expected path. This avoids copying large frame folders into Colab's
temporary disk. Change `MyDrive/Course Project` if your Drive folder differs:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Course Project
# Dataset must contain Train/<class>/... and Test/<class>/..., as it does now.
!python scripts/setup_environment.py
```

Then start training with this cell:

```bash
!python scripts/train_supervised.py \
  --data-root data/raw/ucf_crime \
  --train-list data/subset/ucf_crime_train_subset.txt \
  --val-list data/subset/ucf_crime_test_subset.txt \
  --epochs 30 --patience 5 --batch-size 8 --num-workers 2
```

The dataset location in Drive should therefore be
`Course Project/data/raw/ucf_crime/Train/<class>/...` and
`Course Project/data/raw/ucf_crime/Test/<class>/...`. The setup script builds
the two subset-list files from your PNG frame folders before training begins.

## Optional native SORT binding

`sort-tracker` needs a compiler plus OpenCV and Eigen development headers and
is most reliable on Linux:

```bash
sudo apt-get install -y libopencv-dev libeigen3-dev build-essential
bash scripts/setup.sh --native-sort
```

The native module imports as `sort`, but it is **not** the unrelated `sort`
PyPI package. Prefer a pure-Python SORT implementation where portability matters.

## Project layout and running

Setup creates `data/raw`, `data/processed`, `checkpoints`, `outputs`,
`logs`, `configs`, `src`, and `tests`. This project already exposes supervised
and weakly supervised training scripts, a demo inference script, and evaluation.

```powershell
.\scripts\run.ps1 supervised -ExtraArgs @("--train-list", "data/subset/ucf_crime_train_subset.txt", "--val-list", "data/subset/ucf_crime_test_subset.txt", "--data-root", "data/raw/ucf_crime")
.\scripts\run.ps1 infer
```

```bash
bash scripts/run.sh supervised --train-list data/subset/ucf_crime_train_subset.txt --val-list data/subset/ucf_crime_test_subset.txt --data-root data/raw/ucf_crime
bash scripts/run.sh infer
```

## Supervised-training monitoring

Supervised training shows a live batch progress bar with loss, accuracy, and
learning rate. After every epoch it evaluates the validation list, writes a
one-vs-rest multiclass ROC curve to `results/roc_curve.png`, and writes the
latest metrics (including macro AUC-ROC) to `results/metrics.json`.

Use `--patience` to enable validation-loss early stopping and `--epochs` to
set an upper training limit. For example:

```powershell
.\scripts\run.ps1 supervised -ExtraArgs @(
  "--train-list", "data/subset/ucf_crime_train_subset.txt",
  "--val-list", "data/subset/ucf_crime_test_subset.txt",
  "--data-root", "data/raw/ucf_crime",
  "--epochs", "30",
  "--patience", "5",
  "--batch-size", "8",
  "--num-workers", "4"
)
```

The best validation-loss checkpoint is saved as `checkpoints/sup_best.pth`.

Use `weakly` instead of `supervised` for the CLIP model, and `evaluate` with
`--data-list` and `--data-root` to calculate metrics. The current demo selects
the first video from `data/subset/ucf_crime_test_subset.txt`.

## Conflict handling

The setup scripts upgrade packaging tools first and use pip's
`--upgrade-strategy only-if-needed`, preserving compatible packages. If you
change PyTorch backends, delete only `.venv` and rerun setup; do not mix CUDA,
ROCm, and CPU wheels in one environment. Run `python -m pip check` after
adding project-specific packages.


# Video Anomaly Detection Project — Complete Handoff

> **Purpose:** This is a factual technical handoff for collaborators, report writers, or a new AI assistant. It records the project state, how it was run, observed results, known limitations, and what must be done before making research claims. It is safe to paste this whole file into a fresh AI chat.

## 1. Project at a glance

This is a Python/PyTorch course project for recognising surveillance-video event categories using the **UCF-Crime** dataset. The primary implemented and tested path is a **supervised 3D video classifier**:

- Input: RGB video clips with shape `(batch, 3, 16, 112, 112)`.
- Backbone: pretrained torchvision **R3D-18**, initialized with Kinetics-400 weights.
- Output: one of 14 classes (13 anomalous event classes plus `NormalVideos`).
- Loss: multiclass cross-entropy.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Hardware target: Google Colab **Tesla T4 CUDA GPU**.
- Training amenities: CUDA mixed precision, progress bar, validation, early stopping, best-checkpoint saving, per-class one-vs-rest ROC curves, macro AUC-ROC.

The repository also contains building blocks intended for a broader/hybrid anomaly-detection system: CLIP temporal encoding, weak supervision, motion filtering, model routing, SORT-related dependencies, and 3D Grad-CAM explanations. However, only the supervised R3D-18 route has been verified end-to-end in the current Colab work. Do **not** claim that a full hybrid model was trained and evaluated unless that integration is completed and measured.

## 2. Important terminology and task definition

The current supervised script is primarily **14-class event recognition**, not a fully developed temporal anomaly-localization system.

Classes discovered from `Train/` and `Test/`:

1. `Abuse`
2. `Arrest`
3. `Arson`
4. `Assault`
5. `Burglary`
6. `Explosion`
7. `Fighting`
8. `NormalVideos`
9. `RoadAccidents`
10. `Robbery`
11. `Shooting`
12. `Shoplifting`
13. `Stealing`
14. `Vandalism`

The classifier predicts a class for each 16-frame clip. It does not currently use temporal ground-truth anomaly start/end annotations, and it does not produce a rigorously evaluated binary anomaly score over a complete video.

## 3. Project tree and responsibilities

```text
Course Project/
├── src/
│   ├── datasets.py              # Frame/video dataset and clip sampling
│   ├── supervised.py            # R3D-18 supervised classifier
│   ├── weakly_supervised.py     # Frozen OpenAI CLIP + temporal transformer components
│   ├── routing.py               # Proposed supervised/weak score router
│   ├── motion_filter.py         # Farneback optical-flow motion gate
│   ├── device.py                # CUDA-only Colab device selection
│   └── explain/gradcam.py       # 3D Grad-CAM implementation
├── scripts/
│   ├── setup_environment.py     # Build train/test subset lists; optionally downloads Kaggle data
│   ├── setup_colab.sh           # Installs non-PyTorch packages and verifies CUDA/R3D-18
│   ├── verify_device.py         # CUDA tensor and R3D-18 forward-pass check
│   ├── train_supervised.py      # Main verified training entry point
│   ├── train_weakly.py          # Weak training entry point (not validated end-to-end)
│   ├── evaluate.py              # Hybrid-oriented evaluation draft (not validated end-to-end)
│   └── demo.py                  # Supervised prediction + Grad-CAM demo
├── notebooks/
│   ├── check_colab_project_files.ipynb
│   ├── download_ucf_crime_from_kaggle.ipynb
│   └── train_tesla_t4_colab.ipynb
├── requirements.txt
├── requirements-colab.txt
├── checkpoints/                 # Created during training; best model is sup_best.pth
├── results/                     # ROC curve, JSON metrics, demo/evaluation outputs
└── data/subset/                 # Generated small text lists, not the source data itself
```

## 4. Dataset source and observed layout

### Kaggle source used in Colab

```python
import kagglehub
dataset_path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")
```

The confirmed Colab cache root was:

```text
/root/.cache/kagglehub/datasets/odins0n/ucf-crime-dataset/versions/1
```

The dataset-root inspection found a usable pair of folders:

```text
<dataset_path>/Train/
<dataset_path>/Test/
```

The data should be read from this local Colab cache, **not copied to mounted Google Drive**, because local cache access is much faster for massive sets of PNG images. The Kaggle cache disappears if the Colab runtime is reset, so the Kaggle download cell may need to be rerun on a new runtime.

### Expected folder semantics

The loader supports either videos (`.avi`, `.mp4`, `.mov`, `.mkv`, etc.) or a directory whose immediate files are image frames (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`). A directory containing frames is treated as **one video sequence**.

The initially observed user layout was effectively:

```text
Train/
  Abuse/         # PNG frames directly here
  Arrest/        # PNG frames directly here
  ...
Test/
  Abuse/         # PNG frames directly here
  Arrest/        # PNG frames directly here
  ...
```

That layout can make each entire class folder appear to the loader as only one long sequence. It is much less suitable than a per-video directory layout such as:

```text
Train/
  Abuse/
    Abuse001/
      frame_0001.png
      ...
    Abuse002/
      frame_0001.png
      ...
```

Before reporting final model performance, collaborators must count the number of independent videos/sequences per class and confirm there is no accidental concatenation of many videos into one class-level frame directory.

## 5. Data-list generation and sampling

`scripts/setup_environment.py` creates:

```text
data/subset/ucf_crime_train_subset.txt
data/subset/ucf_crime_test_subset.txt
```

For an existing Kaggle cache root, the intended command is:

```bash
python scripts/setup_environment.py --data-root "<dataset_path>"
```

The list generator:

- discovers the class folders under `Train/`;
- scans for video files and directories containing image files;
- uses up to 30 sequences per training class;
- uses up to 200 test sequences overall;
- uses deterministic Python random seed `42` before shuffling.

The dataset class turns each sequence into sliding 16-frame clips. Repository defaults are:

```text
train clip length = 16 frames
train stride      = 8 frames
test clip length  = 16 frames
test stride       = 16 frames
```

Stride is crucial: stride 8 creates heavily overlapping clips. Increasing it to 16 approximately halves the clip count; 32 approximately quarters it. The current local repository defaults remain 8 (training) and 16 (validation), but a Colab session may have been manually changed to 16 and 32 while testing speed. Always print/check the active script before reproducing an experiment.

## 6. Main model and training procedure

### Supervised R3D-18 model

`src/supervised.py` builds `torchvision.models.video.r3d_18` with `R3D_18_Weights.KINETICS400_V1` by default. The original classification layer is replaced by:

```python
nn.Identity()                  # remove R3D's original head
nn.Linear(512, num_classes)    # 14-class project head
```

R3D-18 is a 3D convolutional neural network. Unlike an image-by-image 2D network, its convolutions operate over **time, height, and width**, so it can model short motion patterns across the 16 input frames.

### Preprocessing

Each frame is read with OpenCV, converted from BGR to RGB, converted to float `[0, 1]`, then each clip is resized frame-by-frame to `112 × 112`. Normalization is:

```text
mean = [0.45, 0.45, 0.45]
std  = [0.225, 0.225, 0.225]
```

### Hardware path

`src/device.py` requires CUDA for train/demo/evaluation execution. In Colab the confirmed environment reported:

```text
GPU: Tesla T4
PyTorch: 2.11.0+cu130
CUDA: 13.0
Python: 3.13.16
```

The following test has been successfully reached in the project workflow:

- CUDA tensor operation succeeds.
- R3D-18 forward pass on CUDA is the intended verification.

On the user's Windows AMD RX 7600, `torch-directml` successfully ran simple tensors but did **not** support the Conv3D operations required by R3D-18. Therefore the project was moved to CUDA/Tesla-T4 Colab for the 3D model.

### Optimisation and monitoring

The supervised script uses:

- `torch.autocast(... dtype=torch.float16)` CUDA mixed precision;
- `torch.amp.GradScaler("cuda")`;
- DataLoader pinned memory;
- persistent workers if workers are nonzero;
- a `tqdm` batch progress bar showing running loss, training accuracy, and learning rate;
- validation after every epoch;
- early stopping based on **validation loss**;
- `checkpoints/sup_best.pth` saved when validation loss improves;
- `results/metrics.json` overwritten with the latest epoch metrics;
- `results/roc_curve.png` overwritten each epoch.
- `results/confusion_matrix.png` overwritten each epoch with a labeled, row-normalized heatmap;
- `results/training_curves.png` overwritten each epoch with train/validation loss, train/validation accuracy, and validation macro-AUC curves;
- `results/training_history.json` contains all completed epoch metrics, including the classification report and raw confusion matrix.

The current Colab session warned that its recommended maximum worker count is 2. Therefore use:

```text
--num-workers 2
```

The T4 reached roughly 14.3 GB VRAM use at batch size 128, which is close to its practical memory capacity. A safer high batch size is **96**. Note that memory use is not the same thing as GPU compute utilization. PNG decoding and CPU-side resizing can still bottleneck the pipeline.

## 7. Correct Colab workflow

1. Select **Runtime → Change runtime type → T4 GPU**.
2. Mount Google Drive and enter the project:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Course Project
```

3. Download Kaggle data locally:

```python
import kagglehub
dataset_path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")
print(dataset_path)
```

4. Install dependencies and run device verification:

```bash
!bash scripts/setup_colab.sh
!python scripts/verify_device.py
```

5. Create the train/test list files. Do not write `{dataset_path}` literally inside a shell command unless using an IPython `!` variable expansion. The robust approach is Python `subprocess`:

```python
import subprocess

subprocess.run(
    ["python", "scripts/setup_environment.py", "--data-root", str(dataset_path)],
    check=True,
)
```

6. Train (example with safer high batch size):

```python
subprocess.run(
    [
        "python", "scripts/train_supervised.py",
        "--data-root", str(dataset_path),
        "--train-list", "data/subset/ucf_crime_train_subset.txt",
        "--val-list", "data/subset/ucf_crime_test_subset.txt",
        "--epochs", "10",
        "--patience", "3",
        "--batch-size", "96",
        "--num-workers", "2",
    ],
    check=True,
)
```

The previous command style below is also valid in an IPython/Colab code cell because IPython expands `{dataset_path}`:

```python
!python scripts/train_supervised.py --data-root "{dataset_path}" --train-list data/subset/ucf_crime_train_subset.txt --val-list data/subset/ucf_crime_test_subset.txt --epochs 10 --patience 3 --batch-size 96 --num-workers 2
```

## 8. Colab issues already encountered and fixes

| Issue | Cause | Fix |
|---|---|---|
| `Expected Train and Test folders under --data-root: .../{dataset_path}` | Literal braces were passed through `subprocess`/shell incorrectly. | Pass `str(dataset_path)` in a Python argument list. |
| `ImportError: cannot import name UCFClipDataset from datasets` | Colab has Hugging Face package `datasets`, which shadowed `src/datasets.py`. | Training, demo, weak training, and evaluation scripts use `sys.path.insert(0, 'src')`, not `append`. |
| Tensor normalization dimension mismatch | Original frame transform used a 2D normalization form on a `(C,T,H,W)` clip. | Current transform uses channel mean/std shaped `(3,1,1,1)`. |
| AMD DirectML failure with R3D-18 | DirectML did not support Conv3D for this setup. | Use CUDA/T4 Colab for the 3D model. |
| Very slow training | Stride 8 creates many overlapping clips from very long PNG sequences. | Use a larger stride only after documenting it; avoid silently reporting a capped sample run as a full experiment. |
| Colab warns about 4 DataLoader workers | Runtime CPU allocation recommends at most 2. | Use `--num-workers 2`. |

## 9. Observed experiment and interpretation

### Fast diagnostic run (not valid final research result)

A temporary per-video cap was applied for a quick functional check. It created only:

```text
Training clips:   56
Validation clips: 28
```

This is roughly 4 train clips and 2 validation clips per class. Results:

| Epoch | Train accuracy | Validation loss | Validation accuracy | Macro one-vs-rest AUC |
|---:|---:|---:|---:|---:|
| 1 | 10.7% | 2.9708 | 10.7% | 0.578 |
| 2 | 98.2% | 2.8671 | 7.1% | 0.555 |
| 3 | 100.0% | 2.7575 | 7.1% | 0.587 |
| 4 | 100.0% | 2.8042 | 7.1% | 0.574 |
| 5 | 100.0% | 2.8889 | 7.1% | 0.563 |
| 6 | 100.0% | 2.9426 | 14.3% | 0.540 |
| 7 | 100.0% | 2.9922 | 14.3% | 0.539 |
| 8 | 100.0% | 3.0332 | 7.1% | 0.518 |

Early stopping triggered after epoch 8 because there were five epochs without validation-loss improvement.

### Correct interpretation

This demonstrates that the pipeline runs: CUDA, data loading, model training, progress display, validation, checkpointing, early stopping, ROC creation, and metrics output all functioned.

It does **not** demonstrate good model performance. With 14 classes, random accuracy is approximately `1/14 = 7.14%`. The observed 100% training accuracy with 7.1–14.3% validation accuracy is strong overfitting. Macro AUC around 0.5–0.59 is close to random and unstable with only 28 validation clips.

Do not present these values as final performance in a paper. Label them as a **sanity-check / pilot experiment** if they must be mentioned.

## 10. Hybrid components: what exists versus what is proven

### Existing components

1. **Supervised stream:** R3D-18 produces 14-class event logits.
2. **Weak/vision-language stream:** `CLIPTSAEncoder` loads frozen OpenAI CLIP, encodes frames, and feeds a Transformer encoder over time.
3. **MIL ranking-loss function:** `mil_ranking_loss` is intended to compare normal vs anomalous bags.
4. **Motion filter:** Farneback optical flow gives a basic activity score/gate.
5. **Router:** combines supervised confidence and a weak anomaly score based on thresholds.
6. **Explanation:** Grad-CAM3D creates per-frame spatial saliency maps from R3D-18 layer `layer4[-1].conv2`.
7. **Tracking dependencies:** `filterpy` and `lapx` support a potential SORT implementation. The unrelated PyPI package named `sort` is intentionally not used.

### Caveats

- `train_weakly.py`, `evaluate.py`, and the hybrid router have not been demonstrated as a complete reproducible experiment in this work session.
- `evaluate.py` assumes weak-model checkpoints (`weak_enc_best.pth`, `weak_scorer_best.pth`) that may not exist.
- The weak encoder uses a CLIP model/checkpoint convention that needs explicit compatibility testing.
- The routing thresholds (`sup_thresh=0.85`, `weak_thresh=0.5`) are hand-set, not calibrated on a validation set.
- Motion filtering and SORT tracking are utilities, not currently connected to the verified supervised training pipeline.

Therefore, the defensible current description is: **a supervised 3D video-classification baseline with prototype hybrid components present in the codebase**. It is not yet defensible to describe the evaluation as a validated end-to-end hybrid anomaly-detection system.

## 11. What an IEEE/course report can accurately claim now

### Safe claims

- The project implements a pretrained R3D-18 3D CNN baseline for 14-class surveillance-event recognition on UCF-Crime-style data.
- The implementation processes 16-frame RGB clips resized to 112×112.
- Training uses transfer learning, AdamW, cross-entropy, mixed precision, early stopping, and checkpoint selection by validation loss.
- The project produces per-class one-vs-rest ROC curves and macro AUC-ROC in addition to accuracy/loss.
- A Tesla T4 Colab environment was configured and CUDA execution was verified.
- The codebase includes exploratory weak-supervision, motion, routing, tracking, and Grad-CAM modules.
- The pilot experiment exposed significant overfitting due to insufficient independent training/validation sequences; this motivated the documented need for a correct video-level split and a larger experiment.

### Claims to avoid until validated

- “State-of-the-art” or “highly accurate” anomaly detection.
- A full hybrid model outperforms an R3D baseline.
- Grad-CAM localizes anomalies accurately.
- SORT tracking improves anomaly recognition.
- The 56/28 clip pilot scores represent genuine generalization.
- Comparison with other papers without matching protocol, split, labels, and evaluation metric.

## 12. Recommended report / IEEE-paper structure

1. **Abstract:** surveillance anomaly/event recognition motivation, R3D-18 baseline, UCF-Crime, CUDA implementation, and honest pilot status.
2. **Introduction:** public-safety relevance, challenges of temporal context, surveillance quality, class imbalance, long videos.
3. **Related Work:** 3D CNNs, UCF-Crime, weakly supervised anomaly detection, CLIP/video-language representations, explainability. Cite primary papers; do not invent results.
4. **Methodology:** dataset preparation, clip sampling, R3D-18 transfer learning, preprocessing, optimization, early stopping, ROC/AUC, Grad-CAM.
5. **Hybrid Design Proposal:** clearly mark CLIP temporal encoder, motion gate, routing, and tracking as proposed/prototype unless they have completed experiments.
6. **Experimental Setup:** Tesla T4, PyTorch/CUDA versions, seed(s), exact lists, split rules, resolution, clip length/stride, batch size, epochs, patience.
7. **Results:** only use final experiments with enough independent videos. Include train/validation curves, class support, confusion matrix, macro/per-class AUC, and confidence intervals where possible.
8. **Error Analysis:** class confusion, false positives/negatives, effect of frame rate/stride, cases where Grad-CAM highlights irrelevant background.
9. **Limitations and Ethics:** surveillance privacy, bias, real-world false alarms, dataset license/provenance, pilot sample limitations.
10. **Conclusion/Future Work:** correct video-level split, complete the hybrid evaluation, temporal localization, calibration, external validation.

## 13. Required work before a credible final experiment

1. **Audit the source structure.** Count independent videos under every class in Train/Test. Ensure each source video has its own directory/video file.
2. **Prevent leakage.** Clips from one original video must never appear in both train and validation/test sets.
3. **Use enough independent sequences.** Do not report a result based on 56 training and 28 validation clips.
4. **Fix the experiment protocol before tuning.** Decide whether the task is 14-class event recognition, binary anomaly detection, or anomaly temporal localization.
5. **Run repeatable experiments.** Set random seeds; record data lists, hyperparameters, Git/code snapshot, versions, and hardware.
6. **Report class support.** Per-class ROC/AUC is not reliable when a class has very few validation videos.
7. **Compare baselines fairly.** At least compare pretrained R3D-18 with a simple 2D frame-level baseline or an ablation without pretraining/motion/CLIP if hybrid work is completed.
8. **Keep a final artifact set.** Save the best checkpoint, complete config, ROC plot, confusion matrix, metrics JSON, logs, and exact train/validation list files for every reported run.

## 14. Key files/outputs to preserve for collaborators

After a valid run, preserve and share:

```text
checkpoints/sup_best.pth
results/metrics.json
results/roc_curve.png
data/subset/ucf_crime_train_subset.txt
data/subset/ucf_crime_test_subset.txt
scripts/train_supervised.py
src/datasets.py
src/supervised.py
requirements.txt
requirements-colab.txt
PROJECT_HANDOFF.md
```

Do **not** upload `.venv/` or `.venv-directml/` to Colab/Drive; they are machine-specific and unnecessary. Do not treat the large Kaggle cache as persistent Drive data without intentionally exporting it.

## 15. Immediate status at handoff

- Code: primary supervised path runs on CUDA/Tesla-T4 Colab.
- Dataset: Kaggle download found a root containing Train/Test.
- Training: progress bar, early stopping, checkpoint and ROC outputs implemented.
- Device: Tesla T4 was detected; high batch sizes consume substantial VRAM.
- Correctness concern: dataset granularity/sampling must be audited before final metrics are trusted.
- Current local code: full sliding-clip default has been restored; no per-video clip cap is committed. Colab runtime files can diverge from local Drive files because edits were applied interactively. Always inspect/copy the active `scripts/train_supervised.py` and `src/datasets.py` before a reported experiment.


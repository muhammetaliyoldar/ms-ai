# MS-AI Runbook

> Complete reference for reproducing the MS-AI pipeline: MS lesion segmentation → lesion feature extraction → EDSS regression.

---

## 1. Project Overview

MS-AI is a **two-stage medical imaging pipeline** for the MSLesSeg Dataset:

1. **Stage 1 — Segmentation**: A 3D UNet (MONAI) segments MS lesions from multi-channel MRI (FLAIR, T1, T2).
2. **Stage 2 — EDSS Regression**: Lesion volumetric features are extracted from segmentation masks and merged with clinical data (age, sex, MS type) to predict EDSS disability scores using a Random Forest regressor.

The pipeline supports two modes:
- **GT mode** (`--use gt`): Uses ground-truth masks — no segmentation training required.
- **Pred mode** (`--use pred`): Uses predicted masks from a trained segmentation model.

---

## 2. Environment Setup

### Requirements
| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 |
| CUDA | 12.1 |
| GPU | NVIDIA RTX 2070 (8 GB) — primary dev machine |

### Installation

```bash
# Clone / enter repo
cd ~/projects/ms-ai

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### Verify installation
```bash
python -m ms_ai.cli --help
python -m ms_ai.cli check-data
```

---

## 3. Dataset Structure

**Location (default):**
```
/mnt/ortak/MS-AI/döküman/27919209/MSLesSeg Dataset/MSLesSeg Dataset/
```

Override with `--data-root "/path/to/dataset"` on any CLI command.

### Layout

```
MSLesSeg Dataset/
├── train/                          # 93 cases, multi-timepoint
│   ├── P1/
│   │   ├── T1/                     # Timepoint 1
│   │   │   ├── P1_T1_FLAIR.nii.gz
│   │   │   ├── P1_T1_T1.nii.gz
│   │   │   ├── P1_T1_T2.nii.gz
│   │   │   └── P1_T1_MASK.nii.gz
│   │   ├── T2/                     # Timepoint 2
│   │   └── T3/                     # Timepoint 3
│   ├── P2/ ...
│   └── P53/
├── test/                           # 22 cases, single-timepoint (flat)
│   ├── P54/
│   │   ├── P54_FLAIR.nii.gz
│   │   ├── P54_T1.nii.gz
│   │   ├── P54_T2.nii.gz
│   │   └── P54_MASK.nii.gz
│   └── P75/
└── info_dataset/
    └── clinical_data.csv           # Patient;Timepoint;Age;Sex;MS Type;EDSS;...
```

- **Train (Layout A):** `train/P*/T*/` — patient folders contain timepoint subdirectories.
- **Test (Layout B):** `test/P*/` — files directly in patient folder; treated as timepoint `T1`.

---

## 4. Clean State

After `rm -rf runs/seg` and removing pred artifacts, the repo is "training-ready":

```
artifacts/
├── lesion_features_gt_train.csv        # ✓ GT pipeline output
├── lesion_features_gt_test.csv         # ✓ GT pipeline output
├── features_with_edss_gt_train.csv     # ✓ GT pipeline output
├── edss_model_gt.pkl                   # ✓ GT EDSS model
└── edss_model_gt_meta.json             # ✓ GT model metadata
```

The following are **absent until segmentation training + inference**:
- `runs/seg/best_model.pth` — created by `train-seg`
- `runs/seg/preds/` — created by `infer-seg`
- `artifacts/lesion_features_pred_*` — created by `extract-features --use pred`
- `artifacts/features_with_edss_pred_*` — created by `merge-edss --use pred`
- `artifacts/edss_model_pred.*` — created by `train-edss --use pred`

---

## 5. Pipeline Commands

### 5.1 GT Pipeline (no segmentation training needed)

```bash
# 1. Verify dataset structure
python -m ms_ai.cli check-data

# 2. Extract lesion features from ground-truth masks
python -m ms_ai.cli extract-features --use gt --split train
python -m ms_ai.cli extract-features --use gt --split test

# 3. Merge features with clinical EDSS data
python -m ms_ai.cli merge-edss --use gt --split train

# 4. Train EDSS regressor (5-fold patient-level CV + final model)
python -m ms_ai.cli train-edss --use gt --split train
```

**Outputs:**
| File | Description |
|------|-------------|
| `artifacts/lesion_features_gt_train.csv` | 93 rows of lesion metrics |
| `artifacts/lesion_features_gt_test.csv` | 22 rows of lesion metrics |
| `artifacts/features_with_edss_gt_train.csv` | Features + clinical data merged |
| `artifacts/edss_model_gt.pkl` | Trained Random Forest model |
| `artifacts/edss_model_gt_meta.json` | Feature list, timestamps, config |

### 5.2 Segmentation Pipeline (training-ready)

```bash
# 1. Train 3D UNet segmentation model
#    NOTE: Creates runs/seg/best_model.pth (~18MB)
#    ~5 min/epoch on RTX 2070 (8GB), 50 epochs ≈ 4 hours
python -m ms_ai.cli train-seg --epochs 50

# 2. Run inference on test split
python -m ms_ai.cli infer-seg --split test --checkpoint runs/seg/best_model.pth

# 3. (Optional) Run inference on train split — needed for pred EDSS pipeline
python -m ms_ai.cli infer-seg --split train --checkpoint runs/seg/best_model.pth
```

**Outputs:**
| File | Description |
|------|-------------|
| `runs/seg/best_model.pth` | Best UNet checkpoint (by val Dice) |
| `runs/seg/train_log.jsonl` | Per-epoch log: `{epoch, loss, val_dice}` |
| `runs/seg/preds/<split>/P*/T*/*_MASK_PRED.nii.gz` | Predicted binary masks |

### 5.3 Pred EDSS Pipeline (requires inference outputs)

```bash
# 1. Extract features from predicted masks
python -m ms_ai.cli extract-features --use pred --split test
python -m ms_ai.cli extract-features --use pred --split train

# 2. Merge pred features with clinical EDSS
python -m ms_ai.cli merge-edss --use pred --split train

# 3. Train EDSS model on predicted features
python -m ms_ai.cli train-edss --use pred --split train
```

**Outputs:**
| File | Description |
|------|-------------|
| `artifacts/lesion_features_pred_*.csv` | Lesion metrics from predicted masks |
| `artifacts/features_with_edss_pred_train.csv` | Pred features + clinical data |
| `artifacts/edss_model_pred.pkl` | EDSS model trained on predicted features |
| `artifacts/edss_model_pred_meta.json` | Model metadata |

---

## 6. Output Locations Summary

```
ms-ai/
├── artifacts/                      # CSV files + EDSS models
│   ├── lesion_features_*.csv       # Per-case volumetric features
│   ├── features_with_edss_*.csv    # Features merged with clinical data
│   ├── edss_model_*.pkl            # Trained RF models (pickle)
│   └── edss_model_*_meta.json      # Model metadata (JSON)
├── runs/seg/                       # Segmentation outputs
│   ├── best_model.pth              # UNet checkpoint
│   ├── train_log.jsonl             # Training log
│   └── preds/                      # Predicted masks
│       ├── train/P*/T*/*.nii.gz
│       └── test/P*/T*/*.nii.gz
└── logs/
    └── daily_log.csv               # Run audit trail (see §8)
```

---

## 7. Moving to Second PC (RTX 4060, Windows)

### Hardware
- **GPU**: NVIDIA RTX 4060 (8 GB VRAM) — pipeline is 8GB-safe by design.
- **Note**: GPU is RTX 4060, not 4070.

### Setup

```powershell
# 1. Clone repo
git clone <repo-url> C:\projects\ms-ai
cd C:\projects\ms-ai

# 2. Create venv
python -m venv .venv
.venv\Scripts\activate

# 3. Install PyTorch (CUDA 12.1 — same as dev machine)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt
```

### Running on Windows

Most commands are identical. Key differences:

1. **Data path quoting** — Windows paths with spaces/special chars need quoting:
   ```powershell
   python -m ms_ai.cli check-data --data-root "D:\data\MSLesSeg Dataset\MSLesSeg Dataset"
   ```

2. **venv activation** uses `Scripts\activate` instead of `source .venv/bin/activate`.

3. **CUDA should auto-detect**. Verify with:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

### Recommended workflow on second PC

```powershell
# Verify setup
python -m ms_ai.cli check-data --data-root "D:\path\to\dataset"

# Train segmentation (adjust --data-root as needed)
python -m ms_ai.cli train-seg --epochs 50 --data-root "D:\path\to\dataset"

# Inference + pred pipeline
python -m ms_ai.cli infer-seg --split test --checkpoint runs\seg\best_model.pth --data-root "D:\path\to\dataset"
python -m ms_ai.cli extract-features --use pred --split test --data-root "D:\path\to\dataset"
# ... (follow §5.3 above)
```

---

## 8. Daily Log (`logs/daily_log.csv`)

Every CLI command appends a row to `logs/daily_log.csv` with these fields:

| Field | Description |
|-------|-------------|
| `timestamp_utc` | ISO 8601 UTC timestamp |
| `timestamp_tr` | ISO 8601 Europe/Istanbul (TR) timestamp |
| `command` | CLI command name (e.g., `extract-features`) |
| `use` | `gt` or `pred` (if applicable) |
| `split` | `train`, `test`, or `all` |
| `rows_written` | Number of output rows/files |
| `output_path` | Path to output file |
| `status` | `OK` or `FAIL` |

The file is created automatically on first run. Example:
```csv
timestamp_utc,timestamp_tr,command,use,split,rows_written,output_path,status
2026-02-26T08:30:00+00:00,2026-02-26T11:30:00+03:00,check-data,,,,,OK
2026-02-26T08:31:00+00:00,2026-02-26T11:31:00+03:00,extract-features,gt,train,93,artifacts/lesion_features_gt_train.csv,OK
```

---

## 9. Single-Case Visualization

Use `scripts/viz_one_case.py` to render a single FLAIR slice with GT mask (red) and optional predicted mask (green) overlays.

### Basic usage

```bash
# Train case – default mid-slice
python scripts/viz_one_case.py --split train --patient P1 --timepoint T1 --out P1_T1_overlay.png

# With predicted mask overlay
python scripts/viz_one_case.py --split train --patient P1 --timepoint T1 --use-pred --out P1_T1_overlay.png

# Auto-select slice with maximum GT lesion area
python scripts/viz_one_case.py --split train --patient P1 --auto-slice --out P1_T1_auto.png

# Explicit z-slice
python scripts/viz_one_case.py --split train --patient P1 --slice 60 --out P1_T1_s60.png

# Test case (no timepoint subfolder)
python scripts/viz_one_case.py --split test --patient P54 --use-pred --out P54_overlay.png

# Custom data root
python scripts/viz_one_case.py --data-root "D:\data\MSLesSeg Dataset\MSLesSeg Dataset" --split train --patient P1 --out P1.png
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | config.yaml / built-in | Path to MSLesSeg dataset root |
| `--split` | `train` | `train` or `test` |
| `--patient` | *(required)* | Patient ID (e.g., `P1`) |
| `--timepoint` | `T1` | Timepoint ID (e.g., `T1`) |
| `--use-pred` | off | Also overlay predicted mask (green) |
| `--preds-dir` | `runs/seg/preds` | Root dir for predicted masks |
| `--out` | `overlay.png` | Output PNG path |
| `--slice` | mid-slice | Explicit z-slice index |
| `--auto-slice` | off | Choose slice with max GT lesion pixels |

---

## 10. Architecture Reference

```
ms_ai/
├── cli.py                     # Click CLI — all 6 commands
├── logging_utils.py           # Daily CSV logger
├── data/                      # (reserved for data utilities)
├── features/
│   └── lesion_features.py     # 3D connected-component lesion metrics
├── edss/
│   ├── merge.py               # Clinical CSV loader + EDSS merge
│   └── train.py               # RF regressor + GroupKFold CV
└── seg/
    ├── train.py               # MONAI 3D UNet training pipeline
    └── infer.py               # Sliding-window inference
```

### Model specifications

**Segmentation (3D UNet)**
- Input: 3 channels (FLAIR, T1, T2)
- Output: 2 channels (background, lesion)
- Architecture: `channels=(16,32,64,128,256)`, `strides=(2,2,2,2)`, `num_res_units=2`
- Loss: DiceCE; Optimizer: AdamW lr=1e-4; AMP enabled
- Training: 96³ patches, batch_size=1, sliding-window validation

**EDSS Regression (Random Forest)**
- Features: lesion_volume_mm3, lesion_count, mean/max_lesion_size_mm3, age, sex, ms_type (one-hot)
- CV: GroupKFold(5) by patient
- Estimators: 600 trees, min_samples_leaf=2

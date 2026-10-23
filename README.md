# MS-AI

Two-stage medical imaging pipeline for the **MSLesSeg Dataset** (Multiple Sclerosis Lesion Segmentation):

1. **Segmentation**: a 3D UNet (MONAI / PyTorch) segments MS lesions from multi-channel MRI (FLAIR, T1, T2).
2. **EDSS regression**: lesion volumetric features plus clinical data (age, sex, MS type) are used to predict the EDSS disability score with a Random Forest.

> See **[RUNBOOK.md](RUNBOOK.md)** for the full reproduction guide, expected outputs, the daily log format and the notes for moving to a second (Windows) machine.

## Features

- **Two modes** for the EDSS stage: `--use gt` works on ground-truth masks (no segmentation training needed), `--use pred` works on masks predicted by the trained UNet.
- **Both dataset layouts** are read: multi-timepoint train cases (`train/P*/T*/`) and flat single-timepoint test cases (`test/P*/`, treated as `T1`).
- **Lesion features** from 3D connected components: total lesion volume (mm³), lesion count, mean and max lesion size.
- **Patient-level splits**: the segmentation validation split is made per patient, and EDSS cross-validation uses `GroupKFold(5)` grouped by patient, so timepoints of one patient never leak between folds.
- **8 GB-GPU friendly segmentation**: 96³ patches, batch size 1, mixed precision on CUDA, sliding-window validation and inference.
- **Run audit log**: every CLI command appends a row (UTC and Europe/Istanbul timestamps, command, split, output, status) to `logs/daily_log.csv`.

## Tech stack

Python 3.10, PyTorch (CUDA 12.1 wheels), MONAI, nibabel, NumPy, pandas, SciPy, scikit-learn, Click.

## Project structure

```
ms_ai/
├── cli.py                     # Click CLI, all 6 commands
├── logging_utils.py           # daily CSV run logger
├── data/                      # reserved for data utilities (empty)
├── features/
│   └── lesion_features.py     # 3D connected-component lesion metrics
├── edss/
│   ├── merge.py               # clinical CSV loader + EDSS merge
│   └── train.py               # Random Forest regressor + GroupKFold CV
└── seg/
    ├── train.py               # MONAI 3D UNet training
    └── infer.py               # sliding-window inference
scripts/smoke_test.sh          # checks that the CLI loads and the dataset is valid
config.yaml                    # dataset path and processing notes (not read by the code yet)
RUNBOOK.md                     # full reproduction guide
```

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate

# PyTorch is not in requirements.txt; install the CUDA 12.1 build first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The dataset is **not** included in the repository. Every command that reads it takes `--data-root`; the default is the path on the original development machine:

```
/mnt/ortak/MS-AI/döküman/27919209/MSLesSeg Dataset/MSLesSeg Dataset
```

## Dataset layout

```
MSLesSeg Dataset/
├── train/P*/T*/P*_T*_{FLAIR,T1,T2,MASK}.nii.gz   # multi-timepoint
├── test/P*/P*_{FLAIR,T1,T2,MASK}.nii.gz          # single timepoint (treated as T1)
└── info_dataset/clinical_data.csv                # Patient;Timepoint;Age;Sex;MS Type;EDSS;...
```

## Usage

```bash
# Verify the dataset
python -m ms_ai.cli check-data --data-root "/path/to/MSLesSeg Dataset"

# GT pipeline (no training required)
python -m ms_ai.cli extract-features --use gt --split train
python -m ms_ai.cli merge-edss --use gt --split train
python -m ms_ai.cli train-edss --use gt --split train

# Segmentation training (creates runs/seg/best_model.pth)
python -m ms_ai.cli train-seg --epochs 50

# Inference, then the pred pipeline
python -m ms_ai.cli infer-seg --split test --checkpoint runs/seg/best_model.pth
python -m ms_ai.cli extract-features --use pred --split test
```

Quick check that everything is wired up: `bash scripts/smoke_test.sh`.

## CLI commands

| Command | Options |
|---------|---------|
| `check-data` | `--data-root`, `--max-print` (default 5) |
| `extract-features` | `--use {gt\|pred}`, `--split {train\|test\|all}`, `--data-root`, `--pred-root` (default `runs/seg/preds`) |
| `merge-edss` | `--use {gt\|pred}`, `--split {train\|test\|all}`, `--data-root` |
| `train-edss` | `--use {gt\|pred}`, `--split {train\|test\|all}` |
| `train-seg` | `--epochs` (default 50), `--val-split` (default 0.2), `--out-dir` (default `runs/seg`), `--data-root` |
| `infer-seg` | `--split {train\|test}`, `--checkpoint PATH` (required), `--out-dir` (default `runs/seg/preds`), `--data-root` |

## Outputs

All outputs are git-ignored (`artifacts/`, `runs/`, `logs/`, `*.csv`, `*.pkl`, `*.pth`).

| Path | Description |
|------|-------------|
| `artifacts/lesion_features_{gt,pred}_<split>.csv` | Per-case lesion metrics |
| `artifacts/features_with_edss_{gt,pred}_<split>.csv` | Features merged with clinical EDSS |
| `artifacts/edss_model_{gt,pred}.pkl` + `_meta.json` | Trained RF model and its metadata |
| `runs/seg/best_model.pth` | Best UNet checkpoint (by validation Dice) |
| `runs/seg/train_log.jsonl` | Per-epoch loss and validation Dice |
| `runs/seg/preds/<split>/P*/T*/*_MASK_PRED.nii.gz` | Predicted binary masks |
| `logs/daily_log.csv` | Run audit trail |

## Model details

- **3D UNet**: 3 input channels, 2 output channels, `channels=(16, 32, 64, 128, 256)`, `strides=(2, 2, 2, 2)`, `num_res_units=2`; DiceCE loss, AdamW (lr 1e-4).
- **EDSS Random Forest**: 600 trees, `min_samples_leaf=2`; features are lesion volume, count, mean/max lesion size, age, sex and one-hot MS type.

## Project status

A single commit on 26 February 2026 contains the full pipeline and the runbook. It was developed on Ubuntu 22.04 with an RTX 2070 (8 GB); the runbook also covers a Windows machine with an RTX 4060. There is no license file in the repository.

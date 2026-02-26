# MS-AI Project

Two-stage medical imaging pipeline for the **MSLesSeg Dataset** (Multiple Sclerosis Lesion Segmentation):

1. **Segmentation** — 3D UNet (MONAI/PyTorch) segments MS lesions from multi-channel MRI.
2. **EDSS Regression** — Lesion features + clinical data → EDSS disability score prediction.

> 📖 See **[RUNBOOK.md](RUNBOOK.md)** for full setup, reproduction steps, and migration guide.

## Quick Start

```bash
source .venv/bin/activate

# Verify dataset
python -m ms_ai.cli check-data

# GT pipeline (no training required)
python -m ms_ai.cli extract-features --use gt --split train
python -m ms_ai.cli merge-edss --use gt --split train
python -m ms_ai.cli train-edss --use gt --split train

# Segmentation training (creates runs/seg/best_model.pth)
python -m ms_ai.cli train-seg --epochs 50

# Inference → pred pipeline
python -m ms_ai.cli infer-seg --split test --checkpoint runs/seg/best_model.pth
python -m ms_ai.cli extract-features --use pred --split test
```

## Data Structure

- **Train (Layout A):** `train/P*/T*/` — multi-timepoint (e.g., `train/P1/T1`)
- **Test (Layout B):** `test/P*/` — single timepoint (e.g., `test/P54`, treated as `T1`)

## Project Structure

```
ms_ai/
├── cli.py                     # CLI commands (click)
├── logging_utils.py           # Daily run audit logger
├── features/
│   └── lesion_features.py     # 3D connected-component lesion metrics
├── edss/
│   ├── merge.py               # Clinical CSV loader + EDSS merge
│   └── train.py               # EDSS regression (RF + GroupKFold CV)
└── seg/
    ├── train.py               # MONAI 3D UNet training pipeline
    └── infer.py               # Sliding-window inference
```

## Outputs

| Path | Description |
|------|-------------|
| `artifacts/lesion_features_*.csv` | Per-case lesion metrics |
| `artifacts/features_with_edss_*.csv` | Features + clinical EDSS |
| `artifacts/edss_model_*.pkl` | Trained RF model |
| `runs/seg/best_model.pth` | Best segmentation checkpoint |
| `runs/seg/preds/<split>/P*/T*/*.nii.gz` | Predicted masks |
| `logs/daily_log.csv` | Run audit trail |

## CLI Commands

| Command | Description |
|---------|-------------|
| `check-data` | Verify dataset structure |
| `extract-features --use {gt\|pred} --split {train\|test\|all}` | Extract lesion features |
| `merge-edss --use {gt\|pred} --split {train\|test\|all}` | Merge with clinical EDSS |
| `train-edss --use {gt\|pred} --split {train\|test\|all}` | Train EDSS regressor |
| `train-seg --epochs N` | Train 3D UNet segmentation |
| `infer-seg --split {train\|test} --checkpoint PATH` | Run segmentation inference |

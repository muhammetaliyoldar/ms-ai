"""
MONAI-based 3D UNet segmentation training for MS lesions.

8GB-safe configuration:
  - batch_size=1
  - patch-based training (96³)
  - sliding-window validation
  - mixed-precision (AMP)
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)
from monai.utils import set_determinism

log = logging.getLogger("ms_ai.seg.train")


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_data_dicts(cases: list) -> List[dict]:
    """Convert scanner case tuples to MONAI-style data dicts."""
    dicts = []
    for (_split, patient, timepoint, flair, t1, t2, mask) in cases:
        dicts.append({
            "image": [str(flair), str(t1), str(t2)],
            "label": str(mask),
            "patient": patient,
            "timepoint": timepoint,
        })
    return dicts


def _patient_split(
    data_dicts: List[dict],
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Split data dicts by patient (no leakage)."""
    patients = sorted({d["patient"] for d in data_dicts})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])
    train_list = [d for d in data_dicts if d["patient"] not in val_patients]
    val_list = [d for d in data_dicts if d["patient"] in val_patients]
    return train_list, val_list


# ── Transforms ─────────────────────────────────────────────────────────────

def get_train_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(96, 96, 96),
            random_size=False,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


# ── Model ──────────────────────────────────────────────────────────────────

def build_model() -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


# ── Training loop ─────────────────────────────────────────────────────────

def run_training(
    cases: list,
    epochs: int = 50,
    val_split: float = 0.2,
    out_dir: str = "runs/seg",
):
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Prepare data
    data_dicts = _build_data_dicts(cases)
    train_dicts, val_dicts = _patient_split(data_dicts, val_frac=val_split)
    log.info("Train cases: %d, Val cases: %d", len(train_dicts), len(val_dicts))

    train_ds = CacheDataset(train_dicts, transform=get_train_transforms(), cache_rate=0.0, num_workers=2)
    val_ds = CacheDataset(val_dicts, transform=get_val_transforms(), cache_rate=0.0, num_workers=2)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=list_data_collate,
    )

    # Model, loss, optimizer
    model = build_model().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = -1.0
    log_path = out_path / "train_log.jsonl"
    log_file = open(log_path, "w")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size=(96, 96, 96), sw_batch_size=2,
                        predictor=model, overlap=0.25,
                    )
                val_preds = torch.argmax(val_outputs, dim=1, keepdim=True)
                dice_metric(y_pred=val_preds, y=val_labels)

            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()

        # ── Logging ──
        log_entry = {"epoch": epoch, "loss": round(avg_loss, 5), "val_dice": round(val_dice, 5)}
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        log.info("Epoch %d/%d  loss=%.4f  val_dice=%.4f", epoch, epochs, avg_loss, val_dice)

        # ── Checkpoint ──
        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = out_path / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            log.info("  ↑ New best dice=%.4f → saved %s", best_dice, ckpt_path)

    log_file.close()
    log.info("Training complete. Best val dice=%.4f", best_dice)
    return best_dice

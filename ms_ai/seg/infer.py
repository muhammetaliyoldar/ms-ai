"""
Sliding-window inference for MS lesion segmentation.

Loads a trained MONAI 3D UNet and produces binary mask predictions.
"""

import logging
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

from ms_ai.seg.train import build_model

log = logging.getLogger("ms_ai.seg.infer")


def get_infer_transforms() -> Compose:
    """Preprocessing transforms matching validation (no augmentation)."""
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="LAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])


def run_inference(
    cases: list,
    split: str,
    checkpoint: str,
    out_dir: str = "runs/seg/preds",
) -> int:
    """
    Run sliding-window inference on all cases and save predicted masks.

    Returns number of files saved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded checkpoint: %s", checkpoint)

    transforms = get_infer_transforms()
    saved = 0

    for (_split, patient, timepoint, flair, t1, t2, mask) in cases:
        data = {"image": [str(flair), str(t1), str(t2)]}
        data = transforms(data)

        # Reference NIfTI for affine / header
        ref_nii = nib.load(str(mask))

        img_tensor = data["image"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output = sliding_window_inference(
                img_tensor, roi_size=(96, 96, 96), sw_batch_size=2,
                predictor=model, overlap=0.25,
            )

        # argmax → binary mask (class 1 = lesion)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # Save
        out_base = Path(out_dir) / split / patient / timepoint
        out_base.mkdir(parents=True, exist_ok=True)
        out_file = out_base / f"{patient}_{timepoint}_MASK_PRED.nii.gz"

        pred_nii = nib.Nifti1Image(pred, affine=ref_nii.affine, header=ref_nii.header)
        pred_nii.header.set_data_dtype(np.uint8)
        nib.save(pred_nii, str(out_file))

        saved += 1
        log.info("  [%d] %s/%s → %s", saved, patient, timepoint, out_file.name)

    log.info("Inference complete. Saved %d predictions.", saved)
    return saved

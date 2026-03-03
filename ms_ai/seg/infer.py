"""
Sliding-window inference for MS lesion segmentation.

Loads a trained MONAI 3D UNet and produces binary mask predictions.
Predictions are resampled back to the original GT voxel space so that
the saved NIfTI has the same shape / affine as the ground-truth mask.
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
from scipy.ndimage import affine_transform

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


def _resample_pred_to_ref(
    pred: np.ndarray,
    pred_affine: np.ndarray,
    ref_nii: nib.Nifti1Image,
) -> np.ndarray:
    """
    Resample a predicted mask from inference space back to the reference
    (ground-truth) voxel space using nearest-neighbour interpolation.

    Args:
        pred: Binary predicted mask in inference space.
        pred_affine: 4×4 affine matrix of the inference-space volume.
        ref_nii: Reference NIfTI (ground-truth) whose shape/affine we target.

    Returns:
        Binary mask with the same shape as ref_nii.
    """
    ref_affine = ref_nii.affine
    ref_shape = ref_nii.shape[:3]

    # Mapping:  ref_voxel → world → pred_voxel
    #   ref_voxel → world   :  ref_affine
    #   world → pred_voxel  :  inv(pred_affine)
    #   combined             :  inv(pred_affine) @ ref_affine
    mapping = np.linalg.inv(pred_affine) @ ref_affine
    matrix = mapping[:3, :3]
    offset = mapping[:3, 3]

    resampled = affine_transform(
        pred.astype(np.float32),
        matrix=matrix,
        offset=offset,
        output_shape=ref_shape,
        order=0,          # nearest-neighbour for binary masks
        mode="constant",
        cval=0.0,
    )

    return (resampled > 0.5).astype(np.uint8)


def run_inference(
    cases: list,
    split: str,
    checkpoint: str,
    out_dir: str = "runs/seg/preds",
) -> int:
    """
    Run sliding-window inference on all cases and save predicted masks.

    Predicted masks are resampled back to the original GT voxel space
    so they have the same shape and affine as the ground-truth NIfTI.

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

        # Reference NIfTI (GT) — target space for the output
        ref_nii = nib.load(str(mask))

        # Get the affine of the preprocessed (transformed) image.
        # MONAI meta tensors track the affine through all spatial transforms.
        img_meta = data["image"]
        if hasattr(img_meta, "affine"):
            pred_affine = img_meta.affine.numpy()
        else:
            # Fallback: use ref affine (no resampling will occur)
            log.warning("  [WARN] No meta affine for %s/%s — using ref affine", patient, timepoint)
            pred_affine = ref_nii.affine

        img_tensor = data["image"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output = sliding_window_inference(
                img_tensor, roi_size=(96, 96, 96), sw_batch_size=2,
                predictor=model, overlap=0.25,
            )

        # argmax → binary mask (class 1 = lesion)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # Resample prediction back to GT voxel space
        pred_resampled = _resample_pred_to_ref(pred, pred_affine, ref_nii)

        # Save with GT affine / header so shape matches exactly
        out_base = Path(out_dir) / split / patient / timepoint
        out_base.mkdir(parents=True, exist_ok=True)
        out_file = out_base / f"{patient}_{timepoint}_MASK_PRED.nii.gz"

        pred_nii = nib.Nifti1Image(pred_resampled, affine=ref_nii.affine, header=ref_nii.header)
        pred_nii.header.set_data_dtype(np.uint8)
        nib.save(pred_nii, str(out_file))

        saved += 1
        log.info("  [%d] %s/%s → %s (shape=%s)", saved, patient, timepoint,
                 out_file.name, pred_resampled.shape)

    log.info("Inference complete. Saved %d predictions.", saved)
    return saved

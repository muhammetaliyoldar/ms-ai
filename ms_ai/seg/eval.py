"""
Segmentation evaluation: compare predicted masks against ground-truth masks.

Computes per-case and aggregate metrics:
  - Dice coefficient
  - IoU (Jaccard)
  - Sensitivity (recall / true-positive rate)
  - Specificity (true-negative rate)
  - Precision (positive predictive value)
  - Hausdorff Distance 95th percentile (HD95)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

log = logging.getLogger("ms_ai.seg.eval")


def _load_binary(path: Path) -> np.ndarray:
    """Load a NIfTI mask and binarise to {0, 1}."""
    data = nib.load(str(path)).get_fdata()
    return (data > 0.5).astype(np.uint8)


def _compute_metrics(gt: np.ndarray, pred: np.ndarray, voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    Compute segmentation metrics between a GT mask and a predicted mask.

    Both arrays are expected to be binary (0/1) with the same shape.
    """
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)

    tp = np.logical_and(gt_bool, pred_bool).sum()
    fp = np.logical_and(~gt_bool, pred_bool).sum()
    fn = np.logical_and(gt_bool, ~pred_bool).sum()
    tn = np.logical_and(~gt_bool, ~pred_bool).sum()

    # Dice
    denom_dice = 2 * tp + fp + fn
    dice = (2 * tp / denom_dice) if denom_dice > 0 else 1.0  # both empty → perfect

    # IoU (Jaccard)
    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou > 0 else 1.0

    # Sensitivity (TPR / Recall)
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0

    # Specificity (TNR)
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0

    # Precision (PPV)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0

    # Hausdorff Distance 95 (HD95)
    hd95 = _hausdorff_95(gt_bool, pred_bool, voxel_spacing)

    return {
        "dice": round(float(dice), 5),
        "iou": round(float(iou), 5),
        "sensitivity": round(float(sensitivity), 5),
        "specificity": round(float(specificity), 5),
        "precision": round(float(precision), 5),
        "hd95_mm": round(float(hd95), 3),
    }


def _hausdorff_95(gt: np.ndarray, pred: np.ndarray, spacing: Tuple[float, ...]) -> float:
    """
    Compute 95th-percentile Hausdorff distance (mm).

    If either mask is empty, returns NaN.
    """
    from scipy.ndimage import distance_transform_edt

    if not gt.any() or not pred.any():
        return float("nan")

    # Distance transform of the complement
    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)

    # Surface distances: distances from each surface voxel to the other surface
    d_pred_to_gt = dt_gt[pred]   # distance from each pred voxel to nearest GT
    d_gt_to_pred = dt_pred[gt]   # distance from each GT voxel to nearest pred

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def _resample_pred_to_gt(pred_path: Path, gt_path: Path) -> np.ndarray:
    """
    Resample predicted mask into GT voxel space (nearest-neighbour).

    The predicted mask may have a different shape/affine due to
    CropForeground + Spacing transforms applied during inference.
    """
    from scipy.ndimage import affine_transform

    gt_nii = nib.load(str(gt_path))
    pred_nii = nib.load(str(pred_path))

    gt_data = gt_nii.get_fdata()
    pred_data = pred_nii.get_fdata()

    # If shapes already match, return directly
    if gt_data.shape == pred_data.shape:
        return (pred_data > 0.5).astype(np.uint8)

    # Compute the mapping from GT voxel coords → pred voxel coords
    # GT_voxel → world: gt_affine
    # world → pred_voxel: inv(pred_affine)
    gt_aff = gt_nii.affine
    pred_aff = pred_nii.affine
    # Combined: gt_vox → pred_vox = inv(pred_aff) @ gt_aff
    mapping = np.linalg.inv(pred_aff) @ gt_aff

    # affine_transform needs the matrix and offset
    matrix = mapping[:3, :3]
    offset = mapping[:3, 3]

    resampled = affine_transform(
        pred_data,
        matrix=matrix,
        offset=offset,
        output_shape=gt_data.shape,
        order=0,  # nearest-neighbour for binary masks
        mode='constant',
        cval=0.0,
    )

    return (resampled > 0.5).astype(np.uint8)


def evaluate_segmentation(
    cases: list,
    split: str,
    pred_root: str = "runs/seg/preds",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluate predicted masks against GT masks for all cases.

    Args:
        cases: List of case tuples from _iter_cases.
        split: 'train' or 'test'.
        pred_root: Root directory for predicted masks.

    Returns:
        (per_case_df, aggregate_dict)
    """
    pred_base = Path(pred_root)
    rows = []

    for (_split, patient, timepoint, _flair, _t1, _t2, gt_mask) in cases:
        # Locate predicted mask
        pred_dir = pred_base / split / patient / timepoint
        pred_files = sorted(pred_dir.glob("*_MASK_PRED.nii.gz"))

        if not pred_files:
            log.warning("  [SKIP] No predicted mask for %s/%s", patient, timepoint)
            continue

        pred_path = pred_files[0]

        # Load GT mask
        gt_nii = nib.load(str(gt_mask))
        gt = (gt_nii.get_fdata() > 0.5).astype(np.uint8)
        voxel_spacing = tuple(gt_nii.header.get_zooms()[:3])

        # Load and resample predicted mask to GT space
        pred = _resample_pred_to_gt(pred_path, gt_mask)

        metrics = _compute_metrics(gt, pred, voxel_spacing)
        log.info(
            "  %s/%s  dice=%.4f  iou=%.4f  sens=%.4f  hd95=%.1f",
            patient, timepoint, metrics["dice"], metrics["iou"],
            metrics["sensitivity"], metrics["hd95_mm"],
        )

        rows.append({
            "split": split,
            "patient": patient,
            "timepoint": timepoint,
            **metrics,
        })

    df = pd.DataFrame(rows)

    # Aggregate
    if len(df) > 0:
        agg = {
            "n_cases": len(df),
            "mean_dice": round(df["dice"].mean(), 5),
            "std_dice": round(df["dice"].std(), 5),
            "median_dice": round(df["dice"].median(), 5),
            "mean_iou": round(df["iou"].mean(), 5),
            "mean_sensitivity": round(df["sensitivity"].mean(), 5),
            "mean_specificity": round(df["specificity"].mean(), 5),
            "mean_precision": round(df["precision"].mean(), 5),
            "mean_hd95_mm": round(df["hd95_mm"].dropna().mean(), 3) if df["hd95_mm"].notna().any() else float("nan"),
        }
    else:
        agg = {"n_cases": 0}

    return df, agg

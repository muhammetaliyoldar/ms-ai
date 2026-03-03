#!/usr/bin/env python
"""
viz_one_case.py – Visualize a single MRI case with GT (and optionally predicted)
mask overlays on a FLAIR background slice.

Output: a single PNG image.

Usage examples:
    # Train case, mid-slice (default)
    python scripts/viz_one_case.py --split train --patient P1 --timepoint T1 --out P1_T1_overlay.png

    # With predicted mask overlay
    python scripts/viz_one_case.py --split train --patient P1 --timepoint T1 --use-pred --out P1_T1_overlay.png

    # Auto-select slice with maximum GT lesion area
    python scripts/viz_one_case.py --split train --patient P1 --auto-slice --out P1_T1_auto.png

    # Specific slice index
    python scripts/viz_one_case.py --split train --patient P1 --slice 60 --out P1_T1_s60.png

    # Test case (no timepoint subfolder)
    python scripts/viz_one_case.py --split test --patient P54 --use-pred --out P54_overlay.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("ERROR: nibabel is required.  Install with:  pip install nibabel")

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend – safe for headless servers
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    sys.exit("ERROR: matplotlib is required.  Install with:  pip install matplotlib")


# ── Default data root (mirrors config.yaml / CLI default) ───────────────────
_DEFAULT_DATA_ROOT = (
    "/mnt/ortak/MS-AI/döküman/27919209/MSLesSeg Dataset/MSLesSeg Dataset"
)
_DEFAULT_PREDS_DIR = "runs/seg/preds"


# ── Helpers ─────────────────────────────────────────────────────────────────
def _load_yaml_data_root() -> str | None:
    """Try to read data.root_dir from config.yaml if it exists."""
    try:
        import yaml

        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("data", {}).get("root_dir")
    except Exception:
        pass
    return None


def _resolve_data_root(cli_value: str | None) -> Path:
    """Return the data root, preferring CLI arg > config.yaml > hard-coded default."""
    if cli_value:
        return Path(cli_value)
    yaml_root = _load_yaml_data_root()
    if yaml_root:
        return Path(yaml_root)
    return Path(_DEFAULT_DATA_ROOT)


def _resolve_case_dir(data_root: Path, split: str, patient: str, timepoint: str) -> Path:
    """
    Resolve the directory that contains the NIfTI files for a case.

    Train (Layout A): data_root/train/P1/T1/
    Test  (Layout B): data_root/test/P54/      (flat – no timepoint subfolder)
    """
    if split == "train":
        case_dir = data_root / split / patient / timepoint
    else:
        # Test layout: try with timepoint first, then flat
        case_dir = data_root / split / patient / timepoint
        if not case_dir.exists():
            case_dir = data_root / split / patient
    return case_dir


def _find_file(folder: Path, suffix_pattern: str) -> Path | None:
    """Find the first file matching a glob suffix pattern, or None."""
    hits = sorted(folder.glob(suffix_pattern))
    return hits[0] if hits else None


def _robust_normalize(vol: np.ndarray, plow: float = 1.0, phigh: float = 99.0) -> np.ndarray:
    """Percentile-based intensity normalization to [0, 1]."""
    lo, hi = np.percentile(vol, [plow, phigh])
    if hi - lo < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    clipped = np.clip(vol, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _pick_slice(vol: np.ndarray, gt_mask: np.ndarray | None, requested: int | None,
                auto: bool) -> int:
    """
    Decide which z-slice to display.

    Priority: explicit --slice > --auto-slice > mid-slice.
    """
    n_slices = vol.shape[2]
    if requested is not None:
        idx = max(0, min(requested, n_slices - 1))
        return idx

    if auto and gt_mask is not None:
        counts = gt_mask.sum(axis=(0, 1))  # per-slice lesion pixel counts
        if counts.max() > 0:
            return int(np.argmax(counts))

    # Fallback: mid-slice
    return n_slices // 2


# ── Main rendering ─────────────────────────────────────────────────────────
def render_overlay(
    flair_path: Path,
    gt_mask_path: Path,
    pred_mask_path: Path | None,
    slice_idx: int | None,
    auto_slice: bool,
    out_path: Path,
) -> None:
    """Load volumes, build overlay, save PNG."""

    # Load FLAIR
    flair_nii = nib.load(str(flair_path))
    flair_vol = np.asarray(flair_nii.dataobj, dtype=np.float32)

    # Load GT mask
    gt_nii = nib.load(str(gt_mask_path))
    gt_vol = np.asarray(gt_nii.dataobj).astype(np.float32)
    gt_vol = (gt_vol > 0.5).astype(np.float32)

    # Load pred mask (optional)
    pred_vol = None
    if pred_mask_path is not None and pred_mask_path.exists():
        pred_nii = nib.load(str(pred_mask_path))
        pred_vol = np.asarray(pred_nii.dataobj).astype(np.float32)
        pred_vol = (pred_vol > 0.5).astype(np.float32)

    # Normalize FLAIR
    flair_norm = _robust_normalize(flair_vol)

    # Pick slice
    s = _pick_slice(flair_norm, gt_vol, slice_idx, auto_slice)
    print(f"  Slice index: {s} / {flair_norm.shape[2] - 1}")

    # Extract slices (transpose for correct orientation: rows=Y, cols=X)
    bg = np.rot90(flair_norm[:, :, s])
    gt_s = np.rot90(gt_vol[:, :, s])
    pred_s = np.rot90(pred_vol[:, :, s]) if pred_vol is not None else None

    # ── Build figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)

    # GT overlay (red)
    gt_alpha = np.where(gt_s > 0.5, 0.35, 0.0)
    gt_rgba = np.zeros((*gt_s.shape, 4), dtype=np.float32)
    gt_rgba[..., 0] = 1.0   # R
    gt_rgba[..., 3] = gt_alpha
    ax.imshow(gt_rgba)

    # Pred overlay (green)
    if pred_s is not None:
        pred_alpha = np.where(pred_s > 0.5, 0.35, 0.0)
        pred_rgba = np.zeros((*pred_s.shape, 4), dtype=np.float32)
        pred_rgba[..., 1] = 1.0   # G
        pred_rgba[..., 3] = pred_alpha
        ax.imshow(pred_rgba)

    # Title
    title_parts = [flair_path.stem]
    title_parts.append(f"slice {s}")
    ax.set_title(" | ".join(title_parts), fontsize=9, color="#333333")

    # Legend patches
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor="red", alpha=0.5, label="GT mask")]
    if pred_s is not None:
        legend_items.append(Patch(facecolor="green", alpha=0.5, label="Pred mask"))
    ax.legend(handles=legend_items, loc="lower right", fontsize=7,
              framealpha=0.7, edgecolor="none")

    ax.axis("off")
    fig.tight_layout(pad=0.5)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  PNG saved -> {out_path.resolve()}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize a single MRI case with GT/Pred mask overlays on FLAIR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data-root", default=None,
                    help="Path to MSLesSeg dataset root (default: from config.yaml or built-in)")
    p.add_argument("--split", choices=["train", "test"], default="train",
                    help="Dataset split (default: train)")
    p.add_argument("--patient", required=True,
                    help="Patient ID, e.g. P1")
    p.add_argument("--timepoint", default="T1",
                    help="Timepoint ID, e.g. T1 (default: T1). Ignored for test-flat layout.")
    p.add_argument("--use-pred", action="store_true",
                    help="If set, try to load and overlay the predicted mask.")
    p.add_argument("--preds-dir", default=_DEFAULT_PREDS_DIR,
                    help=f"Root directory for predicted masks (default: {_DEFAULT_PREDS_DIR})")
    p.add_argument("--out", default="overlay.png",
                    help="Output PNG path (default: overlay.png)")
    p.add_argument("--slice", type=int, default=None, dest="slice_idx",
                    help="Explicit z-slice index. If omitted, use mid-slice (or --auto-slice).")
    p.add_argument("--auto-slice", action="store_true",
                    help="Automatically choose the slice with the most GT lesion pixels.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    data_root = _resolve_data_root(args.data_root)
    print(f"Data root : {data_root}")
    print(f"Split     : {args.split}")
    print(f"Patient   : {args.patient}")
    print(f"Timepoint : {args.timepoint}")

    # ── Resolve case directory ──────────────────────────────────────────
    case_dir = _resolve_case_dir(data_root, args.split, args.patient, args.timepoint)
    if not case_dir.exists():
        sys.exit(f"ERROR: Case directory not found: {case_dir}")

    # ── Locate FLAIR ────────────────────────────────────────────────────
    flair_path = _find_file(case_dir, "*_FLAIR.nii.gz")
    if flair_path is None:
        sys.exit(f"ERROR: No FLAIR file found in {case_dir}")
    print(f"FLAIR     : {flair_path}")

    # ── Locate GT mask ──────────────────────────────────────────────────
    # GT mask file – avoid matching *_MASK_PRED.nii.gz
    gt_candidates = sorted(case_dir.glob("*_MASK.nii.gz"))
    gt_candidates = [p for p in gt_candidates if "_MASK_PRED" not in p.name]
    if not gt_candidates:
        sys.exit(f"ERROR: No GT mask file found in {case_dir}")
    gt_mask_path = gt_candidates[0]
    print(f"GT mask   : {gt_mask_path}")

    # ── Locate predicted mask (optional) ────────────────────────────────
    pred_mask_path = None
    if args.use_pred:
        # Predicted masks always stored as <preds-dir>/<split>/<patient>/<timepoint>/<patient>_<timepoint>_MASK_PRED.nii.gz
        pred_dir = Path(args.preds_dir) / args.split / args.patient / args.timepoint
        pred_name = f"{args.patient}_{args.timepoint}_MASK_PRED.nii.gz"
        pred_path_candidate = pred_dir / pred_name
        if pred_path_candidate.exists():
            pred_mask_path = pred_path_candidate
            print(f"Pred mask : {pred_mask_path}")
        else:
            # Fallback: try glob in pred_dir
            alt = _find_file(pred_dir, "*_MASK_PRED.nii.gz")
            if alt is not None:
                pred_mask_path = alt
                print(f"Pred mask : {pred_mask_path} (alt)")
            else:
                print(f"WARNING: Predicted mask not found at {pred_path_candidate}")
                print("         Continuing with GT overlay only.")

    # ── Render ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    render_overlay(
        flair_path=flair_path,
        gt_mask_path=gt_mask_path,
        pred_mask_path=pred_mask_path,
        slice_idx=args.slice_idx,
        auto_slice=args.auto_slice,
        out_path=out_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()

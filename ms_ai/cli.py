import click
from pathlib import Path
import logging

from ms_ai.logging_utils import log_run

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ms_ai.check_data")


def _pick_one(folder: Path, pattern: str) -> Path:
    """Pick the first file matching an exact-suffix glob pattern."""
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Missing file matching {pattern} in {folder}")
    return hits[0]


# Exact-suffix patterns – avoids e.g. *_T1*.nii.gz matching P1_T1_FLAIR.nii.gz
_PATTERNS = {
    "flair": "*_FLAIR.nii.gz",
    "t1":    "*_T1.nii.gz",
    "t2":    "*_T2.nii.gz",
    "mask":  "*_MASK.nii.gz",
}


def _scan_folder(folder: Path, split: str, patient: str, timepoint: str):
    """Yield a single case tuple from *folder* using exact suffix patterns."""
    flair = _pick_one(folder, _PATTERNS["flair"])
    t1    = _pick_one(folder, _PATTERNS["t1"])
    t2    = _pick_one(folder, _PATTERNS["t2"])
    mask  = _pick_one(folder, _PATTERNS["mask"])
    return (split, patient, timepoint, flair, t1, t2, mask)


def _iter_cases(data_root: Path, split: str):
    """
    Iterate over cases in a split directory.

    Supports two layouts:
      Layout A (train): <split>/P*/T*/  – each T* subfolder has 4 files
      Layout B (test):  <split>/P*/     – patient folder has 4 files directly
                        (no T* subfolders → treated as single timepoint "T1")
    """
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    for pdir in sorted(split_dir.glob("P*")):
        if not pdir.is_dir():
            continue
        patient = pdir.name

        # Check for T* subdirectories (Layout A)
        tp_dirs = sorted([d for d in pdir.glob("T*") if d.is_dir()])

        if tp_dirs:
            # Layout A: iterate over timepoint sub-folders
            for tp_dir in tp_dirs:
                timepoint = tp_dir.name
                yield _scan_folder(tp_dir, split, patient, timepoint)
        else:
            # Layout B: files live directly in patient folder, single timepoint
            yield _scan_folder(pdir, split, patient, "T1")


@click.group()
def cli():
    """MS-AI Command Line Interface."""
    pass


@cli.command("check-data")
@click.option(
    "--data-root",
    default="/mnt/ortak/MS-AI/döküman/27919209/MSLesSeg Dataset/MSLesSeg Dataset",
    show_default=True,
)
@click.option("--max-print", default=5, show_default=True)
def check_data(data_root: str, max_print: int):
    """
    Check dataset structure: train/test/info_dataset + (P*/T*/ FL, T1, T2, MASK) files.
    """
    root = Path(data_root)

    issues = 0
    click.echo(f"DATA_ROOT = {root}")

    # required dirs
    for name in ["train", "test", "info_dataset"]:
        p = root / name
        if not p.exists():
            click.echo(f"[ERROR] Missing folder: {p}")
            issues += 1

    clinical = root / "info_dataset" / "clinical_data.csv"
    if not clinical.exists():
        click.echo(f"[ERROR] Missing: {clinical}")
        issues += 1
    else:
        click.echo(f"[OK] Found clinical: {clinical}")

    for split in ["train", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            # Already flagged above; skip scanning
            continue

        try:
            cases = list(_iter_cases(root, split))
        except Exception as e:
            click.echo(f"[ERROR] Failed scanning split={split}: {e}")
            if split == "train":
                raise SystemExit(2)
            else:
                # test scan failure is non-fatal
                click.echo(f"[WARN] Skipping split={split} due to scan error")
                continue

        click.echo(f"[OK] Split={split} cases={len(cases)}")
        if len(cases) == 0:
            if split == "test":
                click.echo(f"[WARN] No cases found in split={split} (non-fatal)")
            else:
                click.echo(f"[ERROR] No cases found in split={split}")
                issues += 1
            continue

        for i, (sp, patient, tp, flair, t1, t2, mask) in enumerate(cases[:max_print], 1):
            click.echo(f"  Example {i}: {patient}/{tp} -> {flair.name}, {t1.name}, {t2.name}, {mask.name}")

    if issues == 0:
        click.echo("Dataset check: OK ✅")
        log_run("check-data", status="OK")
        raise SystemExit(0)
    else:
        click.echo(f"Dataset check: FAILED (issues={issues}) ❌")
        log_run("check-data", status="FAIL")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# extract-features
# ---------------------------------------------------------------------------
@cli.command("extract-features")
@click.option(
    "--data-root",
    default="/mnt/ortak/MS-AI/döküman/27919209/MSLesSeg Dataset/MSLesSeg Dataset",
    show_default=True,
)
@click.option(
    "--use",
    "mask_source",
    type=click.Choice(["gt", "pred"], case_sensitive=False),
    required=True,
    help="Mask source: gt (ground-truth) or pred (predicted).",
)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "all"], case_sensitive=False),
    required=True,
)
@click.option("--pred-root", default="runs/seg/preds", show_default=True)
def extract_features(data_root: str, mask_source: str, split: str, pred_root: str):
    """
    Extract lesion features from GT or predicted masks and write CSV.
    """
    import pandas as pd
    from ms_ai.features.lesion_features import compute_lesion_features

    root = Path(data_root)
    pred_base = Path(pred_root)
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    splits = ["train", "test"] if split == "all" else [split]

    all_rows = []
    skipped = 0
    for sp in splits:
        split_dir = root / sp
        if not split_dir.exists():
            log.warning("Split folder %s not found, skipping.", split_dir)
            continue

        cases = list(_iter_cases(root, sp))
        log.info("Split=%s  cases=%d  mask_source=%s", sp, len(cases), mask_source)

        for _sp, patient, timepoint, _flair, _t1, _t2, gt_mask in cases:
            # Resolve mask path
            if mask_source == "gt":
                mask_path = gt_mask
            else:
                try:
                    mask_path = _pick_one(
                        pred_base / sp / patient / timepoint,
                        "*_MASK_PRED.nii.gz",
                    )
                except FileNotFoundError:
                    log.warning("  [SKIP] No predicted mask for %s/%s", patient, timepoint)
                    skipped += 1
                    continue

            log.info("  %s/%s  mask=%s", patient, timepoint, mask_path.name)
            feats = compute_lesion_features(mask_path)
            row = {
                "split": sp,
                "patient": patient,
                "timepoint": timepoint,
                "mask_file": mask_path.name,
                **feats,
            }
            all_rows.append(row)

    if skipped:
        click.echo(f"[WARN] Skipped {skipped} cases (missing predicted masks)")

    df = pd.DataFrame(all_rows)
    out_name = f"lesion_features_{mask_source}_{split}.csv"
    out_path = artifacts / out_name
    df.to_csv(out_path, index=False)
    click.echo(f"[OK] Wrote {len(df)} rows → {out_path}")
    log_run("extract-features", use=mask_source, split=split,
            rows_written=len(df), output_path=str(out_path))


# ---------------------------------------------------------------------------
# merge-edss
# ---------------------------------------------------------------------------
@cli.command("merge-edss")
@click.option(
    "--data-root",
    default="/mnt/ortak/MS-AI/döküman/27919209/MSLesSeg Dataset/MSLesSeg Dataset",
    show_default=True,
)
@click.option(
    "--use",
    "mask_source",
    type=click.Choice(["gt", "pred"], case_sensitive=False),
    required=True,
    help="Which features file to merge (gt or pred).",
)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "all"], case_sensitive=False),
    required=True,
)
def merge_edss(data_root: str, mask_source: str, split: str):
    """
    Merge extracted lesion features with clinical EDSS data.
    """
    import pandas as pd
    from ms_ai.edss.merge import merge_features_with_edss

    root = Path(data_root)
    artifacts = Path("artifacts")

    features_file = artifacts / f"lesion_features_{mask_source}_{split}.csv"
    if not features_file.exists():
        click.echo(f"[ERROR] Features file not found: {features_file}")
        click.echo("Run extract-features first.")
        raise SystemExit(1)

    clinical_path = root / "info_dataset" / "clinical_data.csv"
    if not clinical_path.exists():
        click.echo(f"[ERROR] Clinical CSV not found: {clinical_path}")
        raise SystemExit(1)

    features_df = pd.read_csv(features_file)
    merged = merge_features_with_edss(features_df, clinical_path)

    out_name = f"features_with_edss_{mask_source}_{split}.csv"
    out_path = artifacts / out_name
    merged.to_csv(out_path, index=False)

    n_edss = merged["edss"].notna().sum()
    click.echo(f"[OK] Wrote {len(merged)} rows → {out_path}")
    click.echo(f"     Rows with non-null EDSS: {n_edss}/{len(merged)}")
    log_run("merge-edss", use=mask_source, split=split,
            rows_written=len(merged), output_path=str(out_path))


# ---------------------------------------------------------------------------
# train-edss
# ---------------------------------------------------------------------------
@cli.command("train-edss")
@click.option(
    "--use",
    "mask_source",
    type=click.Choice(["gt", "pred"], case_sensitive=False),
    required=True,
    help="Which features to use (gt or pred).",
)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "all"], case_sensitive=False),
    required=True,
)
def train_edss(mask_source: str, split: str):
    """
    Train EDSS regression model with patient-level GroupKFold CV.
    """
    import pandas as pd
    from ms_ai.edss.train import cross_validate, train_final_model

    artifacts = Path("artifacts")

    # Resolve input file(s)
    if split == "all":
        parts = []
        for sp in ["train", "test"]:
            f = artifacts / f"features_with_edss_{mask_source}_{sp}.csv"
            if f.exists():
                parts.append(pd.read_csv(f))
            else:
                log.warning("File not found, skipping: %s", f)
        if not parts:
            click.echo("[ERROR] No merged feature files found.")
            raise SystemExit(1)
        df = pd.concat(parts, ignore_index=True)
    else:
        f = artifacts / f"features_with_edss_{mask_source}_{split}.csv"
        if not f.exists():
            click.echo(f"[ERROR] Merged file not found: {f}")
            click.echo("Run merge-edss first.")
            raise SystemExit(1)
        df = pd.read_csv(f)

    # Keep only rows with non-null EDSS
    before = len(df)
    df = df.dropna(subset=["edss"]).reset_index(drop=True)
    click.echo(f"Loaded {before} rows, {len(df)} with non-null EDSS")

    if len(df) == 0:
        click.echo("[ERROR] No trainable rows (all EDSS values are null).")
        raise SystemExit(1)

    # Cross-validation
    click.echo("\n── Patient-level GroupKFold (5 folds) ──")
    fold_maes, mean_mae, std_mae = cross_validate(df, n_splits=5)
    click.echo(f"\nPer-fold MAE: {fold_maes}")
    click.echo(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")

    # Train final model on all data
    click.echo("\n── Training final model on all data ──")
    model_path = train_final_model(df, mask_source, split, artifacts)
    click.echo(f"[OK] Model saved → {model_path}")
    click.echo(f"[OK] Metadata saved → {model_path.with_name(model_path.stem + '_meta.json')}")
    log_run("train-edss", use=mask_source, split=split,
            output_path=str(model_path))


# ---------------------------------------------------------------------------
# train-seg
# ---------------------------------------------------------------------------
@cli.command("train-seg")
@click.option(
    "--data-root",
    default="/mnt/ortak/MS-AI/döküman/27919209/MSLesSeg Dataset/MSLesSeg Dataset",
    show_default=True,
)
@click.option("--epochs", default=50, show_default=True)
@click.option("--val-split", default=0.2, show_default=True)
@click.option("--out-dir", default="runs/seg", show_default=True)
def train_seg(data_root: str, epochs: int, val_split: float, out_dir: str):
    """
    Train a 3D UNet for MS lesion segmentation (MONAI).
    """
    from ms_ai.seg.train import run_training

    root = Path(data_root)
    cases = list(_iter_cases(root, "train"))
    click.echo(f"Train split: {len(cases)} cases")
    click.echo(f"Epochs: {epochs}, Val split: {val_split}")

    best_dice = run_training(cases, epochs=epochs, val_split=val_split, out_dir=out_dir)
    click.echo(f"\n[OK] Training complete. Best val Dice = {best_dice:.4f}")
    log_run("train-seg", split="train", rows_written=len(cases),
            output_path=f"{out_dir}/best_model.pth")


# ---------------------------------------------------------------------------
# infer-seg
# ---------------------------------------------------------------------------
@cli.command("infer-seg")
@click.option(
    "--data-root",
    default="/mnt/ortak/MS-AI/döküman/27919209/MSLesSeg Dataset/MSLesSeg Dataset",
    show_default=True,
)
@click.option(
    "--split",
    type=click.Choice(["train", "test"], case_sensitive=False),
    required=True,
)
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--out-dir", default="runs/seg/preds", show_default=True)
def infer_seg(data_root: str, split: str, checkpoint: str, out_dir: str):
    """
    Run inference with a trained segmentation model and save predicted masks.
    """
    from ms_ai.seg.infer import run_inference

    root = Path(data_root)
    cases = list(_iter_cases(root, split))
    click.echo(f"Split={split}: {len(cases)} cases")
    click.echo(f"Checkpoint: {checkpoint}")

    n_saved = run_inference(cases, split=split, checkpoint=checkpoint, out_dir=out_dir)
    click.echo(f"\n[OK] Saved {n_saved} predicted masks to {out_dir}/{split}/")
    log_run("infer-seg", split=split, rows_written=n_saved,
            output_path=f"{out_dir}/{split}/")


if __name__ == "__main__":
    cli()
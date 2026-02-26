"""
Clinical data loader and EDSS merge utilities.

Handles the MSLesSeg clinical CSV quirks:
  • BOM (\\ufeff) prefix
  • semicolon delimiter
  • comma-as-decimal (e.g. ``28,09``)
  • column normalisation (lowercase + underscores)
"""

from pathlib import Path

import pandas as pd


def load_clinical(clinical_path: Path) -> pd.DataFrame:
    """
    Load and normalise the clinical CSV.

    Returns a DataFrame with lowercase/underscore column names.
    Numeric columns ``age`` and ``edss`` have comma→dot conversion.
    ``MS Type`` is renamed to ``ms_type``.
    """
    df = pd.read_csv(
        clinical_path,
        sep=";",
        encoding="utf-8-sig",  # strips BOM
        engine="python",
    )

    # Drop fully-empty / unnamed trailing columns created by trailing semicolons
    df = df.loc[:, [c for c in df.columns
                     if str(c).strip() != "" and not str(c).startswith("Unnamed")]]

    # Normalise column names: strip, lowercase, spaces → underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Comma-decimal → dot for numeric fields, then convert
    for col in ["age", "edss"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_features_with_edss(
    features_df: pd.DataFrame,
    clinical_path: Path,
) -> pd.DataFrame:
    """
    Left-join lesion features with clinical data on (patient, timepoint).

    Test-split cases without EDSS will have ``edss = NaN`` (expected).
    """
    clinical = load_clinical(clinical_path)
    merged = features_df.merge(
        clinical,
        on=["patient", "timepoint"],
        how="left",
    )
    return merged

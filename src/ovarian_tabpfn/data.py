"""Loading and cleaning of the Lu et al. (2020) ovarian tumour dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXCLUDED_COLUMNS, ID_COLUMN, PLAUSIBLE_RANGES, TARGET


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the spreadsheet and convert text-encoded numbers (some cells carry tab characters)."""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns[df.dtypes == object]:
        cleaned = df[col].astype(str).str.replace("\t", "", regex=False).str.strip()
        df[col] = pd.to_numeric(cleaned.replace({"nan": np.nan, "": np.nan}), errors="coerce")
    return df


def remove_implausible_values(df: pd.DataFrame, ranges=PLAUSIBLE_RANGES) -> pd.DataFrame:
    df = df.copy()
    for col, (low, high) in ranges.items():
        if col in df:
            df.loc[(df[col] < low) | (df[col] > high), col] = np.nan
    return df


def cancer_label(df: pd.DataFrame) -> pd.Series:
    """1 = ovarian cancer, 0 = benign tumour.

    The file does not document the coding of TYPE. Cancer patients are older
    (median 53 vs 36 years in the original publication), which fixes the orientation.
    """
    cancer_code = df.groupby(TARGET)["Age"].median().idxmax()
    return (df[TARGET] == cancer_code).astype(int).rename("cancer")


def prepare(path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return predictors X (47 columns), outcome y and the cleaned full table."""
    df = remove_implausible_values(load_dataset(path))
    y = cancer_label(df)
    drop = [ID_COLUMN, TARGET, *EXCLUDED_COLUMNS]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return X, y, df

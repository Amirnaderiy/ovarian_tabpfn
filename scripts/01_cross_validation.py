"""Repeated 10 x 10 stratified cross-validation of all models.

Writes results/repeated_cv_predictions.csv: one row per patient with the mean
out-of-fold probability of cancer for each model.

    python scripts/01_cross_validation.py --data "data/Supplementary data 1.xlsx" \
        --model-path models/tabpfn-v3-classifier-v3_20260417_binary.ckpt
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ovarian_tabpfn import models
from ovarian_tabpfn.cv import out_of_fold, splits
from ovarian_tabpfn.data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/Supplementary data 1.xlsx")
    parser.add_argument("--model-path", help="TabPFN-3 checkpoint (or set TABPFN_MODEL_PATH)")
    parser.add_argument("--models", nargs="+", default=models.MODEL_NAMES, choices=models.MODEL_NAMES)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, _ = prepare(args.data)
    print(f"{len(y)} patients ({y.sum()} cancers), {X.shape[1]} predictors")

    folds = splits(X, y)
    table = pd.DataFrame({"SUBJECT_ID": np.arange(len(y)), "true_label": y.values,
                          "true_class": np.where(y == 1, "OC", "BOT")})
    for name in args.models:
        print(f"{name}:")
        table[f"{name}_prob"] = out_of_fold(partial(models.build, name, args.model_path), X, y, folds).round(4)

    path = out / "repeated_cv_predictions.csv"
    table.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

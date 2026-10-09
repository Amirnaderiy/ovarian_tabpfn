"""Repeated stratified cross-validation with per-patient averaging of out-of-fold predictions."""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import N_REPEATS, N_SPLITS, SEED


def splits(X, y, n_repeats: int = N_REPEATS):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=SEED)
    return list(cv.split(X, y))


def out_of_fold(make_model: Callable, X: pd.DataFrame, y: pd.Series, folds=None,
                log_every: int = 10) -> np.ndarray:
    """Mean out-of-fold probability of the positive class for every patient.

    Each patient is in the test fold once per repeat, so with 10 repeats the
    returned value averages ten predictions from models that never saw the patient.
    """
    folds = folds if folds is not None else splits(X, y)
    total = np.zeros(len(y))
    count = np.zeros(len(y))
    y = np.asarray(y)
    for i, (train, test) in enumerate(folds, start=1):
        model = make_model().fit(X.iloc[train], y[train])
        total[test] += model.predict_proba(X.iloc[test])[:, 1]
        count[test] += 1
        if log_every and i % log_every == 0:
            print(f"  fold {i}/{len(folds)}", flush=True)
    return total / count

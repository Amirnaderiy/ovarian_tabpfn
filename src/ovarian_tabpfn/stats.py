"""Discrimination, calibration and clinical-utility statistics for patient-level predictions."""

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, roc_curve

from .config import N_BOOTSTRAP, SEED


# --- DeLong test (fast algorithm of Sun & Xu, 2014) -------------------------------------------

def _midranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks
    return out


def delong(y, p1, p2) -> float:
    """Two-sided p-value for H0: AUC(p1) == AUC(p2), both evaluated on the same patients."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midranks(p[:m]) for p in preds])
    ty = np.array([_midranks(p[m:]) for p in preds])
    tz = np.array([_midranks(p) for p in preds])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    z = (auc[0] - auc[1]) / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    return float(2 * stats.norm.sf(abs(z)))


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


# --- Uncertainty -------------------------------------------------------------------------------

def bootstrap_ci(y, p, statistic=roc_auc_score, n: int = N_BOOTSTRAP, seed: int = SEED):
    """Percentile 95% CI with patients resampled with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y))
        if y[idx].min() != y[idx].max():
            values.append(statistic(y[idx], p[idx]))
    return tuple(np.percentile(values, [2.5, 97.5]))


# --- Operating points --------------------------------------------------------------------------

def sensitivity_at_specificity(y, p, specificity: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[(1 - fpr) >= specificity].max())


def specificity_at_sensitivity(y, p, sensitivity: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(1 - fpr[np.argmax(tpr >= sensitivity)])


def threshold_metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (sens + spec) / 2,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "sensitivity": sens,
        "specificity": spec,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "f1": f1_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


# --- Calibration and decision curves -----------------------------------------------------------

def calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Calibration-in-the-large (slope fixed at 1) and calibration slope (logistic recalibration)."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    slope = LogisticRegression(C=1e9, max_iter=1000).fit(logit[:, None], y).coef_[0, 0]
    intercept = 0.0
    for _ in range(50):  # Newton steps for the offset model
        q = 1 / (1 + np.exp(-(intercept + logit)))
        intercept += (y - q).sum() / (q * (1 - q)).sum()
    return float(intercept), float(slope)


def net_benefit(y, p, thresholds) -> np.ndarray:
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    out = []
    for t in thresholds:
        positive = p >= t
        tp = np.sum(positive & (y == 1))
        fp = np.sum(positive & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def net_benefit_treat_all(y, thresholds) -> np.ndarray:
    prevalence = np.mean(y)
    thresholds = np.asarray(thresholds)
    return prevalence - (1 - prevalence) * thresholds / (1 - thresholds)


# --- Clinical reference --------------------------------------------------------------------------

def roma(he4, ca125, postmenopausal) -> np.ndarray:
    """Risk of Ovarian Malignancy Algorithm (Moore et al., Gynecol Oncol 2009)."""
    he4, ca125 = np.log(np.asarray(he4, float)), np.log(np.asarray(ca125, float))
    index = np.where(postmenopausal,
                     -8.09 + 1.04 * he4 + 0.732 * ca125,
                     -12.0 + 2.38 * he4 + 0.0626 * ca125)
    return 1 / (1 + np.exp(-index))

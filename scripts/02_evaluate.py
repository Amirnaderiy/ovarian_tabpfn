"""Statistics, tables and figures from the patient-level predictions.

Needs only results/repeated_cv_predictions.csv (no TabPFN, no raw data).

Outputs (in --out):
    table2_performance.csv       ROC-AUC with bootstrap CI, PR-AUC, metrics at threshold 0.60
    table3_operating_points.csv  sensitivity/specificity at 0.90, Brier score, calibration, metrics at 0.50
    delong_holm.csv              DeLong tests vs TabPFN-3 with Holm adjustment
    decision_curve.csv           net benefit across thresholds
    fig2_roc.png, fig4_calibration_decision.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ovarian_tabpfn import plots, stats
from ovarian_tabpfn.config import DECISION_THRESHOLD

MODELS = ["TabPFN3", "RandomForest", "XGBoost", "LogReg", "CART"]


def performance(y, p):
    auc_low, auc_high = stats.bootstrap_ci(y, p)
    row = {"roc_auc": roc_auc_score(y, p), "roc_auc_low": auc_low, "roc_auc_high": auc_high,
           "pr_auc": average_precision_score(y, p)}
    row.update(stats.threshold_metrics(y, p, DECISION_THRESHOLD))
    return row


def operating_points(y, p):
    intercept, slope = stats.calibration_intercept_slope(y, p)
    row = {
        "sens_at_spec90": stats.sensitivity_at_specificity(y, p),
        "spec_at_sens90": stats.specificity_at_sensitivity(y, p),
        "brier": brier_score_loss(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    row["sens_at_spec90_low"], row["sens_at_spec90_high"] = stats.bootstrap_ci(
        y, p, stats.sensitivity_at_specificity)
    row["spec_at_sens90_low"], row["spec_at_sens90_high"] = stats.bootstrap_ci(
        y, p, stats.specificity_at_sensitivity)
    row["brier_low"], row["brier_high"] = stats.bootstrap_ci(y, p, brier_score_loss)
    at_half = stats.threshold_metrics(y, p, 0.50)
    row.update({f"{k}_at_0.50": at_half[k] for k in ("accuracy", "sensitivity", "specificity", "f1", "mcc")})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", default="results/repeated_cv_predictions.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.preds)
    y = df["true_label"].to_numpy()
    preds = {m: df[f"{m}_prob"].to_numpy() for m in MODELS}

    table2 = pd.DataFrame({m: performance(y, preds[m]) for m in MODELS}).T
    table3 = pd.DataFrame({m: operating_points(y, preds[m]) for m in MODELS}).T
    table2.round(4).to_csv(out / "table2_performance.csv")
    table3.round(4).to_csv(out / "table3_operating_points.csv")

    others = MODELS[1:]
    p_raw = [stats.delong(y, preds["TabPFN3"], preds[m]) for m in others]
    tests = pd.DataFrame({"comparison": [f"TabPFN3 vs {m}" for m in others],
                          "p": p_raw, "p_holm": stats.holm(p_raw)})
    tests.round(4).to_csv(out / "delong_holm.csv", index=False)

    thresholds = np.round(np.arange(0.05, 0.81, 0.01), 2)
    dca = pd.DataFrame({"threshold": thresholds, "treat_all": stats.net_benefit_treat_all(y, thresholds)})
    for m in MODELS:
        dca[m] = stats.net_benefit(y, preds[m], thresholds)
    dca.round(4).to_csv(out / "decision_curve.csv", index=False)

    plots.roc_figure(y, preds, table2, out / "fig2_roc.png")
    plots.calibration_and_decision_figure(y, preds, table3, dca, out / "fig4_calibration_decision.png")

    with pd.option_context("display.width", 120, "display.precision", 3):
        print(table2[["roc_auc", "roc_auc_low", "roc_auc_high", "sensitivity", "specificity", "mcc"]])
        print(tests)


if __name__ == "__main__":
    main()

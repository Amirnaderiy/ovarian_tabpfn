"""Publication figures (600 dpi)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

LABELS = {"TabPFN3": "TabPFN-3", "XGBoost": "XGBoost", "RandomForest": "Random forest",
          "LogReg": "Logistic regression", "CART": "Decision tree"}
COLORS = {"TabPFN3": "#0072B2", "XGBoost": "#D55E00", "RandomForest": "#009E73",
          "LogReg": "#CC79A7", "CART": "#E69F00"}  # Okabe-Ito
ORDER = ["TabPFN3", "XGBoost", "RandomForest", "LogReg", "CART"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "savefig.bbox": "tight"})


def _width(name):
    return 2.2 if name == "TabPFN3" else 1.5


def roc_figure(y, preds: dict, auc_table: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.3))
    for name in ORDER:
        fpr, tpr, _ = roc_curve(y, preds[name])
        auc, lo, hi = auc_table.loc[name, ["roc_auc", "roc_auc_low", "roc_auc_high"]]
        ax.plot(fpr, tpr, color=COLORS[name], lw=_width(name),
                label=f"{LABELS[name]}  AUC {auc:.2f} ({lo:.2f}–{hi:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.savefig(path)
    plt.close(fig)


def calibration_and_decision_figure(y, preds: dict, calib: pd.DataFrame, dca: pd.DataFrame, path):
    y = np.asarray(y)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.8))

    left.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="Ideal")
    for name in ORDER:
        p = preds[name]
        decile = pd.qcut(p, 10, labels=False, duplicates="drop")
        groups = np.unique(decile)
        left.plot([p[decile == g].mean() for g in groups], [y[decile == g].mean() for g in groups],
                  "o-", ms=4, color=COLORS[name], lw=_width(name) - 0.2,
                  label=f"{LABELS[name]} (Brier {calib.loc[name, 'brier']:.3f}; "
                        f"slope {calib.loc[name, 'calibration_slope']:.2f})")
    left.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
             ylabel="Observed proportion with cancer")
    left.set_aspect("equal")
    left.grid(alpha=0.2)
    left.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    left.text(-0.16, 1.02, "a", transform=left.transAxes, fontsize=13, fontweight="bold")

    t = dca["threshold"]
    right.plot(t, dca["treat_all"], color="0.35", lw=1.2, label="Treat all")
    right.axhline(0, color="k", lw=1, label="Treat none")
    for name in ORDER:
        right.plot(t, dca[name], color=COLORS[name], lw=_width(name), label=LABELS[name])
    right.set(xlim=(0.05, 0.8), ylim=(-0.05, 0.55), xlabel="Threshold probability", ylabel="Net benefit")
    right.grid(alpha=0.2)
    right.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
    right.text(-0.14, 1.02, "b", transform=right.transAxes, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def shap_figures(values, data, features, bar_path, beeswarm_path, n_bar=15, n_beeswarm=20):
    import shap

    importance = np.abs(values).mean(axis=0)
    order = np.argsort(-importance)

    top = order[:n_bar][::-1]
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    ax.barh([features[i] for i in top], importance[top], color=COLORS["TabPFN3"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(bar_path)
    plt.close(fig)

    top = order[:n_beeswarm]
    shap.summary_plot(values[:, top], features=data[:, top], feature_names=[features[i] for i in top],
                      plot_type="dot", show=False, plot_size=(5.6, 6.4), color_bar_label="Feature value")
    plt.gcf().savefig(beeswarm_path)
    plt.close("all")

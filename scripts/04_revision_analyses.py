"""Additional analyses requested during peer review.

1. Clinical references: ROMA (fixed published formula; patients with HE4 and CA125
   measured) and a logistic model of log-HE4, log-CA125 and menopausal status
   (same cross-validation folds), each compared with TabPFN-3 (DeLong, Holm).
2. TabPFN-3 without age and menopausal status (same folds).
3. Stability of SHAP rankings: out-of-fold Kernel SHAP within the ten folds of the
   first repeat, for the full model and the model without age and menopausal status.

Step 0 re-runs TabPFN-3 and checks that it reproduces the saved predictions.
Runtime is dominated by step 3 (several hours on CPU; --skip-shap to omit it).
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ovarian_tabpfn import models, stats
from ovarian_tabpfn.config import SEED
from ovarian_tabpfn.cv import out_of_fold, splits
from ovarian_tabpfn.data import prepare

DEMOGRAPHICS = ["Age", "Menopause"]


def auc_with_ci(y, p):
    low, high = stats.bootstrap_ci(y, p)
    return {"auc": round(roc_auc_score(y, p), 4), "ci": [round(low, 4), round(high, 4)]}


def marker_model():
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def postmenopausal(df):
    # The older group is postmenopausal (coded 1 in this dataset).
    post_code = df.groupby("Menopause")["Age"].median().idxmax()
    return (df["Menopause"] == post_code).to_numpy()


def oof_shap(X, y, folds, model_path, nsamples):
    """Kernel SHAP for each patient, explained by the model of the fold in which it was held out."""
    import shap

    values = np.zeros(X.shape)
    per_fold = []
    for i, (train, test) in enumerate(folds, start=1):
        model = models.tabpfn(model_path).fit(X.iloc[train], y.iloc[train])
        background = shap.kmeans(X.iloc[train].fillna(X.iloc[train].mean()), 10)
        explainer = shap.KernelExplainer(
            lambda d: model.predict_proba(pd.DataFrame(d, columns=X.columns))[:, 1], background)
        fold_values = np.asarray(explainer.shap_values(X.iloc[test], nsamples=nsamples, silent=True))
        values[test] = fold_values
        per_fold.append(np.abs(fold_values).mean(axis=0))
        print(f"  SHAP fold {i}/{len(folds)}", flush=True)
    return values, pd.DataFrame(per_fold, columns=X.columns)


def stability(per_fold: pd.DataFrame, features=("HE4", "CA125", "CEA", "Age")):
    ranks = per_fold.rank(axis=1, ascending=False)
    rho = [spearmanr(per_fold.iloc[i], per_fold.iloc[j])[0]
           for i in range(len(per_fold)) for j in range(i + 1, len(per_fold))]
    summary = {f: {"rank_1": int((ranks[f] == 1).sum()), "top_3": int((ranks[f] <= 3).sum()),
                   "median_rank": float(ranks[f].median())}
               for f in features if f in ranks}
    summary["mean_pairwise_spearman"] = round(float(np.mean(rho)), 3)
    summary["min_pairwise_spearman"] = round(float(np.min(rho)), 3)
    summary["top_features_overall"] = list(per_fold.mean().sort_values(ascending=False).index[:10])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/Supplementary data 1.xlsx")
    parser.add_argument("--preds", default="results/repeated_cv_predictions.csv")
    parser.add_argument("--model-path", help="TabPFN-3 checkpoint (or set TABPFN_MODEL_PATH)")
    parser.add_argument("--out", default="results/revision")
    parser.add_argument("--nsamples", type=int, default=200)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y, df = prepare(args.data)
    saved = pd.read_csv(args.preds)
    if not np.array_equal(saved["true_label"].to_numpy(), y.to_numpy()):
        raise SystemExit("prediction file and dataset are not in the same row order")
    p_tabpfn = saved["TabPFN3_prob"].to_numpy()
    folds = splits(X, y)
    tabpfn = partial(models.tabpfn, args.model_path)
    results = {}

    if not args.skip_check:
        print("Checking that the TabPFN-3 checkpoint reproduces the saved predictions:")
        diff = np.abs(out_of_fold(tabpfn, X, y, folds).round(4) - p_tabpfn).max()
        print(f"  max difference {diff:.4f}")
        if diff > 0.01:
            raise SystemExit("checkpoint does not reproduce results/repeated_cv_predictions.csv")

    # 1. clinical references
    post = postmenopausal(df)
    roma = stats.roma(df["HE4"], df["CA125"], post)
    measured = ~np.isnan(roma)
    X_markers = pd.DataFrame({"log_HE4": np.log(df["HE4"]), "log_CA125": np.log(df["CA125"]),
                              "postmenopausal": post.astype(float)})
    print("HE4 + CA125 + menopause logistic model:")
    p_markers = out_of_fold(marker_model, X_markers, y, folds)

    p_values = [stats.delong(y[measured], p_tabpfn[measured], roma[measured]),
                stats.delong(y, p_tabpfn, p_markers)]
    holm = stats.holm(p_values)
    results["roma"] = {"n": int(measured.sum()), "cancers": int(y[measured].sum()),
                       **auc_with_ci(y[measured], roma[measured]),
                       "tabpfn_same_patients": auc_with_ci(y[measured], p_tabpfn[measured]),
                       "p": round(p_values[0], 4), "p_holm": round(holm[0], 4)}
    results["he4_ca125_menopause"] = {**auc_with_ci(y, p_markers),
                                      "p": round(p_values[1], 4), "p_holm": round(holm[1], 4)}

    # 2. without demographics
    X_reduced = X.drop(columns=DEMOGRAPHICS)
    print("TabPFN-3 without age and menopausal status:")
    p_reduced = out_of_fold(tabpfn, X_reduced, y, folds)
    results["without_demographics"] = {**auc_with_ci(y, p_reduced),
                                       "p_vs_full": round(stats.delong(y, p_tabpfn, p_reduced), 4)}

    # 3. SHAP stability
    if not args.skip_shap:
        first_repeat = folds[:10]
        for label, data in [("full", X), ("without_demographics", X_reduced)]:
            print(f"Out-of-fold SHAP ({label}):")
            values, per_fold = oof_shap(data, y, first_repeat, args.model_path, args.nsamples)
            per_fold.to_csv(out / f"oof_shap_{label}_fold_importance.csv", index=False)
            np.savez(out / f"oof_shap_{label}.npz", values=values, features=np.array(data.columns))
            results[f"shap_stability_{label}"] = stability(per_fold)

    with open(out / "revision_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

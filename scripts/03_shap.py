"""SHAP explanation of TabPFN-3 fitted to all patients (Figure 3).

Kernel SHAP with a background of 50 randomly sampled patients and 200 coalition
samples per explanation, applied to 100 randomly sampled patients.

For Figure 3 the model was loaded with the default weights of the installed
tabpfn package (no --model-path). Pass --model-path to explain the checkpoint
used for cross-validation instead.
"""

import argparse
from pathlib import Path

import numpy as np

from ovarian_tabpfn import models, plots
from ovarian_tabpfn.config import SEED
from ovarian_tabpfn.data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/Supplementary data 1.xlsx")
    parser.add_argument("--model-path")
    parser.add_argument("--background", type=int, default=50)
    parser.add_argument("--explain", type=int, default=100)
    parser.add_argument("--nsamples", type=int, default=200)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    import shap

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    X, y, _ = prepare(args.data)
    model = models.tabpfn(args.model_path).fit(X, y)

    background = shap.sample(X, args.background, random_state=SEED)
    explained = shap.sample(X, args.explain, random_state=SEED).reset_index(drop=True)
    explainer = shap.KernelExplainer(lambda d: model.predict_proba(d)[:, 1], background)
    values = np.asarray(explainer.shap_values(explained, nsamples=args.nsamples))

    features = list(X.columns)
    np.savez(out / "shap_values.npz", values=values, data=explained.to_numpy(), features=np.array(features))

    importance = np.abs(values).mean(axis=0)
    for i in np.argsort(-importance)[:10]:
        print(f"  {features[i]:<10s} {importance[i]:.4f}")

    plots.shap_figures(values, explained.to_numpy(dtype=float), features,
                       out / "fig3a_shap_importance.png", out / "fig3b_shap_beeswarm.png")


if __name__ == "__main__":
    main()

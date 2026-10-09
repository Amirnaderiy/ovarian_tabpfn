"""Model definitions. TabPFN-3 receives raw data; all other models use fold-wise mean imputation."""

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import DECISION_TREE, RANDOM_FOREST, SEED, XGBOOST

MODEL_NAMES = ["TabPFN3", "LogReg", "RandomForest", "CART", "XGBoost"]


def tabpfn(model_path: str | None = None):
    import torch
    from tabpfn import TabPFNClassifier

    model_path = model_path or os.environ.get("TABPFN_MODEL_PATH")
    kwargs = {"model_path": model_path} if model_path else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNClassifier(device=device, random_state=SEED, **kwargs)


def build(name: str, model_path: str | None = None):
    """Return a fresh, unfitted model."""
    if name == "TabPFN3":
        return tabpfn(model_path)
    if name == "LogReg":
        return make_pipeline(SimpleImputer(), StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=SEED))
    if name == "RandomForest":
        return make_pipeline(SimpleImputer(),
                             RandomForestClassifier(random_state=SEED, n_jobs=-1, **RANDOM_FOREST))
    if name == "CART":
        return make_pipeline(SimpleImputer(), DecisionTreeClassifier(random_state=SEED, **DECISION_TREE))
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return make_pipeline(SimpleImputer(), XGBClassifier(eval_metric="logloss", random_state=SEED,
                                                            n_jobs=-1, **XGBOOST))
    raise ValueError(f"unknown model: {name}")

"""Baseline for the churn data (E0.6).

Runs 5-fold stratified CV (SEED) on data/train.csv and prints the mean and std
ROC-AUC of a DummyClassifier(strategy="prior") and of the throwaway LogReg
pipeline from spike_pipeline.py. Run with PYTHONPATH=components:tools.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import FEATURES, RAW_DIR_LOCAL, SEED, TARGET
from spike_pipeline import build_pipeline


def main():
    df = pd.read_csv(f"{RAW_DIR_LOCAL}/train.csv")
    X = df[FEATURES].to_numpy(dtype=object)
    y = df[TARGET].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    models = {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "LogReg pipeline": build_pipeline(),
    }
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        print(f"{name}: ROC-AUC mean={scores.mean():.4f} std={scores.std():.4f}")


if __name__ == "__main__":
    main()

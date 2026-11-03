"""Preprocess step: drop identifiers, make the stratified train/hold-out split and
write both files to the PVC. No fitting happens here (see preprocessing.py)."""
from typing import NamedTuple

from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def preprocess(raw_path: str) -> NamedTuple(
    "Outputs", [("train_path", str), ("holdout_path", str), ("n_train", int), ("n_holdout", int)]
):
    # Imports go inside the body: KFP serializes only this function, and `config`
    # comes from the ConfigMap mounted at /code (PYTHONPATH=/code).
    import os
    from collections import namedtuple

    import pandas as pd
    from sklearn.model_selection import train_test_split

    import config

    df = pd.read_csv(raw_path)
    df = df[config.FEATURES + [config.TARGET]]  # drops id, CustomerId, Surname; fixes column order

    train_df, holdout_df = train_test_split(
        df,
        test_size=config.HOLDOUT_FRACTION,
        stratify=df[config.TARGET],
        random_state=config.SEED,
    )

    os.makedirs(config.PROCESSED_DIR_PVC, exist_ok=True)
    train_path = os.path.join(config.PROCESSED_DIR_PVC, "train.csv")
    holdout_path = os.path.join(config.PROCESSED_DIR_PVC, "holdout.csv")
    train_df.to_csv(train_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)

    print(
        f"preprocess ok: train {len(train_df)} rows ({train_df[config.TARGET].mean():.4%} positive), "
        f"holdout {len(holdout_df)} rows ({holdout_df[config.TARGET].mean():.4%} positive)"
    )
    Outputs = namedtuple("Outputs", ["train_path", "holdout_path", "n_train", "n_holdout"])
    return Outputs(train_path, holdout_path, len(train_df), len(holdout_df))

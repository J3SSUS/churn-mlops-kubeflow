"""Ingest step: read the raw train.csv from the PVC and fail the run if the data
breaks the contract. Returns the raw file path for the next step."""
from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def ingest() -> str:
    # Imports go inside the body: KFP serializes only this function, and `config`
    # comes from the ConfigMap mounted at /code (PYTHONPATH=/code).
    import os

    import pandas as pd
    from pandas.api.types import is_numeric_dtype, is_string_dtype

    import config

    path = os.path.join(config.RAW_DIR_PVC, "train.csv")
    df = pd.read_csv(path)

    categorical = [config.FEATURES[i] for i in config.CATEGORICAL_IDX]
    numeric = [config.ID_COL] + [config.FEATURES[i] for i in config.NUMERIC_IDX] + [config.TARGET]

    missing = [c for c in [config.ID_COL, *config.FEATURES, config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    bad_numeric = [c for c in numeric if not is_numeric_dtype(df[c])]
    bad_text = [c for c in categorical if not is_string_dtype(df[c]) or is_numeric_dtype(df[c])]
    if bad_numeric or bad_text:
        raise ValueError(f"wrong dtype: expected numeric {bad_numeric}, expected text {bad_text}")

    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        raise ValueError(f"nulls found: {nulls.to_dict()}")

    bad_target = sorted(set(df[config.TARGET].unique()) - {0, 1})
    if bad_target:
        raise ValueError(f"{config.TARGET} must be 0 or 1, found {bad_target}")

    print(f"ingest ok: {len(df)} rows, {df.shape[1]} columns, 0 nulls, from {path}")
    return path

"""Fetch champion step (Pipeline 2): save the registered champion as `model.joblib` on the PVC,
in a folder named after its version, where the KServe sklearn server will find it.

It reads the version number behind the alias once and then loads THAT version, so the folder
`v<N>` and the file inside it cannot disagree if the alias moves while the pipeline runs."""
from typing import NamedTuple

from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def fetch_champion() -> NamedTuple(
    "Outputs", [("model_version", str), ("model_dir", str), ("storage_uri", str)],
):
    # Imports go inside the body: KFP serializes only this function, and `config` comes from
    # the ConfigMap mounted at /code (PYTHONPATH=/code).
    import os
    from collections import namedtuple

    import joblib
    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    import config

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
    except MlflowException as e:
        raise RuntimeError(
            f"the alias '{config.CHAMPION_ALIAS}' of model '{config.MODEL_NAME}' does not exist "
            f"(has Pipeline 1 registered a champion?): {e}"
        ) from e
    version = str(champion.version)
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{version}")  # by version, not by alias

    # One folder per version, holding exactly one model file: sklearnserver takes one per folder.
    # The file is written under a name the server ignores and renamed, so a predictor pod that
    # restarts during a second run for the same version never reads a half-written file.
    model_dir = f"{config.MODELS_DIR_PVC}/{config.MODEL_NAME}/v{version}"
    os.makedirs(model_dir, exist_ok=True)
    final = f"{model_dir}/model.joblib"
    joblib.dump(model, f"{final}.tmp")
    os.replace(f"{final}.tmp", final)

    # Reload the file that was written, as the server will, on raw rows in FEATURES order.
    rows = pd.read_csv(f"{config.PROCESSED_DIR_PVC}/train.csv")[config.FEATURES].head(config.SMOKE_ROWS)
    proba = np.asarray(joblib.load(final).predict_proba(rows))
    if proba.shape != (len(rows), 2):
        raise RuntimeError(f"{final} returned predict_proba of shape {proba.shape}, expected {(len(rows), 2)}")

    # pvc://<claim>/<path inside the volume>: the path starts after the mount point of the PVC.
    relative = os.path.relpath(model_dir, os.path.dirname(config.MODELS_DIR_PVC))
    storage_uri = f"pvc://{config.PVC_CLAIM_NAME}/{relative}"
    print(f"fetch ok: {config.MODEL_NAME} v{version} -> {final}, storage_uri {storage_uri}")
    Outputs = namedtuple("Outputs", ["model_version", "model_dir", "storage_uri"])
    return Outputs(version, model_dir, storage_uri)

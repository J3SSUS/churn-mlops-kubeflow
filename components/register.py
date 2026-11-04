"""Register step: refit the winner on train + hold-out, log it in MLflow and register it as
`churn`, then move the alias `champion` to the new version. No metric is computed on the
refit model: the hold-out is now part of its training data, so the number that travels with
the version is the pre-refit one from Evaluate."""
from typing import NamedTuple

from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def register(
    winner_family: str,
    winner_params: str,
    winner_cv_auc: float,
    winner_holdout_auc: float,
    winner_run_id: str,
    kfp_run_id: str,
    train_path: str,
    holdout_path: str,
) -> NamedTuple("Outputs", [("model_version", str), ("model_uri", str)]):
    # Imports go inside the body: KFP serializes only this function, and `config` and
    # `models` come from the ConfigMap mounted at /code (PYTHONPATH=/code).
    import json
    from collections import namedtuple

    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    import config
    import models

    cfg = models.load_tuning_config()
    if winner_family not in cfg["families"]:
        raise ValueError(f"family '{winner_family}' is not in the tuning config: {list(cfg['families'])}")
    params = json.loads(winner_params)

    full = pd.concat([pd.read_csv(train_path), pd.read_csv(holdout_path)], ignore_index=True)
    X, y = full[config.FEATURES], full[config.TARGET]
    model = models.build_pipeline(cfg["families"][winner_family], params)
    model.fit(X, y)  # train + hold-out, preprocessor fitted on both

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    models.use_experiment(config.MLFLOW_EXPERIMENT)
    client = MlflowClient()

    example = X.head(5)  # raw rows in FEATURES order, as Pipeline 2 will serve them
    signature = infer_signature(example, model.predict_proba(example))
    tags = {"kfp_run_id": kfp_run_id, "kind": "register", "family": winner_family}
    with mlflow.start_run(run_name="register", tags=tags):
        mlflow.log_param("family", winner_family)
        mlflow.log_params({f"best.{k}": v for k, v in params.items()})
        mlflow.log_metric("winner_holdout_auc", winner_holdout_auc)  # the pre-refit number from Evaluate
        info = mlflow.sklearn.log_model(
            model,
            name="model",  # MLflow 3: a logged model named "model", linked to this run
            serialization_format="cloudpickle",  # skops refuses xgboost and lightgbm types (see DECISIONS.md)
            signature=signature,
            input_example=example,
            registered_model_name=config.MODEL_NAME,
        )
    version = str(info.registered_model_version)

    # Tags first, alias last: a failure in between can never leave a champion without the
    # tag that the next gate reads.
    for key, value in {
        "holdout_roc_auc": str(winner_holdout_auc),
        "winner_run_id": winner_run_id,
        "family": winner_family,
        "kfp_run_id": kfp_run_id,
    }.items():
        client.set_model_version_tag(config.MODEL_NAME, version, key, value)
    client.set_registered_model_alias(config.MODEL_NAME, config.CHAMPION_ALIAS, version)

    # Round trip through the registry, as Pipeline 2 will do it.
    champion = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
    if str(champion.version) != version:
        raise RuntimeError(f"alias '{config.CHAMPION_ALIAS}' points to version {champion.version}, expected {version}")
    loaded = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}")
    proba = np.asarray(loaded.predict_proba(X.head(5)))
    if proba.shape != (5, 2):
        raise RuntimeError(f"reloaded champion returned predict_proba of shape {proba.shape}, expected (5, 2)")

    print(f"register ok: {config.MODEL_NAME} v{version} ({winner_family}), hold-out AUC before refit {winner_holdout_auc:.4f}, "
          f"alias '{config.CHAMPION_ALIAS}' -> v{version}")
    Outputs = namedtuple("Outputs", ["model_version", "model_uri"])
    return Outputs(version, f"models:/{config.MODEL_NAME}/{version}")

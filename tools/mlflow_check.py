"""MLflow connectivity and registry checks (SPEC 03).

Runs inside a churn-base pod, with MLFLOW_TRACKING_URI pointing at the server.
Modes: connectivity (E1.3), registry (E1.4), cleanup.
"""
import argparse

import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

EXPERIMENT = "connectivity-test"
ARTIFACT_NAME = "hello.txt"
ARTIFACT_TEXT = "hello from mlflow-check"
MODEL_NAME = "registry-test"
ALIAS = "check"


def connectivity():
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_param("check", "connectivity")
        mlflow.log_metric("answer", 42.0)
        mlflow.log_text(ARTIFACT_TEXT, ARTIFACT_NAME)
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=ARTIFACT_NAME)
    with open(path) as f:
        content = f.read()
    assert content == ARTIFACT_TEXT, f"artifact mismatch: {content!r}"

    print(f"run id: {run_id}")
    print(f"artifact uri: {artifact_uri}")
    print("artifact read back: OK")


def registry():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    y = np.array([0, 0, 0, 1])
    model = DummyClassifier(strategy="prior").fit(X, y)
    proba = model.predict_proba(X)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X, proba),
            input_example=X[:2],
            registered_model_name=MODEL_NAME,
        )
    version = info.registered_model_version

    MlflowClient().set_registered_model_alias(MODEL_NAME, ALIAS, version)
    loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    diff = float(np.abs(loaded.predict_proba(X) - proba).max())
    assert diff < 1e-6, f"predict_proba differs by {diff}"

    print(f"model version: {version}")
    print(f"max absolute difference: {diff}")


def cleanup():
    client = MlflowClient()

    client.delete_registered_model(MODEL_NAME)
    print(f"deleted registered model: {MODEL_NAME}")

    experiment = client.get_experiment_by_name(EXPERIMENT)
    for run in client.search_runs([experiment.experiment_id]):
        client.delete_run(run.info.run_id)
        print(f"deleted run: {run.info.run_id}")
    client.delete_experiment(experiment.experiment_id)
    print(f"deleted experiment: {EXPERIMENT} (id {experiment.experiment_id})")


MODES = {"connectivity": connectivity, "registry": registry, "cleanup": cleanup}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    MODES[parser.parse_args().mode]()


if __name__ == "__main__":
    main()

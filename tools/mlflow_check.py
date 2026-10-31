"""MLflow connectivity and registry checks (SPEC 03).

Runs inside a churn-base pod, with MLFLOW_TRACKING_URI pointing at the server.
Modes: connectivity (E1.3).
"""
import argparse

import mlflow

EXPERIMENT = "connectivity-test"
ARTIFACT_NAME = "hello.txt"
ARTIFACT_TEXT = "hello from mlflow-check"


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


MODES = {"connectivity": connectivity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    MODES[parser.parse_args().mode]()


if __name__ == "__main__":
    main()

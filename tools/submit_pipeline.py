"""Sync the shared code, compile the training pipeline and submit a run (SPEC 04).

Needs a port-forward to the KFP API:
    kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from kfp import Client, compiler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.training_pipeline import training_pipeline  # noqa: E402

PACKAGE = ROOT / "pipelines" / "training_pipeline.yaml"
EXPERIMENT = "churn-training"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8888")
    parser.add_argument("--compile-only", action="store_true", help="compile the YAML and stop")
    parser.add_argument("--wait", action="store_true", help="wait until the run finishes")
    args = parser.parse_args()

    compiler.Compiler().compile(training_pipeline, str(PACKAGE))
    print(f"compiled: {PACKAGE.relative_to(ROOT)}")
    if args.compile_only:
        return

    subprocess.run([str(ROOT / "tools" / "sync_code.sh")], check=True)

    client = Client(host=args.host)
    run = client.create_run_from_pipeline_package(
        str(PACKAGE),
        arguments={},
        run_name=f"churn-training-{datetime.now():%Y%m%d-%H%M%S}",
        experiment_name=EXPERIMENT,
    )
    print(f"run submitted: {run.run_id}")
    if args.wait:
        result = client.wait_for_run_completion(run.run_id, timeout=900)
        print(f"run finished: {result.state}")
        sys.exit(0 if result.state == "SUCCEEDED" else 1)


if __name__ == "__main__":
    main()

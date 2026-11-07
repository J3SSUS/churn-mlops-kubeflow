"""Sync the shared code, compile a pipeline and submit a run (SPEC 04, SPEC 07).

    uv run python tools/submit_pipeline.py --wait                      # Pipeline 1, training (the default)
    uv run python tools/submit_pipeline.py --pipeline deploy --wait    # Pipeline 2, deploy

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

from pipelines.deploy_pipeline import deploy_pipeline  # noqa: E402
from pipelines.training_pipeline import training_pipeline  # noqa: E402

# name -> (pipeline function, KFP experiment, compiled package). The YAMLs are git-ignored.
PIPELINES = {
    "training": (training_pipeline, "churn-training", ROOT / "pipelines" / "training_pipeline.yaml"),
    "deploy": (deploy_pipeline, "churn-deploy", ROOT / "pipelines" / "deploy_pipeline.yaml"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="training", help="which pipeline (default: training)")
    parser.add_argument("--host", default="http://localhost:8888")
    parser.add_argument("--compile-only", action="store_true", help="compile the YAML and stop")
    parser.add_argument("--wait", action="store_true", help="wait until the run finishes")
    args = parser.parse_args()
    pipeline_func, experiment, package = PIPELINES[args.pipeline]

    compiler.Compiler().compile(pipeline_func, str(package))
    print(f"compiled: {package.relative_to(ROOT)}")
    if args.compile_only:
        return

    subprocess.run([str(ROOT / "tools" / "sync_code.sh")], check=True)

    client = Client(host=args.host)
    run = client.create_run_from_pipeline_package(
        str(package),
        arguments={},
        run_name=f"churn-{args.pipeline}-{datetime.now():%Y%m%d-%H%M%S}",
        experiment_name=experiment,
    )
    print(f"run submitted: {run.run_id}")
    if args.wait:
        result = client.wait_for_run_completion(run.run_id, timeout=900)
        print(f"run finished: {result.state}")
        sys.exit(0 if result.state == "SUCCEEDED" else 1)


if __name__ == "__main__":
    main()

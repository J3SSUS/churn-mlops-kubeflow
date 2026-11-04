"""Pipeline 1 — Training (KFP v2). SPEC 05: Ingest -> Preprocess -> Load config ->
ParallelFor(Tune per family) -> Evaluate. SPEC 06: Gate -> If(passed) Register.

Run from the repo root so `components` is importable. Compile and submit with
tools/submit_pipeline.py.
"""
from kfp import dsl, kubernetes

from components.evaluate import evaluate
from components.gate import gate
from components.ingest import ingest
from components.load_config import load_config
from components.preprocess import preprocess
from components.register import register
from components.tune import tune

PVC_NAME = "churn-data"
PVC_MOUNT = "/mnt/data"
CODE_CONFIGMAP = "churn-code"
CODE_MOUNT = "/code"


def _configure(task, cpu="100m", memory="256Mi", memory_limit="1Gi"):
    """Settings shared by every step: code from the ConfigMap, data from the PVC,
    local image, no caching (the cache key ignores the ConfigMap content), and more
    MLflow client retries for the SQLite server."""
    task.set_caching_options(False)
    task.set_cpu_request(cpu).set_memory_request(memory).set_memory_limit(memory_limit)
    task.set_env_variable("PYTHONPATH", CODE_MOUNT)
    task.set_env_variable("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "10")
    task.set_env_variable("OMP_NUM_THREADS", "1")  # one thread per pod; HGB has no n_jobs
    kubernetes.set_image_pull_policy(task, "IfNotPresent")
    kubernetes.mount_pvc(task, pvc_name=PVC_NAME, mount_path=PVC_MOUNT)
    kubernetes.use_config_map_as_volume(task, config_map_name=CODE_CONFIGMAP, mount_path=CODE_MOUNT)
    return task


@dsl.pipeline(name="churn-training", description="Churn training: ingest, preprocess, tune per family, evaluate, gate, register (SPEC 06)")
def training_pipeline():
    ingest_task = _configure(ingest())
    preprocess_task = _configure(preprocess(raw_path=ingest_task.output))
    families = _configure(load_config())
    families.after(preprocess_task)  # the spec runs it in sequence after Preprocess

    with dsl.ParallelFor(items=families.output, parallelism=2) as family:
        tune_task = _configure(
            tune(
                family=family,
                kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
                train_path=preprocess_task.outputs["train_path"],
            ),
            cpu="500m", memory="512Mi", memory_limit="1536Mi",
        )

    evaluate_task = _configure(evaluate(
        tune_results=dsl.Collected(tune_task.output),
        kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        train_path=preprocess_task.outputs["train_path"],
        holdout_path=preprocess_task.outputs["holdout_path"],
    ))

    gate_task = _configure(gate(
        winner_holdout_auc=evaluate_task.outputs["winner_holdout_auc"],
        winner_family=evaluate_task.outputs["winner_family"],
        winner_run_id=evaluate_task.outputs["winner_run_id"],
        kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    ))

    # A rejection is a normal result: the gate succeeds with passed=False and Register is skipped.
    with dsl.If(gate_task.outputs["passed"] == True):  # noqa: E712 (KFP builds the condition from ==)
        _configure(
            register(
                winner_family=evaluate_task.outputs["winner_family"],
                winner_params=evaluate_task.outputs["winner_params"],
                winner_cv_auc=evaluate_task.outputs["winner_cv_auc"],
                winner_holdout_auc=evaluate_task.outputs["winner_holdout_auc"],
                winner_run_id=evaluate_task.outputs["winner_run_id"],
                kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
                train_path=preprocess_task.outputs["train_path"],
                holdout_path=preprocess_task.outputs["holdout_path"],
            ),
            cpu="500m", memory="512Mi", memory_limit="1536Mi",
        )

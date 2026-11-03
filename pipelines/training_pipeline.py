"""Pipeline 1 — Training (KFP v2). SPEC 04: Ingest -> Preprocess only.

Run from the repo root so `components` is importable. Compile and submit with
tools/submit_pipeline.py.
"""
from kfp import dsl, kubernetes

from components.ingest import ingest
from components.preprocess import preprocess

PVC_NAME = "churn-data"
PVC_MOUNT = "/mnt/data"
CODE_CONFIGMAP = "churn-code"
CODE_MOUNT = "/code"


def _configure(task):
    """Settings shared by every step: code from the ConfigMap, data from the PVC,
    local image, no caching (the cache key ignores the ConfigMap content)."""
    task.set_caching_options(False)
    task.set_cpu_request("100m").set_memory_request("256Mi").set_memory_limit("1Gi")
    task.set_env_variable("PYTHONPATH", CODE_MOUNT)
    kubernetes.set_image_pull_policy(task, "IfNotPresent")
    kubernetes.mount_pvc(task, pvc_name=PVC_NAME, mount_path=PVC_MOUNT)
    kubernetes.use_config_map_as_volume(task, config_map_name=CODE_CONFIGMAP, mount_path=CODE_MOUNT)
    return task


@dsl.pipeline(name="churn-training", description="Churn training: ingest and preprocess (SPEC 04)")
def training_pipeline():
    ingest_task = _configure(ingest())
    _configure(preprocess(raw_path=ingest_task.output))

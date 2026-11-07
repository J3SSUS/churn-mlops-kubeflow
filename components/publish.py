"""Publish step (Pipeline 2): create or patch the KServe InferenceService `churn` so that it serves
the model folder that Fetch wrote, then wait until it is Ready.

The manifest is built here and not kept as a YAML in the repo, because the `storageUri` changes
with every model version. The runtime `churn-sklearnserver` is checked first, so a missing
runtime fails in seconds with a clear message and nothing is created."""
from typing import NamedTuple

from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def publish(storage_uri: str) -> NamedTuple("Outputs", [("endpoint", str)]):
    # Imports go inside the body: KFP serializes only this function, and `config` comes from
    # the ConfigMap mounted at /code (PYTHONPATH=/code).
    import re
    import time
    from collections import namedtuple

    from kubernetes import client
    from kubernetes import config as k8s_config
    from kubernetes.client.rest import ApiException

    import config

    match = re.search(r"/v(\d+)$", storage_uri)
    if not match:
        raise ValueError(f"storage_uri '{storage_uri}' does not end in /v<N>")
    version = match.group(1)

    namespace, name = config.K8S_NAMESPACE, config.ISVC_NAME
    manifest = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": {"churn/model-version": version},  # for humans: which version is served
        },
        "spec": {
            "predictor": {
                "minReplicas": 1,
                "model": {
                    "modelFormat": {"name": "sklearn"},
                    "runtime": config.SERVING_RUNTIME_NAME,
                    "storageUri": storage_uri,
                    "env": [{"name": "PREDICT_PROBA", "value": "true"}],
                    "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}, "limits": {"memory": "1Gi"}},
                },
            },
        },
    }

    k8s_config.load_incluster_config()  # the service account of the step pod
    api = client.CustomObjectsApi()

    def get(group, api_version, plural, object_name):
        """The object, or None if it does not exist. Any other API error is raised."""
        try:
            return api.get_namespaced_custom_object(group, api_version, namespace, plural, object_name)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    if get("serving.kserve.io", "v1alpha1", "servingruntimes", config.SERVING_RUNTIME_NAME) is None:
        raise RuntimeError(
            f"ServingRuntime '{config.SERVING_RUNTIME_NAME}' does not exist in namespace '{namespace}' "
            "(apply k8s/serving-runtime.yaml first)"
        )

    # Merge patch with the whole desired state: idempotent, and the same call that a first run would create.
    if get("serving.kserve.io", "v1beta1", "inferenceservices", name) is None:
        api.create_namespaced_custom_object("serving.kserve.io", "v1beta1", namespace, "inferenceservices", manifest)
        print(f"publish: created InferenceService {namespace}/{name} -> {storage_uri}")
    else:
        api.patch_namespaced_custom_object("serving.kserve.io", "v1beta1", namespace, "inferenceservices", name, manifest)
        print(f"publish: patched InferenceService {namespace}/{name} -> {storage_uri}")

    # Ready is not enough on its own: right after a patch the object can still report Ready from the
    # old revision. So also wait for the controller to have seen the current generation.
    deadline = time.monotonic() + config.PUBLISH_TIMEOUT_S
    while True:
        isvc = get("serving.kserve.io", "v1beta1", "inferenceservices", name) or {}
        status = isvc.get("status", {})
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in status.get("conditions", []))
        seen = status.get("observedGeneration") == isvc.get("metadata", {}).get("generation")
        if ready and seen:
            break
        if time.monotonic() >= deadline:
            conditions = [(c.get("type"), c.get("status"), c.get("reason")) for c in status.get("conditions", [])]
            raise RuntimeError(
                f"InferenceService {namespace}/{name} was not Ready within {config.PUBLISH_TIMEOUT_S}s "
                f"(generation {isvc.get('metadata', {}).get('generation')}, observedGeneration "
                f"{status.get('observedGeneration')}, conditions {conditions})"
            )
        time.sleep(5)

    endpoint = f"http://{name}-predictor.{namespace}.svc.cluster.local/v1/models/{name}:predict"
    print(f"publish ok: {namespace}/{name} is Ready (model version {version}), endpoint {endpoint}")
    Outputs = namedtuple("Outputs", ["endpoint"])
    return Outputs(endpoint)

"""Pipeline 2 — Deploy (KFP v2), SPEC 07: Fetch champion -> Publish -> Smoke test.

It takes no parameters and always serves `churn@champion`: Fetch saves that version as
`model.joblib` on the PVC, Publish creates or patches the InferenceService `churn` on that
folder, and the smoke test checks that the endpoint answers with that exact model.

Run from the repo root so `components` and `pipelines` are importable. Compile and submit with
tools/submit_pipeline.py --pipeline deploy.
"""
from kfp import dsl

from components.fetch_champion import fetch_champion
from components.publish import publish
from components.smoke_test import smoke_test
from pipelines.training_pipeline import _configure  # same task settings as Pipeline 1


@dsl.pipeline(name="churn-deploy", description="Churn deploy: fetch the champion, publish the InferenceService, smoke test (SPEC 07)")
def deploy_pipeline():
    # Fetch loads the model and fits nothing, but it holds it in memory next to the MLflow client,
    # so it asks the same as Register.
    fetch_task = _configure(fetch_champion(), cpu="500m", memory="512Mi", memory_limit="1536Mi")
    publish_task = _configure(publish(storage_uri=fetch_task.outputs["storage_uri"]))
    _configure(smoke_test(
        endpoint=publish_task.outputs["endpoint"],
        model_dir=fetch_task.outputs["model_dir"],
    ))

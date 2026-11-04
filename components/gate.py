"""Gate step: decide whether the winner of Evaluate may be registered. The rule is the frozen
contract: hold-out ROC-AUC > Dummy + GATE_MARGIN and >= the current champion's hold-out AUC
(if there is one). A rejection is a normal result, not an error."""
from typing import NamedTuple

from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def gate(winner_holdout_auc: float, winner_family: str, winner_run_id: str, kfp_run_id: str) -> NamedTuple(
    "Outputs", [("passed", bool), ("reason", str)],
):
    # Imports go inside the body: KFP serializes only this function, and `config` and
    # `models` come from the ConfigMap mounted at /code (PYTHONPATH=/code).
    from collections import namedtuple

    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    import config
    import models

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Dummy AUC: the run `dummy` that Evaluate created in this same pipeline run.
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{config.MLFLOW_EXPERIMENT}' does not exist")
    dummy_runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.kind = 'dummy' and tags.kfp_run_id = '{kfp_run_id}'",
    )
    if not dummy_runs:
        raise RuntimeError(f"no 'dummy' run for kfp_run_id '{kfp_run_id}' in experiment '{config.MLFLOW_EXPERIMENT}'")
    dummy_auc = dummy_runs[0].data.metrics["holdout_roc_auc"]

    # Champion AUC: the tag that Register wrote on the version behind the alias. A missing
    # alias (or a model that does not exist yet) means there is no champion.
    champion_auc, champion_label = None, "champion"
    try:
        version = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
    except MlflowException as e:
        if e.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise  # MLflow unreachable or another real error: fail the step
        version = None
    if version is not None:
        champion_label = f"champion v{version.version}"
        tag = (version.tags or {}).get("holdout_roc_auc")
        if tag is None:
            raise RuntimeError(
                f"{config.MODEL_NAME} version {version.version} has the alias '{config.CHAMPION_ALIAS}' "
                "but no tag 'holdout_roc_auc', so the gate cannot compare against it"
            )
        champion_auc = float(tag)

    passed, reason = models.gate_decision(winner_holdout_auc, dummy_auc, champion_auc, champion_label)
    client.set_tag(winner_run_id, "gate", "passed" if passed else "rejected")
    print(f"gate {'PASSED' if passed else 'REJECTED'} for {winner_family}: {reason}")

    Outputs = namedtuple("Outputs", ["passed", "reason"])
    return Outputs(bool(passed), reason)

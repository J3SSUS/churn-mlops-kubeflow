"""Evaluate step: refit the best config of every family on all of train.csv, score the
hold-out against a Dummy baseline, and report the winner. The winner is chosen by the
CV AUC from Tune, so the hold-out only reports and takes no part in any selection."""
from typing import List, NamedTuple

from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def evaluate(tune_results: List[str], kfp_run_id: str, train_path: str, holdout_path: str) -> NamedTuple(
    "Outputs",
    [("winner_family", str), ("winner_params", str), ("winner_cv_auc", float),
     ("winner_holdout_auc", float), ("winner_run_id", str)],
):
    # Imports go inside the body: KFP serializes only this function, and `config` and
    # `models` come from the ConfigMap mounted at /code (PYTHONPATH=/code).
    import json
    from collections import namedtuple

    import mlflow
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, recall_score, roc_auc_score

    import config
    import models

    cfg = models.load_tuning_config()
    results = {r["family"]: r for r in (json.loads(s) for s in tune_results)}
    missing = [name for name in cfg["families"] if name not in results]
    if missing:
        raise ValueError(f"no tune result for: {missing} (got {list(results)})")

    train, holdout = pd.read_csv(train_path), pd.read_csv(holdout_path)
    X_train, y_train = train[config.FEATURES], train[config.TARGET]
    X_ho, y_ho = holdout[config.FEATURES], holdout[config.TARGET]

    def score(model):
        proba = model.predict_proba(X_ho)[:, 1]
        pred = (proba >= 0.5).astype(int)  # threshold 0.5 for the label metrics, reporting only
        tn, fp, fn, tp = confusion_matrix(y_ho, pred, labels=[0, 1]).ravel()
        return {"holdout_roc_auc": roc_auc_score(y_ho, proba), "holdout_pr_auc": average_precision_score(y_ho, proba),
                "holdout_recall": recall_score(y_ho, pred, zero_division=0), "holdout_f1": f1_score(y_ho, pred, zero_division=0),
                "holdout_tn": int(tn), "holdout_fp": int(fp), "holdout_fn": int(fn), "holdout_tp": int(tp)}

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    models.use_experiment(config.MLFLOW_EXPERIMENT)

    winner, scores = None, {}
    for name, fam in cfg["families"].items():  # config order, so a tie goes to the first family
        res = results[name]
        model = models.build_pipeline(fam, res["best_params"])
        model.fit(X_train, y_train)  # all of train.csv, preprocessor fitted on it; the CV fold models are not used
        scores[name] = score(model)
        with mlflow.start_run(run_id=res["run_id"]):  # reopen the parent run that Tune closed
            mlflow.log_metrics(scores[name])
        if winner is None or res["best_cv_auc"] > results[winner]["best_cv_auc"]:
            winner = name

    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    with mlflow.start_run(run_name="dummy", tags={"kfp_run_id": kfp_run_id, "kind": "dummy"}):
        dummy_scores = score(dummy)
        mlflow.log_metrics(dummy_scores)

    for name, s in scores.items():
        mark = "  <- winner" if name == winner else ""
        print(f"{name:24} cv {results[name]['best_cv_auc']:.4f}  hold-out AUC {s['holdout_roc_auc']:.4f}{mark}")
    print(f"{'dummy':24} hold-out AUC {dummy_scores['holdout_roc_auc']:.4f}")

    Outputs = namedtuple("Outputs", ["winner_family", "winner_params", "winner_cv_auc",
                                     "winner_holdout_auc", "winner_run_id"])
    w = results[winner]
    return Outputs(winner, json.dumps(w["best_params"]), float(w["best_cv_auc"]),
                   float(scores[winner]["holdout_roc_auc"]), w["run_id"])

"""Tune step: one Optuna study for one model family, tracked in MLflow with a parent run
per family and a nested run per trial. The objective is the mean ROC-AUC of a stratified
CV over the train split only; the preprocessor is fitted inside each fold."""
from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def tune(family: str, kfp_run_id: str, train_path: str) -> str:
    # Imports go inside the body: KFP serializes only this function, and `config` and
    # `models` come from the ConfigMap mounted at /code (PYTHONPATH=/code).
    import json
    import time

    import mlflow
    import numpy as np
    import optuna
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    import config
    import models

    cfg = models.load_tuning_config()
    if family not in cfg["families"]:
        raise ValueError(f"family '{family}' is not in the tuning config: {list(cfg['families'])}")
    fam = cfg["families"][family]
    n_trials = fam.get("n_trials", cfg["n_trials"])
    cv_folds = cfg["cv_folds"]

    df = pd.read_csv(train_path)
    X, y = df[config.FEATURES], df[config.TARGET]
    splits = list(StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.SEED).split(X, y))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    for attempt in range(3):  # two pods may create the experiment at the same moment
        try:
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
            break
        except mlflow.exceptions.MlflowException:
            if attempt == 2:
                raise
            time.sleep(2)

    tags = {"kfp_run_id": kfp_run_id, "family": family}

    def objective(trial):
        params = models.suggest_params(trial, fam["search"])
        aucs, pruned = [], False
        with mlflow.start_run(run_name=f"{family}-trial-{trial.number}", nested=True,
                              tags={**tags, "kind": "trial"}):
            mlflow.log_params(params)
            for step, (fit_idx, val_idx) in enumerate(splits):
                pipe = models.build_pipeline(fam, params)
                pipe.fit(X.iloc[fit_idx], y.iloc[fit_idx])
                aucs.append(roc_auc_score(y.iloc[val_idx], pipe.predict_proba(X.iloc[val_idx])[:, 1]))
                trial.report(float(np.mean(aucs)), step)
                if trial.should_prune():
                    pruned = True
                    break
            mlflow.log_metrics({"cv_roc_auc": float(np.mean(aucs)), "cv_roc_auc_std": float(np.std(aucs)),
                                "folds_done": len(aucs)})
            mlflow.set_tag("trial_state", "PRUNED" if pruned else "COMPLETE")
        if pruned:  # raised outside the run, so MLflow ends it as FINISHED and not FAILED
            raise optuna.TrialPruned()
        return float(np.mean(aucs))

    with mlflow.start_run(run_name=family, tags={**tags, "kind": "family"}) as parent:
        mlflow.log_params({"class": fam["class"], "n_trials": n_trials, "cv_folds": cv_folds})
        mlflow.log_dict(cfg, "tuning_config.json")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED),
                                    pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)

        states = [t.state for t in study.trials]
        n_complete = states.count(optuna.trial.TrialState.COMPLETE)
        n_pruned = states.count(optuna.trial.TrialState.PRUNED)
        if n_complete == 0:
            raise RuntimeError(f"family '{family}': no trial completed ({n_pruned} pruned)")
        mlflow.log_params({f"best.{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({"best_cv_roc_auc": study.best_value, "n_complete": n_complete, "n_pruned": n_pruned})
        run_id = parent.info.run_id

    print(f"tune ok: {family} best CV AUC {study.best_value:.4f} ({n_complete} complete, {n_pruned} pruned)")
    return json.dumps({"family": family, "best_params": study.best_params,
                       "best_cv_auc": study.best_value, "run_id": run_id})

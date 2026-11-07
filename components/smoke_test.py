"""Smoke test step (Pipeline 2): post a few raw rows to the live endpoint and compare its answer
with `predict_proba` of the `model.joblib` that Fetch just wrote.

Shape and range alone would pass for a stale endpoint that still serves the previous version
(the InferenceService can report Ready before the new revision answers). The comparison with the
local file is what proves that the endpoint serves THIS version."""
from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def smoke_test(endpoint: str, model_dir: str):
    # Imports go inside the body: KFP serializes only this function, and `config` comes from
    # the ConfigMap mounted at /code (PYTHONPATH=/code). The HTTP call uses the standard library.
    import json
    import math
    import urllib.error
    import urllib.request

    import joblib
    import numpy as np
    import pandas as pd

    import config

    # The first rows of train.csv in FEATURES order, as a client sends them: numbers as floats and
    # strings as strings. The transformer picks columns by position, so the order is the contract.
    frame = pd.read_csv(f"{config.PROCESSED_DIR_PVC}/train.csv")[config.FEATURES].head(config.SMOKE_ROWS)
    instances = [
        [float(v) if i in config.NUMERIC_IDX else str(v) for i, v in enumerate(row)]
        for row in frame.to_numpy(dtype=object)
    ]

    request = urllib.request.Request(
        endpoint, data=json.dumps({"instances": instances}).encode(), headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"POST {endpoint} answered {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"POST {endpoint} failed: {e.reason}") from e
    if status != 200:
        raise RuntimeError(f"POST {endpoint} answered {status}, expected 200")

    predictions = json.loads(body).get("predictions")
    if not isinstance(predictions, list):
        raise RuntimeError(f"the answer has no 'predictions' list: {body.decode(errors='replace')[:200]}")
    expected_shape = (config.SMOKE_ROWS, 2)
    try:
        answer = np.asarray(predictions, dtype=float)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"'predictions' is not a matrix of numbers: {str(predictions)[:200]}") from e
    if answer.shape != expected_shape:
        raise RuntimeError(f"'predictions' has shape {answer.shape}, expected {expected_shape}")
    if not np.isfinite(answer).all() or answer.min() < 0 or answer.max() > 1:
        raise RuntimeError(f"'predictions' has values outside [0, 1]: {answer.tolist()}")

    # What the file on the PVC predicts for the same rows, as the server receives them.
    local = np.asarray(joblib.load(f"{model_dir}/model.joblib").predict_proba(np.array(instances, dtype=object)))
    diff = float(np.abs(answer - local).max())
    if math.isnan(diff) or diff >= config.SMOKE_TOLERANCE:
        raise RuntimeError(
            f"the endpoint does not serve {model_dir}: max |endpoint - local predict_proba| is {diff:.3e}, "
            f"tolerance {config.SMOKE_TOLERANCE:.0e} (a stale revision still answering?)"
        )
    print(f"smoke test ok: {config.SMOKE_ROWS} rows, status 200, probabilities in [0, 1], "
          f"max diff with {model_dir}/model.joblib {diff:.3e}")

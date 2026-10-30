"""Training step of the serving spike (E0.7).

Runs inside a churn-base:0.1.0 pod with PYTHONPATH=/code. Fits the throwaway
pipeline on all of train.csv, writes models/spike/v1/model.joblib to the PVC and
prints one REFERENCE line with the first 3 test.csv rows and their predict_proba,
so the endpoint answer can be compared against the training environment.
"""
import json
import os

import joblib
import pandas as pd

from config import FEATURES, MODELS_DIR_PVC, NUMERIC_IDX, RAW_DIR_PVC, TARGET
from spike_pipeline import build_pipeline

MODEL_DIR = f"{MODELS_DIR_PVC}/spike/v1"


def main():
    train = pd.read_csv(f"{RAW_DIR_PVC}/train.csv")
    model = build_pipeline().fit(train[FEATURES].to_numpy(dtype=object), train[TARGET].to_numpy())

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/model.joblib")

    test = pd.read_csv(f"{RAW_DIR_PVC}/test.csv").head(3)
    rows = test[FEATURES].to_numpy(dtype=object)
    instances = [[float(v) if i in NUMERIC_IDX else str(v) for i, v in enumerate(row)] for row in rows]
    expected = model.predict_proba(rows).tolist()
    print("REFERENCE " + json.dumps({"instances": instances, "expected": expected}))


if __name__ == "__main__":
    main()

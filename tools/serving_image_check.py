"""Check that a serving image can load a model.joblib of each family (SPEC 07, step 4).

Modes:
  fit    in a churn-base pod: fit the families of tuning_config.json on 2,000 rows of train.csv with
         models.build_pipeline, save models/image-check/<family>/model.joblib, and write
         models/image-check/expected.json with 50 raw rows and the expected predict_proba per family.
  load   in any image (churn-sklearnserver, or the original kserve/sklearnserver): load each
         model.joblib, compare predict_proba with expected.json, print one RESULT line per family.
         Exits 1 if any family fails to load or differs by SMOKE_TOLERANCE or more.
  clean  remove models/image-check from the PVC.

The code comes from the ConfigMap churn-code (PYTHONPATH=/code) and the script from the ConfigMap
serving-image-check. `load` imports only config, joblib and numpy, so it runs in the server image.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

import config

N_FIT_ROWS = 2000
N_CHECK_ROWS = 50
CHECK_DIR = "image-check"


def fit(raw_dir, models_dir, tuning_config):
    import pandas as pd

    import models

    cfg = models.load_tuning_config(tuning_config)
    train = pd.read_csv(Path(raw_dir) / "train.csv").sample(N_FIT_ROWS, random_state=config.SEED)
    X = train[config.FEATURES].to_numpy(dtype=object)
    y = train[config.TARGET].to_numpy()

    # Raw rows as the endpoint receives them: numbers as floats, strings as strings.
    instances = [
        [float(v) if i in config.NUMERIC_IDX else str(v) for i, v in enumerate(row)]
        for row in X[:N_CHECK_ROWS]
    ]
    rows = np.array(instances, dtype=object)

    base = Path(models_dir) / CHECK_DIR
    expected = {}
    for family, block in cfg["families"].items():
        model = models.build_pipeline(block, {}).fit(X, y)
        (base / family).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, base / family / "model.joblib")
        expected[family] = model.predict_proba(rows).tolist()
        print(f"FIT {family}: saved, {block['class']}")

    (base / "expected.json").write_text(json.dumps({"instances": instances, "expected": expected}))
    print(f"FIT done: {len(expected)} families in {base}")
    return 0


def load(models_dir):
    import sklearn

    print(f"ENV python {sys.version.split()[0]} | numpy {np.__version__} | scikit-learn {sklearn.__version__}")
    base = Path(models_dir) / CHECK_DIR
    ref = json.loads((base / "expected.json").read_text())
    rows = np.array(ref["instances"], dtype=object)

    failed = []
    for family, expected in ref["expected"].items():
        try:
            model = joblib.load(base / family / "model.joblib")
            proba = np.asarray(model.predict_proba(rows))
            if proba.shape != (len(rows), 2):
                raise ValueError(f"predict_proba shape {proba.shape}, expected {(len(rows), 2)}")
            diff = float(np.abs(proba - np.asarray(expected)).max())
            ok = diff < config.SMOKE_TOLERANCE
            print(f"RESULT {family} {'ok' if ok else 'MISMATCH'} max_diff={diff:.3e}")
            if not ok:
                failed.append(family)
        except Exception as exc:  # the point of the check is to see which families fail and how
            print(f"RESULT {family} error {type(exc).__name__}: {exc}")
            failed.append(family)
    print(f"SUMMARY failed={failed}")
    return 1 if failed else 0


def clean(models_dir):
    shutil.rmtree(Path(models_dir) / CHECK_DIR, ignore_errors=True)
    print(f"CLEAN removed {Path(models_dir) / CHECK_DIR}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["fit", "load", "clean"])
    parser.add_argument("--raw-dir", default=config.RAW_DIR_PVC)
    parser.add_argument("--models-dir", default=config.MODELS_DIR_PVC)
    parser.add_argument("--tuning-config", default=None, help="default: config.TUNING_CONFIG_PATH")
    args = parser.parse_args()
    if args.mode == "fit":
        return fit(args.raw_dir, args.models_dir, args.tuning_config)
    if args.mode == "load":
        return load(args.models_dir)
    return clean(args.models_dir)


if __name__ == "__main__":
    sys.exit(main())

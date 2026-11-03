"""Model helpers shared by Tune, Evaluate and (SPEC 06) Register: the tuning config
and its validation. Imported inside the component bodies, from /code in the pods."""
import importlib
import inspect
import json

from sklearn.pipeline import Pipeline

import config
from preprocessing import build_preprocessor

ALLOWED_PREFIXES = ("sklearn.", "xgboost.", "lightgbm.")
SEARCH_TYPES = ("int", "float", "categorical")
TOP_KEYS = {"n_trials", "cv_folds", "families"}
FAMILY_KEYS = {"class", "fixed", "search", "n_trials"}

# Parameters that exist but cannot be seen by introspection, because the class takes
# **kwargs. A typo anywhere else still fails; add a legitimate parameter here.
KNOWN_EXTRA_PARAMS = {"lightgbm.LGBMClassifier": {"verbose"}}


def load_tuning_config(path=None):
    """Read the tuning config (default: config.TUNING_CONFIG_PATH) and validate it."""
    with open(path or config.TUNING_CONFIG_PATH) as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _param_names(cls, class_path):
    names = set(inspect.signature(cls.__init__).parameters) - {"self", "args", "kwargs"}
    try:
        names |= set(cls().get_params())
    except Exception:  # a class that needs arguments: the signature is all we have
        pass
    return names | KNOWN_EXTRA_PARAMS.get(class_path, set())


def _check_search_entry(where, spec, errors):
    if not isinstance(spec, dict) or spec.get("type") not in SEARCH_TYPES:
        errors.append(f"{where}: 'type' must be one of {SEARCH_TYPES}")
        return
    if spec["type"] == "categorical":
        if not isinstance(spec.get("choices"), list) or not spec["choices"]:
            errors.append(f"{where}: 'choices' must be a non-empty list")
        return
    low, high = spec.get("low"), spec.get("high")
    if not (_is_number(low) and _is_number(high)):
        errors.append(f"{where}: 'low' and 'high' must be numbers")
    elif low >= high:
        errors.append(f"{where}: 'low' ({low}) must be below 'high' ({high})")
    elif spec["type"] == "int" and not (isinstance(low, int) and isinstance(high, int)):
        errors.append(f"{where}: an int range needs integer 'low' and 'high'")
    elif spec.get("log") and low <= 0:
        errors.append(f"{where}: a log range needs 'low' above 0")


def _check_family(name, block, errors):
    if not isinstance(block, dict):
        errors.append(f"family '{name}': must be an object")
        return
    for key in sorted(set(block) - FAMILY_KEYS):
        errors.append(f"family '{name}': unknown key '{key}'")
    if "n_trials" in block and not (isinstance(block["n_trials"], int) and block["n_trials"] >= 1):
        errors.append(f"family '{name}': 'n_trials' must be an integer >= 1")

    class_path = block.get("class")
    cls = None
    if not isinstance(class_path, str) or not class_path.startswith(ALLOWED_PREFIXES):
        errors.append(f"family '{name}': 'class' must start with one of {ALLOWED_PREFIXES}, got {class_path!r}")
    else:
        try:
            module, _, cls_name = class_path.rpartition(".")
            cls = getattr(importlib.import_module(module), cls_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"family '{name}': cannot import '{class_path}' ({exc})")
        else:
            if not inspect.isclass(cls) or not hasattr(cls, "predict_proba"):
                errors.append(f"family '{name}': '{class_path}' is not a class with predict_proba")
                cls = None

    fixed, search = block.get("fixed", {}), block.get("search", {})
    if not isinstance(fixed, dict) or not isinstance(search, dict):
        errors.append(f"family '{name}': 'fixed' and 'search' must be objects")
        return
    for param in sorted(set(fixed) & set(search)):
        errors.append(f"family '{name}': '{param}' is in both 'fixed' and 'search'")
    if cls is not None:
        valid = _param_names(cls, class_path)
        for param in sorted((set(fixed) | set(search)) - valid):
            errors.append(f"family '{name}': '{class_path}' has no parameter '{param}'")
    for param, spec in search.items():
        _check_search_entry(f"family '{name}', search '{param}'", spec, errors)


def validate_config(cfg):
    """Raise ValueError listing every problem of the tuning config, or return None."""
    errors = []
    if not isinstance(cfg, dict):
        raise ValueError("tuning config: the file must hold a JSON object")
    for key in sorted(set(cfg) - TOP_KEYS):
        errors.append(f"unknown key '{key}'")
    if not (isinstance(cfg.get("n_trials"), int) and cfg["n_trials"] >= 1):
        errors.append("'n_trials' must be an integer >= 1")
    if not (isinstance(cfg.get("cv_folds"), int) and cfg["cv_folds"] >= 2):
        errors.append("'cv_folds' must be an integer >= 2")
    families = cfg.get("families")
    if not isinstance(families, dict) or not families:
        errors.append("'families' must be a non-empty object")
    else:
        for name, block in families.items():
            _check_family(name, block, errors)
    if errors:
        raise ValueError("invalid tuning config:\n  - " + "\n  - ".join(errors))


def suggest_params(trial, search):
    """Ask an Optuna trial for one value per entry of a family's `search` block."""
    params = {}
    for name, spec in search.items():
        if spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return params


def build_pipeline(family_cfg, params):
    """UNFITTED Pipeline: build_preprocessor() plus the family's model, made from the
    family's `fixed` parameters and the given `params`. Sets random_state=SEED when the
    class accepts it and the config does not."""
    class_path = family_cfg["class"]
    module, _, cls_name = class_path.rpartition(".")
    cls = getattr(importlib.import_module(module), cls_name)
    kwargs = {**family_cfg.get("fixed", {}), **params}
    if "random_state" not in kwargs and "random_state" in _param_names(cls, class_path):
        kwargs["random_state"] = config.SEED
    return Pipeline([("prep", build_preprocessor()), ("clf", cls(**kwargs))])

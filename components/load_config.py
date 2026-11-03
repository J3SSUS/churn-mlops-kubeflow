"""Load-config step: read and validate the tuning config from /code and return the
family names, which feed the ParallelFor of the training pipeline."""
from typing import List

from kfp import dsl


@dsl.component(base_image="churn-base:0.3.0")
def load_config() -> List[str]:
    # Imports go inside the body: KFP serializes only this function, and `models`
    # comes from the ConfigMap mounted at /code (PYTHONPATH=/code).
    import models

    cfg = models.load_tuning_config()  # raises with every problem if the file is invalid
    families = list(cfg["families"])
    print(f"load_config ok: {len(families)} families {families}, n_trials {cfg['n_trials']}, cv_folds {cfg['cv_folds']}")
    return families

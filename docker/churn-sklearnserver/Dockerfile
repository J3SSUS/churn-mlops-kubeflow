# Serving image for the InferenceService `churn` (SPEC 07).
# kserve/sklearnserver:v0.21.0 has neither xgboost nor lightgbm, so a registered Pipeline of those
# families cannot load there. This image adds both, with the same versions as churn-base:0.3.0.
#
# Facts about the base image (docs/runbook.md, "Deploy pipeline"): it runs as uid 1000, its venv
# /prod_venv has no pip and the image has no uv, and it has no system libgomp (only a private copy
# inside scikit_learn.libs, which LightGBM and XGBoost do not use).
#
# uv comes from its own image only for the install: it is bind-mounted in one RUN, so it is not
# left in the final image and pip is not added to the venv.
FROM ghcr.io/astral-sh/uv:0.11.29 AS uv

FROM kserve/sklearnserver:v0.21.0

# libgomp1: the OpenMP runtime that LightGBM and XGBoost need. apt needs root.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# --no-deps: the list is complete and pinned, and the base already has numpy, scipy and scikit-learn.
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    uv pip install --no-cache --no-deps --python /prod_venv/bin/python -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Back to the user of the base image: the runtime sets runAsNonRoot, so the user must stay numeric.
USER 1000

# Fails the build, as the runtime user, if a library, one of its dependencies or libgomp is missing.
# (--no-deps skips the resolver, so this import is what proves the pinned list is complete.)
RUN python -c "import numpy, scipy, sklearn, xgboost, lightgbm; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| scikit-learn', sklearn.__version__, '| xgboost', xgboost.__version__, '| lightgbm', lightgbm.__version__)"

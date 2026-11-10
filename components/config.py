"""Shared contract for the churn project: frozen FEATURES order, column indices,
target, seed, PVC paths and MLflow names. Columns are always addressed by position."""
FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL_IDX = [1, 2]                      # Geography, Gender
NUMERIC_IDX = [0, 3, 4, 5, 6, 7, 8, 9]        # includes the 0/1 columns HasCrCard, IsActiveMember
TARGET = "Exited"
ID_COL = "id"
SEED = 42
RAW_DIR_LOCAL = "data"
RAW_DIR_PVC = "/mnt/data/raw"
MODELS_DIR_PVC = "/mnt/data/models"
PROCESSED_DIR_PVC = "/mnt/data/processed"
HOLDOUT_FRACTION = 0.2
MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:5000"
MLFLOW_EXPERIMENT = "churn-training"
MODEL_NAME = "churn"                          # frozen contract
CHAMPION_ALIAS = "champion"                   # frozen contract
GATE_MARGIN = 0.05                            # frozen contract: hold-out AUC must beat Dummy + 0.05
TUNING_CONFIG_PATH = "/code/tuning_config.json"   # the ConfigMap mount; tests point it elsewhere
PVC_CLAIM_NAME = "churn-data"                 # claim behind pvc:// URIs
K8S_NAMESPACE = "kubeflow"                    # where the InferenceService and the runtime live
ISVC_NAME = "churn"
SERVING_RUNTIME_NAME = "churn-sklearnserver"
PUBLISH_TIMEOUT_S = 300                       # how long Publish waits for Ready
SMOKE_ROWS = 3
SMOKE_TOLERANCE = 1e-6                        # max |endpoint - local predict_proba|

# Load generator (SPEC 09). Defaults of the arguments of tools/load_generator.py; the Job can override them.
SUBMISSION_PATH = "/mnt/data/submission.csv"  # on the PVC, written by the load generator
LOAD_BATCH_SIZE = 100                         # rows per request
LOAD_RATE_RPS = 2.0                           # requests per second, one at a time
LOAD_PASSES = 5                               # passes over test.csv; only the first keeps predictions
LOAD_RETRIES = 3                              # attempts per batch after the first one
LOAD_RETRY_PAUSE_S = 2.0                      # pause between attempts of the same batch
LOAD_METRICS_PORT = 9100                      # /metrics of the generator

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
TUNING_CONFIG_PATH = "/code/tuning_config.json"   # the ConfigMap mount; tests point it elsewhere

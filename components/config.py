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

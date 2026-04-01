import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/HAM10000"
CSV_PATH = CSV_PATH = "data/HAM10000/metadata.csv"

MODEL_PATH = "models/model.pth"
METRICS_PATH = "reports/evaluation.json"

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 1

# 🚨 TRUST CONDITIONS
ACCURACY_THRESHOLD = 0.80
FAIRNESS_THRESHOLD = 0.40
UNCERTAINTY_THRESHOLD = 0.13
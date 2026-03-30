import os

def ensure_dirs():
    for d in ["logs", "models", "reports"]:
        os.makedirs(d, exist_ok=True)
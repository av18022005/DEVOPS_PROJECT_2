import os

from src.config import MODEL_DIR, CURRENT_MODEL_FILE


def get_current_model_path():
    if not os.path.exists(CURRENT_MODEL_FILE):
        raise Exception("❌ current_model.txt not found!")

    with open(CURRENT_MODEL_FILE, "r") as f:
        model_name = f.read().strip()

    return os.path.join(MODEL_DIR, model_name)


def set_current_model(model_name):
    with open(CURRENT_MODEL_FILE, "w") as f:
        f.write(model_name)
import torch
import os

from src.config import *
from src.model import MultimodalFairNet
from src.data_loader import get_loaders
from src.preprocess import *
from src.evaluate import evaluate
from src.trust_gate import check_trust
from src.utils import ensure_dirs
from src.model_loader import get_current_model_path, set_current_model
from src.versioning import save_version


def main():

    ensure_dirs()

    # 🔥 LIMIT DATASET (avoid timeout)
    MAX_SAMPLES = 100

    # Load data
    meta = load_metadata(CSV_PATH)
    meta = add_image_paths(meta, DATA_DIR + "/images")

    if len(meta) > MAX_SAMPLES:
        meta = meta.sample(n=MAX_SAMPLES, random_state=42)

    train_df, test_df = split_data(meta)
    train_df, test_df = encode_features(train_df, test_df)

    _, test_loader = get_loaders(train_df, test_df, BATCH_SIZE)

    # 🔥 LOAD CURRENT MODEL
    current_model_path = get_current_model_path()
    print(f"🔁 Current deployed model: {current_model_path}")

    model = MultimodalFairNet()
    model.load_state_dict(torch.load(current_model_path, map_location=DEVICE))
    model.to(DEVICE)

    # Evaluate
    results = evaluate(model, test_loader, DEVICE)

    # Save version
    version_id = save_version(results, current_model_path)
    print(f"📌 Version saved: v{version_id}")

    # Trust decision
    decision = check_trust(
        results["accuracy"],
        results["fairness_gap"],
        results["uncertainty"]
    )

    print("FINAL DECISION:", decision)

    if decision == "REJECT":
        print("⛔ Keeping previous model (Rollback)")
        exit(1)

    print("🚀 Model Approved for Deployment")


if __name__ == "__main__":
    main()
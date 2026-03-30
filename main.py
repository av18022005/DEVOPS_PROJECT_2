import torch

# 📁 Project imports (with src structure)
from src.config import *
from src.utils import ensure_dirs

from src.preprocess import (
    load_metadata,
    add_image_paths,
    split_data,
    encode_features
)

from src.data_loader import get_loaders
from src.model import MultimodalFairNet
from src.train import run_pipeline


def main():

    print("🚀 Starting Full ML Pipeline...\n")

    # ✅ Step 0: Ensure folders exist
    ensure_dirs()

    # ✅ Step 1: Load metadata
    print("📊 Loading metadata...")
    meta = load_metadata(CSV_PATH)

    # ✅ Step 2: Add image paths
    print("🖼️ Resolving image paths...")
    IMAGE_DIR = DATA_DIR + "/images"
    meta = add_image_paths(meta, IMAGE_DIR)

    print(f"✅ Total usable samples: {len(meta)}\n")

    # ✅ Step 3: Train/Test split
    print("🔀 Splitting dataset...")
    train_df, test_df = split_data(meta)

    # ✅ Step 4: Encode categorical features
    print("🔢 Encoding features...")
    train_df, test_df = encode_features(train_df, test_df)

    # ✅ Step 5: Create DataLoaders
    print("📦 Creating DataLoaders...")
    train_loader, test_loader = get_loaders(
        train_df, test_df, BATCH_SIZE
    )

    # ✅ Step 6: Initialize model
    device = DEVICE
    print(f"\n🧠 Using device: {device}")

    model = MultimodalFairNet().to(device)

    # ✅ Step 7: Run FULL pipeline (TRAIN + EVAL + TRUST GATE)
    print("\n⚙️ Running Training + Evaluation Pipeline...")
    decision = run_pipeline(
        model,
        train_loader,
        test_loader,
        device
    )

    print("\n🎯 FINAL DECISION:", decision)
    print("\n✅ Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
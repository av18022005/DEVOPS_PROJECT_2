import torch
import torch.nn as nn
import torch.optim as optim

from src.config import *
from src.model import MultimodalFairNet
from src.data_loader import get_loaders
from src.preprocess import *
from src.utils import ensure_dirs

def train():

    ensure_dirs()

    # Load and preprocess data
    meta = load_metadata(CSV_PATH)
    meta = add_image_paths(meta, DATA_DIR + "/images")

    train_df, test_df = split_data(meta)
    train_df, test_df = encode_features(train_df, test_df)

    train_loader, _ = get_loaders(train_df, test_df, BATCH_SIZE)

    # Model
    model = MultimodalFairNet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0

        for imgs, meta, labels, _ in train_loader:
            imgs, meta, labels = imgs.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs, _, _, _ = model(imgs, meta)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    train()
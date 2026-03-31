import torch
import torch.nn as nn

from src.evaluate import evaluate
from src.trust_gate import check_trust
from src.versioning import save_version
from src.config import *


def train_model(model, train_loader, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for imgs, metas, labels, skins in train_loader:
            imgs = imgs.to(device)
            metas = metas.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, _, _, _ = model(imgs, metas)

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    return model


def run_pipeline(model, train_loader, test_loader, device="cpu"):

    # 🏋️ TRAIN
    print("\n🏋️ Training Model...")
    model = train_model(model, train_loader, device)

    # 🔍 EVALUATE
    print("\n🔍 Evaluating Model...")
    results = evaluate(model, test_loader, device)

    print("\n📊 Evaluation Results:")
    print(results)

    # 🧠 TRUST GATE
    decision = check_trust(
        results["accuracy"],
        results["fairness_gap"],
        results["uncertainty"]
    )

    # 🚀 DEPLOYMENT
    if decision == "DEPLOY":

        # ✅ Save model correctly
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)

        # ✅ Save version with SAME path
        save_version(results, MODEL_PATH)

        print("✅ Model Deployed Successfully")

    else:
        print("❌ Model Rejected due to Trust Constraints")

    return decision
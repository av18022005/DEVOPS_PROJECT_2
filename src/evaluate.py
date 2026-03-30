import torch
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, roc_auc_score

from src.fairness import compute_fairness_gap
from src.uncertainty import compute_entropy
from src.config import *


def evaluate(model, loader, device="cpu"):
    model.eval()

    all_probs, all_preds, all_labels, all_skins = [], [], [], []

    with torch.no_grad():
        for imgs, metas, labels, skins in loader:
            imgs, metas = imgs.to(device), metas.to(device)

            logits, _, _, _ = model(imgs, metas)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_skins.extend(skins)

    all_probs = np.vstack(all_probs)

    # ✅ Accuracy
    acc = accuracy_score(all_labels, all_preds)

    # ✅ AUC
    try:
        auc = roc_auc_score(
            np.eye(len(set(all_labels)))[all_labels],
            all_probs,
            multi_class="ovr"
        )
    except:
        auc = None

    # ✅ Fairness
    fairness_gap = compute_fairness_gap(
        all_preds, all_labels, all_skins
    )

    # ✅ Uncertainty
    entropy = compute_entropy(all_probs)

    results = {
        "accuracy": float(acc),
        "auc": float(auc) if auc else None,
        "fairness_gap": float(fairness_gap),
        "uncertainty": float(entropy)
    }

    # ✅ SAVE REPORT (IMPORTANT FOR YOUR PROJECT)
    os.makedirs("reports", exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    return results
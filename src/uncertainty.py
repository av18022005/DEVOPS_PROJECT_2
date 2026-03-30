import torch
import numpy as np

def mc_dropout(model, imgs, metas, T=5):
    model.train()  # activate dropout

    preds = []
    for _ in range(T):
        logits, _, _, _ = model(imgs, metas)
        probs = torch.softmax(logits, dim=1)
        preds.append(probs.unsqueeze(0))

    preds = torch.cat(preds, dim=0)
    mean_pred = preds.mean(dim=0)
    var_pred = preds.var(dim=0)

    return mean_pred, var_pred.mean().item()


def compute_entropy(probs):
    return -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1))
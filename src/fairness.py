# fairness.py
import numpy as np
from collections import defaultdict

def compute_fairness_gap(preds, labels, skins):
    group_correct = defaultdict(int)
    group_total = defaultdict(int)

    for p, l, s in zip(preds, labels, skins):
        group_total[s] += 1
        if p == l:
            group_correct[s] += 1

    group_acc = []
    for g in group_total:
        acc = group_correct[g] / group_total[g]
        group_acc.append(acc)

    return max(group_acc) - min(group_acc)
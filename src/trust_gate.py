# trust_gate.py
from config import *

def check_trust(acc, gap, uncertainty):

    if acc < ACCURACY_THRESHOLD:
        print("❌ Low accuracy")
        return "REJECT"

    if gap > FAIRNESS_THRESHOLD:
        print("❌ Bias detected")
        return "REJECT"

    if uncertainty > UNCERTAINTY_THRESHOLD:
        print("❌ High uncertainty")
        return "REJECT"

    print("✅ Model Approved")
    return "DEPLOY"
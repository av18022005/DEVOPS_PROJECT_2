import torch

from src.config import *
from src.model import MultimodalFairNet
from src.data_loader import get_loaders
from src.preprocess import *
from src.evaluate import evaluate
from src.trust_gate import check_trust
from src.utils import ensure_dirs

def main():

    ensure_dirs()

    # Load data
    meta = load_metadata(CSV_PATH)
    meta = add_image_paths(meta, DATA_DIR + "/images")

    train_df, test_df = split_data(meta)
    train_df, test_df = encode_features(train_df, test_df)

    _, test_loader = get_loaders(train_df, test_df, BATCH_SIZE)

    # Load pretrained model
    model = MultimodalFairNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Evaluate
    results = evaluate(model, test_loader, DEVICE)

    print("Accuracy:", results["accuracy"])
    print("Fairness gap:", results["fairness_gap"])
    print("Uncertainty:", results["uncertainty"])

    # Trust decision
    decision = check_trust(
        results["accuracy"],
        results["fairness_gap"],
        results["uncertainty"]
    )

    print("FINAL DECISION:", decision)

    if decision == "REJECT":
        exit(1)

if __name__ == "__main__":
    main()
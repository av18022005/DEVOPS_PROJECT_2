import json
import os

VERSION_FILE = "logs/versions.json"


def save_version(results, model_path):

    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            versions = json.load(f)
    else:
        versions = []

    version_id = len(versions) + 1

    entry = {
        "version": version_id,
        "model_path": model_path,
        "metrics": results
    }

    versions.append(entry)

    with open(VERSION_FILE, "w") as f:
        json.dump(versions, f, indent=4)

    return version_id


def get_best_model():
    if not os.path.exists(VERSION_FILE):
        return None

    with open(VERSION_FILE) as f:
        versions = json.load(f)

    best = max(versions, key=lambda x: x["metrics"]["accuracy"])
    return best["model_path"]
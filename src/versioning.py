# versioning.py
import json
import os

def save_version(results, model_path):
    version_file = "logs/versions.json"

    if os.path.exists(version_file):
        with open(version_file, "r") as f:
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

    with open(version_file, "w") as f:
        json.dump(versions, f, indent=4)

    return version_id


def compare_versions():
    if not os.path.exists("logs/versions.json"):
        return None

    with open("logs/versions.json") as f:
        versions = json.load(f)

    best = max(versions, key=lambda x: x["metrics"]["accuracy"])
    return best
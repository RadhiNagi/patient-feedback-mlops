"""
Simple Model Registry - tracks model versions and metrics
"""

import json
import os
from datetime import datetime


def register_model(metrics, registry_path="models/model_registry.json"):
    """Register a new model version with its metrics"""

    # Load existing registry or create new
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    # Create new version entry
    version = len(registry["models"]) + 1
    entry = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "accuracy": metrics["accuracy"],
        "f1_weighted": metrics["f1_weighted"],
        "f1_per_class": metrics["f1_per_class"],
        "status": "active"
    }

    # Mark previous models as archived
    for model in registry["models"]:
        model["status"] = "archived"

    registry["models"].append(entry)
    registry["current_version"] = version

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\n📋 Model v{version} registered!")
    print(f"   Accuracy: {metrics['accuracy']}")
    print(f"   F1 Score: {metrics['f1_weighted']}")

    return version


if __name__ == "__main__":
    # Test with current metrics
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        register_model(metrics)
    else:
        print("No metrics found. Train model first.")
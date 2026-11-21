"""
One-time script: logs existing model weights into MLflow and registers them.
Run once to seed the Model Registry with current production models.

Usage:
    python D:\audit\register_models.py
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

HERE = os.path.dirname(os.path.abspath(__file__))
DB   = f"sqlite:///{os.path.join(HERE, 'mlflow.db')}"

mlflow.set_tracking_uri(DB)
client = MlflowClient(tracking_uri=DB)

MODELS = [
    {
        "name":        "GogglesDetector",
        "path":        os.path.join(HERE, "models", "goggles_model.pt"),
        "experiment":  "model-registry-seed",
        "description": "Custom YOLOv8n goggles/sunglasses detector",
        "tags":        {"type": "yolo", "version": "v1", "classes": "specs,goggles"},
    },
    {
        "name":        "GenderClassifier",
        "path":        os.path.join(HERE, "models", "best_convnext_large.pth"),
        "experiment":  "model-registry-seed",
        "description": "ConvNeXt-Large gender classifier (male/female)",
        "tags":        {"type": "convnext", "version": "v1", "classes": "male,female"},
    },
    {
        "name":        "AppearanceModel",
        "path":        os.path.join(HERE, "models", "appearance_model.pt"),
        "experiment":  "model-registry-seed",
        "description": "CLIP-based appearance category classifier (8 classes)",
        "tags":        {"type": "clip", "version": "v1"},
    },
    {
        "name":        "OIV7Detector",
        "path":        os.path.join(HERE, "models", "yolov8m-oiv7.pt"),
        "experiment":  "model-registry-seed",
        "description": "YOLOv8m OIV7 object detector — filters non-person images",
        "tags":        {"type": "yolo", "version": "oiv7"},
    },
]


def _ensure_registered(name: str, description: str):
    """Create the registered model entry if it doesn't exist yet."""
    try:
        client.get_registered_model(name)
        print(f"  Registry entry already exists: {name}")
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(name, description=description)
        print(f"  Created registry entry: {name}")


def register(entry: dict) -> str:
    mlflow.set_experiment(entry["experiment"])
    _ensure_registered(entry["name"], entry["description"])

    with mlflow.start_run(run_name=f"seed_{entry['name']}") as run:
        mlflow.set_tags(entry.get("tags", {}))
        mlflow.log_artifact(entry["path"], artifact_path="weights")

        model_uri = f"runs:/{run.info.run_id}/weights/{os.path.basename(entry['path'])}"
        mv = client.create_model_version(
            name=entry["name"],
            source=f"runs:/{run.info.run_id}/weights",
            run_id=run.info.run_id,
            description=entry["description"],
        )

    # promote straight to Production (these are the current working models)
    client.set_registered_model_alias(entry["name"], "production", mv.version)
    print(f"  v{mv.version} → alias 'production'")
    return mv.version


print(f"\nMLflow registry seed — {DB}\n{'─'*50}")
for m in MODELS:
    print(f"\n[{m['name']}]  {m['path']}")
    if not os.path.exists(m["path"]):
        print(f"  SKIP — file not found")
        continue
    v = register(m)
    print(f"  Done: {m['name']} v{v}")

print(f"\n{'─'*50}")
print("Registry seeded. Open http://localhost:5000 → Models tab to view.\n")

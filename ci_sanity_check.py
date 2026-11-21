"""
Fast CI sanity check — runs in ~10 seconds.
Verifies imports work and models are accessible without loading them all.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Checking Python imports...")
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import mlflow
print("  core imports OK")

print("Checking model files exist...")
HERE = os.path.dirname(os.path.abspath(__file__))
models = [
    "models/goggles_model.pt",
    "models/best_convnext_large.pth",
    "models/appearance_model.pt",
    "models/yolov8m-oiv7.pt",
]
for m in models:
    path = os.path.join(HERE, m)
    if not os.path.exists(path):
        print(f"  MISSING: {m}")
        sys.exit(1)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  OK  {m}  ({size_mb:.0f} MB)")

print("Checking dataset exists...")
test_dir = os.path.join(HERE, "dataset", "test")
if not os.path.isdir(test_dir):
    print(f"  MISSING: dataset/test")
    sys.exit(1)
cats = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
total_imgs = sum(
    len([f for f in os.listdir(os.path.join(test_dir, c))
         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    for c in cats
)
print(f"  OK  {len(cats)} categories, {total_imgs} images")

print("Checking MLflow registry...")
try:
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=f"sqlite:///{os.path.join(HERE, 'mlflow.db')}")
    for name in ["GogglesDetector", "GenderClassifier", "AppearanceModel", "OIV7Detector"]:
        mv = client.get_model_version_by_alias(name, "production")
        print(f"  OK  {name} v{mv.version} @ production")
except Exception as e:
    print(f"  WARNING: MLflow check failed: {e}")

print("\nSanity check PASSED")

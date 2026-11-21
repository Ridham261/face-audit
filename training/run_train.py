from pathlib import Path
from ultralytics import YOLO
import mlflow

HERE     = Path(r"D:\audit")
yaml     = HERE / "yolo_glasses_dataset" / "data.yaml"

EPOCHS  = 30
IMGSZ   = 320
BATCH   = 32
DEVICE  = "cpu"

mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
mlflow.set_experiment("goggles-detector")

with mlflow.start_run():
    mlflow.log_params({
        "epochs":    EPOCHS,
        "imgsz":     IMGSZ,
        "batch":     BATCH,
        "device":    DEVICE,
        "base_model": "yolov8n.pt",
        "dataset":   "goggles_glasses_1251imgs",
    })

    model   = YOLO("yolov8n.pt")
    results = model.train(
        data    = str(yaml),
        epochs  = EPOCHS,
        imgsz   = IMGSZ,
        batch   = BATCH,
        name    = "glasses_detector",
        project = str(HERE / "runs"),
        patience= 10,
        workers = 0,
        device  = DEVICE,
        verbose = True,
    )

    # log final epoch metrics from YOLO results dict
    rd = results.results_dict or {}
    mlflow.log_metrics({
        "mAP50":     rd.get("metrics/mAP50(B)",    0),
        "mAP50_95":  rd.get("metrics/mAP50-95(B)", 0),
        "precision": rd.get("metrics/precision(B)", 0),
        "recall":    rd.get("metrics/recall(B)",    0),
        "fitness":   rd.get("fitness",              0),
    })

    best = HERE / "runs" / "glasses_detector" / "weights" / "best.pt"
    if best.exists():
        mlflow.log_artifact(str(best), artifact_path="weights")

    print(f"\nDone. Best weights: {best}")
    print(f"MLflow run: {mlflow.active_run().info.run_id}")

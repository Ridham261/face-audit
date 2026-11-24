from pathlib import Path
from ultralytics import YOLO
import mlflow
from mlflow.tracking import MlflowClient

HERE     = Path(r"D:\audit")
yaml     = HERE / "yolo_glasses_dataset" / "data.yaml"

EPOCHS            = 30
IMGSZ             = 320
BATCH             = 32
DEVICE            = "cpu"
PROMOTE_THRESHOLD = 0.90   # mAP50 must exceed this to go to Production

DB = f"sqlite:///{HERE / 'mlflow.db'}"
mlflow.set_tracking_uri(DB)
mlflow.set_experiment("goggles-detector")
client = MlflowClient(tracking_uri=DB)

with mlflow.start_run() as run:
    mlflow.log_params({
        "epochs":     EPOCHS,
        "imgsz":      IMGSZ,
        "batch":      BATCH,
        "device":     DEVICE,
        "base_model": "yolov8n.pt",
        "dataset":    "goggles_glasses_1251imgs",
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

    rd     = results.results_dict or {}
    map50  = rd.get("metrics/mAP50(B)", 0)
    mlflow.log_metrics({
        "mAP50":     map50,
        "mAP50_95":  rd.get("metrics/mAP50-95(B)", 0),
        "precision": rd.get("metrics/precision(B)", 0),
        "recall":    rd.get("metrics/recall(B)",    0),
        "fitness":   rd.get("fitness",              0),
    })

    best = HERE / "runs" / "glasses_detector" / "weights" / "best.pt"
    if best.exists():
        mlflow.log_artifact(str(best), artifact_path="weights")

        # register new version in MLflow Model Registry
        try:
            client.get_registered_model("GogglesDetector")
        except mlflow.exceptions.MlflowException:
            client.create_registered_model("GogglesDetector",
                description="Custom YOLOv8n goggles/sunglasses detector")

        mv = client.create_model_version(
            name="GogglesDetector",
            source=f"runs:/{run.info.run_id}/weights",
            run_id=run.info.run_id,
        )
        print(f"\nRegistered: GogglesDetector v{mv.version}  (mAP50={map50:.4f})")

        if map50 >= PROMOTE_THRESHOLD:
            client.set_registered_model_alias("GogglesDetector", "production", mv.version)
            print(f"Promoted to Production (mAP50 {map50:.4f} >= {PROMOTE_THRESHOLD})")
        else:
            print(f"NOT promoted — mAP50 {map50:.4f} < threshold {PROMOTE_THRESHOLD}")

    print(f"\nDone. Best weights: {best}")
    print(f"MLflow run: {run.info.run_id}")

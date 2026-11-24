"""
Goggles Wrapper API - Port 8000
1. Receives image on POST /validate
2. Runs custom YOLO goggles model (class 1 = goggles/sunglasses)
3. Rejects immediately if goggles detected (conf >= 0.50)
4. Otherwise forwards to face validation engine on port 8001

Run with:
    conda activate face-validation && python D:\audit\goggles_wrapper_api.py
"""
import io
import os

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

import mlflow
from mlflow.tracking import MlflowClient
from monitoring.prediction_logger import log as _log_prediction

HERE           = os.path.dirname(os.path.abspath(__file__))
_FALLBACK_PATH = os.path.join(HERE, "models", "goggles_model.pt")
_MLFLOW_DB     = f"sqlite:///{os.path.join(HERE, 'mlflow.db')}"

VALIDATION_URL       = "http://localhost:8001/validate"
GOGGLES_CLASS_ID     = 1
GOGGLES_CONF_THRESHOLD = 0.50


def _load_goggles_model() -> YOLO:
    """Load GogglesDetector@production from MLflow registry; fall back to local file."""
    mlflow.set_tracking_uri(_MLFLOW_DB)
    try:
        client   = MlflowClient(tracking_uri=_MLFLOW_DB)
        alias_mv = client.get_model_version_by_alias("GogglesDetector", "production")
        uri      = mlflow.artifacts.download_artifacts(
            f"models:/GogglesDetector@production",
            tracking_uri=_MLFLOW_DB,
        )
        # artifact dir contains the .pt file
        pt_files = [f for f in os.listdir(uri) if f.endswith(".pt")]
        if pt_files:
            model_path = os.path.join(uri, pt_files[0])
            print(f"Loading GogglesDetector v{alias_mv.version} from MLflow registry")
            return YOLO(model_path)
    except Exception as exc:
        print(f"Registry load failed ({exc}) — falling back to local file")

    if not os.path.exists(_FALLBACK_PATH):
        raise FileNotFoundError(f"Goggles model not found: {_FALLBACK_PATH}")
    print(f"Loading goggles model from local file: {_FALLBACK_PATH}")
    return YOLO(_FALLBACK_PATH)


goggles_model = _load_goggles_model()
print("Goggles model loaded OK")

app = FastAPI(title="Goggles Wrapper API", version="1.0")


def _check_goggles(bgr: np.ndarray) -> float:
    """Return highest confidence goggles detection, 0.0 if none."""
    results = goggles_model(bgr, verbose=False)
    best_conf = 0.0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == GOGGLES_CLASS_ID:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
    return best_conf


@app.post("/validate")
async def validate_face(file: UploadFile = File(...)):
    contents = await file.read()
    pil = ImageOps.exif_transpose(
        Image.open(io.BytesIO(contents))
    ).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    goggles_conf = _check_goggles(bgr)

    if goggles_conf >= GOGGLES_CONF_THRESHOLD:
        _log_prediction("rejected", "goggles", False, 0.0)
        return JSONResponse(content={
            "status": "rejected",
            "is_valid": 0,
            "audit_category": "goggles",
            "result": {
                "code": "goggles",
                "title": "Sunglasses / Goggles Detected",
                "message": "Photo rejected: sunglasses or goggles detected."
            },
            "checks": {
                "no_sunglasses": {
                    "ok": False,
                    "note": f"Goggles detected (conf={goggles_conf:.2f})"
                }
            },
            "warnings": ["Goggles or sunglasses detected by custom YOLO model"],
            "scores": {"checks": {"goggles": round(goggles_conf, 4)}},
            "goggles_check": {
                "detected": True,
                "confidence": round(goggles_conf, 4),
                "model": "custom_yolo_goggles_v1"
            }
        })

    # No goggles — forward to face validation engine
    try:
        resp = requests.post(
            VALIDATION_URL,
            files={"file": (file.filename or "image.jpg", contents,
                            file.content_type or "image/jpeg")},
            timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Face validation server on port 8001 is not running. "
                   "Start it first: conda activate face-validation && "
                   "python D:\\audit\\validation_server_310.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result["goggles_check"] = {
        "detected": False,
        "confidence": round(goggles_conf, 4),
        "model": "custom_yolo_goggles_v1"
    }
    _log_prediction(
        status=result.get("status", ""),
        result_code=(result.get("result") or {}).get("code", ""),
        is_valid=result.get("is_valid", False),
        processing_ms=result.get("processing_ms", 0.0),
        appearance_label=(result.get("appearance") or {}).get("label", ""),
        gender=(result.get("face") or {}).get("gender", ""),
    )
    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "ok", "service": "goggles-wrapper", "port": 8000}


if __name__ == "__main__":
    print("Starting Goggles Wrapper API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

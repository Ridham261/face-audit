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

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "goggles_model.pt")
VALIDATION_URL = "http://localhost:8001/validate"
GOGGLES_CLASS_ID = 1       # class 0 = specs (OK), class 1 = goggles (reject)
GOGGLES_CONF_THRESHOLD = 0.50

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Goggles model not found: {MODEL_PATH}")

print(f"Loading goggles model from: {MODEL_PATH}")
goggles_model = YOLO(MODEL_PATH)
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
                   "Start it first: conda activate faceswapih && "
                   "python D:\\audit\\validation_server_310.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result["goggles_check"] = {
        "detected": False,
        "confidence": round(goggles_conf, 4),
        "model": "custom_yolo_goggles_v1"
    }
    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "ok", "service": "goggles-wrapper", "port": 8000}


if __name__ == "__main__":
    print("Starting Goggles Wrapper API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

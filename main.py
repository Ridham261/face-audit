"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║              FACE VALIDATION ENGINE  —  main.py  —  v2.5.0                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

OVERVIEW
────────
A production-ready FastAPI service that performs multi-layer facial validation on
uploaded images and CSV bulk inputs. Every image passes through a strict priority
chain of checks and is assigned exactly one category, mapped to a meaningful HTTP
status code so callers can act on the result directly.

VALIDATION PRIORITY CHAIN
──────────────────────────
Each image is evaluated in this order — the first failing check wins:

  1. Dimension Guard     → image too small or too large                → irrelevant
  2. No Human           → MTCNN found no face above confidence 0.90   → no_human
  3. Irrelevant         → no face AND skin coverage < 3 %             → irrelevant
  4. Multiple Person    → MediaPipe or FaceMesh detected > 1 face     → multiple_person
  5. No Landmarks       → MTCNN found faces but FaceMesh returned none → no_landmarks
  6. Sunglasses         → dark goggles/sunglasses detected on any face → sunglasses
  7. Eyes Closed        → both eyes fully shut (EAR two-gate check)   → eyes_closed
  8. Pose               → all remaining faces classified as Frontal or Side
                           all frontal          → front   (HTTP 200 ✅)
                           all side             → side    (HTTP 400)
                           mix of both          → mixed   (HTTP 400)

OUTPUT CATEGORIES & HTTP STATUS CODES
──────────────────────────────────────
  front           200   Valid frontal face — the only passing result
  side            400   Face is turned to the side
  mixed           400   Some faces frontal, some side
  eyes_closed     422   Both eyes fully shut
  sunglasses      423   Dark goggles or sunglasses detected
  multiple_person 409   More than one person in the image
  no_human        404   No human face detected
  irrelevant      415   Image contains no human content at all
  no_landmarks    422   MTCNN detected a face but MediaPipe could not extract landmarks

MODULE BREAKDOWN
────────────────
  MODULE 0 — Image Pre-processing
      validate_image_dimensions()   Rejects images outside 1×1 to 8000×8000 px.
      _resize_for_processing()      Proportionally downscales the longest side to
                                    1600 px for all MediaPipe operations, returning
                                    the frame and a scale factor for coordinate mapping.

  MODULE 1 — No Human / Irrelevant Detection
      _skin_ratio()                 Computes the fraction of pixels that fall within
                                    HSV skin-tone ranges covering light and dark skin.
      detect_no_human()             Returns is_human=False when no MTCNN bbox passes
                                    the confidence + size filter chain.
      detect_irrelevant_image()     Distinguishes no_human (skin present but no face)
                                    from irrelevant (no face AND very low skin coverage).

  MODULE 2 — Multiple Person Detection
      detect_persons_and_size()     Runs MediaPipe FaceDetection on the downscaled frame.
                                    Returns multiple_person if more than one face is found.

  MODULE 3 — Sunglasses / Dark-Goggle Detection
      _classical_sunglasses_crop_evidence() Detects aligned dark lens regions in an enlarged eye band.
      _extract_eye_roi()            Extracts a tight BGR/HSV crop around each eye polygon.
      detect_sunglasses()           YOLO on face + enlarged eye-band, paired-lens fallback,
                                    then per-eye and upper-face-band corroboration:
                                      S5 — iris guard (hard veto if both eyes show iris)
                                      S1 — mean grey brightness  < SG_BRIGHTNESS_MAX
                                      S2 — grey std-dev contrast < SG_CONTRAST_MAX
                                      S3 — HSV-V channel mean    < SG_HSV_V_MAX
                                      S4 — dark pixel coverage   >= SG_DARK_COVERAGE_MIN
                                    All 4 votes (S1–S4) must pass for detection.
                                    Both eyes must be measurable (SG_REQUIRE_BOTH_EYES).

  MODULE 4 — Closed-Eye Detection (EAR)
      _compute_ear()                Eye Aspect Ratio = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
                                    using 6 Soukupova-Cech landmarks per eye.
      _estimate_open_ear_baseline() Estimates the expected open-eye EAR from horizontal
                                    eye width × EAR_OPEN_BASELINE_FACTOR. Person-neutral,
                                    no prior frame required.
      detect_closed_eyes()          Two-gate system (both must pass):
                                      Gate 1 — absolute: both EARs < 0.15 (deeply shut)
                                      Gate 2 — relative: both EARs <= 35% of their baseline
                                    Skips faces smaller than EAR_MIN_RESOLUTION (80 px) to
                                    avoid unreliable measurements on tiny detections.

  MODULE 5 — Head-Pose Estimation
      _np_angle()                   Angle in degrees at vertex b between rays ba and bc.
      _estimate_focal_length()      Focal length via 70-degree diagonal-FOV heuristic,
                                    rotation-invariant across portrait/landscape images.
      _get_mediapipe_angles()       Full SolvePnP head-pose: maps 6 FaceMesh landmarks
                                    to 3-D reference model, runs SOLVEPNP_SQPNP,
                                    applies 180-degree flip guard, returns yaw/pitch/roll.
      _get_nose_visibility_ratio()  Nose-wing width / cheek-to-cheek width ratio.
                                    ~0.25–0.30 for frontal, < 0.08 for full side profile.
      classify_pose_strong()        Weighted 4-signal pose classifier:
                                      S1 (×1.0) — MTCNN eye-to-nose angle range
                                      S2 (×1.5) — left/right angle asymmetry
                                      S3 (×2.5) — MediaPipe yaw (abstains if unavailable)
                                      S4 (×1.5) — nose visibility ratio
                                    total_score >= 0 → Frontal, else → Side.
                                    Confidence = |score| / dynamic_max × 100.

  MODULE 6 — MTCNN ↔ MediaPipe Face Matching
      _match_landmarks_to_faces()   For each MTCNN bbox, finds the nearest FaceMesh face
                                    by Euclidean distance from bbox centre to nose-tip
                                    landmark, with coordinate scaling to proc-frame space.

  CORE PIPELINE
      run_full_validation()         Orchestrates all modules in priority order.
                                    Accepts a PIL image + OpenCV BGR frame.
                                    Returns a structured dict with category, face list,
                                    per-face pose/eye/sunglasses detail, and timing.

  ANNOTATION
      draw_annotations()            Draws bounding boxes, landmark dots, triangle lines,
                                    pose score, EAR value, yaw/pitch/roll, and a summary
                                    banner onto the frame using colour-coded overlays.

API ENDPOINTS
─────────────
  Single-Image
    POST /validate-face             Validate one uploaded image (JPEG/PNG).
                                    Optional ?save=true saves annotated output to disk.
                                    Returns JSON with category, faces, and processing_ms.
    GET  /get-image                 Retrieve a saved output image by filename + category.
    GET  /get-latest                Return the most recently saved image for a category.
    GET  /list-outputs              List all saved images grouped by category with counts.
    GET  /health                    Service health check — version, device, GPU status.

  Folder Batch
    POST /validate-folder           Validate every JPEG/PNG in a local folder path.
                                    Skips files > 25 MB. Optional ?save=true.
                                    Returns a JSON summary with per-image results.

  CSV Batch — Background Job (no timeout risk)
    POST /validate-csv              Upload a CSV (must have `photograph` + `revice_photograph`
                                    columns). Returns a job_id immediately; validation
                                    runs in the background.
    GET  /job-status/{job_id}       Poll for live progress and per-row results so far.
                                    Status values: queued → processing → done | error.
    GET  /job-download/{job_id}     Download the final validated CSV (only when done).

  CSV Batch — Live Stream (row-by-row NDJSON)
    POST /validate-csv-stream       Upload a CSV. Streams one NDJSON line per row as
                                    each result is produced. Also writes every row to a
                                    partial CSV on disk immediately — safe to disconnect.
                                    First line contains stream_id for download tracking.
    GET  /stream-download/{id}      Download the partial or complete result CSV at any
                                    time, even while the stream is still running.
    GET  /stream-status/{id}        Check stream progress and download availability.

IMAGE URL RESOLUTION (CSV endpoints)
──────────────────────────────────────
  Each CSV row is resolved in this priority:
    1. revice_photograph  → PREFIX_REVISE   + filename
    2. photograph         → PREFIX_ORIGINAL + filename
  If both columns are empty/null the row is marked no_human with a comment.

CONCURRENCY & PERFORMANCE
──────────────────────────
  - MTCNN runs on GPU if CUDA is available, otherwise CPU.
  - MediaPipe instances are thread-local (one per worker thread) to avoid
    contention across concurrent requests.
  - CSV batch processing uses asyncio semaphores + ThreadPoolExecutor to
    download and validate up to MAX_CONCURRENT_DOWNLOADS images in parallel.
  - Streaming endpoint uses a single semaphore (concurrency=1) for low load.
  - All MediaPipe operations run on images downscaled to 1600 px max dimension;
    MTCNN runs on the original full-resolution image for maximum accuracy.

DEPENDENCIES
────────────
  fastapi, uvicorn, opencv-python (cv2), mediapipe, facenet-pytorch (MTCNN),
  torch, Pillow, numpy, aiohttp, pandas

REQUIRED CSV COLUMNS
─────────────────────
  photograph          filename of the original doctor photo
  revice_photograph   filename of the revised/updated photo (takes priority)


  Two pixel-size thresholds decide low_resolution:

BLUR_MIN_FACE_PX  = 40   # face width OR height below this → low_resolution immediately
BLUR_FACE_LOW_RES = 50   # face min(width, height) below this → low_resolution

Decision flow (detect_blur_and_resolution, line ~2190):

face_w, face_h = (bbox x2-x1), (bbox y2-y1)

Step 1 — hard minimum (40px):
  if face_w < 40 OR face_h < 40:
      category = "low_resolution"    ← no blur check at all, too small

Step 2 — soft minimum (80px):
  elif min(face_w, face_h) < 80:
      category = "low_resolution"    ← face pixels too few for reliable blur score

Step 3 — blur score check:
  elif blur_score >= BLUR_SCORE_THRESHOLD (65.0) AND decision == "blurred":
      category = "blurred"

Step 4 — sharp:
  elif decision == "not_blurry":
      category = "not_blurry"

Step 5 — uncertain:
  else:
      category = "blur_uncertain"

Image-level resolution tiers (_resolution_metadata):

Total pixels	Tier
< 300,000 (0.3MP)	very_low_resolution
< 1,000,000 (1MP)	low_resolution
< 2,000,000 (2MP)	moderate_resolution
< 8,000,000 (8MP)	high_resolution
≥ 8MP	very_high_resolution

So category = "low_resolution" fires when:

The detected face in the image is smaller than 80×80 pixels (or 40×40 for the hard floor)
The image itself may be high-res, but the face crop must be large enough to measure sharpness reliably

If you want to change these thresholds, edit BLUR_MIN_FACE_PX (line 842) and BLUR_FACE_LOW_RES (line 843).

"""

import os
import io
import base64
import cv2
import math
import time
import logging
import threading
import numpy as np
import torch
import uvicorn
import mediapipe as mp

from contextlib import asynccontextmanager
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional

import asyncio, io, os, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import aiohttp, pandas as pd

# ─────────────────────────────────────────────────────────────
# Optional ML packages — graceful fallback when not installed
# ─────────────────────────────────────────────────────────────
try:
    from skimage.measure import blur_effect as _skimage_blur_effect
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False

try:
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import YOLO as _UltralyticsYOLO
    _ULTRALYTICS_OK = True
except ImportError:
    _ULTRALYTICS_OK = False

# YOLOv8n Open-Images-V7 eyewear class IDs
_OIV7_SUNGLASSES_CLS = 505   # dark/tinted sunglasses
_OIV7_GOGGLES_CLS    = 220   # opaque goggles (ski, swim, safety)
_OIV7_GLASSES_CLS    = 217   # clear prescription glasses — NOT flagged

_OIV7_HAT_CLS        = 243   # Hat (baseball cap, sports cap, bucket hat …)
_OIV7_HELMET_CLS     = 248   # Helmet (safety/hard hat, military helmet …)
_OIV7_BIKE_HELMET    = 43    # Bicycle helmet
_OIV7_FOOTBALL_HELM  = 202   # Football helmet
_OIV7_SWIM_CAP       = 509   # Swim cap

_yolo_oiv7_model  = None   # lazy-loaded on first request
_yolo_load_failed = False
_yolo_warned: set = set()

_sg_clf        = None   # lazy-loaded SunglassesClassifier (glasses-detector library)
_sg_clf_failed = False

_gender_model        = None   # lazy-loaded ConvNeXt-Large (gender + cap/hijab/turban)
_gender_model_failed = False
_gender_model_error  = None


def _yolo_warn_once(key: str, msg: str):
    if key not in _yolo_warned:
        _yolo_warned.add(key)
        logger.warning(f"[sunglasses/YOLO] {msg}")


def _eyewear_kind(model, class_id: int) -> str:
    """Resolve eyewear class by the model's actual label names, not guessed IDs."""
    try:
        names = model.names
        label = names.get(class_id, "") if isinstance(names, dict) else names[class_id]
        label = str(label).strip().lower().replace("_", " ").replace("-", " ")
        if "sunglass" in label or "sun glasses" in label:
            return "sunglasses"
        if "goggle" in label:
            return "goggles"
        if "glass" in label or "eyeglass" in label or "spectacle" in label:
            return "clear_glasses"
    except Exception:
        pass
    # Fallback for known exported Open Images label maps only.
    if class_id == _OIV7_SUNGLASSES_CLS:
        return "sunglasses"
    if class_id == _OIV7_GOGGLES_CLS:
        return "goggles"
    if class_id == _OIV7_GLASSES_CLS:
        return "clear_glasses"
    return "other"


def _headwear_kind(model, class_id: int) -> str:
    try:
        names = model.names
        label = names.get(class_id, "") if isinstance(names, dict) else names[class_id]
        label = str(label).strip().lower().replace("_", " ").replace("-", " ")
        if "helmet" in label:
            return "helmet"
        if any(k in label for k in ("hat", " cap", "beanie", "fedora", "sombrero", "beret", "turban", "headband")):
            return "cap"
    except Exception:
        pass
    if class_id in (_OIV7_HELMET_CLS, _OIV7_BIKE_HELMET, _OIV7_FOOTBALL_HELM):
        return "helmet"
    if class_id in (_OIV7_HAT_CLS, _OIV7_SWIM_CAP):
        return "cap"
    return "other"


def _load_yolo_oiv7():
    global _yolo_oiv7_model, _yolo_load_failed
    if _yolo_oiv7_model is None and _ULTRALYTICS_OK and not _yolo_load_failed:
        try:
            _yolo_oiv7_model = _UltralyticsYOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolov8m-oiv7.pt"))
        except Exception as exc:
            _yolo_load_failed = True
            _yolo_warn_once("load", f"model load failed, classical fallback only: {exc!r}")
    return _yolo_oiv7_model


def _load_sg_clf():
    global _sg_clf, _sg_clf_failed
    if _sg_clf is None and not _sg_clf_failed:
        try:
            import sys as _sys
            if "albumentations" not in _sys.modules:
                import os as _os
                _audit_dir = _os.path.dirname(_os.path.abspath(__file__))
                if _audit_dir not in _sys.path:
                    _sys.path.insert(0, _audit_dir)
                import fake_albumentations as _fa
                _sys.modules["albumentations"] = _fa
                _sys.modules["albumentations.pytorch"] = _fa.pytorch
            from glasses_detector import SunglassesClassifier as _SGC
            _sg_clf = _SGC()
            logger.info("[sunglasses/CLF] SunglassesClassifier loaded")
        except Exception as _exc:
            _sg_clf_failed = True
            logger.warning(f"[sunglasses/CLF] load failed, will skip: {_exc!r}")
    return _sg_clf


def _detect_sg_classifier(frame_cv2_orig, orig_bbox) -> Optional[float]:
    """Face-crop probability from SunglassesClassifier; returns None on any failure."""
    try:
        clf = _load_sg_clf()
        if clf is None or frame_cv2_orig is None or orig_bbox is None:
            return None
        img_h, img_w = frame_cv2_orig.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        pad = int(round(max(fx2 - fx1, fy2 - fy1, 1.0) * 0.15))
        x1, y1 = max(0, int(fx1) - pad), max(0, int(fy1) - pad)
        x2, y2 = min(img_w, int(fx2) + pad), min(img_h, int(fy2) + pad)
        face_crop = frame_cv2_orig[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        # Upscale tiny crops with Lanczos before handing to PIL — the classifier
        # input is 224×224 anyway; explicit high-quality upscale beats the model's
        # own bilinear resize from a 20px source.
        ch, cw = face_crop.shape[:2]
        if min(ch, cw) < 50:
            scale = max(224 / max(ch, cw), 2.0)
            new_w, new_h = int(cw * scale), int(ch * scale)
            face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        return float(clf.predict(pil_img, label_type="proba"))
    except Exception as _exc:
        logger.warning(f"[sunglasses/CLF] inference error: {_exc!r}")
        return None


# ─────────────────────────────────────────────────────────────
# Gender + headwear detector: ConvNeXt-Large (timm), 5 classes:
#   cap, female, hijab, male, turban  (sigmoid outputs, one per class)
# Local weights: models/best_convnext_large.pth
# The cap/hijab/turban probabilities are also used by the appearance classifier.
# ─────────────────────────────────────────────────────────────

_gender_model        = None
_gender_model_failed = False
_gender_model_error  = None
_gender_model_arch   = "ConvNeXt-Large"
_GENDER_MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_convnext_large.pth")
_CONVNEXT_CLASSES    = ["cap", "female", "hijab", "male", "turban"]
_CONVNEXT_DEVICE     = None  # set lazily on first load


def _load_gender_model():
    global _gender_model, _gender_model_failed, _gender_model_error, _CONVNEXT_DEVICE
    if _gender_model is not None:
        return _gender_model
    if _gender_model_failed:
        return None
    if not os.path.isfile(_GENDER_MODEL_PATH):
        _gender_model_failed = True
        _gender_model_error = f"model_file_not_found: {_GENDER_MODEL_PATH}"
        logger.error("[gender] %s", _gender_model_error)
        return None
    try:
        import torch, timm
        from torchvision import transforms as _tv_transforms
        _CONVNEXT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model("convnext_large", pretrained=False, num_classes=len(_CONVNEXT_CLASSES))
        checkpoint = torch.load(_GENDER_MODEL_PATH, map_location=_CONVNEXT_DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(_CONVNEXT_DEVICE)
        model.eval()
        _transform = _tv_transforms.Compose([
            _tv_transforms.Resize((224, 224)),
            _tv_transforms.ToTensor(),
            _tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _gender_model = (model, _transform)
        _gender_model_error = None
        logger.info("[gender] loaded ConvNeXt-Large from %s on %s", _GENDER_MODEL_PATH, _CONVNEXT_DEVICE)
        return _gender_model
    except Exception as exc:
        _gender_model_failed = True
        _gender_model_error = f"{type(exc).__name__}: {exc}"
        logger.exception("[gender] model load failed: %s", _gender_model_error)
        return None


def _gender_unavailable(status, reason, **extra):
    return {"gender": None, "status": status, "reason": str(reason),
            "confidence": None, "probs": None, **extra}


def detect_gender(frame_cv2_orig: np.ndarray, orig_bbox) -> dict:
    loaded = _load_gender_model()
    if loaded is None:
        return _gender_unavailable("model_unavailable", _gender_model_error or "Gender model unavailable")
    model, transform = loaded
    try:
        import torch
        if not isinstance(frame_cv2_orig, np.ndarray) or frame_cv2_orig.ndim != 3:
            raise ValueError("Expected HxWx3 BGR image")
        if orig_bbox is None or len(orig_bbox) < 4:
            raise ValueError("Bounding box must contain x1,y1,x2,y2")
        ih, iw = frame_cv2_orig.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in orig_bbox[:4]]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid face bounding box")
        pad = max(x2 - x1, y2 - y1, 1.0) * 0.10
        xa = max(0, min(iw, int(np.floor(x1 - pad))))
        ya = max(0, min(ih, int(np.floor(y1 - pad))))
        xb = max(0, min(iw, int(np.ceil(x2 + pad))))
        yb = max(0, min(ih, int(np.ceil(y2 + pad))))
        if xb <= xa or yb <= ya:
            raise ValueError("Face crop is empty")
        crop = frame_cv2_orig[ya:yb, xa:xb]
        if crop.size == 0:
            raise ValueError("Face crop is empty")
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_crop).unsqueeze(0).to(_CONVNEXT_DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output[0]).tolist()
        prob_dict = {cls: round(probs[i], 4) for i, cls in enumerate(_CONVNEXT_CLASSES)}
        male_p   = prob_dict["male"]
        female_p = prob_dict["female"]
        # gender: use whichever is higher; flag manual_review if neither is > 0.9
        if male_p > female_p:
            gender, confidence = "male", male_p
        else:
            gender, confidence = "female", female_p
        manual_review = 0 if confidence > 0.9 else 1
        # headwear: cap / hijab / turban if max > 0.85
        hw_names  = ["cap", "hijab", "turban"]
        hw_probs  = [prob_dict[h] for h in hw_names]
        max_hw    = max(hw_probs)
        headwear  = hw_names[hw_probs.index(max_hw)] if max_hw > 0.40 else None
        return {
            "gender":        gender,
            "status":        "predicted",
            "reason":        None,
            "confidence":    round(confidence, 4),
            "probs":         {"male": round(male_p, 4), "female": round(female_p, 4)},
            "all_probs":     prob_dict,
            "manual_review": manual_review,
            "headwear":      headwear,
            "model":         _gender_model_arch,
            "crop_bbox":     [xa, ya, xb, yb],
            "crop_size":     [int(crop.shape[1]), int(crop.shape[0])],
        }
    except Exception as exc:
        logger.exception("[gender] inference failed")
        return _gender_unavailable("inference_error", f"{type(exc).__name__}: {exc}")




# ═══════════════════════════════════════════════════════════════
# MODULE A — Hair color detection
# ═══════════════════════════════════════════════════════════════

def _hair_zone_stats(frame_cv2_orig, zy1, zy2, zx1, zx2):
    """Extract hair statistics from one image zone after masking skin and background."""
    if zy2 <= zy1 + 2 or zx2 <= zx1 + 2:
        return None
    crop = frame_cv2_orig[zy1:zy2, zx1:zx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    # Skin mask: warm hue (0-25 or 160-180), moderate sat, not too dark
    skin_mask = (((H < 25) | (H > 155)) & (S > 25) & (V > 55))
    # Background mask: very bright+desaturated (white wall) OR very dark (<20)
    bg_mask = ((V > 225) & (S < 25)) | (V < 20)
    hair_mask = ~skin_mask & ~bg_mask
    n = int(np.sum(hair_mask))
    if n < 20:
        return None
    sv = S[hair_mask]; vv = V[hair_mask]
    s_mean = float(np.mean(sv))
    s_med  = float(np.median(sv))
    v_mean = float(np.mean(vv))
    v_med  = float(np.median(vv))
    v_std  = float(np.std(vv))   # texture signal — white hair has variation, flat bg is uniform
    v_p25  = float(np.percentile(vv, 25))
    v_p75  = float(np.percentile(vv, 75))
    # Achromatic fraction: pixels with S<40 AND V>45 (grey/white, not shadow)
    ach_pct = float(np.mean((sv < 40) & (vv > 45)))
    # Texture: Laplacian variance on full crop (high=hair strands, low=flat wall/bg)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap_var = float(np.var(cv2.Laplacian(gray_crop, cv2.CV_32F)))
    # Dark pixel fraction from ALL pixels: high means bimodal/mixed zone
    v_dark_pct = float(np.mean(V < 80))
    # Vertical brightness gradient: compare top half vs bottom half brightness
    mid_y = (zy2 - zy1) // 2
    if mid_y > 2 and (zy2 - zy1 - mid_y) > 2:
        bot_v = cv2.cvtColor(frame_cv2_orig[zy1 + mid_y:zy2, zx1:zx2],
                             cv2.COLOR_BGR2HSV).astype(np.float32)[:, :, 2]
        top_v = cv2.cvtColor(frame_cv2_orig[zy1:zy1 + mid_y, zx1:zx2],
                             cv2.COLOR_BGR2HSV).astype(np.float32)[:, :, 2]
        v_med_bot = float(np.median(bot_v))
        v_med_top = float(np.median(top_v))
    else:
        v_med_bot = v_med_top = v_med
    return {"s_mean": s_mean, "s_med": s_med, "v_mean": v_mean,
            "v_med": v_med, "v_std": v_std, "v_p25": v_p25, "v_p75": v_p75,
            "ach_pct": ach_pct, "n": n,
            "lap_var": lap_var, "v_dark_pct": v_dark_pct,
            "v_med_bot": v_med_bot, "v_med_top": v_med_top}


def _classify_zone(z):
    """Classify one zone as 'white', 'black', or 'unknown'."""
    if z is None:
        return "unknown"
    lap_var    = z.get("lap_var", 0)
    dark_pct   = z.get("v_dark_pct", 0)
    v_p25      = z.get("v_p25", z["v_med"])
    v_p75      = z.get("v_p75", z["v_med"])
    iqr        = v_p75 - v_p25
    v_med_bot  = z.get("v_med_bot", z["v_med"])
    v_med_top  = z.get("v_med_top", z["v_med"])
    # Vertical gradient guard: if top half much darker than bottom AND zone is mostly dark,
    # the top is capturing dark background above the head — use bottom-half brightness.
    v_med_eff = (v_med_bot if (v_med_top < v_med_bot - 25 and dark_pct > 0.20)
                 else z["v_med"])
    # Rule 1: very dark → black hair
    if v_med_eff < 80:
        return "black"
    # Rule 2: high saturation → dark/colored hair
    if z["s_mean"] > 55:
        return "black"
    # Guard: very bright + very low saturation = white wall/background, NOT hair
    # BYPASS only if the zone is textured (real hair strands) AND clean (no dark mixing)
    if z["v_med"] > 178 and z["s_mean"] < 22:
        is_textured_hair = (dark_pct < 0.05 and lap_var > 30)
        if not is_textured_hair:
            return "unknown"
    # Dark-zone filter: >40% dark pixels → bimodal/mixed zone (dark hair + bright bg)
    if dark_pct > 0.40:
        return "unknown"
    # IQR filter: strongly bimodal brightness (dark cluster + bright cluster)
    if iqr > 100:
        return "unknown"
    # Rule 3: strong achromatic evidence → white/grey hair
    if z["s_med"] < 38 and z["ach_pct"] > 0.45 and z["v_std"] > 10:
        return "white"
    # Rule 4: bright + mostly desaturated + texture → grey/white hair
    if z["v_mean"] > 145 and z["s_mean"] < 48 and z["ach_pct"] > 0.35 and z["v_std"] > 10:
        return "white"
    # Rule 5: moderate saturation, not enough achromatic → black fallback
    if z["s_mean"] > 35:
        return "black"
    return "unknown"


_FACE_PARSER_DIR = os.environ.get(
    "FACE_PARSER_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_parsing"))
_face_parser        = None   # (processor, model) — SegFormer fine-tuned on CelebAMask-HQ
_face_parser_failed = False
_FP_HAIR, _FP_HAT = 13, 14   # CelebAMask-HQ label ids used by jonathandinu/face-parsing


def _load_face_parser():
    """Optional hair/hat segmentation model. Only used if the folder exists locally
    (download: huggingface.co/jonathandinu/face-parsing -> models/face_parsing).
    Without it, detect_hair_color falls back to the HSV-band heuristic."""
    global _face_parser, _face_parser_failed
    if _face_parser is None and not _face_parser_failed:
        if not os.path.isdir(_FACE_PARSER_DIR):
            _face_parser_failed = True
            return None
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            proc  = SegformerImageProcessor.from_pretrained(_FACE_PARSER_DIR)
            model = SegformerForSemanticSegmentation.from_pretrained(_FACE_PARSER_DIR).eval().to(device)
            _face_parser = (proc, model)
            logger.info("[hair] face-parsing model loaded from %s", _FACE_PARSER_DIR)
        except Exception as exc:
            _face_parser_failed = True
            logger.warning("[hair] face-parsing unavailable, using HSV heuristic: %r", exc)
    return _face_parser


def _hair_color_from_parsing(frame_cv2_orig: np.ndarray, orig_bbox) -> Optional[dict]:
    """Segment the hair pixels precisely, then classify colour from hair pixels only."""
    fp = _load_face_parser()
    if fp is None:
        return None
    try:
        import torch as _torch
        proc, model = fp
        ih, iw = frame_cv2_orig.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in orig_bbox[:4]]
        fw, fh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 - fw * 0.6)); cx2 = min(iw, int(x2 + fw * 0.6))
        cy1 = max(0, int(y1 - fh * 0.9)); cy2 = min(ih, int(y2 + fh * 0.3))
        crop = frame_cv2_orig[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        inputs = proc(images=pil, return_tensors="pt").to(device)
        with _torch.no_grad():
            logits = model(**inputs).logits
        up = _torch.nn.functional.interpolate(
            logits, size=crop.shape[:2], mode="bilinear", align_corners=False)
        seg = up.argmax(dim=1)[0].cpu().numpy()
        hair_m = (seg == _FP_HAIR).astype(np.uint8)
        hat_frac = float(np.mean(seg == _FP_HAT))
        hair_m = cv2.erode(hair_m, np.ones((3, 3), np.uint8)).astype(bool)
        n = int(hair_m.sum())
        if n < 150:
            return {"color": None, "hair_px": n, "hat_frac": round(hat_frac, 3)}
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        S = hsv[:, :, 1][hair_m]; V = hsv[:, :, 2][hair_m]
        grey_frac = float(np.mean((S < 55) & (V > 120)))      # grey/white strands
        v_med, s_med = float(np.median(V)), float(np.median(S))
        white = grey_frac >= 0.40 or (v_med >= 165 and s_med < 60)
        return {"color": "white" if white else "black", "hair_px": n,
                "grey_frac": round(grey_frac, 3), "v_med": round(v_med, 1),
                "hat_frac": round(hat_frac, 3)}
    except Exception as exc:
        logger.debug("[hair] parsing failed: %r", exc)
        return None


def detect_hair_color(frame_cv2_orig: np.ndarray, orig_bbox) -> dict:
    """
    Detect hair color using multi-zone HSV analysis.

    Two zones are analysed independently:
      Zone A (crown)  — top 55% of hair region: catches what is on top of the head
      Zone B (temple) — bottom 45% of hair region: catches the actual hair at the sides

    Both zones vote. Agreement → confident result.
    Disagreement → use the zone with more pixels, or fall back to conservative (black).
    """
    try:
        if orig_bbox is None or len(orig_bbox) < 4:
            return {"hair_color": None, "status": "no_bbox"}
        ih, iw = frame_cv2_orig.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in orig_bbox[:4]]
        face_h = y2 - y1
        face_w = x2 - x1

        # Overall hair band: from 90% above face top to 5% into the face
        pad_x = face_w * 0.12
        bx1 = max(0, int(x1 - pad_x))
        bx2 = min(iw, int(x2 + pad_x))
        band_top = max(0, int(y1 - face_h * 0.90))
        band_bot = max(0, int(y1 + face_h * 0.05))

        if band_bot <= band_top + 4:
            return {"hair_color": None, "status": "hair_zone_too_small",
                    "top_zone_achromatic": False}

        band_mid = band_top + int((band_bot - band_top) * 0.55)

        # Zone A: crown (top 55% of hair band)
        za = _hair_zone_stats(frame_cv2_orig, band_top, band_mid, bx1, bx2)
        # Zone B: temple/sides (bottom 45% of hair band)
        zb = _hair_zone_stats(frame_cv2_orig, band_mid, band_bot, bx1, bx2)

        label_a = _classify_zone(za)
        label_b = _classify_zone(zb)

        # Zone C: ear-side bilateral zones — most reliable, unaffected by dark bg above head
        cs_y1 = max(0, int(y1 - face_h * 0.20))
        cs_y2 = min(ih, int(y1 + face_h * 0.45))
        csl_x1 = max(0, int(x1 - face_w * 0.35)); csl_x2 = min(iw, int(x1 + face_w * 0.08))
        csr_x1 = max(0, int(x2 - face_w * 0.08)); csr_x2 = min(iw, int(x2 + face_w * 0.35))
        zc_l = _hair_zone_stats(frame_cv2_orig, cs_y1, cs_y2, csl_x1, csl_x2)
        zc_r = _hair_zone_stats(frame_cv2_orig, cs_y1, cs_y2, csr_x1, csr_x2)
        lc_l = _classify_zone(zc_l); lc_r = _classify_zone(zc_r)
        if lc_l == lc_r and lc_l != "unknown":
            label_c = lc_l
        elif lc_l == "unknown":
            label_c = lc_r
        elif lc_r == "unknown":
            label_c = lc_l
        else:
            label_c = "unknown"

        # Weighted voting: Zone C = 2pts (bilateral agreement), A/B = 1pt each
        def _pts(lbl, w): return w if lbl == "white" else 0, w if lbl == "black" else 0
        wa, ba = _pts(label_a, 1); wb, bb = _pts(label_b, 1); wc, bc = _pts(label_c, 2)
        white_pts = wa + wb + wc; black_pts = ba + bb + bc
        if white_pts == 0 and black_pts == 0:
            color = "black"           # all unknown → conservative fallback
        elif white_pts > black_pts:
            color = "white"
        elif black_pts > white_pts:
            color = "black"
        else:
            # Tie: prefer Zone C (most reliable), then B, then A
            color = label_c if label_c != "unknown" else (
                    label_b if label_b != "unknown" else (
                    label_a if label_a != "unknown" else "black"))

        # Aggregate stats for logging/return (zones can be None -> guard empty lists)
        _zs = [z for z in (za, zb, zc_l, zc_r) if z]
        s_mean = round(float(np.mean([z["s_mean"] for z in _zs])), 1) if _zs else 0.0
        v_med  = round(float(np.median([z["v_med"] for z in _zs])), 1) if _zs else 0.0
        ach    = round(float(np.mean([z["ach_pct"] for z in _zs])), 3) if _zs else 0.0
        total_px = sum(z["n"] for z in _zs)

        # Preferred: segmentation-based hair colour (if the face-parsing model is installed)
        method, parsed = "hsv_band", _hair_color_from_parsing(frame_cv2_orig, orig_bbox)
        if parsed and parsed.get("color"):
            color, method = parsed["color"], "face_parsing"
            ach, total_px = parsed["grey_frac"], parsed["hair_px"]

        logger.debug("[hair] zone_a=%s zone_b=%s zone_c=%s -> %s | s=%.1f v_med=%.1f ach=%.2f px=%d",
                     label_a, label_b, label_c, color, s_mean, v_med, ach, total_px)

        # Top-zone achromatic signal: very top 25% of hair band — bright + desaturated
        # Signals that something white/bright is on top of the head (cap fabric, turban)
        try:
            tz_bot = band_top + max(1, (band_bot - band_top) // 4)
            top_zone = frame_cv2_orig[band_top:tz_bot, bx1:bx2]
            if top_zone.size > 0:
                tz_hsv = cv2.cvtColor(top_zone, cv2.COLOR_BGR2HSV).astype(np.float32)
                tz_v = float(np.mean(tz_hsv[:, :, 2]))
                tz_s = float(np.mean(tz_hsv[:, :, 1]))
                top_zone_achromatic = bool(tz_v > 170 and tz_s < 85)
            else:
                top_zone_achromatic = False
        except Exception:
            top_zone_achromatic = False

        return {
            "hair_color":          color,
            "status":              "predicted",
            "zone_a":              label_a,
            "zone_b":              label_b,
            "zone_c":              label_c,
            "s_mean":              s_mean,
            "v_med":               v_med,
            "achromatic_pct":      ach,
            "hair_px":             total_px,
            "top_zone_achromatic": top_zone_achromatic,
            "method":              method,
            "parse_hat_frac":      (parsed or {}).get("hat_frac"),
            "hair_bbox":           [bx1, band_top, bx2, band_bot],
        }
    except Exception as exc:
        logger.warning("[hair] detection failed: %r", exc)
        return {"hair_color": None, "status": f"error: {exc}",
                "top_zone_achromatic": False}


# ═══════════════════════════════════════════════════════════════
# MODULE B — Spectacles / eyeglasses detection
# ═══════════════════════════════════════════════════════════════

# CLIP prompts for spectacles detection
# Two positive prompts (glasses/specs) + one strong negative (no glasses)
_SPECS_PROMPTS = [
    "a person wearing prescription eyeglasses or spectacles",
    "clear corrective glasses with thin frames resting on the nose bridge",
    "a face with no glasses, no spectacles, bare eyes",
]

_gl_clf        = None   # lazy-loaded glasses-detector GlassesClassifier(kind="eyeglasses")
_gl_clf_failed = False


def _load_glasses_clf():
    """Eyeglasses (clear/prescription) classifier from the `glasses-detector` package
    (github.com/mantasu/glasses-detector). Same shim/loading pattern as _load_sg_clf."""
    global _gl_clf, _gl_clf_failed
    if _gl_clf is None and not _gl_clf_failed:
        try:
            import sys as _sys
            if "albumentations" not in _sys.modules:
                _d = os.path.dirname(os.path.abspath(__file__))
                if _d not in _sys.path:
                    _sys.path.insert(0, _d)
                import fake_albumentations as _fa
                _sys.modules["albumentations"] = _fa
                _sys.modules["albumentations.pytorch"] = _fa.pytorch
            from glasses_detector import GlassesClassifier as _GC
            _gl_clf = _GC(kind="eyeglasses")
            logger.info("[specs/CLF] GlassesClassifier(eyeglasses) loaded")
        except Exception as exc:
            _gl_clf_failed = True
            logger.warning("[specs/CLF] unavailable, using heuristics only: %r", exc)
    return _gl_clf


def _detect_eyeglasses_classifier(frame_cv2_orig, orig_bbox) -> Optional[float]:
    """Probability (0-1) that the face crop wears eyeglasses; None if unavailable."""
    try:
        clf = _load_glasses_clf()
        if clf is None:
            return None
        ih, iw = frame_cv2_orig.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        pad = int(round(max(fx2 - fx1, fy2 - fy1, 1.0) * 0.15))
        x1, y1 = max(0, int(fx1) - pad), max(0, int(fy1) - pad)
        x2, y2 = min(iw, int(fx2) + pad), min(ih, int(fy2) + pad)
        crop = frame_cv2_orig[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        if min(ch, cw) < 50:
            s = max(224 / max(ch, cw), 2.0)
            crop = cv2.resize(crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_LANCZOS4)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return float(clf.predict(pil, label_type="proba"))
    except Exception as exc:
        logger.debug("[specs/CLF] inference error: %r", exc)
        return None


def detect_spectacles(frame_cv2_orig: np.ndarray, orig_bbox) -> dict:
    """
    Detect eyeglasses/spectacles using 4 independent signals with weighted voting.

    Signal 1 — YOLO:        sunglasses model repurposed for specs (strong=2pts, soft=1pt)
    Signal 2 — Eye shadow:  specs darken the eye socket — compare eye-band to cheek brightness
    Signal 3 — Frame edges: strong horizontal edges in the nose-bridge + eye-corner zones
    Signal 4 — CLIP:        zero-shot language-vision model (1pt)

    Decision: YOLO-strong alone → detected
              Otherwise: total votes >= 2 → detected
    """
    try:
        if orig_bbox is None or len(orig_bbox) < 4:
            return {"detected": False, "status": "no_bbox"}
        ih, iw = frame_cv2_orig.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in orig_bbox[:4]]
        face_h = y2 - y1
        face_w = x2 - x1
        signals = {}

        # ── Signal 1: YOLO ────────────────────────────────────────────────────
        sg = _detect_sunglasses_yolo(frame_cv2_orig, orig_bbox)
        yolo_strong = bool(sg and sg.get("yolo_detected"))
        yolo_soft   = bool(sg and sg.get("yolo_soft"))
        signals["yolo_strong"] = yolo_strong
        signals["yolo_soft"]   = yolo_soft

        # ── Signal 2: Eye-socket brightness dip ──────────────────────────────
        # Spectacle frames + lenses reduce brightness in the eye-socket band.
        # Compare mean V (brightness) of eye band vs cheek band.
        # Specs: eye_V is noticeably darker relative to cheeks → ratio < ~0.88
        shadow_hit  = False
        shadow_ratio = 1.0
        try:
            eye_y1 = max(0, int(y1 + face_h * 0.22))
            eye_y2 = min(ih, int(y1 + face_h * 0.50))
            chk_y1 = min(ih, int(y1 + face_h * 0.60))
            chk_y2 = min(ih, int(y1 + face_h * 0.82))
            # Use centre 70% of face width to avoid ears/hair edges
            mx = int((x1 + x2) / 2)
            hw = int(face_w * 0.35)
            lx = max(0, mx - hw); rx = min(iw, mx + hw)
            if eye_y2 > eye_y1 + 4 and chk_y2 > chk_y1 + 4 and rx > lx + 4:
                eye_crop = frame_cv2_orig[eye_y1:eye_y2, lx:rx]
                chk_crop = frame_cv2_orig[chk_y1:chk_y2, lx:rx]
                eye_v = float(np.mean(
                    cv2.cvtColor(eye_crop, cv2.COLOR_BGR2HSV)[:, :, 2]))
                chk_v = float(np.mean(
                    cv2.cvtColor(chk_crop, cv2.COLOR_BGR2HSV)[:, :, 2]))
                shadow_ratio = round(eye_v / (chk_v + 1.0), 3)
                # Threshold: specs darken eyes relative to cheeks
                shadow_hit = bool(shadow_ratio < 0.86 and chk_v > 60)
        except Exception as _se:
            logger.debug("[specs] shadow check failed: %r", _se)
        signals["eye_shadow"]    = shadow_hit
        signals["shadow_ratio"]  = shadow_ratio

        # ── Signal 3: Nose-bridge + eye-frame horizontal edges ────────────────
        # Spectacle frames leave strong horizontal Sobel edges in the nose-bridge
        # zone (y=33-50%). Compare vs upper-forehead (no-glasses reference zone).
        frame_hit  = False
        frame_ratio = 0.0
        try:
            nb_y1 = max(0, int(y1 + face_h * 0.33))
            nb_y2 = min(ih, int(y1 + face_h * 0.50))
            fh_y1 = max(0, int(y1 + face_h * 0.05))
            fh_y2 = min(ih, int(y1 + face_h * 0.22))
            nb_cx = (x1 + x2) / 2
            nb_x1 = max(0, int(nb_cx - face_w * 0.32))
            nb_x2 = min(iw, int(nb_cx + face_w * 0.32))
            if (nb_y2 > nb_y1 + 4 and nb_x2 > nb_x1 + 4
                    and fh_y2 > fh_y1 + 4):
                nb_gray = cv2.cvtColor(
                    frame_cv2_orig[nb_y1:nb_y2, nb_x1:nb_x2],
                    cv2.COLOR_BGR2GRAY)
                fh_gray = cv2.cvtColor(
                    frame_cv2_orig[fh_y1:fh_y2, nb_x1:nb_x2],
                    cv2.COLOR_BGR2GRAY)
                nb_gy = cv2.Sobel(nb_gray, cv2.CV_32F, 0, 1, ksize=3)
                fh_gy = cv2.Sobel(fh_gray, cv2.CV_32F, 0, 1, ksize=3)
                nb_edge = float(np.mean(np.abs(nb_gy) > 18))
                fh_edge = float(np.mean(np.abs(fh_gy) > 18))
                frame_ratio = round(nb_edge / (fh_edge + 0.01), 3)
                frame_hit = bool(frame_ratio > 1.7 and nb_edge > 0.10)
        except Exception as _fe:
            logger.debug("[specs] frame-edge check failed: %r", _fe)
        signals["frame_edge"]   = frame_hit
        signals["frame_ratio"]  = frame_ratio

        # ── Signal 3a: Eye-frame top bar edges ───────────────────────────────
        # The top bar of glasses frames leaves horizontal edges at y=20-36% of face.
        # Compare vs forehead reference (y=3-18%) as baseline.
        frame_top_hit = False; frame_top_ratio = 0.0
        try:
            ft_y1 = max(0, int(y1 + face_h * 0.20)); ft_y2 = min(ih, int(y1 + face_h * 0.36))
            fr_y1 = max(0, int(y1 + face_h * 0.03)); fr_y2 = min(ih, int(y1 + face_h * 0.18))
            ft_x1 = max(0, int(x1 + face_w * 0.10)); ft_x2 = min(iw, int(x2 - face_w * 0.10))
            if ft_y2 > ft_y1 + 4 and fr_y2 > fr_y1 + 4 and ft_x2 > ft_x1 + 4:
                ft_gray = cv2.cvtColor(frame_cv2_orig[ft_y1:ft_y2, ft_x1:ft_x2], cv2.COLOR_BGR2GRAY)
                fr_gray = cv2.cvtColor(frame_cv2_orig[fr_y1:fr_y2, ft_x1:ft_x2], cv2.COLOR_BGR2GRAY)
                ft_gy = cv2.Sobel(ft_gray, cv2.CV_32F, 0, 1, ksize=3)
                fr_gy = cv2.Sobel(fr_gray, cv2.CV_32F, 0, 1, ksize=3)
                ft_edge = float(np.mean(np.abs(ft_gy) > 18))
                fr_edge = float(np.mean(np.abs(fr_gy) > 18))
                frame_top_ratio = round(ft_edge / (fr_edge + 0.01), 3)
                frame_top_hit = bool(frame_top_ratio > 1.5 and ft_edge > 0.09)
        except Exception as _fte:
            logger.debug("[specs] frame-top check failed: %r", _fte)
        signals["frame_top_edge"]  = frame_top_hit
        signals["frame_top_ratio"] = frame_top_ratio

        # ── Signal 4: CLIP zero-shot ──────────────────────────────────────────
        clip_hit = False; clip_score = 0.0
        st = _load_scene_clip()
        if st is not None:
            try:
                import torch as _torch
                pad = max(face_w, face_h) * 0.12
                cx1 = max(0, int(x1 - pad)); cy1 = max(0, int(y1 - pad))
                cx2 = min(iw, int(x2 + pad)); cy2 = min(ih, int(y2 + pad))
                pil_face = Image.fromarray(
                    cv2.cvtColor(frame_cv2_orig[cy1:cy2, cx1:cx2],
                                 cv2.COLOR_BGR2RGB))
                proc = st["proc"]; mdl = st["model"]
                pix  = proc.image_processor(
                    images=pil_face, return_tensors="pt")["pixel_values"].to(device)
                tok  = proc.tokenizer(
                    _SPECS_PROMPTS, padding=True, return_tensors="pt")
                text = {k: v.to(device) for k, v in tok.items()
                        if k in ("input_ids", "attention_mask")}
                with _torch.no_grad():
                    i_feat = mdl.get_image_features(pixel_values=pix)
                    t_feat = mdl.get_text_features(**text)
                    i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
                    t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
                    sims = (i_feat @ t_feat.T).squeeze(0).softmax(dim=-1).tolist()
                # sims[0]+sims[1] = both positive prompts; sims[2] = negative
                clip_score = round(sims[0] + sims[1], 3)
                clip_hit   = clip_score > 0.52
            except Exception as _ce:
                logger.debug("[specs] CLIP failed: %r", _ce)
        signals["clip"]       = clip_hit
        signals["clip_score"] = clip_score

        # ── Signal 5: Nose-bridge intensity dip ──────────────────────────────
        # Glasses bridge casts a slight shadow/dark strip across the nose bridge.
        # Compare centre bridge zone brightness vs adjacent lens zones.
        bridge_hit = False; bridge_ratio = 0.0
        try:
            br_y1 = max(0, int(y1 + face_h * 0.30)); br_y2 = min(ih, int(y1 + face_h * 0.42))
            cx = (x1 + x2) / 2; bw = face_w * 0.10
            br_x1 = max(0, int(cx - bw)); br_x2 = min(iw, int(cx + bw))
            ln_x1 = max(0, int(x1 + face_w * 0.15)); ln_x2 = min(iw, int(x2 - face_w * 0.15))
            if br_y2 > br_y1 + 4 and br_x2 > br_x1 + 2 and ln_x2 > ln_x1 + 4:
                bridge_v = float(np.mean(
                    cv2.cvtColor(frame_cv2_orig[br_y1:br_y2, br_x1:br_x2],
                                 cv2.COLOR_BGR2HSV)[:, :, 2]))
                lens_v = float(np.mean(
                    cv2.cvtColor(frame_cv2_orig[br_y1:br_y2, ln_x1:ln_x2],
                                 cv2.COLOR_BGR2HSV)[:, :, 2]))
                bridge_ratio = round(bridge_v / (lens_v + 1.0), 3)
                bridge_hit = bool(bridge_ratio < 0.84 and lens_v > 60)
        except Exception as _be:
            logger.debug("[specs] bridge-dip check failed: %r", _be)
        signals["bridge_dip"]       = bridge_hit
        signals["bridge_dip_ratio"] = bridge_ratio

        # ── Signal 6: Temple arm edges ────────────────────────────────────────
        # Glasses arms run horizontally along the outer face sides (y=25-55%).
        # Both sides must show horizontal Sobel edges stronger than inner face.
        temple_hit = False; temple_ratio = 0.0
        try:
            te_y1 = max(0, int(y1 + face_h * 0.25)); te_y2 = min(ih, int(y1 + face_h * 0.55))
            tl_x1 = max(0, int(x1 - face_w * 0.15)); tl_x2 = min(iw, int(x1 + face_w * 0.00))
            tr_x1 = max(0, int(x2 - face_w * 0.00)); tr_x2 = min(iw, int(x2 + face_w * 0.15))
            ic_x1 = max(0, int(x1 + face_w * 0.20)); ic_x2 = min(iw, int(x2 - face_w * 0.20))
            if (te_y2 > te_y1 + 4 and tl_x2 > tl_x1 + 2 and tr_x2 > tr_x1 + 2
                    and ic_x2 > ic_x1 + 4):
                def _hsobel_mean(crop):
                    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    return float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) > 18))
                tl_edge = _hsobel_mean(frame_cv2_orig[te_y1:te_y2, tl_x1:tl_x2])
                tr_edge = _hsobel_mean(frame_cv2_orig[te_y1:te_y2, tr_x1:tr_x2])
                ic_edge = _hsobel_mean(frame_cv2_orig[te_y1:te_y2, ic_x1:ic_x2])
                both_avg = (tl_edge + tr_edge) / 2.0
                temple_ratio = round(both_avg / (ic_edge + 0.01), 3)
                temple_hit = bool(temple_ratio > 1.5 and both_avg > 0.07
                                  and tl_edge > 0.05 and tr_edge > 0.05)
        except Exception as _te:
            logger.debug("[specs] temple-arm check failed: %r", _te)
        signals["temple_arm"]   = temple_hit
        signals["temple_ratio"] = temple_ratio

        # ── Voting ────────────────────────────────────────────────────────────
        # YOLO strong = 2 votes (most reliable signal)
        # Each other signal = 1 vote
        # Detect if: YOLO strong alone  OR  total votes >= 2
        # Dedicated eyeglasses classifier (most reliable): >=0.60 counts as 2 votes,
        # <=0.15 means "confidently no glasses" so weak heuristics alone cannot fire.
        clf_prob   = _detect_eyeglasses_classifier(frame_cv2_orig, orig_bbox)
        clf_strong = clf_prob is not None and clf_prob >= 0.60
        clf_clear  = clf_prob is not None and clf_prob <= 0.15
        signals["clf_prob"] = None if clf_prob is None else round(clf_prob, 3)

        heur_votes = (int(shadow_hit) + int(frame_hit) + int(frame_top_hit)
                      + int(bridge_hit) + int(temple_hit) + int(clip_hit))
        vote_count = ((2 if yolo_strong else int(yolo_soft))
                      + heur_votes + (2 if clf_strong else 0))
        needed   = 3 if clf_clear else 2
        detected = bool(yolo_strong or clf_strong or vote_count >= needed)

        logger.debug(
            "[specs] yolo=%s/%s shadow=%s(%.2f) frame=%s(%.2f) frame_top=%s "
            "bridge=%s temple=%s clip=%s(%.2f) votes=%d -> %s",
            yolo_strong, yolo_soft,
            shadow_hit, shadow_ratio,
            frame_hit,  frame_ratio,
            frame_top_hit, bridge_hit, temple_hit,
            clip_hit,   clip_score,
            vote_count, detected)

        return {"detected": detected, "votes": vote_count,
                "signals": signals, "status": "predicted"}
    except Exception as exc:
        logger.warning("[specs] detection failed: %r", exc)
        return {"detected": False, "status": f"error: {exc}"}


# ═══════════════════════════════════════════════════════════════
# MODULE C — Appearance category combiner
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# MODULE C2 — Trained appearance classifier (8 categories)
# Trained on the labelled doctor photos (see training/). Three small heads on
# CLIP features + the ConvNeXt cap/hijab/turban probabilities:
#   headwear (none|cap|turban)  specs (no|yes)  hair (black|white)
# Pure-numpy inference; falls back to the rule-based labels if anything is missing.
# ═══════════════════════════════════════════════════════════════
APP_ML_ENABLED  = os.environ.get("APPEARANCE_ML", "1") == "1"
APP_ML_MIN_CONF = float(os.environ.get("APPEARANCE_ML_MIN_CONF", "0.60"))
_APP_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "appearance_head.npz")
_app_model        = None
_app_model_failed = False
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _load_app_model():
    global _app_model, _app_model_failed
    if _app_model is None and not _app_model_failed:
        try:
            z = np.load(_APP_MODEL_PATH, allow_pickle=False)
            _app_model = {h: {"classes": [str(c) for c in z[f"{h}__classes"]],
                              "coef": z[f"{h}__coef"].astype(np.float64),
                              "bias": z[f"{h}__intercept"].astype(np.float64)}
                          for h in ("headwear", "specs", "hair")}
            _app_model["feature_set"] = str(z["feature_set"]) if "feature_set" in z else "clip+cnxprob"
            logger.info("[appearance] trained model loaded (%s) feature_set=%s", _APP_MODEL_PATH, _app_model["feature_set"])
        except Exception as exc:
            _app_model_failed = True
            logger.warning("[appearance] trained model unavailable, using rules: %r", exc)
    return _app_model


def _lr_proba(head: dict, x: np.ndarray) -> np.ndarray:
    z = head["coef"] @ x + head["bias"]
    if z.shape[0] == 1:                         # binary logistic regression
        p = 1.0 / (1.0 + np.exp(-z[0]))
        return np.array([1.0 - p, p])
    e = np.exp(z - z.max())
    return e / e.sum()


def _letterbox_224(pil_img):
    from PIL import ImageOps as _IO
    return _IO.pad(pil_img, (224, 224), color=(128, 128, 128))


def _clip_embed_images(pil_list):
    """L2-normalised CLIP image embeddings (n,512) as a float32 numpy array, or None."""
    import torch as _torch
    model, _proc = _load_clip()
    if model is None:
        return None
    arr = np.stack([(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 - _CLIP_MEAN) / _CLIP_STD
                    for im in pil_list]).transpose(0, 3, 1, 2).copy()
    try:
        px = _torch.from_numpy(arr)
    except Exception:                           # numpy<->torch interop unavailable
        px = _torch.tensor(arr.tolist())
    with _torch.no_grad():
        f = model.get_image_features(pixel_values=px)
        if not _torch.is_tensor(f):
            f = f.pooler_output
    v = np.array(f.tolist(), dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def _app_features(frame_cv2: np.ndarray, bbox, gender_res: dict):
    """Rebuild exactly the feature vector used in training (training/extract_features.py).
    Supports feature_set='clip' (1024-d) and 'clip+cnxprob' (1048-d)."""
    from PIL import ImageOps as _IO
    m = _load_app_model()
    fset = m.get("feature_set", "clip+cnxprob") if isinstance(m, dict) else "clip+cnxprob"
    probs = (gender_res or {}).get("all_probs")
    if fset != "clip" and not probs:
        return None                             # ConvNeXt output missing -> cannot build features
    ih, iw = frame_cv2.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    fw, fh = x2 - x1, y2 - y1
    box = (int(max(0, x1 - 0.6 * fw)), int(max(0, y1 - 0.9 * fh)),
           int(min(iw, x2 + 0.6 * fw)), int(min(ih, y2 + 0.35 * fh)))
    if box[2] <= box[0] + 2 or box[3] <= box[1] + 2:
        return None
    whole = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
    head  = whole.crop(box)
    emb = _clip_embed_images([_letterbox_224(head), _letterbox_224(_IO.mirror(head)), _letterbox_224(whole)])
    if emb is None:
        return None
    h = emb[:2].mean(axis=0)
    h = h / (np.linalg.norm(h) + 1e-8)
    if fset == "clip":
        return np.concatenate([h, emb[2]]).astype(np.float64)      # 1024-d
    cnx = np.repeat(np.array([probs.get("cap", 0.0), probs.get("hijab", 0.0), probs.get("turban", 0.0)],
                             dtype=np.float32), 8)
    return np.concatenate([h, emb[2], cnx]).astype(np.float64)     # 1048-d


def _compose_appearance(hw: str, specs: bool, hair: str) -> str:
    if hw == "turban":
        return "turban_specs" if specs else "turban"
    if hw == "cap":
        return "doctor_cap_specs" if specs else "doctor_cap"
    base = "white_hair" if hair == "white" else "black_hair"
    return ("specs_" + base) if specs else base


def predict_appearance_ml(frame_cv2: np.ndarray, bbox, gender_res: dict) -> Optional[dict]:
    """Trained 8-category appearance prediction for one face; None if unavailable."""
    if not APP_ML_ENABLED:
        return None
    try:
        m = _load_app_model()
        if m is None or bbox is None:
            return None
        x = _app_features(frame_cv2, bbox, gender_res)
        if x is None or x.shape[0] != m["headwear"]["coef"].shape[1]:
            return None
        p_hw  = _lr_proba(m["headwear"], x)
        p_sp  = _lr_proba(m["specs"], x)
        p_hr  = _lr_proba(m["hair"], x)
        hw    = m["headwear"]["classes"][int(p_hw.argmax())]
        specs = m["specs"]["classes"][int(p_sp.argmax())] == "1"
        hair  = m["hair"]["classes"][int(p_hr.argmax())]
        conf  = float(min(p_hw.max(), p_sp.max(), p_hr.max() if hw == "none" else 1.0))
        # joint probability of each of the 8 labels (headwear x specs x hair) - sums to 1
        cat_probs: dict = {}
        for _hw, _ph in zip(m["headwear"]["classes"], p_hw):
            for _sp, _ps in zip(m["specs"]["classes"], p_sp):
                if _hw == "none":
                    for _hr, _pr in zip(m["hair"]["classes"], p_hr):
                        _k = _compose_appearance("none", _sp == "1", _hr)
                        cat_probs[_k] = cat_probs.get(_k, 0.0) + float(_ph * _ps * _pr)
                else:
                    _k = _compose_appearance(_hw, _sp == "1", "black")
                    cat_probs[_k] = cat_probs.get(_k, 0.0) + float(_ph * _ps)
        return {
            "category_probs":   {k: round(v, 4) for k, v in cat_probs.items()},
            "appearance_label": _compose_appearance(hw, specs, hair),
            "hair_color":       hair if hw == "none" else None,
            "specs_detected":   bool(specs),
            "headwear_kind":    None if hw == "none" else hw,
            "confidence":       round(conf, 3),
            "uncertain":        conf < APP_ML_MIN_CONF,
            "source":           "ml",
            "head_probs":       {"headwear": {c: round(float(v), 3) for c, v in zip(m["headwear"]["classes"], p_hw)},
                                 "specs":    round(float(p_sp[1]), 3),
                                 "white_hair": round(float(p_hr[1]), 3)},
        }
    except Exception as exc:
        logger.warning("[appearance] ML prediction failed, using rules: %r", exc)
        return None


def _appearance_for_face(frame_cv2, bbox, gender_res, hair_res, specs_res, hw_res) -> dict:
    """Trained model first; rule-based labels as fallback (and kept for comparison)."""
    rules = _combine_appearance_labels(hair_res or {}, specs_res or {}, hw_res or {})
    ml = predict_appearance_ml(frame_cv2, bbox, gender_res)
    if ml is None:
        rules["source"] = "rules"
        return rules
    ml["rule_label"] = rules.get("appearance_label")
    return ml


def _combine_appearance_labels(hair_res: dict, specs_res: dict, hw_res: dict) -> dict:
    """
    Combine hair color + spectacles + headwear into the 8 appearance categories:
    black_hair | specs_black_hair | white_hair | specs_white_hair
    doctor_cap | doctor_cap_specs | turban | turban_specs

    Robust logic:
    - Headwear (cap/turban/hijab) wins over hair ONLY when confidence is sufficient
    - ConvNeXt-detected headwear requires prob >= 0.45 to override hair classification
    - White hair requires strong achromatic signal (achromatic_pct > 0.35)
    - "other" hair falls back to black_hair (safe default for dark-haired doctors)
    """
    hair    = (hair_res  or {}).get("hair_color")
    specs   = bool((specs_res or {}).get("detected"))
    hw_det  = bool((hw_res   or {}).get("detected"))
    hw_kind = ((hw_res or {}).get("kind") or "").lower()
    hw_src  = (hw_res or {}).get("source", "yolo")  # "yolo" or "convnext"

    # ConvNeXt probability for the detected headwear class
    # Used to gate low-confidence headwear detections
    cnx_prob = float((hw_res or {}).get("convnext_prob") or 0.0)

    # Achromatic fraction from hair detection
    # High achromatic_pct (>0.35) means truly white/grey hair region
    achromatic_pct = float((hair_res or {}).get("achromatic_pct") or 0.0)

    # Top-zone achromatic signal from hair detection (high-V, low-S top region)
    top_achromatic = bool((hair_res or {}).get("top_zone_achromatic"))

    # --- Headwear confidence gating ---
    # If headwear was detected ONLY by ConvNeXt (not YOLO) and confidence is low,
    # treat it as uncertain — fall through to hair-based classification
    hw_confident = False
    if hw_det and hw_kind in ("cap", "turban", "hijab"):
        if hw_src == "yolo":
            hw_confident = True       # YOLO detection → always trust
        elif hw_src == "convnext" and cnx_prob >= 0.45:
            hw_confident = True       # ConvNeXt high confidence
        elif hw_src == "convnext" and cnx_prob >= 0.30:
            # Borderline ConvNeXt — confirm with hair region signal
            # Cap: white fabric on top → top_achromatic should be true
            # Turban: typically colorful or white → check top zone
            hw_confident = top_achromatic
        # else: cnx_prob < 0.30 → not confident enough

    label = None
    uncertain = False
    if hw_det and not hw_confident:
        uncertain = True            # headwear suspected but not trusted -> hair label used
    if hw_confident and hw_kind == "cap":
        label = "doctor_cap_specs" if specs else "doctor_cap"
    elif hw_confident and hw_kind == "turban":
        label = "turban_specs" if specs else "turban"
    elif hw_confident and hw_kind == "hijab":
        label = "hijab_specs" if specs else "hijab"
    else:
        # Hair-based classification
        # White/grey hair requires strong achromatic evidence
        is_white = (hair == "white") and (achromatic_pct > 0.35 or achromatic_pct == 0.0)
        # achromatic_pct == 0.0 means old result without field — trust hair_color as-is
        if is_white:
            label = "specs_white_hair" if specs else "white_hair"
        else:
            # Black is the default family (also when hair colour could not be measured,
            # e.g. hair_color None) so every face lands in one of the 8 categories.
            label = "specs_black_hair" if specs else "black_hair"
            if hair is None:
                uncertain = True

    return {"appearance_label": label, "hair_color": hair,
            "specs_detected": specs, "headwear_kind": hw_kind if hw_det else None,
            "uncertain": uncertain,
            "hair_method": (hair_res or {}).get("method")}

def _detect_sunglasses_yolo(frame_cv2: np.ndarray, orig_bbox) -> Optional[dict]:
    """Detect sunglasses on both a face crop and an enlarged eye-band crop.

    The second pass is important for distant/small faces: sunglasses may occupy
    only a few pixels in the full face crop. We upscale the upper-face region
    before inference; this improves detector input scale but cannot recreate
    detail absent from the original image.
    """
    try:
        if not _ULTRALYTICS_OK:
            _yolo_warn_once("missing", "ultralytics not importable — classical fallback only")
            return None
        if orig_bbox is None or frame_cv2 is None:
            return None
        model = _load_yolo_oiv7()
        if model is None:
            return None

        img_h, img_w = frame_cv2.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
        fx2, fy2 = min(float(img_w), fx2), min(float(img_h), fy2)
        fw, fh = max(fx2 - fx1, 1.0), max(fy2 - fy1, 1.0)
        pad = int(round(max(fw, fh) * YOLO_CROP_PAD_FRAC))
        x1, y1 = max(0, int(fx1) - pad), max(0, int(fy1) - pad)
        x2, y2 = min(img_w, int(fx2) + pad), min(img_h, int(fy2) + pad)
        face_crop = frame_cv2[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        # Build multiple upper-face crops. Small sunglasses can be only 10-20 px
        # wide in the source, so a single eye-band crop is not reliable enough.
        rel_fx1 = max(0, int(round(fx1 - x1)))
        rel_fx2 = min(face_crop.shape[1], int(round(fx2 - x1)))
        rel_fy1 = max(0, int(round(fy1 - y1)))
        rel_fy2 = min(face_crop.shape[0], int(round(fy2 - y1)))
        eye_x_pad = int(round(0.22 * fw))
        eye_x1 = max(0, rel_fx1 - eye_x_pad)
        eye_x2 = min(face_crop.shape[1], rel_fx2 + eye_x_pad)
        # Two slightly different vertical windows tolerate face-box variation.
        eye_crops = []
        bbox_touches_top = fy1 <= max(2.0, 0.015 * img_h)
        eye_windows = (
            ((0.02, 0.36, "zoomed_eye_band_top"),
             (0.06, 0.44, "zoomed_eye_band_top_wide"),
             (0.12, 0.50, "zoomed_eye_band"),
             (0.20, 0.60, "zoomed_eye_band_wide"),
             (0.28, 0.68, "zoomed_eye_band_low"))
            if bbox_touches_top else
            ((0.08, 0.44, "zoomed_eye_band_top"),
             (0.12, 0.50, "zoomed_eye_band"),
             (0.18, 0.58, "zoomed_eye_band_wide"),
             (0.26, 0.66, "zoomed_eye_band_low"))
        )
        for top_frac, bottom_frac, crop_name in eye_windows:
            cy1 = max(0, int(round(rel_fy1 + top_frac * fh)))
            cy2 = min(face_crop.shape[0], int(round(rel_fy1 + bottom_frac * fh)))
            crop = face_crop[cy1:cy2, eye_x1:eye_x2]
            if crop.size:
                eh, ew = crop.shape[:2]
                # Upscale aggressively for small/distant faces; interpolation does
                # not create detail, but lets the detector use more input pixels.
                scale = min(5.0, 1024.0 / max(eh, ew, 1))
                if scale > 1.0:
                    crop = cv2.resize(crop, (max(1, int(ew * scale)), max(1, int(eh * scale))),
                                      interpolation=cv2.INTER_CUBIC)
                eye_crops.append((crop_name, crop, 1024))
        if not eye_crops:
            eye_crops = [("zoomed_eye_band", face_crop, 768)]

        # Run the full face plus multiple zoomed eye-region passes.
        passes = [("face", face_crop, 640)] + eye_crops
        scale = max((max(crop.shape[:2]) / max(1, max(face_crop.shape[:2])) for _, crop, _ in eye_crops), default=1.0)
        sg = gg = cl = 0.0
        best_source = {"sunglasses": None, "goggles": None, "clear_glasses": None}
        for source, crop, imgsz in passes:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            results = model(pil_img, verbose=False, conf=0.05, iou=0.45, imgsz=imgsz)
            crop_h, crop_w = crop.shape[:2]
            for r in results:
                if r.boxes is None:
                    continue
                boxes_xyxy = r.boxes.xyxy.tolist() if r.boxes.xyxy is not None else []
                for idx, (cid_raw, cf_raw) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
                    cid, cf = int(cid_raw), float(cf_raw)
                    kind = _eyewear_kind(model, cid)
                    if kind not in ("sunglasses", "goggles", "clear_glasses"):
                        continue
                    # Box-position validation: know WHERE the box falls (ModaNet lesson —
                    # each class has an expected anatomical region; detections outside it are noise).
                    # High-confidence hits (≥ YOLO_SG_CONF_HARD) bypass this — YOLO is trusted.
                    if idx < len(boxes_xyxy) and cf < YOLO_SG_CONF_HARD:
                        bx1_b, by1_b, bx2_b, by2_b = boxes_xyxy[idx]
                        bw_box = bx2_b - bx1_b
                        bcy = (by1_b + by2_b) / 2.0
                        if source == "face" and kind in ("sunglasses", "goggles"):
                            # Sunglasses box center must be roughly in the eye zone.
                            # Allow -15% above face top (glasses can sit on forehead)
                            # and up to 75% down (wrap-around frames, large lenses).
                            face_h_crop = max(1.0, float(rel_fy2 - rel_fy1))
                            face_w_crop = max(1.0, float(rel_fx2 - rel_fx1))
                            box_face_frac_y = (bcy - rel_fy1) / face_h_crop
                            box_face_frac_w = bw_box / face_w_crop
                            if not (-0.15 <= box_face_frac_y <= 0.75 and box_face_frac_w >= 0.10):
                                continue
                        elif source.startswith("zoomed") and kind in ("sunglasses", "goggles"):
                            # Eye band crops: reject detections narrower than 6% of crop — noise
                            if bw_box < 0.06 * max(1, crop_w):
                                continue
                    if kind == "sunglasses" and cf > sg:
                        sg, best_source["sunglasses"] = cf, source
                    elif kind == "goggles" and cf > gg:
                        gg, best_source["goggles"] = cf, source
                    elif kind == "clear_glasses" and cf > cl:
                        cl, best_source["clear_glasses"] = cf, source

        dark = max(sg, gg)
        # A positive on a tight, enlarged eye crop is more informative than the
        # same low confidence on the whole image. Keep a separate calibrated floor.
        zoom_hit = any(best_source.get(k) and best_source[k].startswith("zoomed_eye_band")
                       for k in ("sunglasses", "goggles"))
        # A zoomed crop is not independent evidence: interpolation can amplify
        # noise. Require the same calibrated confidence floor on every crop.
        zoom_hard_floor = YOLO_SG_CONF_HARD
        hard = dark >= YOLO_SG_CONF_HARD
        soft = (not hard) and dark >= YOLO_SG_CONF_SOFT
        veto = "clear_glasses" if (cl >= YOLO_CLEAR_VETO and not hard and not soft) else None
        return {
            "yolo_detected": bool(hard),
            "yolo_soft": bool(soft),
            "yolo_sunglasses": round(sg, 3),
            "yolo_goggles": round(gg, 3),
            "yolo_clear_glasses": round(cl, 3),
            "yolo_veto": veto,
            "yolo_best_source": best_source,
            "yolo_zoom_hit": bool(zoom_hit),
            "yolo_effective_hard_threshold": YOLO_SG_CONF_HARD,
            "yolo_class_labels": (list(model.names.values()) if isinstance(getattr(model, "names", None), dict)
                                  else list(getattr(model, "names", []) or [])),
            "zoomed_eye_band_scale": round(float(scale), 2),
            "zoomed_eye_band_sizes": [
                [int(crop.shape[1]), int(crop.shape[0])]
                for source, crop, _ in passes if source.startswith("zoomed_eye_band")
            ],
        }
    except Exception as exc:
        _yolo_warn_once("infer", f"inference failed, classical fallback only: {exc!r}")
        return None


def _detect_headwear_yolo(frame_cv2: np.ndarray, orig_bbox) -> Optional[dict]:
    try:
        if not _ULTRALYTICS_OK or orig_bbox is None or frame_cv2 is None:
            return None
        model = _load_yolo_oiv7()
        if model is None:
            return None

        img_h, img_w = frame_cv2.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
        fx2, fy2 = min(float(img_w), fx2), min(float(img_h), fy2)
        fw, fh = max(fx2 - fx1, 1.0), max(fy2 - fy1, 1.0)

        # Hats/helmets sit ABOVE the face bbox — extend crop upward aggressively.
        top_ext  = int(fh * HW_CROP_PAD_TOP_FRAC)
        side_pad = int(fw * HW_CROP_PAD_SIDE)
        x1 = max(0, int(fx1) - side_pad)
        y1 = max(0, int(fy1) - top_ext)
        x2 = min(img_w, int(fx2) + side_pad)
        y2 = min(img_h, int(fy2) + int(fh * 0.15))

        head_crop = frame_cv2[y1:y2, x1:x2]
        if head_crop.size == 0:
            return None

        ch, cw = head_crop.shape[:2]
        scale = min(4.0, 1024.0 / max(ch, cw, 1))
        if scale > 1.1:
            head_crop = cv2.resize(head_crop,
                                   (max(1, int(cw * scale)), max(1, int(ch * scale))),
                                   interpolation=cv2.INTER_CUBIC)

        pil_img = Image.fromarray(cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB))
        results = model(pil_img, verbose=False, conf=0.05, iou=0.45, imgsz=640)

        best_helmet = 0.0
        best_cap    = 0.0
        for r in results:
            if r.boxes is None:
                continue
            for cid_raw, cf_raw in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                cid, cf = int(cid_raw), float(cf_raw)
                kind = _headwear_kind(model, cid)
                if kind == "helmet" and cf > best_helmet:
                    best_helmet = cf
                elif kind == "cap" and cf > best_cap:
                    best_cap = cf

        best = max(best_helmet, best_cap)
        kind_best = "helmet" if best_helmet >= best_cap else "cap"
        hard = best >= HW_YOLO_CONF_HARD
        soft = (not hard) and best >= HW_YOLO_CONF_SOFT
        return {
            "yolo_detected":    bool(hard),
            "yolo_soft":        bool(soft),
            "yolo_helmet_conf": round(best_helmet, 3),
            "yolo_cap_conf":    round(best_cap, 3),
            "yolo_best_kind":   kind_best if (hard or soft) else None,
        }
    except Exception as exc:
        _yolo_warn_once("hw_infer", f"headwear YOLO failed: {exc!r}")
        return None


def _detect_headwear_yolo_fullimage(frame_cv2: np.ndarray) -> Optional[dict]:
    """Run OIV7 YOLO on the full image to find helmets/caps when no face is detected.
    Used for full-face helmets (motorcycle, football) that completely hide the face."""
    try:
        if not _ULTRALYTICS_OK or frame_cv2 is None:
            return None
        model = _load_yolo_oiv7()
        if model is None:
            return None

        img_h, img_w = frame_cv2.shape[:2]
        imgsz = min(1280, max(640, max(img_h, img_w)))
        pil_img = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        results = model(pil_img, verbose=False, conf=0.08, iou=0.45, imgsz=imgsz)

        best_helmet = 0.0
        best_cap    = 0.0
        helmet_box  = None
        cap_box     = None
        for r in results:
            if r.boxes is None:
                continue
            for box, cid_raw, cf_raw in zip(r.boxes.xyxy.tolist(),
                                             r.boxes.cls.tolist(),
                                             r.boxes.conf.tolist()):
                cid, cf = int(cid_raw), float(cf_raw)
                kind = _headwear_kind(model, cid)
                if kind == "helmet" and cf > best_helmet:
                    best_helmet = cf
                    helmet_box  = [round(v, 1) for v in box]
                elif kind == "cap" and cf > best_cap:
                    best_cap = cf
                    cap_box  = [round(v, 1) for v in box]

        best = max(best_helmet, best_cap)
        kind_best = "helmet" if best_helmet >= best_cap else "cap"
        hard = best >= HW_YOLO_CONF_HARD
        soft = (not hard) and best >= HW_YOLO_CONF_SOFT
        return {
            "yolo_detected":    bool(hard),
            "yolo_soft":        bool(soft),
            "yolo_helmet_conf": round(best_helmet, 3),
            "yolo_cap_conf":    round(best_cap, 3),
            "yolo_best_kind":   kind_best if (hard or soft) else None,
            "helmet_box":       helmet_box,
            "cap_box":          cap_box,
        }
    except Exception as exc:
        _yolo_warn_once("hw_full_infer", f"full-image headwear YOLO failed: {exc!r}")
        return None


def _detect_face_mask_classical(frame_cv2: np.ndarray, orig_bbox) -> Optional[dict]:
    """Multi-signal face mask detector.
    Compares nose/cheek region (skin) against mouth/chin region (mask if present).
    Uses saturation, texture, uniformity, and hue — works for any mask color.
    """
    try:
        if frame_cv2 is None or orig_bbox is None:
            return None
        img_h, img_w = frame_cv2.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
        fx2, fy2 = min(float(img_w), fx2), min(float(img_h), fy2)
        fw, fh = max(fx2 - fx1, 1.0), max(fy2 - fy1, 1.0)

        # mid = nose-tip / cheek area (always skin)
        # low = mouth + chin (covered by mask if present)
        mid_y1 = int(fy1 + fh * 0.28)
        mid_y2 = int(fy1 + fh * 0.54)
        low_y1 = int(fy1 + fh * 0.54)
        low_y2 = int(min(float(img_h), fy2))
        x1 = max(0, int(fx1 + fw * 0.08))
        x2 = min(img_w, int(fx2 - fw * 0.08))

        if mid_y2 <= mid_y1 or low_y2 <= low_y1 or x2 <= x1:
            return None

        mid_bgr = frame_cv2[mid_y1:mid_y2, x1:x2]
        low_bgr = frame_cv2[low_y1:low_y2, x1:x2]
        if mid_bgr.size == 0 or low_bgr.size == 0:
            return None

        mid_hsv = cv2.cvtColor(mid_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        low_hsv = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        mid_gray = cv2.cvtColor(mid_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        low_gray = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        mid_sat  = float(np.mean(mid_hsv[:, :, 1]))
        low_sat  = float(np.mean(low_hsv[:, :, 1]))
        mid_val  = float(np.mean(mid_hsv[:, :, 2]))
        low_val  = float(np.mean(low_hsv[:, :, 2]))
        low_std  = float(np.std(low_gray))
        mid_std  = float(np.std(mid_gray))
        brightness_diff = abs(mid_val - low_val)

        # Signal 1: saturation drop — skin is warm/saturated; masks are desaturated
        sat_drop = mid_sat - low_sat          # positive = lower face less saturated
        s1_sat   = sat_drop > 18             # surgical/cloth masks: typically large drop

        # Signal 2: texture contrast — skin has fine texture, mask is smooth
        mid_lap = float(cv2.Laplacian(mid_gray.astype(np.uint8), cv2.CV_64F).var())
        low_lap = float(cv2.Laplacian(low_gray.astype(np.uint8), cv2.CV_64F).var())
        tex_ratio = mid_lap / max(low_lap, 0.5)
        s2_tex   = tex_ratio > 2.0            # mid has 2× more texture than lower

        # Signal 3: lower face is uniformly smooth (low std)
        s3_unif  = low_std < 32 and low_std < mid_std * 0.75

        # Signal 4: brightness jump (any mask that differs from skin tone)
        s4_bright = brightness_diff > 22

        # Signal 5: lower-face hue is outside skin-tone range
        # Skin hue in HSV (OpenCV scale 0-180): ~0-25 for warm tones
        low_hue = float(np.mean(low_hsv[:, :, 0]))
        low_sat_val = low_sat
        skin_hue_ok = (low_hue < 28) and (low_sat_val > 28)  # warm+saturated = skin
        s5_nonskin  = not skin_hue_ok                          # True when lower face ≠ skin

        score = sum([s1_sat, s2_tex, s3_unif, s4_bright, s5_nonskin])
        # 2 signals = soft, 3+ = confident mask
        mask_signal = score >= 2

        return {
            "mask_signal":    bool(mask_signal),
            "mask_score":     int(score),
            "s1_sat_drop":    bool(s1_sat),
            "s2_texture":     bool(s2_tex),
            "s3_uniform":     bool(s3_unif),
            "s4_bright_diff": bool(s4_bright),
            "s5_nonskin":     bool(s5_nonskin),
            "sat_drop":       round(sat_drop, 1),
            "tex_ratio":      round(tex_ratio, 2),
            "lower_face_std": round(low_std, 1),
            "brightness_diff":round(brightness_diff, 1),
            "lower_hue":      round(low_hue, 1),
        }
    except Exception:
        return None


def _detect_headwear_classical(frame_cv2: np.ndarray, orig_bbox) -> Optional[dict]:
    try:
        if frame_cv2 is None or orig_bbox is None:
            return None
        img_h, img_w = frame_cv2.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
        fx2, fy2 = min(float(img_w), fx2), min(float(img_h), fy2)
        fw, fh = max(fx2 - fx1, 1.0), max(fy2 - fy1, 1.0)

        # Analyse two regions:
        #  above_region  → where helmet dome / cap body sits (above face top)
        #  forehead_region → top 20% of face bbox (helmet rim sits here)
        above_y1 = max(0, int(fy1 - fh * 0.45))
        above_y2 = max(above_y1 + 1, int(fy1))
        fore_y1  = int(fy1)
        fore_y2  = max(fore_y1 + 1, int(fy1 + fh * 0.22))
        mid_y1   = int(fy1 + fh * 0.30)
        mid_y2   = max(mid_y1 + 1, int(fy1 + fh * 0.60))
        x1 = max(0, int(fx1))
        x2 = min(img_w, int(fx2))

        if x2 <= x1:
            return None

        def _region_stats(y1, y2):
            if y2 <= y1:
                return None
            r = frame_cv2[y1:y2, x1:x2]
            if r.size == 0:
                return None
            g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            return {"mean": float(np.mean(g)), "std": float(np.std(g))}

        above_s = _region_stats(above_y1, above_y2)
        fore_s  = _region_stats(fore_y1,  fore_y2)
        mid_s   = _region_stats(mid_y1,   mid_y2)

        if above_s is None or fore_s is None or mid_s is None:
            return None

        # Signal 1 (dark brim): above/forehead region is dark — classic dark helmet or cap brim
        dark_brim = (fore_s["mean"] < HW_BRIM_DARK_THR and
                     float(np.mean(cv2.cvtColor(
                         frame_cv2[fore_y1:fore_y2, x1:x2], cv2.COLOR_BGR2GRAY
                     ) < HW_BRIM_DARK_THR)) >= HW_BRIM_COVERAGE_MIN)

        # Signal 2 (any-color helmet): above region is notably DIFFERENT from mid-face
        # (skin is consistent; a helmet introduces a new texture/color above the face)
        brightness_diff = abs(above_s["mean"] - mid_s["mean"])
        above_uniform   = above_s["std"] < 35        # helmets are smoother than background
        color_diff_signal = (brightness_diff > 22 and above_uniform)

        # Signal 3: forehead is cut off (helmet rim covers part of forehead — low std)
        fore_cut = fore_s["std"] < 28 and abs(fore_s["mean"] - mid_s["mean"]) > 18

        brim_signal = dark_brim or color_diff_signal or fore_cut

        return {
            "brim_signal":       bool(brim_signal),
            "dark_brim":         bool(dark_brim),
            "color_diff_signal": bool(color_diff_signal),
            "fore_cut":          bool(fore_cut),
            "brim_mean":         round(fore_s["mean"], 1),
            "above_mean":        round(above_s["mean"], 1),
            "mid_mean":          round(mid_s["mean"], 1),
            "brightness_diff":   round(brightness_diff, 1),
            "above_std":         round(above_s["std"], 1),
            "brim_dark_cov":     round(float(np.mean(
                cv2.cvtColor(frame_cv2[fore_y1:fore_y2, x1:x2], cv2.COLOR_BGR2GRAY)
                < HW_BRIM_DARK_THR)) if fore_y2 > fore_y1 else 0.0, 3),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("FaceValidationEngine")


def _exif_upright(img):
    """Apply the EXIF orientation tag so phone photos are not processed sideways
    (a rotated face breaks MTCNN/pose and is wrongly reported as side/no_human)."""
    try:
        from PIL import ImageOps as _ImageOps
        return _ImageOps.exif_transpose(img)
    except Exception:
        return img
# Keep the terminal clean: silence chatty third-party libraries and per-request access lines.
for _noisy in ("ultralytics", "transformers", "PIL", "urllib3", "httpx", "asyncio",
               "matplotlib", "numba", "uvicorn.access", "timm", "filelock", "absl"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {device}")

# ─────────────────────────────────────────────────────────────
# Output folders
# ─────────────────────────────────────────────────────────────
BASE_OUTPUT = "detection_outputs"
CATEGORY_FOLDERS = {
    "front":           os.path.join(BASE_OUTPUT, "front"),
    "side":            os.path.join(BASE_OUTPUT, "side"),
    "mixed":           os.path.join(BASE_OUTPUT, "mixed"),
    "eyes_closed":     os.path.join(BASE_OUTPUT, "eyes_closed"),
    "multiple_person": os.path.join(BASE_OUTPUT, "multiple_person"),
    "sunglasses":      os.path.join(BASE_OUTPUT, "sunglasses"),
    "no_human":        os.path.join(BASE_OUTPUT, "no_human"),
    "irrelevant":      os.path.join(BASE_OUTPUT, "irrelevant"),
    "no_landmarks":    os.path.join(BASE_OUTPUT, "no_landmarks"),
    "blurred":                   os.path.join(BASE_OUTPUT, "blurred"),
    "blur_uncertain":            os.path.join(BASE_OUTPUT, "blur_uncertain"),
    "blur_sunglasses_uncertain": os.path.join(BASE_OUTPUT, "blur_sunglasses_uncertain"),
    "low_resolution":            os.path.join(BASE_OUTPUT, "low_resolution"),
    "focused":                   os.path.join(BASE_OUTPUT, "focused"),
}
for _folder in CATEGORY_FOLDERS.values():
    os.makedirs(_folder, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Constants & thresholds
# ─────────────────────────────────────────────────────────────

# SolvePnP 3-D reference model (generic head, millimetre scale)
# Order: nose-tip, chin, left-eye-corner, right-eye-corner,
#        left-mouth-corner, right-mouth-corner
MODEL_POINTS_3D = np.array([
    (   0.0,    0.0,    0.0),
    (   0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)

# MediaPipe FaceMesh landmark indices
NOSE_TIP     = 1
CHIN         = 152
LEFT_EYE_MP  = 263
RIGHT_EYE_MP = 33
LEFT_MOUTH   = 287
RIGHT_MOUTH  = 57
NOSE_LEFT    = 279
NOSE_RIGHT   = 49
LEFT_CHEEK   = 234
RIGHT_CHEEK  = 454

# EAR landmarks — Soukupova-Cech order:
# [outer, upper-outer, upper-inner, inner, lower-inner, lower-outer]
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

# Sunglasses eye-polygon landmarks
SG_LEFT_EYE_POLY  = [33,  133, 160, 159, 158, 144, 145, 153]
SG_RIGHT_EYE_POLY = [362, 263, 387, 386, 385, 380, 381, 373]

# Detection / filtering
MTCNN_CONFIDENCE_THRESHOLD = 0.90
MIN_FACE_SIZE_PX           = 1     # minimum bbox side (original px space)

# Image dimension limits
IMG_MIN_DIM            = 1
IMG_MAX_DIM            = 8000
IMG_PROCESSING_MAX_DIM = 1600   # downscale target for all MediaPipe ops

# Batch limit
BATCH_MAX_FILE_BYTES = 25 * 1024 * 1024   # 25 MB

# Pose thresholds
YAW_FRONTAL_MAX       = 20.0
YAW_SLIGHTLY_SIDE_MAX = 35.0
ASYM_FRONTAL_MAX      = 12.0
ASYM_SLIGHTLY_MAX     = 22.0
MTCNN_FRONTAL_R_MIN   = 30
MTCNN_FRONTAL_R_MAX   = 62
MTCNN_FRONTAL_L_MIN   = 30
MTCNN_FRONTAL_L_MAX   = 63
NOSE_VISIBILITY_MIN   = 0.08

# EAR  (HIGH-SEVERITY: full/deep closure only, small-eye safe)
EAR_CLOSED_THRESHOLD     = 0.15   # only deeply shut eyes trigger this
EAR_MIN_RESOLUTION       = 80     # min face height in ORIGINAL px
EAR_RELATIVE_DROP_MIN    = 0.35   # EAR must be <= 35% of the eye's own open baseline
EAR_OPEN_BASELINE_FACTOR = 0.85   # eye width * this = estimated open EAR baseline 

# Sunglasses
SG_BRIGHTNESS_MAX    = 62   # dark + tinted lenses: black~10, gray~45, brown~55, amber~65
SG_CONTRAST_MAX      = 22
SG_HSV_V_MAX         = 65   # raised: tinted/mirrored lenses have HSV-V up to ~65
SG_DARK_COVERAGE_MIN = 0.48  # MediaPipe eye ROI is small; 48% coverage is enough
SG_DARK_PIXEL_THR    = 60   # count pixels up to brightness 60 as "dark" for tinted lenses
SG_IRIS_CONTRAST_MIN = 65
SG_IRIS_BRIGHT_MIN   = 70
SG_VOTE_THRESHOLD    = 4    # require all 4 signals — dark circles can't pass all 4
SG_REQUIRE_BOTH_EYES = False

YOLO_SG_CONF_HARD   = 0.50   # real goggles score 0.70+; 0.30–0.49 are dark-frame false positives
YOLO_SG_CONF_SOFT   = 0.25   # raised: marginal scores no longer feed evidence accumulator
YOLO_CLEAR_VETO     = 0.38
YOLO_CROP_PAD_FRAC  = 0.25
YOLO_EYE_BAND_Y     = (0.0, 0.65)
YOLO_EYE_BAND_X     = (-0.10, 1.10)
SG_EYE_VOTES        = 4    # require ALL 4 classical signals per eye
SG_EYE_VOTES_SOFT   = 3    # even with weak YOLO assist, still need 3/4

HW_YOLO_CONF_HARD    = 0.28   # headwear: YOLO hard-detect confidence floor
HW_YOLO_CONF_SOFT    = 0.13   # headwear: YOLO soft-assist floor
HW_VOTE_THRESHOLD    = 3      # votes needed to declare headwear detected
HW_CROP_PAD_TOP_FRAC = 0.60   # extend crop THIS much above face top (hat sits above face)
HW_CROP_PAD_SIDE     = 0.25   # side padding fraction
HW_FOREHEAD_TOP_FRAC = 0.22   # top portion of face bbox used for brim analysis
HW_BRIM_DARK_THR     = 80     # mean brightness below this in forehead → dark brim
HW_BRIM_COVERAGE_MIN = 0.55   # fraction of forehead pixels darker than threshold
HW_MASK_LOWER_FRAC   = 0.45   # lower face portion analysed for mask (from nose down)
HW_MASK_SKIN_DIFF    = 28     # lower-face brightness must differ from mid-face by this
HW_MASK_UNIFORM_THR  = 22     # std-dev of lower face below this → mask-like uniform cover

# Blur detection
BLUR_NORMALIZE_SIZE        = 256   # longest side used for consistent metric extraction
BLUR_SCORE_THRESHOLD       = 78.0  # raised further — valid doctor thumbnails score 44-103
BLUR_WHOLE_IMAGE_THRESHOLD = 65.0  # retained for compatibility; regional consensus is used
BLUR_MIN_FACE_PX           = 40    # below this, face sharpness is not reliable
BLUR_FACE_LOW_RES          = 50    # below this, report low_resolution rather than falsely claiming blur

# Framing / passport composition
FRAMING_TOP_MARGIN_MIN    = 0.02   # space above face top (for hair)
FRAMING_BOTTOM_MARGIN_MIN = 0.005  # space below face bottom (for neck)
FRAMING_FACE_H_MAX        = 0.85   # face_h/img_h > this → too zoomed in
FRAMING_FACE_W_MAX        = 0.80   # face_w/img_w > this → too zoomed in

# Drawing
LINE_COLOR = (255, 255, 0)
FONT       = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1
FONT_THICK = 2

# HTTP status codes
CATEGORY_STATUS_MAP = {
    "front":           200,
    "side":            400,
    "mixed":           400,
    "eyes_closed":     422,
    "sunglasses":      423,
    "blurred":                   422,
    "blur_uncertain":            422,
    "blur_sunglasses_uncertain": 422,
    "low_resolution":            422,
    "focused":                   422,
    "multiple_person": 409,
    "no_human":        404,
    "irrelevant":      415,
    "no_landmarks":    422,
}

# ─────────────────────────────────────────────────────────────
# Thread-safe MediaPipe instances
# ─────────────────────────────────────────────────────────────
_thread_local            = threading.local()
mp_face_detection_module = mp.solutions.face_detection
mp_face_mesh_module      = mp.solutions.face_mesh


def _get_face_detector():
    if not hasattr(_thread_local, "face_detector"):
        _thread_local.face_detector = mp_face_detection_module.FaceDetection(
            model_selection=1, min_detection_confidence=0.70
        )
    return _thread_local.face_detector


def _get_face_mesh():
    if not hasattr(_thread_local, "face_mesh"):
        _thread_local.face_mesh = mp_face_mesh_module.FaceMesh(
            static_image_mode=True, max_num_faces=10, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
    return _thread_local.face_mesh


# ─────────────────────────────────────────────────────────────
# MTCNN global
# ─────────────────────────────────────────────────────────────
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=True, keep_all=True, device=device,
)

# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app_: "FastAPI"):
    logger.info("Face Validation Engine v2.5.0 starting.")
    yield
    logger.info("Face Validation Engine shut down.")

app = FastAPI(
    title="Production Face Validation Engine",
    description="Multi-layer facial validation: pose · eyes · sunglasses · liveness",
    version="2.5.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# MODULE 0 — Image pre-processing
# ═══════════════════════════════════════════════════════════════

def validate_image_dimensions(frame_cv2: np.ndarray) -> Optional[str]:
    h, w = frame_cv2.shape[:2]
    if h < IMG_MIN_DIM or w < IMG_MIN_DIM:
        return f"Image too small: {w}x{h}px (min {IMG_MIN_DIM}px per side)"
    if h > IMG_MAX_DIM or w > IMG_MAX_DIM:
        return f"Image too large: {w}x{h}px (max {IMG_MAX_DIM}px per side)"
    return None


def _resize_for_processing(frame_cv2: np.ndarray) -> tuple:
    """
    Proportionally downscale so longest side <= IMG_PROCESSING_MAX_DIM.
    Returns (proc_frame, proc_scale).   proc_scale=1.0 means no resize.

    Coordinate contract:
      original_coord * proc_scale  = proc_coord
      proc_coord    / proc_scale   = original_coord
    """
    h, w    = frame_cv2.shape[:2]
    max_dim = max(h, w)
    if max_dim <= IMG_PROCESSING_MAX_DIM:
        return frame_cv2, 1.0
    scale   = IMG_PROCESSING_MAX_DIM / max_dim
    new_w   = max(1, int(w * scale))
    new_h   = max(1, int(h * scale))
    resized = cv2.resize(frame_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale



# ═══════════════════════════════════════════════════════════════
# MODULE 1B — Scene check (object / scenery / animal / cartoon / person)
# ═══════════════════════════════════════════════════════════════
import copy as _copy
import hashlib as _hashlib
from collections import OrderedDict as _OrderedDict


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


SCENE_CHECK_ENABLED      = os.environ.get("SCENE_CHECK_ENABLED", "1") == "1"
SCENE_CLIP_ENABLED       = os.environ.get("SCENE_CLIP_ENABLED", "1") == "1"
_LOCAL_CLIP_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clip")
SCENE_CLIP_MODEL         = os.environ.get(
    "SCENE_CLIP_MODEL",
    _LOCAL_CLIP_DIR if os.path.isdir(_LOCAL_CLIP_DIR) else "openai/clip-vit-base-patch32",
)
SCENE_MAX_SIDE           = int(_env_float("SCENE_MAX_SIDE", 640))
SCENE_CACHE_MAX          = int(_env_float("SCENE_CACHE_MAX", 256))

SCENE_PERSON_MIN         = _env_float("SCENE_PERSON_MIN", 0.30)
SCENE_CARTOON_MIN        = _env_float("SCENE_CARTOON_MIN", 0.40)
SCENE_CARTOON_STRONG     = _env_float("SCENE_CARTOON_STRONG", 0.75)
SCENE_HEUR_AGREE         = _env_float("SCENE_HEUR_AGREE", 0.30)
SCENE_HEUR_ALONE         = _env_float("SCENE_HEUR_ALONE", 0.75)
SCENE_UNCERTAIN_MAX      = _env_float("SCENE_UNCERTAIN_MAX", 0.45)
SCENE_MARGIN_MIN         = _env_float("SCENE_MARGIN_MIN", 0.08)
SCENE_YOLO_OTHER_MIN     = _env_float("SCENE_YOLO_OTHER_MIN", 0.35)

SCENE_BORDERLINE_CONF    = _env_float("SCENE_BORDERLINE_CONF", 0.97)
SCENE_SMALL_FACE_PX      = _env_float("SCENE_SMALL_FACE_PX", 40)
SCENE_FACE_CARTOON_MIN   = _env_float("SCENE_FACE_CARTOON_MIN", 0.60)
SCENE_FACE_CARTOON_ALONE = _env_float("SCENE_FACE_CARTOON_ALONE", 0.85)
SCENE_FACE_HEUR_ALONE    = _env_float("SCENE_FACE_HEUR_ALONE", 0.85)
SCENE_FACE_YOLO_MAX      = _env_float("SCENE_FACE_YOLO_MAX", 0.50)

_SCENE_PERSON_NAMES = {"person", "man", "woman", "boy", "girl",
                       "human face", "human head", "human body"}

_SCENE_PROMPTS = {
    "person":     ["a photo of a real person", "a close-up photo of a person's face",
                   "a portrait photograph of a human being"],
    "cartoon":    ["a cartoon character", "an anime illustration",
                   "a drawing or digital illustration of a character"],
    "object":     ["a photo of an object", "a product photo",
                   "a photo of something on a table"],
    "scenery":    ["a photo of scenery", "a landscape photo",
                   "a photo of a building or a street"],
    "animal":     ["a photo of an animal", "a photo of a pet",
                   "a photo of a bird or an insect"],
    "screenshot": ["a screenshot of an app or website", "a logo or text on a plain background",
                   "a scanned document"],
    "statue":     ["a photo of a statue or a mannequin", "a photo of a doll or a toy figure",
                   "a sculpture of a human"],
}

_scene_clip_state      = None
_scene_clip_failed     = False
_scene_clip_fail_count = 0
_scene_clip_load_lock  = threading.Lock()
_scene_clip_infer_lock = threading.Lock()
_scene_yolo_lock       = threading.Lock()
_scene_cache           = _OrderedDict()
_scene_cache_lock      = threading.Lock()


def _scene_unknown(reason: str, ms: float = 0.0) -> dict:
    return {"label": "unknown", "top_score": None, "person_score": 0.0, "yolo_person": 0.0,
            "yolo_other": 0.0, "cartoon_confirmed": False, "cartoon_confirmed_strict": False,
            "uncertain": True, "source": reason, "ms": round(ms, 1)}


def _scene_prepare(frame):
    try:
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return None
        f = frame
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.ndim == 3 and f.shape[2] == 1:
            f = cv2.cvtColor(f[..., 0], cv2.COLOR_GRAY2BGR)
        elif f.ndim == 3 and f.shape[2] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        elif f.ndim != 3 or f.shape[2] != 3:
            return None
        h, w = f.shape[:2]
        if h < 2 or w < 2:
            return None
        scale = 1.0
        if max(h, w) > SCENE_MAX_SIDE:
            scale = SCENE_MAX_SIDE / float(max(h, w))
            f = cv2.resize(f, (max(2, int(w * scale)), max(2, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(f), scale
    except Exception as exc:
        logger.warning("[scene] frame prepare failed: %r", exc)
        return None


def _scene_cache_key(small, extra: str = "") -> str:
    thumb = cv2.resize(small, (96, 96), interpolation=cv2.INTER_AREA)
    return _hashlib.md5(thumb.tobytes()).hexdigest() + f"|{small.shape[0]}x{small.shape[1]}|{extra}"


def _scene_cache_get(key: str):
    with _scene_cache_lock:
        v = _scene_cache.get(key)
        if v is not None:
            _scene_cache.move_to_end(key)
            return _copy.deepcopy(v)
    return None


def _scene_cache_put(key: str, value: dict) -> None:
    with _scene_cache_lock:
        _scene_cache[key] = _copy.deepcopy(value)
        _scene_cache.move_to_end(key)
        while len(_scene_cache) > SCENE_CACHE_MAX:
            _scene_cache.popitem(last=False)


def _load_scene_clip():
    global _scene_clip_state, _scene_clip_failed
    if _scene_clip_state is not None:
        return _scene_clip_state
    if _scene_clip_failed or not SCENE_CLIP_ENABLED:
        return None
    with _scene_clip_load_lock:
        if _scene_clip_state is not None:
            return _scene_clip_state
        if _scene_clip_failed:
            return None
        try:
            from transformers import CLIPModel, CLIPProcessor
            proc  = CLIPProcessor.from_pretrained(SCENE_CLIP_MODEL)
            model = CLIPModel.from_pretrained(SCENE_CLIP_MODEL).eval().to(device)
            classes, owner, prompts = list(_SCENE_PROMPTS.keys()), [], []
            for ci, k in enumerate(classes):
                for p in _SCENE_PROMPTS[k]:
                    owner.append(ci)
                    prompts.append(p)
            tok  = proc.tokenizer(prompts, padding=True, return_tensors="pt")
            text = {k: v.to(device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}
            _scene_clip_state = {"model": model, "proc": proc, "classes": classes,
                                 "owner": np.array(owner), "text": text}
            logger.info("[scene] CLIP loaded: %s | device=%s | prompts=%d",
                        SCENE_CLIP_MODEL, device, len(prompts))
        except Exception as exc:
            _scene_clip_failed = True
            logger.warning("[scene] CLIP unavailable -> YOLO + heuristic only: %r", exc)
            return None
    return _scene_clip_state


def _scene_clip_scores(small_bgr):
    global _scene_clip_failed, _scene_clip_fail_count
    st = _load_scene_clip()
    if st is None:
        return None
    try:
        pil = Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
        pix = st["proc"].image_processor(images=pil, return_tensors="pt")["pixel_values"].to(device)
        with _scene_clip_infer_lock, torch.inference_mode():
            out = st["model"](pixel_values=pix, **st["text"])
            logits = out.logits_per_image[0].float().cpu().numpy()
        cls_logits = np.array([logits[st["owner"] == i].mean() for i in range(len(st["classes"]))])
        e = np.exp(cls_logits - cls_logits.max())
        probs = e / e.sum()
        _scene_clip_fail_count = 0
        return {k: float(p) for k, p in zip(st["classes"], probs)}
    except Exception as exc:
        _scene_clip_fail_count += 1
        logger.warning("[scene] CLIP inference failed (%d): %r", _scene_clip_fail_count, exc)
        if _scene_clip_fail_count >= 3:
            _scene_clip_failed = True
            logger.warning("[scene] CLIP disabled after repeated failures")
        return None


def _scene_yolo_scores(small_bgr) -> dict:
    out = {"person": 0.0, "other": 0.0, "available": False}
    try:
        if not _ULTRALYTICS_OK:
            return out
        model = _load_yolo_oiv7()
        if model is None:
            return out
        names = getattr(model, "names", {}) or {}
        pil = Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
        with _scene_yolo_lock:
            res = model(pil, verbose=False, conf=0.15, iou=0.45, imgsz=640)
        out["available"] = True
        for r in res:
            if r.boxes is None:
                continue
            for cid_raw, cf_raw in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                cid = int(cid_raw)
                nm = str(names[cid] if isinstance(names, dict) else names[cid]).lower()
                cf = float(cf_raw)
                if nm in _SCENE_PERSON_NAMES:
                    out["person"] = max(out["person"], cf)
                else:
                    out["other"] = max(out["other"], cf)
    except Exception as exc:
        logger.warning("[scene] YOLO person scan failed: %r", exc)
    return out


def _cartoon_heuristic(small_bgr, bbox=None) -> dict:
    try:
        img = small_bgr
        if bbox is not None:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            mw, mh = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
            x1, y1 = int(max(0, x1 - mw)), int(max(0, y1 - mh))
            x2, y2 = int(min(w, x2 + mw)), int(min(h, y2 + mh))
            img = img[y1:y2, x1:x2]
        h, w = img.shape[:2]
        if min(h, w) < 24:
            return {"score": 0.0, "skipped": "too_small"}
        s = 128.0 / max(h, w)
        img = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
        den  = cv2.medianBlur(img, 3)
        gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy) / 4.0
        flat_frac    = float(np.mean(mag < 2.0))
        nonflat      = float(np.sum(mag >= 2.0))
        strong_ratio = float(np.sum(mag > 60.0)) / (nonflat + 1.0)
        a = float(np.clip((flat_frac - 0.25) / 0.40, 0.0, 1.0))
        b = float(np.clip((strong_ratio - 0.08) / 0.20, 0.0, 1.0))
        return {"score": round(math.sqrt(a * b), 3), "flat_frac": round(flat_frac, 3),
                "strong_ratio": round(strong_ratio, 3)}
    except Exception as exc:
        logger.warning("[scene] cartoon heuristic failed: %r", exc)
        return {"score": 0.0, "error": repr(exc)}


def scene_check(frame_cv2, face_bbox=None) -> dict:
    t0 = time.time()
    if not SCENE_CHECK_ENABLED:
        return _scene_unknown("disabled")
    prep = _scene_prepare(frame_cv2)
    if prep is None:
        return _scene_unknown("bad_frame", (time.time() - t0) * 1000)
    small, scale = prep

    bbox_s = None
    if face_bbox is not None:
        try:
            bbox_s = [float(v) * scale for v in list(face_bbox)[:4]]
        except Exception:
            bbox_s = None

    key = _scene_cache_key(small, "" if bbox_s is None else ",".join(f"{v:.0f}" for v in bbox_s))
    cached = _scene_cache_get(key)
    if cached is not None:
        cached["cached"] = True
        return cached

    clip = _scene_clip_scores(small)
    yolo = _scene_yolo_scores(small)
    heur = _cartoon_heuristic(small)
    fh   = _cartoon_heuristic(small, bbox_s) if bbox_s is not None else None
    h_sc = heur.get("score", 0.0)
    h_best = max(h_sc, (fh or {}).get("score", 0.0))

    if clip is not None:
        ranked    = sorted(clip.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        margin    = top_score - ranked[1][1]
        c         = clip.get("cartoon", 0.0)
        person_score = max(clip.get("person", 0.0), yolo["person"])
        confirmed = ((c >= SCENE_CARTOON_STRONG)
                     or (c >= SCENE_CARTOON_MIN and h_best >= SCENE_HEUR_AGREE)
                     or (h_best >= SCENE_HEUR_ALONE and yolo["person"] < SCENE_PERSON_MIN))
        strict    = ((c >= SCENE_FACE_CARTOON_ALONE)
                     or (c >= SCENE_FACE_CARTOON_MIN and h_best >= SCENE_HEUR_AGREE)
                     or (yolo["available"] and h_best >= SCENE_FACE_HEUR_ALONE and yolo["person"] < SCENE_PERSON_MIN))
        uncertain = (top_score < SCENE_UNCERTAIN_MAX) or (margin < SCENE_MARGIN_MIN)
        label     = "cartoon" if confirmed else top_label
        if confirmed:
            uncertain = False
        source = "clip+yolo+heuristic"
    else:
        top_score, margin = None, None
        person_score = yolo["person"]
        confirmed = (h_sc >= SCENE_HEUR_ALONE and yolo["person"] < SCENE_PERSON_MIN)
        strict    = (yolo["available"] and h_best >= SCENE_FACE_HEUR_ALONE
                     and yolo["person"] < SCENE_PERSON_MIN)
        if confirmed:
            label, uncertain = "cartoon", False
        elif yolo["person"] >= SCENE_PERSON_MIN:
            label, uncertain = "person", False
        elif yolo["available"] and yolo["other"] >= SCENE_YOLO_OTHER_MIN:
            label, uncertain = "object", False
        else:
            label, uncertain = "unknown", True
        source = "yolo+heuristic"

    out = {"label": label,
           "top_score": None if top_score is None else round(top_score, 3),
           "margin": None if margin is None else round(margin, 3),
           "person_score": round(person_score, 3),
           "yolo_person": round(yolo["person"], 3), "yolo_other": round(yolo["other"], 3),
           "cartoon_confirmed": bool(confirmed), "cartoon_confirmed_strict": bool(strict),
           "uncertain": bool(uncertain), "heuristic": heur, "source": source}
    if fh is not None:
        out["face_heuristic"] = fh
    if clip is not None:
        out["clip"] = {k: round(v, 3) for k, v in clip.items()}
    out["ms"] = round((time.time() - t0) * 1000, 1)
    _scene_cache_put(key, out)
    return out


def scene_check_safe(frame_cv2, face_bbox=None) -> dict:
    t0 = time.time()
    try:
        return scene_check(frame_cv2, face_bbox)
    except Exception as exc:
        logger.warning("[scene] scene_check failed, falling back to old logic: %r", exc)
        d = _scene_unknown("error", (time.time() - t0) * 1000)
        d["error"] = repr(exc)
        return d


def _scene_summary(scene: dict) -> str:
    return ("label=%s top=%s person=%s yolo_p=%s cartoon=%s strict=%s unc=%s heur=%s src=%s ms=%s%s" % (
        scene.get("label"), scene.get("top_score"), scene.get("person_score"),
        scene.get("yolo_person"), scene.get("cartoon_confirmed"),
        scene.get("cartoon_confirmed_strict"), scene.get("uncertain"),
        (scene.get("heuristic") or {}).get("score"), scene.get("source"), scene.get("ms"),
        " (cached)" if scene.get("cached") else ""))


def _scene_verdict_no_face(scene: dict) -> str:
    if scene.get("cartoon_confirmed"):
        return "cartoon"
    if scene.get("person_score", 0.0) >= SCENE_PERSON_MIN:
        return "person_present"
    # Non-person label with low person score → reject even if uncertain
    label = scene.get("label", "unknown")
    if label in ("object", "scenery", "animal", "screenshot", "statue", "cartoon"):
        return "nonperson"
    # Truly unknown (CLIP + YOLO both unavailable) → let old skin rule decide
    if label == "unknown" and scene.get("source") in ("disabled", "bad_frame"):
        return "uncertain"
    # Everything else: no person detected, not a person image → reject
    return "nonperson"


def _scene_reject_face_as_cartoon(scene: dict) -> bool:
    return bool(scene.get("cartoon_confirmed_strict")
                and scene.get("yolo_person", 0.0) < SCENE_FACE_YOLO_MAX)


def run_scene_eval(argv=None):
    import argparse, csv, glob
    ap = argparse.ArgumentParser(prog="main.py --eval-scene")
    ap.add_argument("dataset")
    ap.add_argument("--no-clip", action="store_true", help="YOLO + heuristic only")
    ap.add_argument("--csv", default="eval_report.csv")
    args = ap.parse_args(argv)
    if args.no_clip:
        globals()["SCENE_CLIP_ENABLED"] = False

    classes = ["person", "cartoon", "object", "scenery", "animal", "screenshot", "statue"]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    rows, false_rej, missed = [], [], []
    for cls in classes:
        files = sorted(sum((glob.glob(os.path.join(args.dataset, cls, e)) for e in exts), []))
        for fp in files:
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if img is None:
                print("unreadable:", fp)
                continue
            sc  = scene_check_safe(img)
            v   = _scene_verdict_no_face(sc)
            rej = _scene_reject_face_as_cartoon(sc)
            rows.append(dict(cls=cls, path=fp, verdict=v, face_path_reject=rej, label=sc.get("label"),
                             top=sc.get("top_score"), person=sc.get("person_score"),
                             yolo_person=sc.get("yolo_person"), cartoon_confirmed=sc.get("cartoon_confirmed"),
                             strict=sc.get("cartoon_confirmed_strict"), uncertain=sc.get("uncertain"),
                             heuristic=(sc.get("heuristic") or {}).get("score"),
                             ms=sc.get("ms"), source=sc.get("source")))
            if cls == "person" and (v in ("cartoon", "nonperson") or rej):
                false_rej.append((fp, v, rej, sc.get("label"), sc.get("person_score")))
            if cls in ("cartoon", "object", "scenery", "animal", "screenshot") and v not in ("cartoon", "nonperson"):
                missed.append((fp, cls, v, sc.get("label"), sc.get("person_score")))
    if not rows:
        print("No images found. Expected folders under", args.dataset, ":", ", ".join(classes))
        return 1
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    avg_ms = sum((r["ms"] or 0) for r in rows) / len(rows)
    print(f"\nsource: {rows[0]['source']}   images: {len(rows)}   avg {avg_ms:.0f} ms\n")
    print(f"{'class':11s}{'n':>5s}{'cartoon':>9s}{'nonperson':>11s}{'person':>8s}{'uncertain':>11s}{'face-rej':>10s}")
    for cls in classes:
        R = [r for r in rows if r["cls"] == cls]
        if not R:
            continue
        pct = lambda k: 100.0 * sum(r["verdict"] == k for r in R) / len(R)
        fr  = 100.0 * sum(bool(r["face_path_reject"]) for r in R) / len(R)
        print(f"{cls:11s}{len(R):5d}{pct('cartoon'):8.0f}%{pct('nonperson'):10.0f}%"
              f"{pct('person_present'):7.0f}%{pct('uncertain'):10.0f}%{fr:9.0f}%")
    n_person = sum(r["cls"] == "person" for r in rows)
    if n_person:
        print(f"\nFALSE REJECTS on person/: {len(false_rej)} of {n_person}   <- must be 0")
    for fp, v, rej, lab, ps in false_rej[:15]:
        print(f"   {fp}  verdict={v} face_reject={rej} label={lab} person={ps}")
    print(f"\nMissed (non-person / cartoon not rejected): {len(missed)}")
    for fp, cls, v, lab, ps in missed[:15]:
        print(f"   [{cls}] {fp}  verdict={v} label={lab} person={ps}")
    print(f"\nPer-image scores: {args.csv}")
    return 0


# ═══════════════════════════════════════════════════════════════
# MODULE 1 — No Human / Irrelevant
# ═══════════════════════════════════════════════════════════════

def _skin_ratio(frame_cv2: np.ndarray) -> float:
    """Fraction of pixels that match skin-tone HSV ranges (light and dark skin)."""
    hsv      = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2HSV)
    total_px = frame_cv2.shape[0] * frame_cv2.shape[1]
    mask1    = cv2.inRange(hsv, np.array([0, 20, 60], np.uint8),
                                np.array([25, 255, 255], np.uint8))
    mask2    = cv2.inRange(hsv, np.array([0, 10, 30], np.uint8),
                                np.array([30, 180, 180], np.uint8))
    return float(np.sum(cv2.bitwise_or(mask1, mask2) > 0)) / (total_px + 1e-8)


def detect_no_human(valid_bboxes: list, valid_probs: list) -> dict:
    """Called AFTER size+confidence filtering (single filter chain)."""
    if not valid_bboxes:
        return {"is_human": False, "reason": "No face passed filter",
                "max_confidence": 0.0}
    probs = [float(p) for p in valid_probs if p is not None]
    if not probs:
        return {"is_human": False, "reason": "All detections returned null confidence",
                "max_confidence": 0.0}
    return {"is_human": True, "reason": "Human face detected",
            "max_confidence": round(max(probs), 4)}


def detect_irrelevant_image(valid_bboxes: list, frame_cv2: np.ndarray) -> dict:
    """
    irrelevant : no face AND skin < 3%
    no_human   : no face BUT skin >= 3%
    """
    if valid_bboxes:
        return {"is_irrelevant": False, "reason": "Face detected"}
    skin = _skin_ratio(frame_cv2)
    if skin < 0.03:
        return {"is_irrelevant": True,
                "reason": f"No face + very low skin coverage ({skin:.3f})",
                "skin_ratio": round(skin, 4)}
    return {"is_irrelevant": False,
            "reason": f"No face but skin present ({skin:.3f}) -> no_human",
            "skin_ratio": round(skin, 4)}


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — Multiple person detection
# ═══════════════════════════════════════════════════════════════

def detect_persons_and_size(proc_frame: np.ndarray) -> dict:
    """Uses MediaPipe FaceDetection on the downscaled frame to count faces. Triggers multiple_person only when more than one face is found."""
    face_detector = _get_face_detector()
    rgb           = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    result        = face_detector.process(rgb)
    if not result.detections:
        return {"valid": True, "total_faces": 0,
                "category": "ok", "reason": "No detections — deferred to MTCNN"}
    total = len(result.detections)
    if total > 1:
        return {"valid": False, "total_faces": total,
                "category": "multiple_person",
                "reason": f"{total} persons detected. Only one allowed."}
    return {"valid": True, "total_faces": 1,
            "category": "ok", "reason": "Single face detected."}


# ═══════════════════════════════════════════════════════════════
# MODULE 3 — Sunglasses / dark-goggle detection
# ═══════════════════════════════════════════════════════════════

def _extract_eye_roi(proc_frame: np.ndarray, face_lm, indices: list):
    """Returns (gray, hsv) for the eye polygon, or (None, None) if too small."""
    h, w = proc_frame.shape[:2]
    lm   = face_lm.landmark
    xs   = [int(lm[i].x * w) for i in indices]
    ys   = [int(lm[i].y * h) for i in indices]
    x1, x2 = max(0, min(xs) - 4), min(w, max(xs) + 4)
    y1, y2 = max(0, min(ys) - 4), min(h, max(ys) + 4)
    roi    = proc_frame[y1:y2, x1:x2]
    if roi.size < 16:
        return None, None
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


_clip_model  = None
_clip_proc   = None
_clip_failed = False

def _load_clip():
    global _clip_model, _clip_proc, _clip_failed
    if _clip_failed or _clip_model is not None:
        return _clip_model, _clip_proc
    try:
        from transformers import CLIPProcessor, CLIPModel
        _CLIP_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clip")
        _clip_src   = _CLIP_LOCAL if os.path.isdir(_CLIP_LOCAL) else "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(_clip_src)
        _clip_proc  = CLIPProcessor.from_pretrained(_clip_src)
        _clip_model.eval()
        logger.info("[CLIP] model loaded")
    except Exception as exc:
        _clip_failed = True
        logger.warning(f"[CLIP] load failed: {exc!r}")
    return _clip_model, _clip_proc


def _sg_clip_score(frame_cv2: np.ndarray, orig_bbox) -> Optional[float]:
    """CLIP zero-shot sunglasses probability on face crop.
    Returns probability that the face is wearing sunglasses (0.0–1.0), or None."""
    try:
        model, proc = _load_clip()
        if model is None or frame_cv2 is None or orig_bbox is None:
            return None
        import torch
        ih, iw = frame_cv2.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in orig_bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        crop = frame_cv2[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if min(crop.shape[:2]) < 32:
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        texts = ["a person wearing sunglasses or dark goggles",
                 "a person with no sunglasses"]
        inputs = proc(text=texts, images=pil_crop, return_tensors="pt", padding=True)
        with torch.no_grad():
            out    = model(**inputs)
            logits = out.logits_per_image
            probs  = logits.softmax(dim=1)[0]
        return float(probs[0])
    except Exception as exc:
        logger.debug(f"[CLIP/SG] inference error: {exc!r}")
        return None


def _sg_eye_openness(face_landmarks, proc_frame: np.ndarray) -> Optional[dict]:
    """Eye-openness from FaceMesh EAR (Eye Aspect Ratio).
    Sunglasses block the iris — EAR drops sharply even on open eyes.
    Returns dict with left/right EAR and suppressed flag, or None."""
    try:
        if face_landmarks is None or proc_frame is None:
            return None
        h, w = proc_frame.shape[:2]
        def _lm(idx):
            lm = face_landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
        # MediaPipe FaceMesh EAR landmarks per eye:
        # vertical pairs: (159,145), (158,153), (160,144)  horizontal: (33,133)
        def _ear(v_pairs, h_pair):
            num = sum(np.linalg.norm(_lm(a) - _lm(b)) for a, b in v_pairs)
            den = np.linalg.norm(_lm(h_pair[0]) - _lm(h_pair[1])) * len(v_pairs)
            return num / max(den, 0.001)
        l_ear = _ear([(159, 145), (158, 153), (160, 144)], (33, 133))
        r_ear = _ear([(386, 374), (385, 380), (387, 373)], (362, 263))
        # Normal open eye EAR ≈ 0.25–0.35. Glasses push it below ~0.15
        # because lens occludes the visible eye slit in landmark detection.
        threshold = 0.15
        l_suppressed = l_ear < threshold
        r_suppressed = r_ear < threshold
        both_suppressed = l_suppressed and r_suppressed
        return {"l_ear": round(l_ear, 4), "r_ear": round(r_ear, 4),
                "l_suppressed": l_suppressed, "r_suppressed": r_suppressed,
                "both_suppressed": both_suppressed}
    except Exception:
        return None


def _sg_relative_brightness(frame_cv2: np.ndarray, orig_bbox) -> Optional[dict]:
    """Eye-band brightness relative to cheek skin. Works for all lens types including tinted/amber."""
    try:
        if frame_cv2 is None or orig_bbox is None:
            return None
        ih, iw = frame_cv2.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in orig_bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        ey1 = y1 + int(0.18 * fh); ey2 = y1 + int(0.48 * fh)
        cy1 = y1 + int(0.55 * fh); cy2 = y1 + int(0.78 * fh)
        eye_band   = frame_cv2[max(0, ey1):min(ih, ey2), x1:x2]
        cheek_band = frame_cv2[max(0, cy1):min(ih, cy2), x1:x2]
        if eye_band.size < 100 or cheek_band.size < 100:
            return None
        eye_v   = float(np.mean(cv2.cvtColor(eye_band,   cv2.COLOR_BGR2GRAY)))
        cheek_v = float(np.mean(cv2.cvtColor(cheek_band, cv2.COLOR_BGR2GRAY)))
        ratio   = eye_v / max(cheek_v, 1.0)
        eye_s   = float(np.mean(cv2.cvtColor(eye_band,   cv2.COLOR_BGR2HSV)[:, :, 1]))
        cheek_s = float(np.mean(cv2.cvtColor(cheek_band, cv2.COLOR_BGR2HSV)[:, :, 1]))
        sat_drop = cheek_s - eye_s
        return {"eye_v": round(eye_v, 1), "cheek_v": round(cheek_v, 1),
                "ratio": round(ratio, 3), "eye_s": round(eye_s, 1),
                "cheek_s": round(cheek_s, 1), "sat_drop": round(sat_drop, 1)}
    except Exception:
        return None


def _sg_lens_texture(gray_roi: np.ndarray) -> Optional[float]:
    """Laplacian variance of eye ROI: lens is smooth (<25), bare skin/iris is textured (>50)."""
    try:
        if gray_roi is None or gray_roi.size < 16:
            return None
        g = gray_roi if (gray_roi.shape[0] >= 4 and gray_roi.shape[1] >= 4) else cv2.resize(gray_roi, (20, 20))
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        return None


def _sg_iris_suppression(gray_roi: np.ndarray) -> Optional[bool]:
    """Gradient suppression: a lens hides the iris, flattening the gradient in the eye ROI center."""
    try:
        if gray_roi is None or gray_roi.size < 25:
            return None
        g = cv2.resize(gray_roi, (32, 16)) if gray_roi.shape[1] < 16 else gray_roi
        sobelx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad   = np.sqrt(sobelx**2 + sobely**2)
        h, w   = grad.shape
        cx1, cx2 = w // 4, 3 * w // 4
        cy1, cy2 = h // 4, 3 * h // 4
        center_grad = float(np.mean(grad[cy1:cy2, cx1:cx2]))
        outer_grad  = float(np.mean(grad))
        ratio = center_grad / max(outer_grad, 0.1)
        return ratio < 1.4
    except Exception:
        return None


def _classical_sunglasses_crop_evidence(frame_cv2: np.ndarray, orig_bbox) -> dict:
    """Robust cropped-face sunglasses fallback.

    Searches several overlapping eye-band positions rather than assuming the
    face box contains a complete, normally aligned face. This is important when
    the forehead/top of the head is clipped by the image boundary or detector box.
    Uses paired dark lens components OR strong horizontal lens-band evidence.
    Returns detailed measurements for debugging.
    """
    out = {"detected": False, "method": "multi_band_dark_lenses",
           "reason": "inconclusive", "candidate_bands": []}
    try:
        if frame_cv2 is None or orig_bbox is None or getattr(frame_cv2, "size", 0) == 0:
            out["reason"] = "missing_frame_or_bbox"
            return out

        ih, iw = frame_cv2.shape[:2]
        raw = [float(v) for v in orig_bbox[:4]]
        raw_x1, raw_y1, raw_x2, raw_y2 = raw
        clipped_top = raw_y1 <= max(2.0, 0.015 * ih)
        bx1, by1 = max(0, int(round(raw_x1))), max(0, int(round(raw_y1)))
        bx2, by2 = min(iw, int(round(raw_x2))), min(ih, int(round(raw_y2)))
        fw, fh = bx2 - bx1, by2 - by1
        bbox_aspect = fw / max(float(fh), 1.0)
        out.update({"bbox": [bx1, by1, bx2, by2],
                    "bbox_touches_top": bool(clipped_top),
                    "face_size": [fw, fh],
                    "bbox_aspect_ratio": round(float(bbox_aspect), 3)})
        if fw < 16 or fh < 16:
            out["reason"] = "face_bbox_too_small"
            return out
        # Reject obviously non-face-like boxes (e.g. torso/background boxes),
        # which otherwise can contain two dark shirt panels resembling lenses.
        if bbox_aspect > 1.45:
            out["reason"] = "bbox_not_face_like"
            return out

        # Expand sideways to include sunglass temples. Do not assume y=27%-57%
        # is always the eye line: cropped boxes may place eyes much closer to top.
        xpad = int(round(0.16 * fw))
        x1, x2 = max(0, bx1 - xpad), min(iw, bx2 + xpad)
        if clipped_top:
            y_ranges = (
                (0.02, 0.30), (0.05, 0.38), (0.10, 0.43),
                (0.16, 0.49), (0.22, 0.56), (0.28, 0.64),
                (0.34, 0.70), (0.20, 0.72),
            )
        else:
            y_ranges = (
                (0.10, 0.38), (0.16, 0.46), (0.20, 0.52),
                (0.25, 0.58), (0.30, 0.64), (0.18, 0.70),
            )

        best = None
        for band_index, (top_frac, bottom_frac) in enumerate(y_ranges):
            y1 = max(0, by1 + int(round(top_frac * fh)))
            y2 = min(ih, by1 + max(int(round(bottom_frac * fh)), int(round((top_frac + 0.12) * fh))))
            band = frame_cv2[y1:y2, x1:x2]
            if band.size == 0 or band.shape[0] < 4 or band.shape[1] < 10:
                continue

            original_band_size = [int(band.shape[1]), int(band.shape[0])]
            scale = min(6.0, max(1.0, 320.0 / float(band.shape[1])))
            if scale > 1.0:
                band = cv2.resize(
                    band, (int(round(band.shape[1] * scale)), int(round(band.shape[0] * scale))),
                    interpolation=cv2.INTER_CUBIC
                )
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            bh, bw = gray.shape[:2]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            band_best = None
            band_dark_fraction = float(np.mean(gray < 65))
            band_brightness = float(np.mean(gray))
            max_row_coverage = float(np.max(np.mean(gray < 65, axis=1))) if bh else 0.0

            for threshold in (45, 60, 75):
                mask = (gray < threshold).astype(np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                n, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
                comps = []
                min_area = max(12, int(0.0018 * bw * bh))
                for i in range(1, n):
                    xx, yy, ww, hh, area = [int(v) for v in stats[i]]
                    cx, cy = [float(v) for v in centers[i]]
                    if area < min_area or ww < 0.055 * bw or hh < 0.10 * bh:
                        continue
                    if cy < 0.08 * bh or cy > 0.94 * bh:
                        continue
                    aspect = ww / max(hh, 1)
                    if not (0.45 <= aspect <= 4.5):
                        continue
                    comps.append({"x": xx, "y": yy, "w": ww, "h": hh,
                                  "area": area, "cx": cx, "cy": cy, "aspect": aspect})
                for i, a in enumerate(comps):
                    for b in comps[i + 1:]:
                        left, right = (a, b) if a["cx"] <= b["cx"] else (b, a)
                        gap = right["cx"] - left["cx"]
                        dy = abs(left["cy"] - right["cy"])
                        area_ratio = min(left["area"], right["area"]) / max(left["area"], right["area"])
                        if not (0.12 * bw <= gap <= 0.78 * bw):
                            continue
                        if dy > 0.32 * bh or area_ratio < 0.20:
                            continue
                        if min(left["w"], right["w"]) < 0.065 * bw:
                            continue
                        pair_score = (
                            0.30 * area_ratio
                            + 0.30 * max(0.0, 1.0 - dy / max(0.32 * bh, 1.0))
                            + 0.20 * min(1.0, gap / max(0.35 * bw, 1.0))
                            + 0.20 * min(1.0, (left["area"] + right["area"]) / max(0.10 * bw * bh, 1.0))
                        )
                        if band_best is None or pair_score > band_best["score"]:
                            band_best = {"score": pair_score, "left": left, "right": right,
                                         "threshold": threshold, "kind": "paired_lenses"}

            # Some sunglasses have a connected bridge so both lenses form one
            # component. Use a horizontal dark-strip cue only when it is broad
            # and dark, and record it as weaker evidence than a paired-lens hit.
            strip_score = (
                0.45 * min(1.0, max_row_coverage / 0.78)
                + 0.35 * min(1.0, band_dark_fraction / 0.38)
                + 0.20 * max(0.0, 1.0 - band_brightness / 130.0)
            )
            strip_candidate = (
                max_row_coverage >= 0.66 and
                band_dark_fraction >= 0.16 and
                band_brightness < 105 and
                bw >= 40
            )
            if strip_candidate and (band_best is None or strip_score > band_best["score"]):
                band_best = {"score": strip_score, "threshold": 65, "kind": "horizontal_dark_eye_band"}

            candidate = {
                "band_index": band_index, "vertical_range": [top_frac, bottom_frac],
                "band_size_original": original_band_size, "band_size_processed": [int(bw), int(bh)],
                "brightness": round(band_brightness, 2),
                "dark_fraction": round(band_dark_fraction, 4),
                "max_row_coverage": round(max_row_coverage, 4),
                "candidate": band_best is not None,
                "candidate_kind": band_best["kind"] if band_best else None,
                "score": round(float(band_best["score"]), 4) if band_best else 0.0,
            }
            out["candidate_bands"].append(candidate)
            if band_best is not None:
                item = {**candidate, **band_best, "y_origin": y1, "x_origin": x1}
                if best is None or item["score"] > best["score"]:
                    best = item

        if best:
            detected = (best["kind"] == "paired_lenses" and best["score"] >= 0.22) or (
                best["kind"] == "horizontal_dark_eye_band" and best["score"] >= 0.68
            )
            out.update({
                "detected": bool(detected),
                "confidence": round(float(best["score"]), 3),
                "reason": best["kind"] if detected else "weak_dark_band_candidate",
                "best_band": {
                    "vertical_range": best["vertical_range"],
                    "band_size_original": best["band_size_original"],
                    "brightness": best["brightness"],
                    "dark_fraction": best["dark_fraction"],
                    "max_row_coverage": best["max_row_coverage"],
                    "kind": best["kind"],
                },
            })
            if best["kind"] == "paired_lenses":
                out["lens_boxes"] = [
                    [best["left"]["x"], best["left"]["y"], best["left"]["w"], best["left"]["h"]],
                    [best["right"]["x"], best["right"]["y"], best["right"]["w"], best["right"]["h"]],
                ]
        else:
            out["reason"] = "no_pair_or_horizontal_dark_band"
        return out
    except Exception as exc:
        out["reason"] = "exception"
        out["error"] = repr(exc)
        return out

def detect_sunglasses(proc_frame: np.ndarray, face_landmarks,
                      frame_cv2_orig: np.ndarray = None,
                      orig_bbox=None) -> dict:
    """
    Stage 1  YOLO OIV7: confident hit -> flag; clear-glasses -> veto; weak hit -> 'soft'.
    Stage 2  classical, evaluated PER EYE:
             an eye is dark when >= SG_EYE_VOTES of 4 signals fire (2 when YOLO was soft)
             sunglasses  <=>  EVERY measurable eye is dark
    """
    yolo = _detect_sunglasses_yolo(frame_cv2_orig, orig_bbox)

    # YOLO hard hit → immediate flag
    if yolo is not None and yolo["yolo_detected"]:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": (f"YOLO: sunglasses={yolo['yolo_sunglasses']:.2f} "
                           f"goggles={yolo['yolo_goggles']:.2f}"),
                "details": yolo}

    # Clear-glasses veto → immediate pass
    if yolo is not None and yolo["yolo_veto"] == "clear_glasses":
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": f"YOLO: clear prescription glasses (conf={yolo['yolo_clear_glasses']:.2f})",
                "details": yolo}

    # Hard uncertain floor: face < 20 px is too small for any detector to be reliable.
    if frame_cv2_orig is not None and orig_bbox is not None:
        try:
            _ih0, _iw0 = frame_cv2_orig.shape[:2]
            _fx1, _fy1, _fx2, _fy2 = [float(v) for v in orig_bbox[:4]]
            _fw0 = max(0.0, min(_iw0, _fx2) - max(0.0, _fx1))
            _fh0 = max(0.0, min(_ih0, _fy2) - max(0.0, _fy1))
            if min(_fw0, _fh0) < 20:
                return {"detected": False, "status": "low_resolution_uncertain", "votes": 0,
                        "reason": f"Face too small for any reliable detection ({int(_fw0)}x{int(_fh0)} px)",
                        "details": {"face_size": [int(_fw0), int(_fh0)]}}
        except Exception:
            pass

    soft = bool(yolo and yolo.get("yolo_soft"))

    # glasses-detector classifier — purpose-trained model (F1 ~0.95), run before gate decisions.
    clf_prob  = _detect_sg_classifier(frame_cv2_orig, orig_bbox)
    yolo_info = dict(yolo) if yolo else {}
    clf_info  = {"classifier_prob": round(clf_prob, 3) if clf_prob is not None else None}

    # STAGE 0 — Classifier is now the primary detector (threshold = 0.50, majority vote).
    if clf_prob is not None and clf_prob >= 0.65:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": f"Classifier: sunglasses prob={clf_prob:.2f}",
                "details": {**yolo_info, **clf_info}}

    # Classifier very confident no sunglasses → requires strong corroborating evidence.
    clf_soft_veto = (clf_prob is not None and clf_prob <= 0.20)

    # STAGE 0c — CLIP zero-shot (tiebreaker only — not standalone).
    # CLIP softmax is miscalibrated for binary face classification; it cannot
    # independently declare sunglasses. Use only to break ties when other signals
    # already show partial evidence.
    clip_prob = _sg_clip_score(frame_cv2_orig, orig_bbox)
    clip_info = {"clip_prob": round(clip_prob, 3) if clip_prob is not None else None}

    # YOLO soft + classifier partially confirms → flag.
    if soft and clf_prob is not None and clf_prob >= 0.55:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": f"YOLO soft + Classifier: prob={clf_prob:.2f}",
                "details": {**yolo_info, **clf_info, **clip_info}}
    # YOLO soft + CLIP strongly confirms (threshold raised to avoid false positives).
    if soft and clip_prob is not None and clip_prob >= 0.80:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": f"YOLO soft + CLIP: prob={clip_prob:.2f}",
                "details": {**yolo_info, **clf_info, **clip_info}}

    # Evidence accumulator — replaces the old YOLO hard gate.
    # Accumulates signals; score >= 3 -> detected, independent of YOLO verdict.
    # This handles tinted/amber/low-res cases where YOLO is uncertain.
    ev = {}; ev_score = 0
    if soft:
        ev["yolo_soft"] = True; ev_score += 2

    _rb = _sg_relative_brightness(frame_cv2_orig, orig_bbox)
    ev["relative_brightness"] = _rb
    if _rb is not None:
        if _rb.get("ratio", 1.0) < 0.65:
            ev["rb_dark"] = True; ev_score += 2
        if _rb.get("sat_drop", 0.0) > 30:
            ev["sat_drop_signal"] = True; ev_score += 1

    # Small-face check: if face is tiny and evidence is weak, return uncertain.
    if frame_cv2_orig is not None and orig_bbox is not None:
        try:
            _ih, _iw = frame_cv2_orig.shape[:2]
            _bx1, _by1, _bx2, _by2 = [float(v) for v in orig_bbox[:4]]
            _face_w = max(0.0, min(_iw, _bx2) - max(0.0, _bx1))
            _face_h = max(0.0, min(_ih, _by2) - max(0.0, _by1))
            if min(_face_w, _face_h) < 40 and ev_score < 3:
                _details = {**yolo_info, **clf_info, "evidence": ev, "ev_score": ev_score}
                _details["face_size"] = [int(_face_w), int(_face_h)]
                return {"detected": False, "status": "uncertain", "votes": 0,
                        "reason": "Face too small; insufficient evidence for sunglasses",
                        "details": _details}
        except Exception:
            pass

    # Early evidence decision before landmark extraction.
    if ev_score >= 4 and not clf_soft_veto:
        return {"detected": True, "status": "detected", "votes": ev_score,
                "reason": f"Evidence accumulator: score={ev_score} ({', '.join(k for k in ev if ev[k] is True)})",
                "details": {**yolo_info, **clf_info, "evidence": ev}}

    # Classifier-only path when YOLO is unavailable.
    classical_pair = (
        _classical_sunglasses_crop_evidence(frame_cv2_orig, orig_bbox)
        if yolo is None else {"detected": False, "reason": "skipped — using evidence accumulator"}
    )
    if yolo is None and clf_prob is not None and clf_prob >= 0.65:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": f"Classifier: sunglasses prob={clf_prob:.2f} (YOLO unavailable)",
                "details": clf_info}
        # Keep classical crop evidence for diagnostics only. Dark hair,
        # shadows and eyebrows can create paired components; it must not
        # independently declare sunglasses present.

    if face_landmarks is None:
        details = dict(yolo) if yolo else {}
        details["classical_crop"] = classical_pair
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": "No landmarks; crop-based detector inconclusive",
                "details": details}

    details = dict(yolo) if yolo else {}

    # Analyze the complete upper-face band as well as the tiny eye polygons.
    # Eye polygons alone miss wide sunglasses whose frame spans the bridge and
    # temples. Use original-resolution crop when available; otherwise proc frame.
    band_stats = None
    try:
        source_frame = frame_cv2_orig if frame_cv2_orig is not None else proc_frame
        if orig_bbox is not None and frame_cv2_orig is not None:
            ih, iw = source_frame.shape[:2]
            bx1, by1, bx2, by2 = [int(round(float(v))) for v in orig_bbox[:4]]
            bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
            px = int(0.12 * bw)
            cx1, cx2 = max(0, bx1 - px), min(iw, bx2 + px)
            cy1, cy2 = max(0, by1 + int(0.16 * bh)), min(ih, by1 + int(0.58 * bh))
            band = source_frame[cy1:cy2, cx1:cx2]
        elif face_landmarks is not None:
            fh, fw = proc_frame.shape[:2]
            lm = face_landmarks.landmark
            xs = [lm[i].x * fw for i in (33, 133, 263, 362)]
            ys = [lm[i].y * fh for i in (1, 168, 6, 197)]
            x1 = max(0, int(min(xs) - 0.20 * (max(xs)-min(xs))) )
            x2 = min(fw, int(max(xs) + 0.20 * (max(xs)-min(xs))) )
            y1 = max(0, int(min(ys) - 0.25 * (max(ys)-min(ys))) )
            y2 = min(fh, int(max(ys) + 1.80 * (max(ys)-min(ys))) )
            band = proc_frame[y1:y2, x1:x2]
        else:
            band = None
        if band is not None and band.size:
            bhh, bww = band.shape[:2]
            # Upscaling is for measurement consistency only, not added detail.
            if bww < 160:
                band = cv2.resize(band, (160, max(1, int(bhh * 160 / bww))), interpolation=cv2.INTER_CUBIC)
            gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            hsv_band = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            dark_mask = gray_band < SG_DARK_PIXEL_THR
            row_coverage = np.mean(dark_mask, axis=1)
            # Long, connected dark rows across the eye band are characteristic
            # of tinted lenses; require low luminance and broad horizontal coverage.
            band_stats = {
                "brightness": float(np.mean(gray_band)),
                "contrast": float(np.std(gray_band)),
                "dark_coverage": float(np.mean(dark_mask)),
                "max_row_coverage": float(np.max(row_coverage)) if row_coverage.size else 0.0,
                "hsv_value": float(np.mean(hsv_band[:, :, 2])),
            }
            # Dark sunglasses often form a broad horizontal strip even when
            # MediaPipe's eye polygons are tiny or partially misplaced. The
            # row-coverage test helps distinguish the strip from dark hair.
            band_dark = (
                band_stats["brightness"] < 105 and
                band_stats["dark_coverage"] >= 0.22 and
                band_stats["max_row_coverage"] >= 0.62 and
                band_stats["hsv_value"] < 115
            )
            details["upper_face_band"] = {k: round(v, 3) for k, v in band_stats.items()}
            details["upper_face_band_dark_candidate"] = bool(band_dark)
        else:
            band_dark = False
            details["upper_face_band"] = "inconclusive"
    except Exception as exc:
        band_dark = False
        details["upper_face_band_error"] = str(exc)

    l_gray, l_hsv = _extract_eye_roi(proc_frame, face_landmarks, SG_LEFT_EYE_POLY)
    r_gray, r_hsv = _extract_eye_roi(proc_frame, face_landmarks, SG_RIGHT_EYE_POLY)

    # Add texture + iris suppression signals to evidence score now that we have eye ROIs.
    for _gr in (l_gray, r_gray):
        _tv = _sg_lens_texture(_gr)
        if _tv is not None and _tv < 30:
            ev_score += 1; ev["lens_smooth"] = True; break
    _iri_l = _sg_iris_suppression(l_gray)
    _iri_r = _sg_iris_suppression(r_gray)
    if _iri_l or _iri_r:
        ev_score += 1; ev["iris_suppressed"] = True

    # Eye-openness (EAR) from FaceMesh: sunglasses suppress the detected eye
    # slit even when eyes are open — both EARs drop below 0.20.
    _ear_res = _sg_eye_openness(face_landmarks, proc_frame)
    if _ear_res is not None:
        ev["ear"] = _ear_res
        if _ear_res.get("both_suppressed"):
            ev_score += 1; ev["ear_suppressed"] = True

    details["ev_score"] = ev_score
    details["evidence"] = ev

    # Dark-circle guard for the post-ROI evidence accumulator.
    # Sunglasses cover the FULL eye opening uniformly; under-eye dark circles
    # concentrate brightness BELOW the eyelid, leaving the top of the ROI brighter.
    # If both measured eye ROIs show that pattern (top_mean - bottom_mean > 18),
    # reduce the evidence score so the accumulator does not fire on dark circles.
    if not soft and ev_score >= 4 and (l_gray is not None or r_gray is not None):
        _dc_grays = [g for g in (l_gray, r_gray) if g is not None and g.shape[0] >= 2]
        if _dc_grays:
            _dc_per_eye = []
            for _gr in _dc_grays:
                _mid2 = max(1, _gr.shape[0] // 2)
                _dc_per_eye.append(float(np.mean(_gr[:_mid2])) - float(np.mean(_gr[_mid2:])) > 18)
            if all(_dc_per_eye):
                ev["dark_circle_precheck"] = True
                ev_score = max(0, ev_score - 2)
                details["ev_score"] = ev_score
                details["evidence"] = ev

    if ev_score >= 4 and not clf_soft_veto:
        return {"detected": True, "status": "detected", "votes": ev_score,
                "reason": f"Evidence accumulator (post-ROI): score={ev_score}",
                "details": {**details, **clf_info, **clip_info}}
    if ev_score >= 3 and clf_prob is not None and clf_prob >= 0.60 and not clf_soft_veto:
        return {"detected": True, "status": "detected", "votes": ev_score,
                "reason": f"Evidence+classifier: score={ev_score}, clf={clf_prob:.2f}",
                "details": {**details, **clf_info, **clip_info}}

    def _stats(gray, hsv, side):
        if gray is None:
            details[f"{side}_roi"] = "inconclusive"
            return None
        s = {
            "b":  round(float(np.mean(gray)), 2),
            "c":  round(float(np.std(gray)),  2),
            "v":  round(float(np.mean(hsv[:, :, 2])), 2),
            "dc": round(float(np.sum(gray < SG_DARK_PIXEL_THR)) / (gray.size + 1e-8), 4),
        }
        # Vertical uniformity: sunglasses darken the full ROI top-to-bottom.
        # Dark circles concentrate below the eyelid → top half is brighter than bottom.
        mid = max(1, gray.shape[0] // 2)
        s["vert_diff"] = round(float(np.mean(gray[:mid])) - float(np.mean(gray[mid:])), 2)
        details[f"{side}_brightness"], details[f"{side}_contrast"] = s["b"], s["c"]
        details[f"{side}_v_channel"],  details[f"{side}_dark_coverage"] = s["v"], s["dc"]
        details[f"{side}_vert_diff"] = s["vert_diff"]
        return s

    L, R = _stats(l_gray, l_hsv, "left"), _stats(r_gray, r_hsv, "right")
    eyes = {k: v for k, v in (("left", L), ("right", R)) if v is not None}
    if not eyes:
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": "Both eye ROIs inconclusive", "details": details}

    def _iris(s):
        return s is not None and s["c"] > SG_IRIS_CONTRAST_MIN and s["b"] > SG_IRIS_BRIGHT_MIN
    iris_l, iris_r = _iris(L), _iris(R)
    if iris_l and iris_r and not soft:
        details["s5_iris_guard"] = "iris_visible_both_eyes — hard veto"
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": "Iris visible in both eyes (clear/prescription glass or bare)",
                "details": details}
    details["s5_iris_guard"] = f"iris NOT confirmed both eyes (L={iris_l}, R={iris_r})"

    # Vertical-uniformity veto: if in EVERY measurable eye the top half is
    # significantly brighter than the bottom, the darkness is under-eye skin
    # (dark circles), not a lens covering the full eye opening.
    _VERT_DC_THR = 18  # top_b - bot_b > 18 → darkness concentrated below eyelid
    dc_veto = {side: s["vert_diff"] > _VERT_DC_THR for side, s in eyes.items()}
    details["dark_circle_vert_check"] = dc_veto
    if all(dc_veto.values()) and not soft:
        details["dark_circle_veto"] = "top half brighter in all eyes — darkness below eyelid"
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": "Dark-circle veto: brightness concentrated below eyelid, not a lens",
                "details": details}

    need = SG_EYE_VOTES_SOFT if soft else SG_EYE_VOTES
    per_eye = {}
    for side, s in eyes.items():
        sig = {
            "brightness": s["b"]  <  SG_BRIGHTNESS_MAX,
            "contrast":   s["c"]  <  SG_CONTRAST_MAX,
            "hsv_v":      s["v"]  <  SG_HSV_V_MAX,
            "dark_cov":   s["dc"] >= SG_DARK_COVERAGE_MIN,
        }
        per_eye[side] = sum(sig.values())
        details[f"{side}_signals"] = sig
        details[f"{side}_votes"]   = per_eye[side]

    single = len(eyes) == 1
    eye_detected = all(v >= need for v in per_eye.values())
    # Raised from 2→3: dark circles routinely score 2 (low brightness + low HSV)
    # but rarely pass all 3 classical signals on the top half of the ROI.
    asymmetric_eye_detected = False  # one-eye darkness is not sufficient proof
    # The broad band helps when landmark eye polygons are too tight. Treat it
    # as supporting evidence, not a standalone trigger, to reduce dark-room FPs.
    any_eye_dark = asymmetric_eye_detected
    # The wide-band cue can stand alone only when it has a strong, continuous
    # horizontal dark row. Otherwise require eye votes or a soft model hit.
    band_strong = bool(
        band_dark and band_stats is not None and
        band_stats.get("max_row_coverage", 0.0) >= 0.82 and
        band_stats.get("dark_coverage", 0.0) >= 0.32 and
        band_stats.get("brightness", 255.0) < 75
    )
    # Re-enable band path when YOLO already gave a soft signal (combined evidence)
    # or when the band signal itself is very strong. Stand-alone band is still disabled.
    band_supported = bool(band_dark and (soft or band_strong))
    # Require face-relative brightness to confirm goggles for per-eye and band paths.
    # Dark goggles/sunglasses make the eye-band noticeably darker than the cheeks (ratio < 0.68).
    # Dark-framed clear glasses, dark eyes, heavy eyeliner, shadows, and caps do not.
    # This applies even when YOLO gave a soft signal — the YOLO model can fire on
    # dark frames without the actual lenses being dark.
    rb_confirms = _rb is not None and _rb.get("ratio", 1.0) < 0.68
    if not rb_confirms:
        eye_detected = False
        band_supported = False
    # A two-signal hit in either eye is intentionally accepted for asymmetric
    # views. This addresses side profiles where the far lens is occluded or the
    # landmark ROI contains mostly skin, while the visible lens is clearly dark.
    detected = bool(eye_detected or band_supported)
    votes = max(per_eye.values())
    details.update({"votes": votes, "votes_needed_per_eye": need,
                    "measurable_eyes": len(eyes), "single_eye_only": single,
                    "yolo_soft_assist": soft, "eye_vote_detection": bool(eye_detected),
                    "asymmetric_eye_detection": bool(asymmetric_eye_detected),
                    "band_supported_detection": band_supported})
    suffix = " [asymmetric-view: one eye has >=2 signals]" if (detected and asymmetric_eye_detected and not eye_detected) else (" [single-eye — low confidence]" if (detected and single) else "")
    reason = (f"Dark goggles detected (per-eye votes {per_eye}; asymmetric 2-vote rule enabled){suffix}"
              if detected else f"No dark goggles (per-eye votes {per_eye}, need {need})")
    return {"detected": detected, "votes": votes, "reason": reason, "details": details}


# ═══════════════════════════════════════════════════════════════
# MODULE 3b — Headwear detection (cap / helmet / face mask)
# ═══════════════════════════════════════════════════════════════

def detect_headwear(frame_cv2_orig: np.ndarray, orig_bbox) -> dict:
    votes = 0
    vote_log = {}
    details = {}

    # ── Detector 1a: YOLO on face-extended crop (hat rim area) ──
    hw = _detect_headwear_yolo(frame_cv2_orig, orig_bbox)
    if hw is not None:
        details["crop_yolo"] = hw
        if hw.get("yolo_detected"):
            v = 4; votes += v
            vote_log["crop_yolo_hard"] = f"+{v}"
        elif hw.get("yolo_soft"):
            v = 2; votes += v
            vote_log["crop_yolo_soft"] = f"+{v}"
        # No penalty for no-detect on crop — full-image pass below compensates

    # ── Detector 1b: YOLO on full image (catches helmets that dwarf face bbox) ──
    fi = _detect_headwear_yolo_fullimage(frame_cv2_orig)
    if fi is not None:
        details["full_yolo"] = fi
        if fi.get("yolo_detected"):
            # Only add votes if crop didn't already give a hard hit (avoid double-counting)
            if not (hw or {}).get("yolo_detected"):
                v = 4; votes += v
                vote_log["full_yolo_hard"] = f"+{v}"
        elif fi.get("yolo_soft"):
            if not (hw or {}).get("yolo_detected") and not (hw or {}).get("yolo_soft"):
                v = 2; votes += v
                vote_log["full_yolo_soft"] = f"+{v}"
        elif not (hw or {}).get("yolo_detected") and not (hw or {}).get("yolo_soft"):
            v = -1; votes += v
            vote_log["yolo_no_detect"] = f"{v}"

    # Best YOLO result across both passes
    best_yolo = hw if (hw or {}).get("yolo_detected") else (fi if (fi or {}).get("yolo_detected") else
                hw if (hw or {}).get("yolo_soft")     else (fi if (fi or {}).get("yolo_soft")     else hw))

    # ── Detector 2: classical above-face / brim analysis ──────
    cl = _detect_headwear_classical(frame_cv2_orig, orig_bbox)
    if cl is not None:
        details["classical"] = cl
        if cl.get("brim_signal"):
            v = 2; votes += v
            vote_log["classical_brim"] = (f"+{v} dark={cl.get('dark_brim')} "
                                          f"diff={cl.get('color_diff_signal')} fore={cl.get('fore_cut')}")
        else:
            v = -1; votes += v
            vote_log["classical_no_brim"] = f"{v}"

    # ── Detector 3: face mask (multi-signal classical) ────────
    mk = _detect_face_mask_classical(frame_cv2_orig, orig_bbox)
    mask_detected = False
    if mk is not None:
        details["mask"] = mk
        mask_score = mk.get("mask_score", 0)
        if mk.get("mask_signal"):
            mask_detected = True
            # Confident mask (3+ signals) → early return; YOLO noise can't veto it
            if mask_score >= 3:
                details["vote_log"] = {"mask_early_return": f"score={mask_score}/5"}
                details["total_votes"] = 99
                return {
                    "detected": True, "votes": 99, "kind": "mask", "mask": True,
                    "reason":   f"Face mask detected (score={mask_score}/5 signals)",
                    "details":  details,
                }
            # Soft mask (2 signals) — add to vote pool but needs corroboration
            v = 4; votes += v
            vote_log["mask_signal"] = f"+{v} (score={mask_score}/5 std={mk['lower_face_std']:.1f})"

    detected = votes >= HW_VOTE_THRESHOLD
    yolo_kind = (best_yolo or {}).get("yolo_best_kind") or "cap_or_helmet"
    kind = "mask" if mask_detected else yolo_kind
    details["vote_log"] = vote_log
    details["total_votes"] = votes
    return {
        "detected":    detected,
        "votes":       votes,
        "kind":        kind if detected else None,
        "mask":        mask_detected,
        "reason":      f"Headwear votes={votes}/{HW_VOTE_THRESHOLD} kind={kind}",
        "details":     details,
    }


def _clean_headwear(hw: dict) -> dict:
    if not hw:
        return hw
    d = hw.get("details") or {}
    return {
        "detected": hw.get("detected"),
        "kind":     hw.get("kind"),
        "mask":     hw.get("mask"),
        "reason":   hw.get("reason"),
        "details": {
            "yolo_detected":    d.get("yolo_detected"),
            "yolo_helmet_conf": d.get("yolo_helmet_conf"),
            "yolo_cap_conf":    d.get("yolo_cap_conf"),
            "yolo_best_kind":   d.get("yolo_best_kind"),
            "brim_signal":      d.get("brim_signal"),
            "brim_dark_cov":    d.get("brim_dark_cov"),
            "mask_signal":      (d.get("mask") or {}).get("mask_signal"),
            "lower_face_std":   (d.get("mask") or {}).get("lower_face_std"),
        },
    }


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — Closed-eye detection  (EAR)
# ═══════════════════════════════════════════════════════════════

def _compute_ear(landmarks_2d: list, eye_idx: list) -> Optional[float]:
    """
    EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
    Returns None on failure (callers must handle None explicitly).
    """
    try:
        pts = [landmarks_2d[i] for i in eye_idx]
        v1  = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2  = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h   = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return round(float((v1 + v2) / (2.0 * h)), 4) if h > 0 else None
    except Exception:
        return None


def _estimate_open_ear_baseline(landmarks_2d: list, eye_idx: list) -> Optional[float]:
    """
    Estimates what this person's EAR would be when eyes are open,
    using only the horizontal eye width (pts[0] to pts[3]).
    This is self-contained per image — no prior frame needed.

    Formula:  baseline = eye_width * EAR_OPEN_BASELINE_FACTOR / (2 * eye_width)
                       = EAR_OPEN_BASELINE_FACTOR / 2

    This gives a person-neutral open baseline (~0.42 for normal eyes).
    Small-eyed people have a smaller eye_width, so their EAR is already
    naturally lower — the relative drop check catches only TRUE closure.
    """
    try:
        pts       = [landmarks_2d[i] for i in eye_idx]
        eye_width = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        if eye_width < 1e-8:
            return None
        # Vertical open height estimated as fraction of eye width
        estimated_open_v = eye_width * EAR_OPEN_BASELINE_FACTOR
        return round(float(estimated_open_v / (2.0 * eye_width)), 4)
    except Exception:
        return None


def detect_closed_eyes(
    face_landmarks,
    proc_w: int,
    proc_h: int,
    orig_bbox,     # original-space bbox — for resolution guard only
    proc_bbox,     # unused, kept for call-site compatibility
) -> dict:
    """
    HIGH-SEVERITY closed-eye detection — small-eye safe.

    Two-gate system:
      Gate 1 (absolute): both EARs < EAR_CLOSED_THRESHOLD (0.15 — deeply shut)
      Gate 2 (relative): both EARs <= EAR_RELATIVE_DROP_MIN * their open baseline
                         Protects naturally small/narrow eyes from false positives.

    Both gates must pass for eyes_closed = True.
    """
    result = {"eyes_closed": False, "left_ear": None, "right_ear": None,
              "avg_ear": None, "reason": "Not checked", "checked": False}

    if face_landmarks is None:
        result["reason"] = "MediaPipe landmarks unavailable"
        return result

    # Resolution guard — skip on tiny faces (unreliable EAR)
    if orig_bbox is None:
        result["reason"] = "No bounding box — skipping EAR (unreliable without size guard)"
        return result

    x1, y1, x2, y2 = orig_bbox
    orig_face_h = abs(y2 - y1)
    if orig_face_h < EAR_MIN_RESOLUTION:
        result["reason"] = (
            f"Face too small ({orig_face_h:.0f}px < {EAR_MIN_RESOLUTION}px) — skipped"
        )
        return result

    lm           = face_landmarks.landmark
    landmarks_2d = [(lm[i].x * proc_w, lm[i].y * proc_h) for i in range(len(lm))]

    left_ear  = _compute_ear(landmarks_2d, LEFT_EYE_EAR)
    right_ear = _compute_ear(landmarks_2d, RIGHT_EYE_EAR)

    result.update({"left_ear": left_ear, "right_ear": right_ear, "checked": True})

    if left_ear is None or right_ear is None:
        result["reason"] = (
            f"EAR computation failed (L={left_ear}, R={right_ear}) — treating as open"
        )
        return result

    avg_ear           = round((left_ear + right_ear) / 2.0, 4)
    result["avg_ear"] = avg_ear

    # Gate 1: absolute threshold — must be deeply shut
    gate1_l = left_ear  < EAR_CLOSED_THRESHOLD
    gate1_r = right_ear < EAR_CLOSED_THRESHOLD
    gate1   = gate1_l and gate1_r

    # Gate 2: relative drop — EAR must be <= 35% of estimated open baseline
    # Protects small-eyed people: their baseline is also small, so ratio is fair
    base_l = _estimate_open_ear_baseline(landmarks_2d, LEFT_EYE_EAR)
    base_r = _estimate_open_ear_baseline(landmarks_2d, RIGHT_EYE_EAR)

    if base_l is not None and base_r is not None:
        ratio_l = round(left_ear  / base_l, 4) if base_l > 0 else 1.0
        ratio_r = round(right_ear / base_r, 4) if base_r > 0 else 1.0
        gate2   = (ratio_l <= EAR_RELATIVE_DROP_MIN and
                   ratio_r <= EAR_RELATIVE_DROP_MIN)
        result.update({
            "left_baseline":  base_l, "right_baseline":  base_r,
            "left_ratio":     ratio_l, "right_ratio":    ratio_r,
        })
    else:
        # Baseline unavailable — fall back to gate1 only
        gate2   = True
        ratio_l = ratio_r = None
        result["baseline_note"] = "Baseline unavailable — gate2 bypassed"

    if gate1 and gate2:
        result["eyes_closed"] = True
        result["reason"] = (
            f"Both eyes fully closed: L={left_ear:.3f}, R={right_ear:.3f} "
            f"(threshold={EAR_CLOSED_THRESHOLD}, "
            f"relative L={ratio_l}, R={ratio_r} <= {EAR_RELATIVE_DROP_MIN})"
        )
    elif not gate1:
        open_eyes = []
        if left_ear  >= EAR_CLOSED_THRESHOLD: open_eyes.append(f"L={left_ear:.3f}")
        if right_ear >= EAR_CLOSED_THRESHOLD: open_eyes.append(f"R={right_ear:.3f}")
        result["reason"] = (
            f"Gate1 failed — not deeply closed ({', '.join(open_eyes)}, "
            f"threshold={EAR_CLOSED_THRESHOLD})"
        )
    else:
        result["reason"] = (
            f"Gate2 failed — relative drop insufficient "
            f"(L ratio={ratio_l}, R ratio={ratio_r}, min drop={EAR_RELATIVE_DROP_MIN}): "
            f"likely small-eyed person with open eyes"
        )
    return result


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — Head-pose estimation
# ═══════════════════════════════════════════════════════════════

def _np_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b between rays ba and bc."""
    ba    = a - b;  bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def _estimate_focal_length(img_w: int, img_h: int) -> float:
    """
    Focal length in pixels via 70-degree diagonal-FOV heuristic.
      f = sqrt(w^2+h^2) / (2 * tan(FOV_diag/2))
    Diagonal is rotation-invariant: same f for portrait and landscape.
    """
    diag = math.sqrt(img_w ** 2 + img_h ** 2)
    return diag / (2.0 * math.tan(math.radians(70.0) / 2.0))


def _get_mediapipe_angles(face_landmarks, img_w: int, img_h: int) -> Optional[dict]:
    """
    Computes head yaw/pitch/roll via SolvePnP + ZYX Euler decomposition.

    Camera model: diagonal-FOV focal length, principal point at image centre,
    zero distortion.

    180-degree flip guard: rm[:,2] is the face forward-Z in camera space.
    Normal orientation has forward[2] < 0 (Z points into scene, away from camera).
    If forward[2] > 0 (flipped), negate rm to correct orientation.

    ZYX Euler decomposition:
      sy  = sqrt(R[0,0]^2 + R[1,0]^2)
      if sy > eps:
        pitch = atan2( R[2,1], R[2,2])
        yaw   = atan2(-R[2,0], sy)
        roll  = atan2( R[1,0], R[0,0])
      else (gimbal lock, pitch ~ +-90 deg):
        pitch = atan2(-R[1,2], R[1,1])
        yaw   = atan2(-R[2,0], sy)
        roll  = 0  (lost at gimbal lock — standard behaviour)
    """
    try:
        lm  = face_landmarks.landmark
        pts = np.array([
            (lm[NOSE_TIP].x     * img_w, lm[NOSE_TIP].y     * img_h),
            (lm[CHIN].x         * img_w, lm[CHIN].y          * img_h),
            (lm[LEFT_EYE_MP].x  * img_w, lm[LEFT_EYE_MP].y  * img_h),
            (lm[RIGHT_EYE_MP].x * img_w, lm[RIGHT_EYE_MP].y * img_h),
            (lm[LEFT_MOUTH].x   * img_w, lm[LEFT_MOUTH].y   * img_h),
            (lm[RIGHT_MOUTH].x  * img_w, lm[RIGHT_MOUTH].y  * img_h),
        ], dtype=np.float64)

        fl  = _estimate_focal_length(img_w, img_h)
        cm  = np.array([[fl, 0,  img_w / 2],
                        [0,  fl, img_h / 2],
                        [0,  0,  1.0      ]], dtype=np.float64)

        # Use SOLVEPNP_SQPNP (globally convergent) for reliable pose estimation
        ok, rvec, _tvec = cv2.solvePnP(
            MODEL_POINTS_3D, pts, cm, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            return None

        rm, _ = cv2.Rodrigues(rvec)

        # Do not negate the full rotation matrix: -R is not a valid SO(3)
        # rotation and corrupts the Euler angles. Keep the proper Rodrigues
        # rotation returned by solvePnP and classify using absolute yaw.

        sy = math.sqrt(rm[0, 0] ** 2 + rm[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.atan2( rm[2, 1], rm[2, 2])
            yaw   = math.atan2(-rm[2, 0], sy)
            roll  = math.atan2( rm[1, 0], rm[0, 0])
        else:   # gimbal lock — pitch is near ±90 degrees, roll is undefined
            pitch = math.atan2(-rm[1, 2], rm[1, 1])
            yaw   = math.atan2(-rm[2, 0], sy)
            roll  = 0.0

        return {
            "yaw":   round(math.degrees(yaw),   2),
            "pitch": round(math.degrees(pitch),  2),
            "roll":  round(math.degrees(roll),   2),
        }
    except Exception:
        return None


def _get_nose_visibility_ratio(face_landmarks) -> float:
    """
    Nose-wing width / cheek-to-cheek width — purely normalised ratio.
    ~0.25–0.30 frontal; < 0.08 full side.
    """
    try:
        lm = face_landmarks.landmark
        nw = abs(lm[NOSE_RIGHT].x - lm[NOSE_LEFT].x)
        fw = abs(lm[RIGHT_CHEEK].x - lm[LEFT_CHEEK].x)
        return round(nw / fw, 4) if fw > 1e-8 else 1.0
    except Exception:
        return 1.0


def classify_pose_strong(
    angR: float,
    angL: float,
    yaw_data: Optional[dict],
    nose_ratio: float,
) -> tuple:
    """
    Weighted 4-signal pose classifier.

    Weights:  S1 x1.0  S2 x1.5  S3 x2.5  S4 x1.5   (theoretical max ±6.5)

    S3 abstains when yaw unavailable (was s1*0.5 phantom penalty).
    unavailable S3 slot excluded from dynamic_max.
    classification boundary score >= 0 -> Frontal.

    confidence = |total_score| / dynamic_max * 100
    """
    total_score = 0.0
    dynamic_max = 0.0
    signals     = {}

    # S1 — MTCNN eye-to-nose angle range
    ri, li = int(angR), int(angL)
    if (MTCNN_FRONTAL_R_MIN <= ri <= MTCNN_FRONTAL_R_MAX and
            MTCNN_FRONTAL_L_MIN <= li <= MTCNN_FRONTAL_L_MAX):
        s1 = 1.0;  signals["mtcnn_angle"] = "Frontal"
    elif 25 <= ri <= 68 and 25 <= li <= 70:
        s1 = 0.3;  signals["mtcnn_angle"] = "Slightly Side"
    else:
        s1 = -1.0; signals["mtcnn_angle"] = "Side"
    total_score += s1 * 1.0
    dynamic_max += 1.0

    # S2 — Eye-angle asymmetry
    asym = abs(angR - angL)
    if asym <= ASYM_FRONTAL_MAX:
        s2 = 1.0;  signals["asymmetry"] = f"Frontal ({asym:.1f}deg)"
    elif asym <= ASYM_SLIGHTLY_MAX:
        s2 = 0.3;  signals["asymmetry"] = f"Slightly Side ({asym:.1f}deg)"
    else:
        s2 = -1.0; signals["asymmetry"] = f"Side ({asym:.1f}deg)"
    total_score += s2 * 1.5
    dynamic_max += 1.5

    # S3 — MediaPipe yaw  (highest weight; abstains when unavailable or roll≈180° flip)
    # When SolvePnP returns |roll| > 150°, the solution is in a flipped orientation
    # and the extracted yaw is unreliable — abstain rather than penalise wrongly.
    _roll_ok = yaw_data is not None and abs(yaw_data.get("roll", 0.0)) < 150.0
    if _roll_ok:
        yaw_abs = abs(yaw_data["yaw"])
        signals.update(yaw_data)
        if yaw_abs <= YAW_FRONTAL_MAX:
            s3 = 1.0;  signals["mediapipe_yaw"] = f"Frontal ({yaw_abs}deg)"
        elif yaw_abs <= YAW_SLIGHTLY_SIDE_MAX:
            decay = 1.0 - ((yaw_abs - YAW_FRONTAL_MAX) /
                            (YAW_SLIGHTLY_SIDE_MAX - YAW_FRONTAL_MAX))
            s3 = max(0.1, decay * 0.8)
            signals["mediapipe_yaw"] = f"Slightly Side ({yaw_abs}deg, s3={s3:.3f})"
        else:
            s3 = -1.0; signals["mediapipe_yaw"] = f"Full Side ({yaw_abs}deg)"
        total_score += s3 * 2.5
        dynamic_max += abs(s3) * 2.5
    else:
        if yaw_data is not None:
            signals.update(yaw_data)
        # abstain — no phantom penalty, no dynamic_max contribution
        _roll_val = abs(yaw_data.get("roll", 0.0)) if yaw_data is not None else None
        reason = "unavailable" if yaw_data is None else f"roll={_roll_val:.1f}deg flip"
        signals["mediapipe_yaw"] = f"{reason} — S3 abstained"

    # S4 — Nose visibility ratio
    signals["nose_visibility_ratio"] = nose_ratio
    if nose_ratio >= 0.18:
        s4 = 1.0;  signals["nose_visibility"] = f"Frontal ({nose_ratio:.4f})"
    elif nose_ratio >= NOSE_VISIBILITY_MIN:
        s4 = 0.2;  signals["nose_visibility"] = f"Slightly Side ({nose_ratio:.4f})"
    else:
        s4 = -1.0; signals["nose_visibility"] = f"Full Side ({nose_ratio:.4f})"
    total_score += s4 * 1.5
    dynamic_max += 1.5

    # High-confidence side gates. A true profile typically has a large yaw
    # and/or very little visible nose width. These should not be outvoted by
    # noisy eye-angle measurements from partial occlusion or sunglasses.
    yaw_abs  = abs(float(yaw_data.get("yaw",  0.0))) if yaw_data is not None else None
    roll_abs = abs(float(yaw_data.get("roll", 0.0))) if yaw_data is not None else None
    # Only trust yaw-based gates when SolvePnP roll is in a normal range
    yaw_trust = roll_abs is None or roll_abs < 150.0
    strong_side_reasons = []
    # Three-quarter/profile photos often have yaw around 30-40 degrees.
    # Treat moderate yaw as side when corroborated by reduced visible nose width;
    # this avoids requiring a near-perfect 90-degree profile.
    if yaw_abs is not None and yaw_trust and yaw_abs >= 38.0:
        strong_side_reasons.append(f"yaw={yaw_abs:.1f}deg")
    elif yaw_abs is not None and yaw_trust and yaw_abs >= 30.0 and nose_ratio < 0.16:
        strong_side_reasons.append(f"moderate_yaw={yaw_abs:.1f}deg+nose_ratio={nose_ratio:.4f}")
    if nose_ratio < 0.075:
        strong_side_reasons.append(f"nose_ratio={nose_ratio:.4f}")
    if asym >= 24.0 and nose_ratio < 0.16:
        strong_side_reasons.append(f"eye_asymmetry={asym:.1f}deg+narrow_nose")

    if strong_side_reasons:
        final = "Side"
        signals["strong_side_gate"] = strong_side_reasons
    else:
        final = "Frontal" if total_score >= 0 else "Side"
        signals["strong_side_gate"] = []

    conf_pct = (abs(total_score) / dynamic_max * 100.0) if dynamic_max > 0 else 0.0
    signals.update({
        "total_score": round(total_score, 4),
        "max_score":   round(dynamic_max, 4),
        "confidence":  f"{round(conf_pct, 1)}%",
        "pose_decision": final,
    })
    return final, signals


# ═══════════════════════════════════════════════════════════════
# MODULE 6 — MTCNN <-> MediaPipe face matching
# ═══════════════════════════════════════════════════════════════

def _match_landmarks_to_faces(
    orig_bboxes: list,
    mp_faces:    list,
    proc_w:      int,
    proc_h:      int,
    proc_scale:  float,
) -> list:
    """
    Matches each MTCNN bbox (original space) to the closest MediaPipe face
    by Euclidean distance to the nose-tip landmark.

    bbox centre is multiplied by proc_scale to convert it to
    proc-frame space before computing distance.  Without this, on a 4K image
    (scale=0.5) the centre was ~1000 px from the correct nose-tip.

    Distance formula:
      d = sqrt( (lm.x*proc_w  -  cx_orig*proc_scale)^2
              + (lm.y*proc_h  -  cy_orig*proc_scale)^2 )
    """
    matched = []
    for bbox in orig_bboxes:
        if bbox is None:
            matched.append(None)
            continue
        bx1, by1, bx2, by2 = bbox
        # convert original-space centre -> proc-frame space
        cx_proc = ((bx1 + bx2) / 2.0) * proc_scale
        cy_proc = ((by1 + by2) / 2.0) * proc_scale

        best_mp, best_dist = None, float("inf")
        for mp_face in mp_faces:
            lm   = mp_face.landmark
            dist = math.sqrt(
                (lm[NOSE_TIP].x * proc_w - cx_proc) ** 2 +
                (lm[NOSE_TIP].y * proc_h - cy_proc) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_mp   = mp_face
        matched.append(best_mp)
    return matched


# ═══════════════════════════════════════════════════════════════
# BLUR SCORING
# ═══════════════════════════════════════════════════════════════

def compute_blur_score(crop_bgr: np.ndarray):
    """
    Multi-signal no-reference blur assessment.

    Returns (blur_score, details). blur_score is a HEURISTIC 0..100 index,
    not a probability. A calibrated probability requires a labelled dataset
    and a separately trained/calibrated model.

    Signals: skimage blur-effect (when available), Laplacian variance,
    Tenengrad, Canny edge density, high-frequency energy, regional sharpness,
    luminance, and contrast. Scores are resolution-normalized but thresholds
    still require validation on the deployment dataset.
    """
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return 100.0, {"reason": "empty_crop", "decision": "uncertain",
                       "score_kind": "heuristic_not_probability"}

    # Preserve aspect ratio. Stretching every crop to a square distorts edges
    # and changes sharpness statistics.
    src_h, src_w = crop_bgr.shape[:2]
    if src_h < 2 or src_w < 2:
        return 100.0, {"reason": "crop_too_small", "decision": "uncertain",
                       "score_kind": "heuristic_not_probability"}
    target = max(64, int(BLUR_NORMALIZE_SIZE))
    scale = min(target / float(max(src_h, src_w)), 1.0)
    # Upscale small crops only to stabilize metric dimensions; upscaling does
    # not create real detail and is explicitly recorded below.
    if max(src_h, src_w) < target:
        scale = target / float(max(src_h, src_w))
    nw, nh = max(2, int(round(src_w * scale))), max(2, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    norm = cv2.resize(crop_bgr, (nw, nh), interpolation=interpolation)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY) if norm.ndim == 3 else norm.copy()
    gray_f = gray.astype(np.float32) / 255.0

    # Light denoising limits high-frequency noise, while metrics are also
    # calculated on the un-denoised image to avoid erasing useful fine detail.
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape
    details = {
        "source_dimensions": [int(src_w), int(src_h)],
        "metric_dimensions": [int(w), int(h)],
        "metric_upscaled": bool(scale > 1.0),
        "score_kind": "heuristic_0_100_not_probability",
    }

    # Signal 1: Laplacian variance (more edge detail generally means sharper).
    lap_var = float(cv2.Laplacian(denoised, cv2.CV_64F, ksize=3).var())

    # Signal 2: Tenengrad / Sobel gradient energy.
    sx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    grad_energy = sx * sx + sy * sy
    tenengrad = float(np.mean(grad_energy))

    # Signal 3: edge density, using thresholds adapted to local contrast.
    med = float(np.median(denoised))
    sigma = float(np.std(denoised))
    lower = int(max(10, min(80, 0.66 * med)))
    upper = int(max(lower + 10, min(220, 1.33 * med)))
    edges = cv2.Canny(denoised, lower, upper)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    # Signal 4: high-frequency energy ratio in the Fourier domain.
    f = np.fft.fftshift(np.fft.fft2(gray_f))
    mag = np.abs(f)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    cutoff = 0.18 * min(h, w)
    total_energy = float(np.sum(mag ** 2)) + 1e-12
    high_freq_ratio = float(np.sum((mag ** 2)[radius >= cutoff]) / total_energy)

    # Signal 5: patch-level detail. Median is robust to a few textured patches;
    # 75th percentile indicates whether any regions contain real edges.
    patch_laps = []
    rows = np.array_split(denoised, min(4, h), axis=0)
    for row in rows:
        for patch in np.array_split(row, min(4, w), axis=1):
            if patch.size >= 16:
                patch_laps.append(float(cv2.Laplacian(patch, cv2.CV_64F, ksize=3).var()))
    patch_median = float(np.median(patch_laps)) if patch_laps else 0.0
    patch_p75 = float(np.percentile(patch_laps, 75)) if patch_laps else 0.0

    # Signal 6: optional no-reference blur-effect metric (0 sharp .. 1 blurred).
    skimage_effect = None
    if _SKIMAGE_OK:
        try:
            skimage_effect = float(np.clip(_skimage_blur_effect(gray_f, h_size=11), 0.0, 1.0))
        except Exception:
            skimage_effect = None

    # Exposure/contrast indicators matter because very dark, flat, or washed-out
    # images can look "blurry" to edge metrics even when the focus is unknown.
    mean_luma = float(np.mean(gray))
    p05, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p05)
    clipped_dark = float(np.mean(gray <= 5))
    clipped_bright = float(np.mean(gray >= 250))

    # Convert raw metrics into conservative, interpretable heuristic sharpness
    # signals. These anchors are starting values, not universal thresholds.
    lap_sharp = float(np.clip(lap_var / 900.0, 0.0, 1.0))
    ten_sharp = float(np.clip(tenengrad / 1800.0, 0.0, 1.0))
    edge_sharp = float(np.clip(edge_density / 0.045, 0.0, 1.0))
    hf_sharp = float(np.clip(high_freq_ratio / 0.12, 0.0, 1.0))
    patch_sharp = float(np.clip(patch_median / 900.0, 0.0, 1.0))
    cv_sharp = (0.27 * lap_sharp + 0.23 * ten_sharp +
                0.17 * edge_sharp + 0.18 * hf_sharp + 0.15 * patch_sharp)
    cv_blur = 100.0 * (1.0 - cv_sharp)

    if skimage_effect is not None:
        # Keep the optional signal supplementary so one metric cannot dominate.
        blur_score = float(np.clip(0.75 * cv_blur + 0.25 * skimage_effect * 100.0, 0, 100))
    else:
        blur_score = float(np.clip(cv_blur, 0, 100))

    # Agreement / reliability flags.
    low_contrast = dynamic_range < 25 or sigma < 12
    extreme_exposure = mean_luma < 12 or mean_luma > 245
    noisy_or_textured = lap_var > 2500 and tenengrad > 5000 and high_freq_ratio > 0.10
    low_detail_votes = sum([
        lap_sharp < 0.20,
        ten_sharp < 0.20,
        edge_sharp < 0.20,
        hf_sharp < 0.20,
        patch_sharp < 0.20,
    ])
    high_detail_votes = sum([
        lap_sharp > 0.45,
        ten_sharp > 0.45,
        edge_sharp > 0.35,
        hf_sharp > 0.30,
        patch_sharp > 0.40,
    ])

    if extreme_exposure:
        decision = "uncertain"
        reason = "extreme_exposure"
    # Strong, consistent absence of detail can itself be evidence of blur.
    # Keep truly flat/featureless images uncertain by requiring some tonal range.
    elif blur_score >= 85 and low_detail_votes >= 5 and dynamic_range >= 12 and sigma >= 4:
        decision = "blurred"
        reason = "strong_multi_signal_blur_despite_reduced_contrast"
    elif low_contrast:
        decision = "uncertain"
        reason = "low_contrast_or_featureless_image"
    elif blur_score >= 78 and low_detail_votes >= 5:
        decision = "blurred"
        reason = "multiple_sharpness_signals_low"
    elif blur_score <= 42 and high_detail_votes >= 3:
        decision = "not_blurry"
        reason = "multiple_sharpness_signals_high"
    else:
        decision = "uncertain"
        reason = "mixed_or_borderline_sharpness_signals"

    # Blur subtype is a heuristic hint only, not a guaranteed physical diagnosis.
    gx_energy = float(np.mean(sx * sx)) + 1e-8
    gy_energy = float(np.mean(sy * sy)) + 1e-8
    anisotropy = max(gx_energy, gy_energy) / min(gx_energy, gy_energy)
    if decision == "blurred" and anisotropy >= 2.5:
        blur_type = "possible_motion_blur"
    elif decision == "blurred":
        blur_type = "possible_defocus_or_general_blur"
    else:
        blur_type = None

    details.update({
        "laplacian_var": round(lap_var, 3),
        "tenengrad": round(tenengrad, 3),
        "edge_density": round(edge_density, 5),
        "high_frequency_ratio": round(high_freq_ratio, 5),
        "patch_median_var": round(patch_median, 3),
        "patch_p75_var": round(patch_p75, 3),
        "mean_luminance": round(mean_luma, 2),
        "luminance_dynamic_range_p05_p95": round(dynamic_range, 2),
        "contrast_std": round(sigma, 2),
        "clipped_dark_fraction": round(clipped_dark, 4),
        "clipped_bright_fraction": round(clipped_bright, 4),
        "skimage_blur_effect": round(skimage_effect, 4) if skimage_effect is not None else None,
        "cv_blur_score": round(cv_blur, 2),
        "blur_score": round(blur_score, 2),
        "sharpness_index": round(1.0 - blur_score / 100.0, 4),
        "decision": decision,
        "reason": reason,
        "blur_type_hint": blur_type,
        "low_detail_signal_count": int(low_detail_votes),
        "high_detail_signal_count": int(high_detail_votes),
        "anisotropy": round(anisotropy, 3),
        "low_contrast": bool(low_contrast),
        "extreme_exposure": bool(extreme_exposure),
        "noisy_or_textured": bool(noisy_or_textured),
        "probability_note": "blur_score is heuristic, not a calibrated probability",
    })
    return round(blur_score, 2), details


def _resolution_metadata(width_px: int, height_px: int) -> dict:
    """Describe actual pixel dimensions; never confuse low resolution with blur."""
    width_px, height_px = max(0, int(width_px)), max(0, int(height_px))
    total = width_px * height_px
    megapixels = total / 1_000_000.0
    if total == 0:
        tier, description = "unknown", "Image dimensions are unavailable."
    elif megapixels < 0.3:
        tier, description = "very_low_resolution", "Very few pixels are available; facial details may be lost."
    elif megapixels < 1.0:
        tier, description = "low_resolution", "Limited pixel detail; small facial features may not be reliably assessed."
    elif megapixels < 2.0:
        tier, description = "moderate_resolution", "Adequate for basic inspection, but fine details may be limited."
    elif megapixels < 8.0:
        tier, description = "high_resolution", "Good overall pixel count for facial-detail analysis."
    else:
        tier, description = "very_high_resolution", "High pixel count; actual detail still depends on focus and compression."
    return {
        "width_px": width_px,
        "height_px": height_px,
        "total_pixels": total,
        "megapixels": round(megapixels, 3),
        "resolution_tier": tier,
        "description": description,
    }


def _blur_level_metadata(score, decision: str, resolution: dict = None) -> dict:
    """Map the heuristic score to user-facing 20-point levels, not probabilities."""
    if not isinstance(score, (int, float)):
        return {"blur_level_percent": None, "blur_level_label": "unavailable",
                "blur_description": "A blur level could not be estimated from the available pixels."}
    s = float(max(0.0, min(100.0, score)))
    if decision == "not_blurry" and s <= 42:
        pct, label, desc = 0, "sharp_or_acceptable", "No strong blur evidence was detected. This is a heuristic assessment, not a guarantee."
    elif s <= 20:
        pct, label, desc = 20, "very_slight_blur", "Level 1 of 5: very slight softness; most edges should remain clear."
    elif s <= 40:
        pct, label, desc = 20, "very_slight_blur", "Level 1 of 5: slight softness; fine details may begin to weaken."
    elif s <= 55:
        pct, label, desc = 40, "mild_blur", "Level 2 of 5: mild blur; small details and edges are less crisp."
    elif s <= 70:
        pct, label, desc = 60, "moderate_blur", "Level 3 of 5: moderate blur; facial details may be harder to inspect."
    elif s <= 85:
        pct, label, desc = 80, "heavy_blur", "Level 4 of 5: heavy blur; important facial details are substantially degraded."
    else:
        pct, label, desc = 100, "very_heavy_blur", "Level 5 of 5: very heavy blur; fine facial details are likely unusable."
    if decision in ("uncertain", "blur_uncertain"):
        desc += " The image is inconclusive, so treat this level as an estimate only."
    if resolution and resolution.get("resolution_tier") in ("very_low_resolution", "low_resolution"):
        desc += " Low pixel resolution may mimic blur or prevent a reliable blur measurement."
    return {"blur_level_percent": pct, "blur_level_label": label, "blur_description": desc}


def _detect_blur_on_face(frame_cv2: np.ndarray, bbox_, valid_indices: list) -> dict:
    """Assess blur severity, face resolution, and source-image pixel dimensions."""
    if frame_cv2 is None or not isinstance(frame_cv2, np.ndarray) or frame_cv2.size == 0:
        return {"is_blurred": False, "category": "blur_uncertain", "blur_score": None,
                "blur_level_percent": None, "blur_level_label": "unavailable",
                "blur_description": "No image pixels were available for analysis.",
                "image_resolution": _resolution_metadata(0, 0),
                "details": {"reason": "empty_image"}, "faces": [], "uncertain": True}

    img_h, img_w = frame_cv2.shape[:2]
    image_resolution = _resolution_metadata(img_w, img_h)
    if not valid_indices or bbox_ is None:
        fallback = _check_whole_image_blur(frame_cv2)
        fallback.setdefault("details", {})["reason"] = "no_face_bbox_whole_image_fallback"
        return fallback
    global_score, global_details = compute_blur_score(frame_cv2)
    face_results = []
    for face_number, idx in enumerate(valid_indices or [], start=1):
        try:
            b = bbox_[idx]
            x1 = max(0, min(img_w, int(round(float(b[0])))))
            y1 = max(0, min(img_h, int(round(float(b[1])))))
            x2 = max(0, min(img_w, int(round(float(b[2])))))
            y2 = max(0, min(img_h, int(round(float(b[3])))))
            face_w, face_h = max(0, x2-x1), max(0, y2-y1)
            face_pixels = face_w * face_h
            face_resolution = _resolution_metadata(face_w, face_h)
            item = {"face_id": face_number, "bbox": [x1, y1, x2, y2],
                    "face_width_px": face_w, "face_height_px": face_h,
                    "face_pixel_count": face_pixels,
                    "face_megapixels": round(face_pixels / 1_000_000.0, 5),
                    "face_resolution": face_resolution}
            if face_w < BLUR_MIN_FACE_PX or face_h < BLUR_MIN_FACE_PX:
                item.update({"category": "low_resolution", "is_blurred": False,
                             "blur_score": None, "blur_level_percent": None,
                             "blur_level_label": "unreliable_due_to_low_resolution",
                             "blur_description": "Face crop is too small for a reliable blur estimate. Upload a larger image or crop closer to the face.",
                             "details": {"reason": "face_too_small_for_reliable_blur_measurement"}})
            else:
                crop = frame_cv2[y1:y2, x1:x2]
                score, details = compute_blur_score(crop)
                ey1, ey2 = int(round(face_h*.18)), int(round(face_h*.62))
                ex1, ex2 = int(round(face_w*.08)), int(round(face_w*.92))
                eye_crop = crop[max(0,ey1):min(face_h,ey2), max(0,ex1):min(face_w,ex2)]
                eye_score, eye_details = (compute_blur_score(eye_crop)
                    if eye_crop.size and min(eye_crop.shape[:2]) >= 16
                    else (None, {"decision":"uncertain", "reason":"eye_region_too_small"}))
                eye_w = max(0, min(face_w,ex2)-max(0,ex1))
                eye_h = max(0, min(face_h,ey2)-max(0,ey1))
                eye_resolution = _resolution_metadata(eye_w, eye_h)
                face_is_blurred = (details.get("decision") == "blurred"
                                   and isinstance(score,(int,float)) and score >= BLUR_SCORE_THRESHOLD)
                if min(face_w, face_h) < BLUR_FACE_LOW_RES:
                    category, is_blurred = "low_resolution", False
                    reason = f"Detected face is only {face_w}x{face_h}px; blur confidence is limited."
                elif face_is_blurred:
                    category, is_blurred = "blurred", True
                    reason = "Multiple sharpness signals indicate blur in the face crop."
                elif details.get("decision") == "not_blurry":
                    category, is_blurred = "not_blurry", False
                    reason = "Face crop contains sufficient sharpness evidence."
                else:
                    category, is_blurred = "blur_uncertain", False
                    reason = "Sharpness signals disagree or image contrast is insufficient."
                if category == "low_resolution":
                    level = {"blur_level_percent": None,
                             "blur_level_label": "unreliable_due_to_low_resolution",
                             "blur_description": "Blur percentage is withheld because the face crop is below the recommended pixel size."}
                else:
                    level = _blur_level_metadata(score, details.get("decision"), face_resolution)
                item.update({"category": category, "is_blurred": is_blurred,
                             "blur_score": score, **level, "reason": reason,
                             "details": {**details,
                                 "source_image_resolution": image_resolution,
                                 "face_resolution": face_resolution,
                                 "eye_region_resolution": eye_resolution,
                                 "eye_region_pixel_dimensions": {"width_px": eye_w, "height_px": eye_h},
                                 "resolution_warning": reason if category == "low_resolution" else None},
                             "eye_region_blur_score": eye_score,
                             "eye_region_details": eye_details})
            face_results.append(item)
        except Exception as exc:
            logger.warning(f"Blur analysis failed for face #{face_number}: {exc!r}")
            face_results.append({"face_id": face_number, "category":"blur_uncertain",
                                 "is_blurred":False, "blur_score":None,
                                 "blur_level_percent":None,
                                 "blur_description":"Blur could not be measured for this face.",
                                 "details":{"reason":"blur_analysis_exception", "error":str(exc)}})

    blurred_faces = [f for f in face_results if f.get("category") == "blurred"]
    low_res_faces = [f for f in face_results if f.get("category") == "low_resolution"]
    uncertain_faces = [f for f in face_results if f.get("category") == "blur_uncertain"]
    scores = [f["blur_score"] for f in face_results if isinstance(f.get("blur_score"),(int,float))]
    if blurred_faces: category = "blurred"
    elif low_res_faces: category = "low_resolution"
    elif uncertain_faces: category = "blur_uncertain"
    else: category = "not_blurry"
    primary_score = max(scores) if scores else None
    if category == "low_resolution":
        level = {"blur_level_percent": None, "blur_level_label":"unreliable_due_to_low_resolution",
                 "blur_description":"Blur percentage is withheld because the detected face has too few original pixels. See face_resolution for exact dimensions."}
    else:
        decision = "blurred" if category == "blurred" else ("not_blurry" if category == "not_blurry" else "uncertain")
        level = _blur_level_metadata(primary_score, decision, image_resolution)
    return {"is_blurred": bool(blurred_faces), "category": category,
            "blur_score": primary_score, **level,
            "image_resolution": image_resolution,
            "details": {"global_image_blur_score": global_score,
                        "global_image_metrics": global_details,
                        "source_image_resolution": image_resolution,
                        "faces_analyzed": len(face_results),
                        "blurred_face_count": len(blurred_faces),
                        "low_resolution_face_count": len(low_res_faces),
                        "uncertain_face_count": len(uncertain_faces),
                        "blur_levels": {"0": "Sharp / no strong blur evidence",
                                        "20": "Very slight blur", "40": "Mild blur",
                                        "60": "Moderate blur", "80": "Heavy blur",
                                        "100": "Very heavy blur"},
                        "aggregation": "any clearly blurred face marks image blurred; low resolution is not treated as blur"},
            "faces": face_results, "uncertain": bool(low_res_faces or uncertain_faces),
            "score_kind": "heuristic_0_100_not_probability", "blur_probability": None,
            "blur_level_scale_note": "20/40/60/80/100 are heuristic severity bands; they are not the percentage of pixels blurred or a probability.",
            "probability_note": "Blur level is a heuristic severity estimate, not a calibrated probability."}


def _check_whole_image_blur(frame_cv2: np.ndarray) -> dict:
    """Whole-image blur fallback with regional consensus and pixel diagnostics."""
    if frame_cv2 is None or not isinstance(frame_cv2, np.ndarray) or frame_cv2.size == 0:
        return {"is_blurred":False, "category":"blur_uncertain", "blur_score":None,
                "blur_level_percent":None, "blur_level_label":"unavailable",
                "blur_description":"No image pixels were available for analysis.",
                "image_resolution":_resolution_metadata(0,0),
                "details":{"reason":"empty_image"}, "source":"whole_image_fallback", "uncertain":True}
    h,w = frame_cv2.shape[:2]
    image_resolution = _resolution_metadata(w,h)
    regions = [("center", frame_cv2[int(h*.15):max(int(h*.85),1), int(w*.15):max(int(w*.85),1)])]
    for name,(ya,yb,xa,xb) in {"top_left":(0,.65,0,.65), "top_right":(0,.65,.35,1),
                               "bottom_left":(.35,1,0,.65), "bottom_right":(.35,1,.35,1)}.items():
        regions.append((name,frame_cv2[int(h*ya):max(int(h*yb),1), int(w*xa):max(int(w*xb),1)]))
    measured=[]
    for name,region in regions:
        if region is None or region.size == 0 or min(region.shape[:2]) < 16: continue
        score,details=compute_blur_score(region)
        measured.append({"region":name,"pixel_dimensions":{"width_px":int(region.shape[1]),"height_px":int(region.shape[0]),"pixel_count":int(region.shape[0]*region.shape[1])},
                         "blur_score":score,"decision":details.get("decision"),"details":details})
    blurred=[r for r in measured if r["decision"]=="blurred"]
    sharp=[r for r in measured if r["decision"]=="not_blurry"]
    if len(blurred)>=2 and len(blurred)>len(sharp): category,is_blurred="blurred",True
    elif len(sharp)>=2 and len(sharp)>=len(blurred): category,is_blurred="not_blurry",False
    else: category,is_blurred="blur_uncertain",False
    scores=[r["blur_score"] for r in measured]
    score=round(float(np.median(scores)),2) if scores else None
    level=_blur_level_metadata(score, "blurred" if category=="blurred" else ("not_blurry" if category=="not_blurry" else "uncertain"), image_resolution)
    if category=="not_blurry" and score is not None and score<=42:
        level=_blur_level_metadata(score,"not_blurry",image_resolution)
    return {"is_blurred":is_blurred,"category":category,"blur_score":score,**level,
            "image_resolution":image_resolution,
            "details":{"source_image_resolution":image_resolution,"region_count":len(measured),
                       "blurred_region_count":len(blurred),"sharp_region_count":len(sharp),
                       "regions":measured,"decision_rule":"regional consensus; uncertain when signals disagree"},
            "source":"whole_image_fallback","uncertain":category=="blur_uncertain",
            "score_kind":"heuristic_0_100_not_probability","blur_probability":None,
            "blur_level_scale_note":"20/40/60/80/100 are heuristic severity bands; they are not the percentage of pixels blurred or a probability.",
            "probability_note":"Blur level is a heuristic severity estimate, not a calibrated probability."}


def detect_framing(frame_cv2: np.ndarray, bbox_, valid_indices: list) -> dict:
    """
    Passport-photo framing / composition check.
    MTCNN bbox spans approximately forehead→chin. We check:
      - Top margin  < 6%  → hair/forehead cut off
      - Bottom margin < 8% → neck/chin cut off
      - Face height > 72% of image → too zoomed in
      - Face width  > 65% of image → too zoomed in
    """
    if not valid_indices or bbox_ is None:
        return {"is_focused": False, "issues": [], "details": {}}

    img_h, img_w = frame_cv2.shape[:2]
    b = bbox_[valid_indices[0]]
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

    top_margin    = y1 / img_h
    bottom_margin = (img_h - y2) / img_h
    face_h_ratio  = (y2 - y1) / img_h
    face_w_ratio  = (x2 - x1) / img_w

    issues = []
    if top_margin < FRAMING_TOP_MARGIN_MIN:
        issues.append("forehead or hair cut off — more space needed above head")
    if bottom_margin < FRAMING_BOTTOM_MARGIN_MIN:
        issues.append("neck or chin cut off — more space needed below face")
    if face_h_ratio > FRAMING_FACE_H_MAX:
        issues.append(f"face fills {face_h_ratio:.0%} of image height — too zoomed in")
    if face_w_ratio > FRAMING_FACE_W_MAX:
        issues.append(f"face fills {face_w_ratio:.0%} of image width — too zoomed in")

    return {
        "is_focused": len(issues) > 0,
        "issues":     issues,
        "details": {
            "top_margin":    round(top_margin,    4),
            "bottom_margin": round(bottom_margin, 4),
            "face_h_ratio":  round(face_h_ratio,  4),
            "face_w_ratio":  round(face_w_ratio,  4),
        },
    }


# ═══════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════════════════════════════

def _get_upscaled_face_ctx(frame_cv2: np.ndarray, bbox, min_face_dim: int = 80):
    """For small detected faces, extract and Lanczos-upscale the face region.

    Returns (frame_to_use, bbox_to_use): either the original frame+bbox (face is
    already large enough) or an upscaled face-crop with remapped bbox coordinates.
    All subsequent detectors (YOLO, classifier, blur) operate on the returned pair,
    getting a larger, cleaner input without changing the pipeline structure.
    """
    try:
        ih, iw = frame_cv2.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        if min(fw, fh) >= min_face_dim:
            return frame_cv2, bbox
        pad = int(max(fw, fh) * 0.25)
        cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
        cx2 = min(iw, x2 + pad); cy2 = min(ih, y2 + pad)
        crop = frame_cv2[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return frame_cv2, bbox
        ch, cw = crop.shape[:2]
        scale = max(min_face_dim / max(min(ch, cw), 1), 2.0)
        new_w, new_h = int(round(cw * scale)), int(round(ch * scale))
        upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        new_bbox = [
            int(round((x1 - cx1) * scale)),
            int(round((y1 - cy1) * scale)),
            int(round((x2 - cx1) * scale)),
            int(round((y2 - cy1) * scale)),
        ]
        return upscaled, new_bbox
    except Exception:
        return frame_cv2, bbox


def _upscale_if_needed(frame_cv2: np.ndarray, min_dim: int = 300) -> tuple:
    """Upscale images whose shortest side is below min_dim using Lanczos.
    Returns (upscaled_frame, pil_image, was_upscaled, scale_factor)."""
    h, w = frame_cv2.shape[:2]
    short = min(h, w)
    if short >= min_dim:
        return frame_cv2, None, False, 1.0
    scale = min_dim / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    upscaled = cv2.resize(frame_cv2, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    pil_up = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
    return upscaled, pil_up, True, scale


def run_full_validation(
    pil_image: Image.Image,
    frame_cv2: np.ndarray,
) -> dict:
    """
    Priority chain:
      dimension_guard -> upscale_if_needed -> full_image_prescreen
      -> no_human -> irrelevant -> multiple_person
      -> no_landmarks -> sunglasses -> eyes_closed -> pose
    """
    img_h, img_w = frame_cv2.shape[:2]
    t0           = time.time()

    # Step 0: dimension guard
    dim_error = validate_image_dimensions(frame_cv2)
    if dim_error:
        return {"category": "irrelevant", "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [],
                "validation": {"dimension_error": dim_error},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 0b: upscale tiny images before ALL detection.
    # Detectors (MTCNN, YOLO, MediaPipe) all perform better on larger inputs.
    frame_cv2_up, pil_up, _was_upscaled, _up_scale = _upscale_if_needed(frame_cv2, min_dim=300)
    if _was_upscaled:
        frame_cv2 = frame_cv2_up
        pil_image = pil_up
        img_h, img_w = frame_cv2.shape[:2]
        logger.debug("Pre-screen: upscaled image to %dx%d (scale=%.2f)", img_w, img_h, _up_scale)

    # Step 1: downscale for all MediaPipe operations
    proc_frame, proc_scale = _resize_for_processing(frame_cv2)
    proc_h, proc_w         = proc_frame.shape[:2]

    # Step 2: MTCNN on original full-resolution image
    bbox_, prob_, mtcnn_lms_ = mtcnn.detect(pil_image, landmarks=True)

    # Step 3: confidence + size filter
    valid_indices: list = []
    if bbox_ is not None and prob_ is not None:
        for i, p in enumerate(prob_):
            if p is None or float(p) < MTCNN_CONFIDENCE_THRESHOLD:
                continue
            fw = abs(float(bbox_[i][2]) - float(bbox_[i][0]))
            fh = abs(float(bbox_[i][3]) - float(bbox_[i][1]))
            if fw < MIN_FACE_SIZE_PX or fh < MIN_FACE_SIZE_PX:
                logger.debug(f"Skipping small face #{i+1}: {fw:.0f}x{fh:.0f}px")
                continue
            valid_indices.append(i)

    valid_bboxes = [bbox_[i] for i in valid_indices] if valid_indices else []
    valid_probs  = [prob_[i]  for i in valid_indices] if valid_indices else []

    # Scene check (Module 1B). Face-found path: borderline / tiny detections are
    # re-checked so a cartoon or drawn face cannot pass as a real frontal face.
    scene_info = None
    if valid_bboxes and SCENE_CHECK_ENABLED:
        _min_prob = min(float(p) for p in valid_probs)
        _min_side = min(min(abs(float(b[2]) - float(b[0])), abs(float(b[3]) - float(b[1])))
                        for b in valid_bboxes)
        if _min_prob < SCENE_BORDERLINE_CONF or _min_side < SCENE_SMALL_FACE_PX:
            _weakest   = min(range(len(valid_probs)), key=lambda _k: float(valid_probs[_k]))
            scene_info = scene_check_safe(frame_cv2, valid_bboxes[_weakest])
            logger.debug("Scene check (borderline face) | prob=%.3f side=%.0f | %s",
                        _min_prob, _min_side, _scene_summary(scene_info))
            if _scene_reject_face_as_cartoon(scene_info):
                return {"category": "irrelevant", "primary_category": "irrelevant",
                        "categories": ["irrelevant"], "is_valid": False,
                        "total_faces": 0, "frontal_count": 0, "side_count": 0, "faces": [],
                        "validation": {"scene": scene_info,
                                       "irrelevant": {"is_irrelevant": True,
                                                      "reason": "Scene check: cartoon/illustration face, not a real person"}},
                        "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Early, landmark-independent sunglasses pass. This runs before framing and
    # FaceMesh early returns so clipped/occluded faces can still be categorized.
    # For small faces, use an upscaled face crop so all detectors get a larger input.
    pre_sunglasses = []
    for pre_idx, pre_bbox in enumerate(valid_bboxes):
        _sg_frame, _sg_bbox = _get_upscaled_face_ctx(frame_cv2, pre_bbox)
        pre_result = detect_sunglasses(
            proc_frame, None, frame_cv2_orig=_sg_frame, orig_bbox=_sg_bbox
        )
        pre_sunglasses.append(pre_result)
        logger.debug(
            "Sunglasses precheck | face=%s detected=%s reason=%s details=%s",
            pre_idx + 1, pre_result.get("detected"), pre_result.get("reason"),
            pre_result.get("details", {})
        )
    pre_sunglasses_any = any(r.get("detected", False) for r in pre_sunglasses)

    pre_headwear = []
    for _hw_idx, _hw_bbox in enumerate(valid_bboxes):
        _hw_frame, _hw_up_bbox = _get_upscaled_face_ctx(frame_cv2, _hw_bbox)
        _hw_result = detect_headwear(_hw_frame, _hw_up_bbox)
        pre_headwear.append(_hw_result)
        logger.debug("Headwear precheck | face=%s detected=%s kind=%s",
                    _hw_idx + 1, _hw_result.get("detected"), _hw_result.get("kind"))
    pre_headwear_any = any(r.get("detected", False) for r in pre_headwear)

    # Pre-gender: run on every detected face bbox so gender is available
    # even in early-return paths (multiple_person, blurry, etc.)
    pre_gender = [detect_gender(frame_cv2, b) for b in valid_bboxes]

    # ConvNeXt also predicts headwear (cap/hijab/turban) — supplement YOLO results
    for _gi, _gres in enumerate(pre_gender):
        _cnx_hw = _gres.get("headwear")
        if _cnx_hw and _gi < len(pre_headwear):
            if not pre_headwear[_gi].get("detected"):
                pre_headwear[_gi]["detected"]     = True
                pre_headwear[_gi]["kind"]          = _cnx_hw
                pre_headwear[_gi]["source"]        = "convnext"
                pre_headwear[_gi]["convnext_prob"] = (_gres.get("all_probs") or {}).get(_cnx_hw)
            else:
                # YOLO only knows generic hat/helmet classes, so it cannot tell a doctor cap
                # from a turban. When ConvNeXt (trained on cap/hijab/turban) is confident and
                # disagrees, trust ConvNeXt for the *kind*.
                _p = float((_gres.get("all_probs") or {}).get(_cnx_hw) or 0.0)
                _yk = (pre_headwear[_gi].get("kind") or "").lower()
                if _p >= 0.60 and _yk != _cnx_hw:
                    pre_headwear[_gi]["yolo_kind"]     = _yk
                    pre_headwear[_gi]["kind"]          = _cnx_hw
                    pre_headwear[_gi]["source"]        = "convnext"
                    pre_headwear[_gi]["convnext_prob"] = _p
    pre_headwear_any = any(r.get("detected", False) for r in pre_headwear)

    # Module A: hair color per face
    pre_hair = [detect_hair_color(frame_cv2, b) for b in valid_bboxes]

    # Module B: spectacles per face
    pre_specs = [detect_spectacles(frame_cv2, b) for b in valid_bboxes]

    # Module C: combine into appearance label per face
    pre_appearance = [
        _appearance_for_face(
            frame_cv2, valid_bboxes[_ai],
            pre_gender[_ai]   if _ai < len(pre_gender)   else {},
            pre_hair[_ai]     if _ai < len(pre_hair)     else {},
            pre_specs[_ai]    if _ai < len(pre_specs)    else {},
            pre_headwear[_ai] if _ai < len(pre_headwear) else {},
        )
        for _ai in range(len(valid_bboxes))
    ]
    logger.debug("Appearance labels: %s", [a.get("appearance_label") for a in pre_appearance])

    # Step 4: no-human / irrelevant
    no_human = detect_no_human(valid_bboxes, valid_probs)
    if not no_human["is_human"]:
        # Scene check first: objects / scenery / animals / cartoons return immediately
        # (no blur check, no helmet/goggle YOLO scans). Only images that contain a
        # person (or are uncertain) continue through the old no-face checks.
        scene_info    = scene_info or scene_check_safe(frame_cv2)
        scene_verdict = _scene_verdict_no_face(scene_info)
        logger.debug("Scene check (no face) | verdict=%s | %s", scene_verdict, _scene_summary(scene_info))
        if scene_verdict in ("cartoon", "nonperson"):
            _why = ("cartoon/illustration" if scene_verdict == "cartoon"
                    else f"non-person image ({scene_info.get('label')})")
            return {"category": "irrelevant", "primary_category": "irrelevant",
                    "categories": ["irrelevant"], "is_valid": False,
                    "total_faces": 0, "frontal_count": 0, "side_count": 0, "faces": [],
                    "validation": {"no_human": no_human, "scene": scene_info,
                                   "irrelevant": {"is_irrelevant": True,
                                                  "reason": f"No face + scene check: {_why}"}},
                    "processing_ms": round((time.time() - t0) * 1000, 2)}

        # Blurry images can prevent face detection entirely — check before no_human/irrelevant
        whole_blur = _check_whole_image_blur(frame_cv2)
        if whole_blur["is_blurred"]:
            return {"category": "blurred", "primary_category": "blurred",
                    "categories": ["blurred"], "is_valid": False,
                    "total_faces": 0, "frontal_count": 0, "side_count": 0, "faces": [],
                    "blur": whole_blur,
                    "validation": {"no_human": no_human, "blur": whole_blur,
                                   "pose_unavailable": "No face landmarks were detected; pose cannot be inferred reliably."},
                    "processing_ms": round((time.time() - t0) * 1000, 2)}
        # Full-image pre-screen: check ALL categories on the whole image before
        # giving up with no_human. Catches helmets, goggles and sunglasses that
        # completely hide or are cropped away from the face.
        fi_hw = _detect_headwear_yolo_fullimage(frame_cv2)

        # Full-image sunglasses/goggles YOLO scan (pass whole image as bbox).
        _full_bbox = [0, 0, img_w, img_h]
        fi_sg = _detect_sunglasses_yolo(frame_cv2, _full_bbox)
        fi_sg_hit = bool(fi_sg and (fi_sg.get("yolo_detected") or fi_sg.get("yolo_soft")))

        if fi_sg_hit and fi_sg.get("yolo_detected"):
            sg_kind = "goggles" if (fi_sg.get("yolo_goggles", 0) > fi_sg.get("yolo_sunglasses", 0)) else "sunglasses"
            logger.debug("Full-image sunglasses scan | kind=%s sg=%.3f goggles=%.3f",
                        sg_kind, fi_sg.get("yolo_sunglasses", 0), fi_sg.get("yolo_goggles", 0))
            return {
                "category":          "sunglasses",
                "primary_category":  "sunglasses",
                "categories":        ["sunglasses"],
                "is_valid":          False,
                "total_faces":       0,
                "frontal_count":     0,
                "side_count":        0,
                "faces":             [],
                "sunglasses_detected": True,
                "headwear_detected": False,
                "blur":              whole_blur,
                "validation":        {"no_human": no_human, "full_image_sunglasses": fi_sg},
                "processing_ms":     round((time.time() - t0) * 1000, 2),
            }

        _fi_hw_hit = bool(fi_hw and (fi_hw.get("yolo_detected") or fi_hw.get("yolo_soft")))
        if _fi_hw_hit:
            _hw_kind = fi_hw.get("yolo_best_kind") or "helmet"
            logger.debug("Full-image headwear scan (informational) | kind=%s helmet=%.3f cap=%.3f",
                        _hw_kind, fi_hw.get("yolo_helmet_conf", 0), fi_hw.get("yolo_cap_conf", 0))

        if scene_verdict == "person_present":
            irr = {"is_irrelevant": False,
                   "reason": "Scene check: person present but no usable face -> no_human"}
        else:
            irr = detect_irrelevant_image(valid_bboxes, frame_cv2)   # uncertain -> old skin rule
        category = "irrelevant" if irr["is_irrelevant"] else "no_human"
        # If the whole image is inconclusive, preserve that fact as a secondary
        # quality label. This makes API responses consistent with the face path.
        secondary_categories = (["blur_uncertain"] if whole_blur.get("category") == "blur_uncertain" else [])
        return {"category": category, "primary_category": category,
                "categories": secondary_categories + [category],
                "is_valid": False, "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [],
                "headwear_detected": _fi_hw_hit,
                "headwear_kind":    (_hw_kind if _fi_hw_hit else None),
                "headwear_detail":  (fi_hw if _fi_hw_hit else None),
                "blur": whole_blur,
                "validation": {"no_human": no_human, "irrelevant": irr, "blur": whole_blur, "scene": scene_info,
                               "full_image_headwear": fi_hw},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 5: quality check is an attribute, not an early-exit classification.
    # Continue to pose / person classification even when the face is blurry.
    blur_result = _detect_blur_on_face(frame_cv2, bbox_, valid_indices)
    quality_categories = []
    if blur_result.get("is_blurred"):
        quality_categories.append("blurred")
    elif blur_result.get("category") in ("low_resolution", "blur_uncertain"):
        quality_categories.append(blur_result["category"])
    logger.debug(f"Blur | score={blur_result.get('blur_score', '?')} "
                f"category={blur_result.get('category')} "
                f"is_blurred={blur_result.get('is_blurred')} "
                f"details={blur_result.get('details', {})}")

    # Step 5b: multiple-person check (operates on downscaled proc_frame)
    person_check = detect_persons_and_size(proc_frame)
    if not person_check["valid"]:
        categories = quality_categories + [person_check["category"]]
        if pre_sunglasses_any:
            categories.append("sunglasses")
        if pre_headwear_any:
            categories.append("headwear")
        _pc_gender0 = pre_gender[0] if pre_gender else {}
        if _pc_gender0.get("gender"):
            categories.append(_pc_gender0["gender"])
        return {"category": ("sunglasses" if pre_sunglasses_any else person_check["category"]),
                "categories": list(dict.fromkeys(categories)),
                "primary_category": ("sunglasses" if pre_sunglasses_any else person_check["category"]),
                "sunglasses_detected": pre_sunglasses_any,
                "gender": _pc_gender0.get("gender"),
                "gender_confidence": _pc_gender0.get("confidence"),
                "faces": [{"face_id": i + 1, "bbox": [round(float(v), 2) for v in b],
                           "sunglasses": r, "gender": pre_gender[i] if i < len(pre_gender) else {}, "pose": "Unknown",
                           "appearance": pre_appearance[i] if i < len(pre_appearance) else {},
                           "headwear": pre_headwear[i] if i < len(pre_headwear) else {},
                           "hair": pre_hair[i] if i < len(pre_hair) else {},
                           "specs": pre_specs[i] if i < len(pre_specs) else {}}
                          for i, (b, r) in enumerate(zip(valid_bboxes, pre_sunglasses))],
                "total_faces": person_check["total_faces"],
                "frontal_count": 0, "side_count": 0,
                "blur": blur_result,
                "validation": {"no_human": no_human, "person_check": person_check,
                               "blur": blur_result},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 5c: framing check — passport composition (hair, neck, zoom)
    framing_result = detect_framing(frame_cv2, bbox_, valid_indices)
    if framing_result["is_focused"]:
        categories = quality_categories + ["focused"]
        if pre_sunglasses_any:
            categories.append("sunglasses")
        if pre_headwear_any:
            categories.append("headwear")
        _fr_g = pre_gender[0].get("gender") if pre_gender else None
        if _fr_g:
            categories.append(_fr_g)
        return {
            "category":      ("sunglasses" if pre_sunglasses_any else "focused"),
            "primary_category": ("sunglasses" if pre_sunglasses_any else "focused"),
            "categories":    list(dict.fromkeys(categories)),
            "sunglasses_detected": pre_sunglasses_any,
            "is_valid":      False,
            "total_faces":   len(valid_indices),
            "frontal_count": 0, "side_count": 0,
            "gender": pre_gender[0].get("gender") if pre_gender else None,
            "gender_confidence": pre_gender[0].get("confidence") if pre_gender else None,
            "faces": [{"face_id": i + 1, "bbox": [round(float(v), 2) for v in b],
                       "sunglasses": r, "gender": pre_gender[i] if i < len(pre_gender) else {}, "pose": "Unknown",
                           "appearance": pre_appearance[i] if i < len(pre_appearance) else {},
                           "headwear": pre_headwear[i] if i < len(pre_headwear) else {},
                           "hair": pre_hair[i] if i < len(pre_hair) else {},
                           "specs": pre_specs[i] if i < len(pre_specs) else {}}
                      for i, (b, r) in enumerate(zip(valid_bboxes, pre_sunglasses))],
            "blur":          blur_result,
            "blur_debug":    blur_result,
            "framing":       framing_result,
            "validation":    {"no_human": no_human, "person_check": person_check,
                              "framing": framing_result, "blur": blur_result},
            "processing_ms": round((time.time() - t0) * 1000, 2),
        }

    # Step 6: FaceMesh (proc_frame)
    face_mesh  = _get_face_mesh()
    frame_rgb  = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    mp_results = face_mesh.process(frame_rgb)
    mp_faces   = mp_results.multi_face_landmarks or []
    mp_count   = len(mp_faces)
    mtcnn_count = len(valid_indices)

    if mtcnn_count > 0 and mp_count == 0:
        categories = quality_categories + ["no_landmarks"]
        if pre_sunglasses_any:
            categories.append("sunglasses")
        _nl_g = pre_gender[0].get("gender") if pre_gender else None
        if _nl_g:
            categories.append(_nl_g)
        return {"category": ("sunglasses" if pre_sunglasses_any else "no_landmarks"),
                "primary_category": ("sunglasses" if pre_sunglasses_any else "no_landmarks"),
                "categories": list(dict.fromkeys(categories)),
                "sunglasses_detected": pre_sunglasses_any, "is_valid": False,
                "total_faces": len(valid_indices), "frontal_count": 0, "side_count": 0,
                "gender": pre_gender[0].get("gender") if pre_gender else None,
                "gender_confidence": pre_gender[0].get("confidence") if pre_gender else None,
                "faces": [{"face_id": i + 1, "bbox": [round(float(v), 2) for v in b],
                           "sunglasses": r, "gender": pre_gender[i] if i < len(pre_gender) else {}, "pose": "Unknown",
                           "appearance": pre_appearance[i] if i < len(pre_appearance) else {},
                           "headwear": pre_headwear[i] if i < len(pre_headwear) else {},
                           "hair": pre_hair[i] if i < len(pre_hair) else {},
                           "specs": pre_specs[i] if i < len(pre_specs) else {}}
                          for i, (b, r) in enumerate(zip(valid_bboxes, pre_sunglasses))],
                "blur": blur_result,
                "validation": {"no_human": no_human,
                               "blur": blur_result,
                               "reason": "FaceMesh failed (lighting/occlusion/resolution)"},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    if mp_count > 1:
        categories = quality_categories + ["multiple_person"]
        if pre_sunglasses_any:
            categories.append("sunglasses")
        if pre_headwear_any:
            categories.append("headwear")
        for _mg in pre_gender:
            _g_val = _mg.get("gender")
            if _g_val:
                categories.append(_g_val)
        _mp_primary = ("sunglasses" if pre_sunglasses_any else
                       ("headwear" if pre_headwear_any else "multiple_person"))
        return {"category": _mp_primary,
                "primary_category": _mp_primary,
                "categories": list(dict.fromkeys(categories)),
                "sunglasses_detected": pre_sunglasses_any, "is_valid": False,
                "total_faces": mp_count, "frontal_count": 0, "side_count": 0,
                "gender": pre_gender[0].get("gender") if pre_gender else None,
                "gender_confidence": pre_gender[0].get("confidence") if pre_gender else None,
                "faces": [{"face_id": i + 1, "bbox": [round(float(v), 2) for v in b],
                           "sunglasses": r, "gender": pre_gender[i] if i < len(pre_gender) else {}, "pose": "Unknown",
                           "appearance": pre_appearance[i] if i < len(pre_appearance) else {},
                           "headwear": pre_headwear[i] if i < len(pre_headwear) else {},
                           "hair": pre_hair[i] if i < len(pre_hair) else {},
                           "specs": pre_specs[i] if i < len(pre_specs) else {}}
                          for i, (b, r) in enumerate(zip(valid_bboxes, pre_sunglasses))],
                "blur": blur_result,
                "validation": {"no_human": no_human, "blur": blur_result,
                               "multi_person": {"mtcnn_faces": mtcnn_count,
                                                "mediapipe_faces": mp_count,
                                                "reason": "Multiple faces by FaceMesh"}},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 7: match bboxes -> landmarks
    matched_mp = _match_landmarks_to_faces(
        valid_bboxes, mp_faces, proc_w, proc_h, proc_scale
    )

    # Step 8: per-face analysis
    face_results: list = []
    all_poses:    list = []
    face_counter = 0   

    for local_idx, global_idx in enumerate(valid_indices):
        bbox      = bbox_[global_idx]
        prob      = float(prob_[global_idx])
        mtcnn_lms = mtcnn_lms_[global_idx]
        mp_face   = matched_mp[local_idx]

        if mtcnn_lms is None:
            logger.warning(f"MTCNN landmarks None for detection #{local_idx+1} — skip")
            continue

        face_counter += 1
        face_data = {
            "face_id":    face_counter,
            "confidence": round(prob, 4),
            "bbox":       [round(float(v), 2) for v in bbox],
        }

        angR = _np_angle(np.array(mtcnn_lms[0]), np.array(mtcnn_lms[1]),
                         np.array(mtcnn_lms[2]))
        angL = _np_angle(np.array(mtcnn_lms[1]), np.array(mtcnn_lms[0]),
                         np.array(mtcnn_lms[2]))
        face_data["angle_right_eye"] = round(angR, 2)
        face_data["angle_left_eye"]  = round(angL, 2)
        face_data["landmarks"] = {
            "left_eye":    [float(mtcnn_lms[0][0]), float(mtcnn_lms[0][1])],
            "right_eye":   [float(mtcnn_lms[1][0]), float(mtcnn_lms[1][1])],
            "nose":        [float(mtcnn_lms[2][0]), float(mtcnn_lms[2][1])],
            "left_mouth":  [float(mtcnn_lms[3][0]), float(mtcnn_lms[3][1])],
            "right_mouth": [float(mtcnn_lms[4][0]), float(mtcnn_lms[4][1])],
        }

        # Sunglasses: YOLO on original-res face crop, fallback to signal voting on proc_frame.
        # Use upscaled face context for small faces so all detectors get a larger input.
        _lm_sg_frame, _lm_sg_bbox = _get_upscaled_face_ctx(frame_cv2, bbox)
        landmark_sunglasses = detect_sunglasses(
            proc_frame, mp_face, frame_cv2_orig=_lm_sg_frame, orig_bbox=_lm_sg_bbox
        )
        # OR-combine the landmark-independent crop pass with the landmark-aware
        # pass; a later weak/negative eye-polygon result must not erase a crop hit.
        pre_sg = pre_sunglasses[local_idx] if local_idx < len(pre_sunglasses) else {}
        if pre_sg.get("detected") or landmark_sunglasses.get("detected"):
            chosen = landmark_sunglasses if landmark_sunglasses.get("detected") else pre_sg
            merged_details = dict(chosen.get("details") or {})
            merged_details["pre_landmark_crop_pass"] = pre_sg
            merged_details["landmark_aware_pass"] = landmark_sunglasses
            face_data["sunglasses"] = {
                "detected": True, "votes": max(int(pre_sg.get("votes", 0) or 0),
                                                 int(landmark_sunglasses.get("votes", 0) or 0)),
                "reason": "Detected by independent crop or landmark-aware pass",
                "details": merged_details,
            }
        else:
            face_data["sunglasses"] = landmark_sunglasses

        # Headwear: use pre-pass result (no landmarks needed — crops above face bbox)
        face_data["headwear"] = pre_headwear[local_idx] if local_idx < len(pre_headwear) else {}

        # Closed eyes: pass original bbox for size guard
        proc_bbox = [v * proc_scale for v in bbox] if bbox is not None else None
        face_data["eyes"] = detect_closed_eyes(
            mp_face, proc_w, proc_h,
            orig_bbox=bbox,
            proc_bbox=proc_bbox,
        )

        # Pose: proc-space dims for camera matrix
        yaw_data   = _get_mediapipe_angles(mp_face, proc_w, proc_h) if mp_face else None
        nose_ratio = _get_nose_visibility_ratio(mp_face)             if mp_face else 1.0
        pose, pose_signals = classify_pose_strong(angR, angL, yaw_data, nose_ratio)

        face_data["pose"]         = pose
        face_data["pose_signals"] = pose_signals
        face_data["gender"]       = detect_gender(frame_cv2, bbox)
        face_data["hair"]         = pre_hair[local_idx]       if local_idx < len(pre_hair)       else {}
        face_data["specs"]        = pre_specs[local_idx]      if local_idx < len(pre_specs)      else {}
        face_data["appearance"]   = pre_appearance[local_idx] if local_idx < len(pre_appearance) else {}
        all_poses.append(pose)
        face_results.append(face_data)

    # Step 9: guard
    if not face_results:
        categories = quality_categories + ["no_human"]
        return {"category": "no_human", "primary_category": "no_human",
                "categories": categories, "is_valid": False, "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [], "blur": blur_result,
                "validation": {"no_human": no_human, "person_check": person_check,
                               "blur": blur_result},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 10: priority decision
    sunglasses_any  = any(f["sunglasses"].get("detected", False) for f in face_results)
    headwear_any    = any((f.get("headwear") or {}).get("detected", False) for f in face_results)
    eyes_closed_any = any(f["eyes"].get("eyes_closed",   False) for f in face_results)

    # Pose is an independent result: sunglasses must not hide the fact that
    # the head is turned sideways. Keep `category` as the legacy primary label,
    # but expose every applicable condition in `categories`.
    fc = all_poses.count("Frontal")
    sc = all_poses.count("Side")
    if fc > 0 and sc == 0:
        pose_category = "front"
    elif sc > 0 and fc == 0:
        pose_category = "side"
    else:
        pose_category = "mixed"

    if sunglasses_any:
        final_category = "sunglasses"
    elif eyes_closed_any:
        final_category = "eyes_closed"
    else:
        final_category = pose_category

    # blur_sunglasses_uncertain: image quality is degraded AND sunglasses not confirmed
    # AND there is actual evidence of sunglasses that blur may have suppressed.
    # Without the evidence check, every blurry image without sunglasses would match.
    def _sg_has_evidence(sg):
        if not sg or sg.get("detected"):
            return False
        if sg.get("status") == "low_resolution_uncertain":
            return True
        d = sg.get("details") or {}
        clf = d.get("classifier_prob")
        if clf is not None and clf >= 0.30:
            return True
        if d.get("yolo_soft"):
            return True
        ev = d.get("evidence") or {}
        if d.get("ev_score", 0) >= 2:
            return True
        return False

    blur_sg_uncertain = bool(
        quality_categories and not sunglasses_any and not headwear_any
        and any(_sg_has_evidence(f.get("sunglasses")) for f in face_results)
    )

    categories = list(quality_categories)
    categories.append(pose_category)
    if sunglasses_any:
        categories.append("sunglasses")
    if headwear_any:
        categories.append("headwear")
    if eyes_closed_any:
        categories.append("eyes_closed")
    if blur_sg_uncertain:
        categories.append("blur_sunglasses_uncertain")
        if final_category in ("blurred", "low_resolution", "blur_uncertain", pose_category):
            final_category = "blur_sunglasses_uncertain"
    # Add all unique genders detected across all faces
    for _fr in face_results:
        _g = (_fr.get("gender") or {}).get("gender")
        if _g and _g not in ("unknown", "uncertain"):
            categories.append(_g)
    # Preserve order while removing duplicates.
    categories = list(dict.fromkeys(categories))
    return {
        "category":      final_category,
        "primary_category": final_category,
        "pose_category": pose_category,
        "sunglasses_detected": sunglasses_any,
        "headwear_detected": headwear_any,
        "display_category": (f"{pose_category.lower()}+sunglasses" if sunglasses_any else
                             (f"{pose_category.lower()}+headwear" if headwear_any else pose_category.lower())),
        "categories":    categories,
        "is_valid":      (final_category == "front" and not quality_categories),
        "total_faces":   len(face_results),
        "frontal_count": all_poses.count("Frontal"),
        "side_count":    all_poses.count("Side"),
        "faces":         face_results,
        "gender":        (face_results[0].get("gender") or {}).get("gender") if face_results else None,
        "gender_confidence": (face_results[0].get("gender") or {}).get("confidence") if face_results else None,
        "gender_detail": face_results[0].get("gender") if face_results else None,
        "blur":          blur_result,
        "blur_debug":    blur_result,
        "validation":    {"no_human": no_human, "person_check": person_check,
                           "blur": blur_result},
        "processing_ms": round((time.time() - t0) * 1000, 2),
    }


# ═══════════════════════════════════════════════════════════════
# ANNOTATION
# ═══════════════════════════════════════════════════════════════

def _annotation_frame(frame_cv2: np.ndarray) -> np.ndarray:
    """Image to draw the debug overlay on.

    run_full_validation() enlarges small photos (short side < 300 px) before detecting, so every
    box / landmark it returns is in that enlarged space. Drawing them on the original small image
    put the overlay off to the side of the face. Draw on the same enlarged image instead."""
    up, _pil, was_up, _scale = _upscale_if_needed(frame_cv2, min_dim=300)
    return (up if was_up else frame_cv2).copy()


def draw_annotations(frame: np.ndarray, result: dict) -> np.ndarray:
    h_f, w_f  = frame.shape[:2]
    category  = result["category"]
    color_map = {
        "front": (0,255,0), "side": (0,0,255), "mixed": (255,165,0),
        "eyes_closed": (0,255,255), "multiple_person": (255,0,255),
        "sunglasses": (255,0,255), "no_human": (128,128,128),
        "irrelevant": (128,128,128), "no_landmarks": (100,100,200),
        "blurred": (0,0,255), "blur_uncertain": (0,165,255),
        "blur_sunglasses_uncertain": (0,100,255), "low_resolution": (0,165,255),
    }
    box_color = color_map.get(category, (200,200,200))

    def _cx(v): return max(0, min(int(v), w_f - 1))
    def _cy(v): return max(0, min(int(v), h_f - 1))

    for face in result.get("faces", []):
        bbox = face.get("bbox"); pose = face.get("pose","Unknown")
        eyes = face.get("eyes",{}); sigs = face.get("pose_signals",{})
        lms  = face.get("landmarks",{}); sg = face.get("sunglasses",{})

        # Defaults for faces without a bbox (landmarks still drawn at fallback size)
        _lm_r = 3; _ln_thick = 1

        if bbox:
            x1,y1,x2,y2 = _cx(bbox[0]),_cy(bbox[1]),_cx(bbox[2]),_cy(bbox[3])
            # Scale every drawing parameter to the face pixel width so small/blurry/
            # distant faces get thin marks instead of oversized blobs.
            # Formula: face_px=200 → box_thick=2, lm_r=5, font=0.57
            #          face_px=80  → box_thick=1, lm_r=2, font=0.35 (minimal, unobtrusive)
            #          face_px=400 → box_thick=3, lm_r=7, font=0.85 (clear on large face)
            _face_px   = max(1, x2 - x1)
            _box_thick = max(1, _face_px // 80)
            _lm_r      = max(1, _face_px // 40)
            _ln_thick  = max(1, _face_px // 90)
            _fscale    = max(0.35, min(0.85, _face_px / 350.0))
            _fthick    = max(1, _face_px // 150)
            _lh        = max(10, int(_fscale * 15))
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, _box_thick)
            # When the face touches the top edge, labels are placed below the
            # face box instead of stacking over the eyes/forehead.
            label_y = (y1 - _lh * 5) if y1 >= _lh * 5 + 5 else min(h_f - 8, y2 + _lh)
            label_x = min(x1, max(0, w_f - 180))
            cv2.putText(frame, f"Conf:{face.get('confidence',0):.2f}",
                        (label_x, max(12, label_y)), FONT, _fscale, (200,200,200), 1, cv2.LINE_AA)
            if sg.get("detected"):
                cv2.putText(frame, "SUNGLASSES", (label_x, min(h_f - 8, max(12, label_y + _lh))),
                            FONT, _fscale, (255,0,255), _fthick, cv2.LINE_AA)
            cv2.putText(frame,
                        f"{pose}|Score:{sigs.get('total_score','?')}|{sigs.get('confidence','?')}",
                        (label_x,min(h_f - 8, max(12, label_y + _lh * 2))), FONT, _fscale, box_color, _fthick, cv2.LINE_AA)
            if "yaw" in sigs:
                cv2.putText(frame,
                            f"Yaw:{sigs['yaw']} Pitch:{sigs['pitch']} Roll:{sigs['roll']}",
                            (label_x,min(h_f - 8, max(12, label_y + _lh * 3))), FONT, _fscale, (200,200,200), 1, cv2.LINE_AA)
            ear_text  = "EYES CLOSED" if eyes.get("eyes_closed") else f"EAR:{eyes.get('avg_ear','?')}"
            ear_color = (0,255,255) if eyes.get("eyes_closed") else (150,255,150)
            cv2.putText(frame, ear_text, (label_x,min(h_f - 8, max(12, label_y + _lh * 4))),
                        FONT, _fscale, ear_color, 1, cv2.LINE_AA)

        for key in ("left_eye","right_eye","nose","left_mouth","right_mouth"):
            pt = lms.get(key)
            if pt: cv2.circle(frame, (_cx(pt[0]),_cy(pt[1])), _lm_r, (0,255,255), -1)
        try:
            le,re,ns = lms["left_eye"],lms["right_eye"],lms["nose"]
            for p1,p2 in [(le,re),(le,ns),(re,ns)]:
                cv2.line(frame, (_cx(p1[0]),_cy(p1[1])), (_cx(p2[0]),_cy(p2[1])),
                         LINE_COLOR, _ln_thick)
        except Exception:
            pass

    # Put the status strip at the bottom: a top banner can physically cover
    # sunglasses/eyes when the source face is cropped near the upper edge.
    banner_h = min(34, h_f)
    banner_y = max(0, h_f - banner_h)
    cv2.rectangle(frame, (0, banner_y), (w_f - 1, h_f - 1), (30,30,30), -1)
    status_text = (
        f"Category:{category.upper()} | Faces:{result['total_faces']} | "
        f"F:{result['frontal_count']} S:{result['side_count']} | "
        f"{result['processing_ms']}ms"
    )
    cv2.putText(frame, status_text,
                (8, min(h_f - 6, banner_y + 23)), FONT, FONT_SCALE,
                (255,255,255), FONT_THICK, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════
# Response cleaning
# ═══════════════════════════════════════════════════════════════

def _clean_sunglasses(sg: dict) -> dict:
    if not sg:
        return sg
    d = sg.get("details") or {}
    return {
        "detected": sg.get("detected"),
        "reason":   sg.get("reason"),
        "details": {
            "yolo_detected":      d.get("yolo_detected"),
            "yolo_sunglasses":    d.get("yolo_sunglasses"),
            "yolo_goggles":       d.get("yolo_goggles"),
            "yolo_clear_glasses": d.get("yolo_clear_glasses"),
            "classifier_prob":    d.get("classifier_prob"),
        },
    }


def _clean_blur(b: dict) -> dict:
    if not b:
        return b
    return {
        "is_blurred":       b.get("is_blurred"),
        "blur_score":       b.get("blur_score"),
        "blur_level":       b.get("blur_level_label"),
        "blur_level_pct":   b.get("blur_level_percent"),
        "blur_description": b.get("blur_description"),
    }


def _clean_face(f: dict) -> dict:
    g = f.get("gender") or {}
    return {
        "face_id":           f.get("face_id"),
        "bbox":              f.get("bbox"),
        "pose":              f.get("pose"),
        "gender":            g.get("gender"),
        "gender_status":     g.get("status", "not_run"),
        "gender_reason":     g.get("reason"),
        "gender_confidence": g.get("confidence"),
        "gender_probabilities": g.get("probs"),
        "gender_all_probs":  g.get("all_probs"),
        "gender_model":      g.get("model"),
        "gender_input_preprocessing": g.get("input_preprocessing"),
        "gender_crop_bbox":  g.get("crop_bbox"),
        "sunglasses":        _clean_sunglasses(f.get("sunglasses") or {}),
        "headwear":          _clean_headwear(f.get("headwear") or {}),
        "blur_score":        f.get("blur_score"),
        "blur_level":        f.get("blur_level_label"),
        "hair":              f.get("hair") or {},
        "specs":             f.get("specs") or {},
        "appearance":        f.get("appearance") or {},
    }


_APPEARANCE_ORDER = ["black_hair", "specs_black_hair", "white_hair", "specs_white_hair",
                     "doctor_cap", "doctor_cap_specs", "turban", "turban_specs"]
_APPEARANCE_TITLES = {
    "black_hair":       "Black hair",
    "specs_black_hair": "Spectacles & black hair",
    "white_hair":       "White hair",
    "specs_white_hair": "Spectacles & white hair",
    "doctor_cap":       "Doctor cap",
    "doctor_cap_specs": "Doctor cap & spectacles",
    "turban":           "Turban",
    "turban_specs":     "Turban & spectacles",
}
# code -> (short title, plain-language explanation shown to the user)
_PROBLEM_INFO = {
    "goggles":         ("Sunglasses / goggles", "Sunglasses or dark goggles are covering the eyes. Please upload a photo without them."),
    "multiple_person": ("More than one person", "More than one face was found. Please upload a photo of one person only."),
    "side_face":       ("Side face", "The face is turned to the side. Please upload a front-facing photo."),
    "blur":            ("Blurred photo", "The photo is blurred. Please upload a sharper photo."),
    "low_resolution":  ("Low resolution", "The face is too small or the quality is too low. Please upload a larger, clearer photo."),
    "no_human":        ("No face found", "No human face was detected in this photo."),
    "irrelevant":      ("Not a person photo", "This image does not look like a photo of a person."),
    "no_landmarks":    ("Face not readable", "A face was found but its features could not be read (lighting, angle or something covering it)."),
    "focused":         ("Poor framing", "The face is too close, too far or cropped. Please retake with the whole head and shoulders visible."),
    "front":           ("Front face", "Front-facing photo of one person."),
}
_WARNING_TEXT = {
    "eyes_closed": "Eyes appear to be closed.",
    "focused":     "Face framing is off (too close or cropped). Head and shoulders should be visible.",
    "blurred":     "The photo looks blurred.",
    "blur_uncertain": "The photo may be slightly blurred.",
    "low_resolution": "The photo resolution is low.",
}
_ISSUE_ORDER = ["no_face", "focused", "goggles", "multiple_person", "blur", "low_resolution", "eyes_closed"]


# ── small helpers: JSON must always be valid and the same shape ──────────────
def _num(v, nd: int = 3):
    try:
        v = float(v)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, nd)
    except Exception:
        return None


def _clamp01(v) -> float:
    return max(0.0, min(1.0, float(v)))


def _json_safe(o):
    """numpy -> python, NaN/inf -> None, so jsonable_encoder / clients never choke."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _num(o, 4)
    return o


# ── SECTION 1 builders: a confidence for EVERY category ─────────────────────
def _scores_appearance(app: dict):
    probs = (app or {}).get("category_probs")
    if probs:
        return {k: _num(probs.get(k, 0.0)) for k in _APPEARANCE_ORDER}, "model_probability"
    lab = (app or {}).get("appearance_label")
    if not lab:                                   # appearance was not evaluated -> unknown, not 0%
        return {k: None for k in _APPEARANCE_ORDER}, "not_evaluated"
    return {k: (1.0 if k == lab else 0.0) for k in _APPEARANCE_ORDER}, "rule_based"


def _scores_gender(g: dict) -> dict:
    p = (g or {}).get("all_probs") or {}
    m, f = p.get("male"), p.get("female")
    try:
        s = float(m) + float(f)
        if s > 0:
            return {"male": _num(float(m) / s), "female": _num(float(f) / s)}
    except Exception:
        pass
    return {"male": None, "female": None}


def _scores_pose(faces_raw: list) -> dict:
    front_vals = []
    for f in faces_raw:
        sig = f.get("pose_signals") or {}
        tot, mx = sig.get("total_score"), sig.get("max_score")
        try:
            raw = 0.5 + 0.5 * max(-1.0, min(1.0, float(tot) / float(mx)))
        except Exception:
            continue
        if sig.get("pose_decision") == "Side":
            raw = min(raw, 0.49)
        elif sig.get("pose_decision") == "Frontal":
            raw = max(raw, 0.5)
        front_vals.append(raw)
    if not front_vals:
        return {"front": None, "side": None}
    front = min(front_vals)                      # the worst face decides
    return {"front": _num(front), "side": _num(1.0 - front)}


def _scores_checks(result: dict, faces_raw: list, blur: dict, cat: str, cats: list):
    """Problem categories. Every score is 0-1 and 0.5 is the decision point, so a score >= 0.5
    means the check fired. Returns (scores, flags)."""
    faces_n = result.get("total_faces", len(faces_raw))
    scores, flags = {}, {}

    # goggles / sunglasses
    best, flag = 0.0, bool(result.get("sunglasses_detected")) or cat == "sunglasses"
    for f in faces_raw:
        sg = f.get("sunglasses") or {}
        d = sg.get("details") or {}
        flag = flag or bool(sg.get("detected"))
        for key in ("yolo_sunglasses", "yolo_goggles", "classifier_prob"):
            try:
                best = max(best, float(d.get(key) or 0.0))
            except Exception:
                pass
    fi = ((result.get("validation") or {}).get("full_image_sunglasses")) or {}
    for key in ("yolo_sunglasses", "yolo_goggles"):
        try:
            best = max(best, float(fi.get(key) or 0.0))
        except Exception:
            pass
    scores["goggles"] = _clamp01(max(best, 0.5) if flag else min(best, 0.49))
    flags["goggles"] = flag

    # multiple people
    flag = faces_n > 1 or cat == "multiple_person"
    scores["multiple_person"] = 1.0 if flag else 0.0
    flags["multiple_person"] = flag

    # blur  (blur_score 0-100; the pipeline decides "blurred" at BLUR_SCORE_THRESHOLD -> 0.5)
    flag = bool(blur.get("is_blurred")) or cat == "blurred" or "blurred" in cats
    try:
        sc = _clamp01(0.5 * float(blur.get("blur_score")) / float(BLUR_SCORE_THRESHOLD))
    except Exception:
        sc = 0.0
    # confidence >= 0.40 → count as blur (show in detected)
    if sc >= 0.40:
        flag = True
    scores["blur"] = max(sc, 0.5) if flag else min(sc, 0.49)
    flags["blur"] = flag

    # low resolution (face size in the ORIGINAL photo; 50 px -> 0.5, 100 px or more -> 0)
    try:
        iw, ih = result.get("image_size") or (None, None)
        up = max(1.0, 300.0 / min(iw, ih)) if (iw and ih and min(iw, ih) < 300) else 1.0
        sides = [min(abs(b[2] - b[0]), abs(b[3] - b[1])) / up
                 for b in (f.get("bbox") for f in faces_raw) if b]
        sc = _clamp01(1.0 - min(sides) / (2.0 * BLUR_FACE_LOW_RES)) if sides else 0.0
    except Exception:
        sc = 0.0
    flag = cat == "low_resolution" or "low_resolution" in cats or (faces_n > 0 and sc >= 0.5)
    scores["low_resolution"] = max(sc, 0.5) if flag else min(sc, 0.49)
    flags["low_resolution"] = flag

    # eyes closed (EAR 0.15 -> 0.5)
    sc, flag = 0.0, False
    for f in faces_raw:
        e = f.get("eyes") or {}
        flag = flag or bool(e.get("eyes_closed"))
        try:
            sc = max(sc, _clamp01(1.0 - float(e.get("avg_ear")) / 0.30))
        except Exception:
            pass
    scores["eyes_closed"] = max(sc, 0.5) if flag else min(sc, 0.49)
    flags["eyes_closed"] = flag

    # poor framing (pipeline category "focused": face too close / cropped / badly placed)
    flag = cat == "focused" or "focused" in cats
    scores["focused"] = 1.0 if flag else 0.0
    flags["focused"] = flag

    # headwear on the head (cap / helmet / turban / hijab) - informational, not a rejection by itself
    hw_best, flag = 0.0, cat == "headwear" or "headwear" in cats
    for f in faces_raw:
        try:
            hp = ((f.get("appearance") or {}).get("head_probs") or {}).get("headwear") or {}
            if hp.get("none") is not None:
                hw_best = max(hw_best, 1.0 - float(hp["none"]))
            pr = (f.get("gender") or {}).get("all_probs") or {}
            for k in ("cap", "hijab", "turban"):
                hw_best = max(hw_best, float(pr.get(k) or 0.0))
            hd = f.get("headwear") or {}
            flag = flag or bool(hd.get("detected"))
            for k in ("yolo_helmet_conf", "yolo_cap_conf"):
                hw_best = max(hw_best, float((hd.get("details") or {}).get(k) or 0.0))
        except Exception:
            pass
    scores["headwear"] = max(hw_best, 0.5) if flag else min(hw_best, 0.49)
    flags["headwear"] = flag

    # no face at all
    flag = faces_n == 0
    scores["no_face"] = 1.0 if flag else 0.0
    flags["no_face"] = flag

    return {k: _num(v) for k, v in scores.items()}, flags


def _format_category_report(out: dict) -> str:
    """Plain-text version of the JSON: section 1 = confidence per category, section 2 = matched."""
    sc, mt = out.get("scores") or {}, out.get("matched") or {}
    on = set(mt.get("categories") or []) | set(mt.get("issues") or [])

    def bar(v):
        n = int(round((v or 0.0) * 20))
        return "#" * n + "." * (20 - n)

    def row(name, v):
        pct = "  n/a" if v is None else f"{v * 100:5.1f}%"
        return f"   {'>' if name in on else ' '} {name:18s} {pct}  {bar(v)}"

    L = ["", "=" * 62, " 1) CONFIDENCE PER CATEGORY   ( > = matched )", "=" * 62]
    for group, title in (("appearance", "Appearance"), ("gender", "Gender"), ("pose", "Pose"), ("checks", "Problem checks")):
        L.append(f" {title}")
        d = sc.get(group) or {}
        items = sorted(d.items(), key=lambda kv: -(kv[1] if kv[1] is not None else -1)) if group == "appearance" else d.items()
        L += [row(k, v) for k, v in items]
    L += ["", "=" * 62, " 2) MATCHED CATEGORY", "=" * 62]
    ap = mt.get("appearance")
    L.append(f"   Appearance : {_APPEARANCE_TITLES.get(ap, ap) or '-'}"
             + (f"  ({((sc.get('appearance') or {}).get(ap) or 0) * 100:.1f}%)" if ap and (sc.get('appearance') or {}).get(ap) is not None else ""))
    L.append(f"   Gender     : {mt.get('gender') or '-'}")
    L.append(f"   Pose       : {mt.get('pose') or '-'}")
    L.append(f"   Problems   : {', '.join(mt.get('issues') or []) or 'none'}")
    L.append(f"   All matched: {', '.join(mt.get('categories') or []) or '-'}")
    L.append(f"   RESULT     : {str(out.get('status', '')).upper()} - {(out.get('result') or {}).get('message', '')}")
    L.append("=" * 62)
    return "\n".join(L)


def _print_category_report(out: dict) -> None:
    if os.environ.get("REPORT_PRINT", "1") == "0":
        return
    try:
        print(_format_category_report(out), flush=True)
    except Exception as exc:                      # printing must never break the API
        logger.debug("report print failed: %r", exc)


def _clean_response(result: dict, detail: bool = False) -> dict:
    """
    User-facing JSON for /validate-face. The shape is always the same; a failing section
    degrades to nulls and is listed in "errors" instead of breaking the response.

      status   : "accepted" | "review" | "rejected"
      result   : code + title + one plain sentence
      scores   : SECTION 1 - a confidence (0-1) for EVERY category
                   appearance (8), gender (male/female), pose (front/side),
                   checks (goggles, multiple_person, blur, low_resolution, eyes_closed, no_face)
      matched  : SECTION 2 - which categories are TRUE for this image
      checks / warnings / appearance / face : readable detail
    Old keys (category, audit_category, is_valid, categories, quality, ...) are kept.
    ?detail=true adds the full per-face technical breakdown ("faces").
    """
    errors = []
    faces_raw = result.get("faces") or []
    f0        = faces_raw[0] if faces_raw else {}
    cat       = result.get("category") or ""
    cats      = list(dict.fromkeys(c for c in result.get("categories", []) if c))
    app       = f0.get("appearance") or {}
    hair_res  = f0.get("hair") or {}
    spec_res  = f0.get("specs") or {}
    hw_res    = f0.get("headwear") or {}
    gender0   = f0.get("gender") or {}
    try:
        blur = _clean_blur(result.get("blur") or {}) or {}
    except Exception:
        blur = {}
    label     = app.get("appearance_label")
    ml        = app.get("source") == "ml"
    faces_n   = result.get("total_faces", len(faces_raw))

    # ── result code (kept identical to the old `audit_category`) ───────────────
    if cat == "sunglasses":
        code = "goggles"
    elif cat in ("multiple_person", "no_human", "irrelevant", "no_landmarks", "low_resolution", "focused"):
        code = cat
    elif cat in ("side", "mixed"):
        code = "side_face"
    elif cat in ("blurred", "blur_uncertain", "blur_sunglasses_uncertain"):
        code = "blur"
    else:
        code = label or "front"

    # ── SECTION 1: confidence for every category ──────────────────────────────
    scores = {"appearance": {}, "gender": {}, "pose": {}, "checks": {}}
    flags, basis = {}, None
    try:
        scores["appearance"], basis = _scores_appearance(app)
    except Exception as exc:
        errors.append(f"appearance_scores: {exc!r}")
        scores["appearance"] = {k: None for k in _APPEARANCE_ORDER}
    try:
        scores["gender"] = _scores_gender(gender0)
    except Exception as exc:
        errors.append(f"gender_scores: {exc!r}"); scores["gender"] = {"male": None, "female": None}
    try:
        scores["pose"] = _scores_pose(faces_raw)
    except Exception as exc:
        errors.append(f"pose_scores: {exc!r}"); scores["pose"] = {"front": None, "side": None}
    try:
        scores["checks"], flags = _scores_checks(result, faces_raw, blur, cat, cats)
    except Exception as exc:
        errors.append(f"check_scores: {exc!r}")
        scores["checks"] = {k: None for k in _ISSUE_ORDER}
        flags = {k: False for k in _ISSUE_ORDER}

    # Override code to "blur" if blur flag fired (score >= 0.40) and no higher-priority issue
    if flags.get("blur") and code not in ("goggles", "multiple_person", "no_human", "irrelevant",
                                           "no_landmarks", "low_resolution", "focused", "side_face"):
        code = "blur"

    # ── SECTION 2: which categories are TRUE ──────────────────────────────────
    g = scores["gender"]
    p = scores["pose"]
    m_gender = None if g.get("male") is None else ("male" if g["male"] >= g["female"] else "female")
    m_pose = None if p.get("front") is None else ("front" if p["front"] >= 0.5 else "side")
    issues = [k for k in _ISSUE_ORDER if flags.get(k)]
    hw_flag = bool(flags.get("headwear"))
    matched_all = [x for x in (label, m_gender, m_pose) if x] + issues + (["headwear"] if hw_flag else [])
    matched = {
        "appearance": label,
        "gender":     m_gender,
        "pose":       m_pose,
        "issues":     issues,
        "categories": matched_all,
    }

    # ── warnings: things that did not decide the result but matter ────────────
    eyes_closed = bool(flags.get("eyes_closed"))
    warnings = []
    for key in ("eyes_closed", "focused", "blurred", "blur_uncertain", "low_resolution"):
        if (key == "eyes_closed" and eyes_closed) or (key != "eyes_closed" and (cat == key or key in cats)):
            if code != {"blurred": "blur", "blur_uncertain": "blur", "low_resolution": "low_resolution"}.get(key):
                warnings.append(_WARNING_TEXT[key])
    warnings = list(dict.fromkeys(warnings))

    # ── status + message ──────────────────────────────────────────────────────
    uncertain = bool(app.get("uncertain"))
    if code in _APPEARANCE_TITLES:
        title = _APPEARANCE_TITLES[code]
        if uncertain or warnings:
            status = "review"
            why = "Low confidence - please check manually." if uncertain else warnings[0]
            message = f"Looks like: {title}. {why}"
        else:
            status = "accepted"
            message = f"Accepted. Photo classified as: {title}."
    else:
        title, why = _PROBLEM_INFO.get(code, (code.replace("_", " ").capitalize(), ""))
        status, message = "rejected", why or title
        warnings = []

    # ── checks (one readable line per rule) ───────────────────────────────────
    has_face = faces_n > 0
    poses    = [f.get("pose") for f in faces_raw if f.get("pose")]
    sg_found = bool(flags.get("goggles"))
    blurred  = bool(flags.get("blur"))
    low_res  = bool(flags.get("low_resolution"))
    checks = {
        "one_person":   {"ok": faces_n == 1,
                         "note": f"{faces_n} face(s) found"},
        "front_facing": {"ok": (result.get("frontal_count", 0) >= 1 and result.get("side_count", 0) == 0) if has_face else None,
                         "note": ", ".join(poses) if poses else "not evaluated"},
        "eyes_open":    {"ok": (not eyes_closed) if has_face else None,
                         "note": "eyes closed" if eyes_closed else ("eyes open" if has_face else "not evaluated")},
        "no_sunglasses": {"ok": (not sg_found) if has_face else None,
                          "note": "sunglasses / goggles found" if sg_found else ("none found" if has_face else "not evaluated")},
        "sharp":        {"ok": not blurred,
                         "note": blur.get("blur_level") or ("blurred" if blurred else "ok")},
        "resolution":   {"ok": not low_res,
                         "note": "low resolution" if low_res else "ok"},
        "framing":      {"ok": not (cat == "focused" or "focused" in cats),
                         "note": "poorly framed" if (cat == "focused" or "focused" in cats) else "ok"},
    }

    # ── FIRST in the JSON: everything detected, each with its confidence ──────
    #    e.g. {"male": 0.999, "front": 1.0, "black_hair": 0.67}
    detected = {}
    if m_gender:
        detected[m_gender] = g.get(m_gender)
    if m_pose:
        detected[m_pose] = p.get(m_pose)
    if label:
        detected[label] = scores["appearance"].get(label)
    for k in issues:
        detected[k] = scores["checks"].get(k)
    if hw_flag:
        detected["headwear"] = scores["checks"].get("headwear")
    # checks that were measured (score > 0) but did NOT fire, e.g. {"blur": 0.245}
    not_triggered = {k: v for k, v in scores["checks"].items() if k not in detected and v}

    out = {
        "detected":            detected,
        "not_triggered":       not_triggered,
        "status":              status,
        "result":              {"code": code, "title": title, "message": message},
        "needs_manual_review": status == "review",
        "warnings":            warnings,

        # SECTION 1 - confidence of every category (0-1)
        "scores":  {**scores, "basis": {"appearance": basis,
                                        "gender": "model_probability",
                                        "pose": "rule_score",
                                        "checks": "rule_score (0.5 = decision point)"}},
        # SECTION 2 - which categories are true for this image
        "matched": matched,

        "appearance": {
            "label":       label,
            "title":       _APPEARANCE_TITLES.get(label),
            "hair_color":  (app.get("hair_color") if ml else hair_res.get("hair_color")),
            "specs":       (bool(app.get("specs_detected")) if ml else bool(spec_res.get("detected"))),
            "headwear":    (app.get("headwear_kind") if ml else (hw_res.get("kind") if hw_res.get("detected") else None)),
            "confidence":  app.get("confidence"),
            "uncertain":   app.get("uncertain"),
            "source":      app.get("source"),
        },
        "face": {
            "total":             faces_n,
            "frontal":           result.get("frontal_count", 0),
            "side":              result.get("side_count", 0),
            "pose":              poses[0] if len(poses) == 1 else (poses or None),
            "gender":            m_gender,
            "gender_confidence": (g.get(m_gender) if m_gender else None),
        },
        "checks": checks,
        "quality": {
            "blur_score": blur.get("blur_score"),
            "is_blurred": blurred,
            "low_res":    low_res,
            "sunglasses": sg_found,
        },

        # ── kept from the old format so existing clients keep working ─────────
        "category":       result.get("category"),
        "audit_category": code,
        "is_valid":       result.get("is_valid"),
        "categories":     cats,
        "processing_ms":  result.get("processing_ms"),
        "output_image":   result.get("output_image"),
        "filename":       result.get("filename"),
        "saved":          result.get("saved", False),
        "errors":         errors,
    }
    if detail:
        try:
            out["faces"] = [_clean_face(f) for f in faces_raw]
        except Exception as exc:
            out["errors"].append(f"faces_detail: {exc!r}")
    return _json_safe(out)

# ═══════════════════════════════════════════════════════════════
# API endpoints
# ═══════════════════════════════════════════════════════════════

# Allowed MIME types for /validate-face
_ALLOWED_IMAGE_MIMES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/bmp", "image/gif", "image/tiff", "image/tif",
    "image/x-bmp", "image/x-ms-bmp", "image/avif", "image/heic",
    "image/heif", "image/x-png", "image/pjpeg",
    "application/octet-stream",  # some clients send this for images
}
_ALLOWED_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
    ".tiff", ".tif", ".avif", ".heic", ".heif",
}

@app.post("/validate-face")
async def validate_face(
    file: UploadFile = File(...),
    save: bool       = Query(False),
    detail: bool     = Query(False, description="Include the full per-face technical breakdown ('faces')."),
    annotate: bool   = Query(False, description="Return annotated image as base64 JPEG in 'annotated_image_b64'."),
):
    # Accept by MIME or by file extension
    mime = (file.content_type or "").lower().split(";")[0].strip()
    fname = (file.filename or "").lower()
    ext   = os.path.splitext(fname)[1] if fname else ""
    if mime not in _ALLOWED_IMAGE_MIMES and ext not in _ALLOWED_IMAGE_EXTS:
        raise HTTPException(
            400,
            f"Unsupported file type '{mime}'. "
            "Supported: JPEG, PNG, WEBP, BMP, GIF, TIFF, AVIF, HEIC"
        )
    try:
        contents  = await file.read();  await file.close()
        try:
            pil_image = Image.open(io.BytesIO(contents))
            pil_image.verify()           # catch truncated / corrupt files
            pil_image = _exif_upright(Image.open(io.BytesIO(contents)))  # re-open after verify
        except Exception as img_err:
            raise HTTPException(400, f"Cannot read image file: {img_err}")
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        elif pil_image.mode == "L":
            pil_image = pil_image.convert("RGB")
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        result    = run_full_validation(pil_image, cv2_image)
        category  = result["category"]
        output_path = filename = annotated_b64 = None
        if save or annotate:
            annotated_frame = draw_annotations(_annotation_frame(cv2_image), result)
            if save:
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename    = f"{category}_{timestamp}.jpg"
                output_path = os.path.join(CATEGORY_FOLDERS.get(category, BASE_OUTPUT), filename)
                cv2.imwrite(output_path, annotated_frame)
                logger.info(f"Saved: {filename} -> {category}")
            if annotate:
                _, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                annotated_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        result.update({"output_image": output_path, "filename": filename,
                        "saved": save,
                        "annotated_image_b64": annotated_b64,
                        "success": result.get("is_valid", category == "front")})
        status_code = (CATEGORY_STATUS_MAP.get("blurred", 422)
                       if "blurred" in result.get("categories", [])
                       else CATEGORY_STATUS_MAP.get(category, 400))
        logger.info(f"Done | {category} | HTTP {status_code} | "
                    f"faces={result['total_faces']} | {result['processing_ms']}ms")
        return JSONResponse(status_code=status_code, content=jsonable_encoder(_clean_response(result, detail=detail)))
    except HTTPException:
        raise                      # keep intended 4xx responses (bad/corrupt upload)
    except Exception as exc:
        import traceback as _tb
        logger.error(f"Processing error: {exc}\n{_tb.format_exc()}")
        raise HTTPException(500, str(exc))


@app.get("/get-image")
async def get_image(filename: str = Query(...)):
    path = os.path.join(BASE_OUTPUT, filename)
    if not os.path.exists(path):
        for f in CATEGORY_FOLDERS.values():
            cand = os.path.join(f, filename)
            if os.path.exists(cand): path = cand; break
    if not os.path.exists(path):
        raise HTTPException(404, f"'{filename}' not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/get-latest")
async def get_latest(
    category: str = Query("front", description=(
        "front|side|mixed|eyes_closed|multiple_person|sunglasses|"
        "no_human|irrelevant|no_landmarks"
    ))
):
    folder = CATEGORY_FOLDERS.get(category)
    if not folder: raise HTTPException(400, f"Unknown category: {category}")
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    if not files: raise HTTPException(404, f"No images in '{category}'")
    return FileResponse(os.path.join(folder, files[-1]), media_type="image/jpeg")


@app.get("/list-outputs")
async def list_outputs():
    output = {}; total = 0
    for cat, folder in CATEGORY_FOLDERS.items():
        files = sorted(os.listdir(folder), reverse=True)
        output[cat] = {"count": len(files), "files": files}
        total += len(files)
    return JSONResponse({"success": True, "total": total, "categories": output})


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device),
            "gpu": torch.cuda.is_available(),
            "version": "2.5.0",
            "categories": list(CATEGORY_FOLDERS.keys())}


@app.get("/debug-gender")
async def debug_gender():
    result = {
        "model_path": _GENDER_MODEL_PATH,
        "model_file_exists": os.path.isfile(_GENDER_MODEL_PATH),
        "model_loaded": _gender_model is not None,
        "model_failed": _gender_model_failed,
        "model_error": _gender_model_error,
        "architecture": _gender_model_arch,
    }
    model = _load_gender_model()
    result["model_loaded"] = model is not None
    result["model_failed"] = _gender_model_failed
    result["model_error"] = _gender_model_error
    if model is not None:
        try:
            from PIL import Image as _PIL
            dummy_pil = _PIL.new("RGB", (224, 224), color=(128, 128, 128))
            test_result = model(dummy_pil)
            result["dummy_inference_smoke_test"] = test_result
            result["dummy_inference_passed"] = True
        except Exception as exc:
            result["dummy_inference_passed"] = False
            result["dummy_inference_error"] = f"{type(exc).__name__}: {exc}"
    return result


@app.post("/validate-folder")
async def validate_folder(
    folder_path: str = Query(...),
    save: bool       = Query(False),
):
    if not os.path.exists(folder_path):
        raise HTTPException(404, "Folder does not exist")
    if not os.path.isdir(folder_path):
        raise HTTPException(400, "Path is not a folder")
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not images: raise HTTPException(404, "No supported images found")

    batch_results: list = []; total_processed = 0; t0 = time.time()
    for img_name in sorted(images):
        img_path  = os.path.join(folder_path, img_name)
        file_size = os.path.getsize(img_path)
        if file_size > BATCH_MAX_FILE_BYTES:
            logger.warning(f"Skipping {img_name}: {file_size/1e6:.1f}MB > limit")
            batch_results.append({"original_file": img_name,
                "error": f"File too large ({file_size/1e6:.1f}MB)"})
            continue
        try:
            with Image.open(img_path) as pil_img:
                pil_img = _exif_upright(pil_img)
                if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
                pil_copy = pil_img.copy()
            cv2_img  = cv2.cvtColor(np.array(pil_copy), cv2.COLOR_RGB2BGR)
            result   = run_full_validation(pil_copy, cv2_img)
            category = result["category"]
            saved_as = None
            if save:
                annotated = draw_annotations(_annotation_frame(cv2_img), result)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename  = f"{category}_{timestamp}.jpg"
                out_path  = os.path.join(CATEGORY_FOLDERS.get(category, BASE_OUTPUT), filename)
                if not cv2.imwrite(out_path, annotated):
                    raise RuntimeError("cv2.imwrite failed")
                saved_as = filename
            batch_results.append({"original_file": img_name, "category": category,
                                   "faces_detected": result["total_faces"], "saved_as": saved_as})
            total_processed += 1
        except Exception as exc:
            logger.error(f"Batch error — {img_name}: {exc}")
            batch_results.append({"original_file": img_name, "error": str(exc)})

    return JSONResponse({"success": True,
                         "folder_path": os.path.abspath(folder_path),
                         "total_images_found": len(images),
                         "total_processed": total_processed,
                         "processing_time_ms": round((time.time() - t0)*1000, 2),
                         "results": batch_results})




import asyncio
import io
import os
import tempfile
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import aiohttp
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Known column → URL prefix mapping (applied when value is NOT a full URL)
# Add or change entries here if column names or prefixes change.
# ─────────────────────────────────────────────────────────────
KNOWN_COLUMN_PREFIXES: dict[str, str] = {
    "revice_photograph": "https://ih.imagicahealth.in/teammindnext/doc_kyc/revise_doctors/",
    "photograph":        "https://ih.imagicahealth.in/teammindnext/doctors/",
    # Add more known columns here as needed
}

# ─────────────────────────────────────────────────────────────
# Human-readable comments per category
# ─────────────────────────────────────────────────────────────
CATEGORY_COMMENT: dict[str, str] = {
    "front":            "Valid frontal face",
    "side":             "Face is not frontal — side pose detected",
    "mixed":            "Mixed poses detected",
    "eyes_closed":      "Eyes are closed",
    "sunglasses":       "Dark goggles or sunglasses detected",
    "headwear":         "Cap, helmet, or face mask detected",
    "multiple_person":  "Multiple persons detected in image",
    "no_human":         "No human face detected",
    "irrelevant":       "Image does not contain a human",
    "no_landmarks":     "Face landmarks could not be extracted",
    "blurred":          "Image/face has strong multi-signal blur evidence; pose may still be classified",
    "blur_uncertain":   "Sharpness evidence is inconclusive; manual review or additional analysis recommended",
    "low_resolution":  "Face has insufficient pixels for a reliable sharpness decision",
    "focused":          "Face framing/composition needs correction",
}

# ─────────────────────────────────────────────────────────────
# Concurrency / retry settings
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 10
IMAGE_DOWNLOAD_TIMEOUT   = 15       # seconds per attempt
BATCH_SIZE               = 200
THREAD_WORKERS           = min(8, max(4, os.cpu_count() or 4))
MAX_RETRIES              = 2        # retry on transient failures (timeout/503)
RETRY_DELAY              = 1.5      # seconds between retries

# ─────────────────────────────────────────────────────────────
# In-memory job store
# Replace with Redis / DB for multi-process deployments
# ─────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
# Schema per job:
# {
#   "status":       "pending" | "running" | "done" | "failed",
#   "submitted_at": str,
#   "started_at":   str | None,
#   "finished_at":  str | None,
#   "total_rows":   int,
#   "rows_done":    int,
#   "total_valid":  int | None,
#   "total_invalid":int | None,
#   "elapsed_ms":   float | None,
#   "output_path":  str | None,
#   "output_name":  str | None,
#   "error":        str | None,
# }


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _detect_image_columns(df: pd.DataFrame) -> list[str]:
    """
    Scan all columns and return those that look like image URL/filename columns.
    Detection rules (applied to a sample of non-null values):
      - Value starts with http:// or https://             → URL column
      - Value ends with known image extension             → filename column
      - Column name contains known keywords               → included
    Returns columns in priority order: known columns first (revice > photo), then others.
    """
    import re
    IMG_EXT = re.compile(r"\.(jpe?g|png|webp|gif|bmp|tiff?|avif|heic|heif)$", re.I)
    IMG_URL_RE = re.compile(r"^https?://", re.I)

    def _looks_like_image(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sum(
            1 for v in sample
            if IMG_URL_RE.match(v.strip()) or IMG_EXT.search(v.strip())
        )
        return hits / len(sample) >= 0.10  # at least 10% look like images

    # Priority: known columns first (revice before photo), then auto-detected
    found = []
    for col in KNOWN_COLUMN_PREFIXES:
        if col in df.columns:
            found.append(col)
    for col in df.columns:
        if col not in found and _looks_like_image(df[col]):
            found.append(col)
    return found


def _build_image_url_smart(row: pd.Series, url_cols: list[str]) -> tuple[str | None, str]:
    """
    Try each detected image column in priority order.
    - If value is a full URL (http/https), use as-is.
    - If column is in KNOWN_COLUMN_PREFIXES, prepend that prefix.
    - Otherwise use value as-is (assume full URL).
    """
    for col in url_cols:
        val = str(row.get(col, "") or "").strip()
        if not val or val.lower() in ("nan", "none", ""):
            continue
        if val.startswith("http://") or val.startswith("https://"):
            return val, col
        prefix = KNOWN_COLUMN_PREFIXES.get(col, "")
        return prefix + val, col
    return None, "none"


def _build_image_url(row: pd.Series) -> tuple[str | None, str]:
    """Legacy fallback — used only when url_cols not passed."""
    return _build_image_url_smart(row, list(KNOWN_COLUMN_PREFIXES.keys()))


def _empty_result(comment: str, status_code: int = 404) -> dict:
    return {
        "status_code":       status_code,
        "validation_status": "no_human",
        "is_valid":          0,
        "comment":           comment,
    }


def _run_validation_sync(pil_image: Image.Image) -> dict:
    import cv2
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return run_full_validation(pil_image, cv2_image)   # from your main.py


async def _immediate_result(res: dict) -> dict:
    return res


async def _download_and_validate(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    url:       str,
) -> dict:
    """
    Download one image and validate.
    Retries up to MAX_RETRIES times on timeout / 503.
    Uses asyncio.get_running_loop() for thread-safe execution in async context.
    """
    async with semaphore:
        raw = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):   # attempts: 1, 2, 3
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                ) as resp:
                    if resp.status == 503:
                        last_error = "HTTP 503 Service Unavailable"
                        if attempt <= MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * attempt)
                            continue
                        return _empty_result(last_error, status_code=503)

                    if resp.status != 200:
                        return _empty_result(
                            f"HTTP {resp.status} when downloading image",
                            status_code=resp.status,
                        )
                    raw = await resp.read()
                    break   # success

            except asyncio.TimeoutError:
                last_error = "Image request timed out"
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                    continue
                return _empty_result(last_error, status_code=408)

            except aiohttp.ClientError as exc:
                return _empty_result(f"Network error: {exc}", status_code=503)

            except Exception as exc:
                return _empty_result(f"Download error: {exc}", status_code=500)

    if raw is None:
        return _empty_result(last_error or "Unknown download failure", status_code=500)

    # Decode image
    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_image = _exif_upright(img).convert("RGB")
    except Exception as exc:
        return _empty_result(f"Cannot decode image: {exc}", status_code=422)

    # Run CPU-bound validation in thread pool using the running event loop
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _run_validation_sync, pil_image)
    except Exception as exc:
        return _empty_result(f"Validation error: {exc}", status_code=500)

    category = result.get("category", "no_human")
    categories = result.get("categories", [category])
    quality_issue = any(label in ("blurred", "blur_uncertain", "low_resolution")
                        for label in categories)
    quality_label = next((label for label in ("blurred", "low_resolution", "blur_uncertain")
                          if label in categories), None)
    comments = [CATEGORY_COMMENT.get(label, label) for label in categories]
    # Build audit_category (same logic as _clean_response)
    _faces = result.get("faces") or []
    _app_label = (_faces[0].get("appearance") or {}).get("appearance_label") if _faces else None
    if category == "sunglasses":
        _audit = "goggles"
    elif category == "multiple_person":
        _audit = "multiple_person"
    elif category in ("side", "mixed"):
        _audit = "side_face"
    elif category in ("blurred", "blur_uncertain", "blur_sunglasses_uncertain"):
        _audit = "blur"
    elif category == "low_resolution":
        _audit = "low_resolution"
    elif _app_label:
        _audit = _app_label
    else:
        _audit = "front"
    return {
        "status_code":          CATEGORY_STATUS_MAP.get(quality_label, CATEGORY_STATUS_MAP.get(category, 400)) if quality_issue else CATEGORY_STATUS_MAP.get(category, 400),
        "validation_status":    category,  # backward-compatible single primary label
        "validation_categories": categories,
        "audit_category":       _audit,
        "appearance_label":     _app_label,
        "hair_color":           (_faces[0].get("hair") or {}).get("hair_color") if _faces else None,
        "specs_detected":       (_faces[0].get("specs") or {}).get("detected") if _faces else None,
        "headwear_kind":        (_faces[0].get("headwear") or {}).get("kind") if _faces else None,
        "is_valid":             1 if category == "front" and not quality_issue else 0,
        "comment":              "; ".join(dict.fromkeys(comments)),
    }


async def _process_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    batch:     list[tuple[str | None, str]],
) -> list[dict]:
    tasks = [
        _immediate_result(_empty_result("No image address provided in this row"))
        if url is None
        else _download_and_validate(session, semaphore, executor, url)
        for url, _ in batch
    ]
    return await asyncio.gather(*tasks, return_exceptions=False)


# ═══════════════════════════════════════════════════════════════
# Background worker  (runs after POST returns job_id)
# ═══════════════════════════════════════════════════════════════

async def _run_validation_job(job_id: str, df: pd.DataFrame) -> None:
    job = _jobs[job_id]
    job["status"]     = "running"
    job["started_at"] = datetime.now().isoformat()
    t0 = time.time()

    total_rows = len(df)
    job["total_rows"] = total_rows
    all_results: list[dict] = []

    try:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        executor  = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)

        async with aiohttp.ClientSession(connector=connector) as session:
            # Process rows in batches to keep memory usage low
            for batch_start in range(0, total_rows, BATCH_SIZE):
                batch_df = df.iloc[batch_start : batch_start + BATCH_SIZE]
                batch    = [_build_image_url(row) for _, row in batch_df.iterrows()]

                batch_results = await _process_batch(session, semaphore, executor, batch)
                all_results.extend(batch_results)

                job["rows_done"] = min(batch_start + BATCH_SIZE, total_rows)
                logger.info(
                    f"job={job_id} | progress {job['rows_done']}/{total_rows}"
                )

        executor.shutdown(wait=True)

        # Append result columns
        df["status_code"]       = [r["status_code"]       for r in all_results]
        df["validation_status"] = [r["validation_status"] for r in all_results]
        df["audit_category"]    = [r.get("audit_category")    for r in all_results]
        df["appearance_label"]  = [r.get("appearance_label")  for r in all_results]
        df["hair_color"]        = [r.get("hair_color")        for r in all_results]
        df["specs_detected"]    = [r.get("specs_detected")    for r in all_results]
        df["headwear_kind"]     = [r.get("headwear_kind")     for r in all_results]
        df["is_valid"]          = [r["is_valid"]           for r in all_results]
        df["comment"]           = [r["comment"]            for r in all_results]

        total_valid   = int(df["is_valid"].sum())
        total_invalid = total_rows - total_valid
        elapsed_ms    = round((time.time() - t0) * 1000, 2)

        # Microsecond-precision timestamp ensures zero filename collision risk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_name  = f"validated_doctors_{timestamp}.csv"
        out_path  = os.path.join(tempfile.gettempdir(), out_name)
        df.to_csv(out_path, index=False)

        job.update({
            "status":        "done",
            "finished_at":   datetime.now().isoformat(),
            "total_valid":   total_valid,
            "total_invalid": total_invalid,
            "elapsed_ms":    elapsed_ms,
            "output_path":   out_path,
            "output_name":   out_name,
        })
        logger.info(
            f"job={job_id} complete | rows={total_rows} | "
            f"valid={total_valid} | invalid={total_invalid} | {elapsed_ms}ms"
        )

    except Exception as exc:
        job.update({
            "status":      "failed",
            "finished_at": datetime.now().isoformat(),
            "error":       str(exc),
        })
        logger.error(f"job={job_id} failed: {exc}")



import asyncio
import io
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import numpy as np
import pandas as pd
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Known column → URL prefix mapping (applied when value is NOT a full URL)
# Add or change entries here if column names or prefixes change.
# ─────────────────────────────────────────────────────────────
KNOWN_COLUMN_PREFIXES: dict[str, str] = {
    "revice_photograph": "https://ih.imagicahealth.in/teammindnext/doc_kyc/revise_doctors/",
    "photograph":        "https://ih.imagicahealth.in/teammindnext/doctors/",
    # Add more known columns here as needed
}

# ─────────────────────────────────────────────────────────────
# Human-readable comments per category
# ─────────────────────────────────────────────────────────────
CATEGORY_COMMENT: dict[str, str] = {
    "front":            "Valid frontal face",
    "side":             "Face is not frontal — side pose detected",
    "mixed":            "Mixed poses detected",
    "eyes_closed":      "Eyes are closed",
    "sunglasses":       "Dark goggles or sunglasses detected",
    "headwear":         "Cap, helmet, or face mask detected",
    "multiple_person":  "Multiple persons detected in image",
    "no_human":         "No human face detected",
    "irrelevant":       "Image does not contain a human",
    "no_landmarks":     "Face landmarks could not be extracted",
    "blurred":          "Image/face has strong multi-signal blur evidence; pose may still be classified",
    "blur_uncertain":   "Sharpness evidence is inconclusive; manual review or additional analysis recommended",
    "low_resolution":  "Face has insufficient pixels for a reliable sharpness decision",
    "focused":          "Face framing/composition needs correction",
}

# ─────────────────────────────────────────────────────────────
# Concurrency / retry settings
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 10
IMAGE_DOWNLOAD_TIMEOUT   = 15
BATCH_SIZE               = 200
THREAD_WORKERS           = min(8, max(4, os.cpu_count() or 4))
MAX_RETRIES              = 2
RETRY_DELAY              = 1.5


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _detect_image_columns(df: pd.DataFrame) -> list[str]:
    """
    Scan all columns and return those that look like image URL/filename columns.
    Detection rules (applied to a sample of non-null values):
      - Value starts with http:// or https://             → URL column
      - Value ends with known image extension             → filename column
      - Column name contains known keywords               → included
    Returns columns in priority order: known columns first (revice > photo), then others.
    """
    import re
    IMG_EXT = re.compile(r"\.(jpe?g|png|webp|gif|bmp|tiff?|avif|heic|heif)$", re.I)
    IMG_URL_RE = re.compile(r"^https?://", re.I)

    def _looks_like_image(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sum(
            1 for v in sample
            if IMG_URL_RE.match(v.strip()) or IMG_EXT.search(v.strip())
        )
        return hits / len(sample) >= 0.10  # at least 10% look like images

    # Priority: known columns first (revice before photo), then auto-detected
    found = []
    for col in KNOWN_COLUMN_PREFIXES:
        if col in df.columns:
            found.append(col)
    for col in df.columns:
        if col not in found and _looks_like_image(df[col]):
            found.append(col)
    return found


def _build_image_url_smart(row: pd.Series, url_cols: list[str]) -> tuple[str | None, str]:
    """
    Try each detected image column in priority order.
    - If value is a full URL (http/https), use as-is.
    - If column is in KNOWN_COLUMN_PREFIXES, prepend that prefix.
    - Otherwise use value as-is (assume full URL).
    """
    for col in url_cols:
        val = str(row.get(col, "") or "").strip()
        if not val or val.lower() in ("nan", "none", ""):
            continue
        if val.startswith("http://") or val.startswith("https://"):
            return val, col
        prefix = KNOWN_COLUMN_PREFIXES.get(col, "")
        return prefix + val, col
    return None, "none"


def _build_image_url(row: pd.Series) -> tuple[str | None, str]:
    """Legacy fallback — used only when url_cols not passed."""
    return _build_image_url_smart(row, list(KNOWN_COLUMN_PREFIXES.keys()))


def _empty_result(comment: str, status_code: int = 404) -> dict:
    return {
        "status_code":       status_code,
        "validation_status": "no_human",
        "is_valid":          0,
        "comment":           comment,
    }


def _run_validation_sync(pil_image: Image.Image) -> dict:
    import cv2
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return run_full_validation(pil_image, cv2_image)   # from your main.py


async def _immediate_result(res: dict) -> dict:
    return res


async def _download_and_validate(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    url:       str,
) -> dict:
    async with semaphore:
        raw       = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                ) as resp:
                    if resp.status == 503:
                        last_error = "HTTP 503 Service Unavailable"
                        if attempt <= MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * attempt)
                            continue
                        return _empty_result(last_error, status_code=503)
                    if resp.status != 200:
                        return _empty_result(
                            f"HTTP {resp.status} when downloading image",
                            status_code=resp.status,
                        )
                    raw = await resp.read()
                    break
            except asyncio.TimeoutError:
                last_error = "Image request timed out"
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                    continue
                return _empty_result(last_error, status_code=408)
            except aiohttp.ClientError as exc:
                return _empty_result(f"Network error: {exc}", status_code=503)
            except Exception as exc:
                return _empty_result(f"Download error: {exc}", status_code=500)

    if raw is None:
        return _empty_result(last_error or "Unknown download failure", status_code=500)

    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_image = _exif_upright(img).convert("RGB")
    except Exception as exc:
        return _empty_result(f"Cannot decode image: {exc}", status_code=422)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _run_validation_sync, pil_image)
    except Exception as exc:
        return _empty_result(f"Validation error: {exc}", status_code=500)

    category = result.get("category", "no_human")
    categories = result.get("categories", [category])
    quality_issue = any(label in ("blurred", "blur_uncertain", "low_resolution")
                        for label in categories)
    quality_label = next((label for label in ("blurred", "low_resolution", "blur_uncertain")
                          if label in categories), None)
    comments = [CATEGORY_COMMENT.get(label, label) for label in categories]
    # Build audit_category (same logic as _clean_response)
    _faces = result.get("faces") or []
    _app_label = (_faces[0].get("appearance") or {}).get("appearance_label") if _faces else None
    if category == "sunglasses":
        _audit = "goggles"
    elif category == "multiple_person":
        _audit = "multiple_person"
    elif category in ("side", "mixed"):
        _audit = "side_face"
    elif category in ("blurred", "blur_uncertain", "blur_sunglasses_uncertain"):
        _audit = "blur"
    elif category == "low_resolution":
        _audit = "low_resolution"
    elif _app_label:
        _audit = _app_label
    else:
        _audit = "front"
    return {
        "status_code":          CATEGORY_STATUS_MAP.get(quality_label, CATEGORY_STATUS_MAP.get(category, 400)) if quality_issue else CATEGORY_STATUS_MAP.get(category, 400),
        "validation_status":    category,  # backward-compatible single primary label
        "validation_categories": categories,
        "audit_category":       _audit,
        "appearance_label":     _app_label,
        "hair_color":           (_faces[0].get("hair") or {}).get("hair_color") if _faces else None,
        "specs_detected":       (_faces[0].get("specs") or {}).get("detected") if _faces else None,
        "headwear_kind":        (_faces[0].get("headwear") or {}).get("kind") if _faces else None,
        "is_valid":             1 if category == "front" and not quality_issue else 0,
        "comment":              "; ".join(dict.fromkeys(comments)),
    }


async def _process_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    batch:     list[tuple[str | None, str]],
) -> list[dict]:
    tasks = [
        _immediate_result(_empty_result("No image address provided in this row"))
        if url is None
        else _download_and_validate(session, semaphore, executor, url)
        for url, _ in batch
    ]
    return await asyncio.gather(*tasks, return_exceptions=False)



import asyncio
import io
import os
import tempfile
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Known column → URL prefix mapping (applied when value is NOT a full URL)
# Add or change entries here if column names or prefixes change.
# ─────────────────────────────────────────────────────────────
KNOWN_COLUMN_PREFIXES: dict[str, str] = {
    "revice_photograph": "https://ih.imagicahealth.in/teammindnext/doc_kyc/revise_doctors/",
    "photograph":        "https://ih.imagicahealth.in/teammindnext/doctors/",
    # Add more known columns here as needed
}

# ─────────────────────────────────────────────────────────────
# Human-readable comments per category
# ─────────────────────────────────────────────────────────────
CATEGORY_COMMENT: dict[str, str] = {
    "front":            "Valid frontal face",
    "side":             "Face is not frontal — side pose detected",
    "mixed":            "Mixed poses detected",
    "eyes_closed":      "Eyes are closed",
    "sunglasses":       "Dark goggles or sunglasses detected",
    "headwear":         "Cap, helmet, or face mask detected",
    "multiple_person":  "Multiple persons detected in image",
    "no_human":         "No human face detected",
    "irrelevant":       "Image does not contain a human",
    "no_landmarks":     "Face landmarks could not be extracted",
    "blurred":          "Image/face has strong multi-signal blur evidence; pose may still be classified",
    "blur_uncertain":   "Sharpness evidence is inconclusive; manual review or additional analysis recommended",
    "low_resolution":  "Face has insufficient pixels for a reliable sharpness decision",
    "focused":          "Face framing/composition needs correction",
}

# ─────────────────────────────────────────────────────────────
# Concurrency / retry settings
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 10
IMAGE_DOWNLOAD_TIMEOUT   = 15
BATCH_SIZE               = 200
THREAD_WORKERS           = min(8, max(4, os.cpu_count() or 4))
MAX_RETRIES              = 2
RETRY_DELAY              = 1.5

# ─────────────────────────────────────────────────────────────
# In-memory job store  {job_id: job_dict}
# ─────────────────────────────────────────────────────────────
_JOBS: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _detect_image_columns(df: pd.DataFrame) -> list[str]:
    """
    Scan all columns and return those that look like image URL/filename columns.
    Detection rules (applied to a sample of non-null values):
      - Value starts with http:// or https://             → URL column
      - Value ends with known image extension             → filename column
      - Column name contains known keywords               → included
    Returns columns in priority order: known columns first (revice > photo), then others.
    """
    import re
    IMG_EXT = re.compile(r"\.(jpe?g|png|webp|gif|bmp|tiff?|avif|heic|heif)$", re.I)
    IMG_URL_RE = re.compile(r"^https?://", re.I)

    def _looks_like_image(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sum(
            1 for v in sample
            if IMG_URL_RE.match(v.strip()) or IMG_EXT.search(v.strip())
        )
        return hits / len(sample) >= 0.10  # at least 10% look like images

    # Priority: known columns first (revice before photo), then auto-detected
    found = []
    for col in KNOWN_COLUMN_PREFIXES:
        if col in df.columns:
            found.append(col)
    for col in df.columns:
        if col not in found and _looks_like_image(df[col]):
            found.append(col)
    return found


def _build_image_url_smart(row: pd.Series, url_cols: list[str]) -> tuple[str | None, str]:
    """
    Try each detected image column in priority order.
    - If value is a full URL (http/https), use as-is.
    - If column is in KNOWN_COLUMN_PREFIXES, prepend that prefix.
    - Otherwise use value as-is (assume full URL).
    """
    for col in url_cols:
        val = str(row.get(col, "") or "").strip()
        if not val or val.lower() in ("nan", "none", ""):
            continue
        if val.startswith("http://") or val.startswith("https://"):
            return val, col
        prefix = KNOWN_COLUMN_PREFIXES.get(col, "")
        return prefix + val, col
    return None, "none"


def _build_image_url(row: pd.Series) -> tuple[str | None, str]:
    """Legacy fallback — used only when url_cols not passed."""
    return _build_image_url_smart(row, list(KNOWN_COLUMN_PREFIXES.keys()))


def _empty_result(comment: str, status_code: int = 404) -> dict:
    return {
        "status_code":       status_code,
        "validation_status": "no_human",
        "is_valid":          0,
        "comment":           comment,
    }


def _run_validation_sync(pil_image: Image.Image) -> dict:
    import cv2
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return run_full_validation(pil_image, cv2_image)   # from your main.py


async def _immediate_result(res: dict) -> dict:
    return res


async def _download_and_validate(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    url:       str,
) -> dict:
    """Download one image with retry, then run face validation."""
    async with semaphore:
        raw        = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):   # attempts: 1, 2, 3
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                ) as resp:
                    if resp.status == 503:
                        last_error = "HTTP 503 Service Unavailable"
                        if attempt <= MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * attempt)
                            continue
                        return _empty_result(last_error, status_code=503)
                    if resp.status != 200:
                        return _empty_result(
                            f"HTTP {resp.status} when downloading image",
                            status_code=resp.status,
                        )
                    raw = await resp.read()
                    break   # success

            except asyncio.TimeoutError:
                last_error = f"Image request timed out (attempt {attempt})"
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                    continue
                return _empty_result(last_error, status_code=408)
            except aiohttp.ClientError as exc:
                return _empty_result(f"Network error: {exc}", status_code=503)
            except Exception as exc:
                return _empty_result(f"Download error: {exc}", status_code=500)

    if raw is None:
        return _empty_result(last_error or "Unknown download failure", status_code=500)

    # Decode image
    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_image = _exif_upright(img).convert("RGB")
    except Exception as exc:
        return _empty_result(f"Cannot decode image: {exc}", status_code=422)

    # Run CPU-bound validation in thread pool
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _run_validation_sync, pil_image)
    except Exception as exc:
        return _empty_result(f"Validation error: {exc}", status_code=500)

    category = result.get("category", "no_human")
    categories = result.get("categories", [category])
    quality_issue = any(label in ("blurred", "blur_uncertain", "low_resolution")
                        for label in categories)
    quality_label = next((label for label in ("blurred", "low_resolution", "blur_uncertain")
                          if label in categories), None)
    comments = [CATEGORY_COMMENT.get(label, label) for label in categories]
    # Build audit_category (same logic as _clean_response)
    _faces = result.get("faces") or []
    _app_label = (_faces[0].get("appearance") or {}).get("appearance_label") if _faces else None
    if category == "sunglasses":
        _audit = "goggles"
    elif category == "multiple_person":
        _audit = "multiple_person"
    elif category in ("side", "mixed"):
        _audit = "side_face"
    elif category in ("blurred", "blur_uncertain", "blur_sunglasses_uncertain"):
        _audit = "blur"
    elif category == "low_resolution":
        _audit = "low_resolution"
    elif _app_label:
        _audit = _app_label
    else:
        _audit = "front"
    return {
        "status_code":          CATEGORY_STATUS_MAP.get(quality_label, CATEGORY_STATUS_MAP.get(category, 400)) if quality_issue else CATEGORY_STATUS_MAP.get(category, 400),
        "validation_status":    category,  # backward-compatible single primary label
        "validation_categories": categories,
        "audit_category":       _audit,
        "appearance_label":     _app_label,
        "hair_color":           (_faces[0].get("hair") or {}).get("hair_color") if _faces else None,
        "specs_detected":       (_faces[0].get("specs") or {}).get("detected") if _faces else None,
        "headwear_kind":        (_faces[0].get("headwear") or {}).get("kind") if _faces else None,
        "is_valid":             1 if category == "front" and not quality_issue else 0,
        "comment":              "; ".join(dict.fromkeys(comments)),
    }


async def _process_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    batch_rows: list[pd.Series],
    url_cols: list[str] | None = None,
) -> list[dict]:
    """Run a batch of rows concurrently."""
    _cols = url_cols or list(KNOWN_COLUMN_PREFIXES.keys())
    tasks = []
    for row in batch_rows:
        url, _ = _build_image_url_smart(row, _cols)
        tasks.append(
            _immediate_result(_empty_result("No image address provided in this row"))
            if url is None
            else _download_and_validate(session, semaphore, executor, url)
        )
    return await asyncio.gather(*tasks, return_exceptions=False)


# ═══════════════════════════════════════════════════════════════
# Background worker
# ═══════════════════════════════════════════════════════════════

async def _run_validation_job(job_id: str, df: pd.DataFrame) -> None:
    """
    Runs in background after POST returns.
    - Processes rows in BATCH_SIZE chunks (memory safe).
    - Appends each result to job["results"] immediately (live polling).
    - Saves final CSV to temp dir for download.
    """
    job   = _JOBS[job_id]
    t0    = time.time()
    total = len(df)
    rows     = [row for _, row in df.iterrows()]
    url_cols = job.get("url_cols") or list(KNOWN_COLUMN_PREFIXES.keys())

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    executor  = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)

    job["status"]    = "processing"
    job["processed"] = 0
    job["progress"]  = 0.0
    job["results"]   = []   # live per-row results, readable via /job-status at any time

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            for batch_start in range(0, total, BATCH_SIZE):
                chunk         = rows[batch_start : batch_start + BATCH_SIZE]
                batch_results = await _process_batch(session, semaphore, executor, chunk, url_cols)

                # Append each row result immediately so /job-status can serve live data
                for i, result in enumerate(batch_results):
                    row        = chunk[i]
                    row_number = batch_start + i + 1
                    # Include all detected image column values in live status
                    img_vals = {col: str(row.get(col, "") or "").strip() for col in url_cols}
                    job["results"].append({
                        "row":               row_number,
                        **img_vals,
                        "status_code":       result["status_code"],
                        "validation_status": result["validation_status"],
                        "audit_category":    result.get("audit_category"),
                        "appearance_label":  result.get("appearance_label"),
                        "hair_color":        result.get("hair_color"),
                        "specs_detected":    result.get("specs_detected"),
                        "headwear_kind":     result.get("headwear_kind"),
                        "is_valid":          result["is_valid"],
                        "comment":           result["comment"],
                    })

                done             = min(batch_start + BATCH_SIZE, total)
                job["processed"] = done
                job["progress"]  = round(done / total * 100, 1)
                logger.info(f"job={job_id} | {done}/{total} rows done")

        executor.shutdown(wait=True)

        # ── Build summary ─────────────────────────────────────
        total_valid   = sum(r["is_valid"] for r in job["results"])
        total_invalid = total - total_valid
        elapsed_ms    = round((time.time() - t0) * 1000, 2)

        # ── Save result CSV (microsecond timestamp = no collision) ──
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_name  = f"validated_doctors_{timestamp}.csv"
        out_path  = os.path.join(tempfile.gettempdir(), out_name)

        result_df = df.copy()
        result_df["status_code"]       = [r["status_code"]       for r in job["results"]]
        result_df["validation_status"] = [r["validation_status"] for r in job["results"]]
        result_df["audit_category"]    = [r.get("audit_category")    for r in job["results"]]
        result_df["appearance_label"]  = [r.get("appearance_label")  for r in job["results"]]
        result_df["hair_color"]        = [r.get("hair_color")        for r in job["results"]]
        result_df["specs_detected"]    = [r.get("specs_detected")    for r in job["results"]]
        result_df["headwear_kind"]     = [r.get("headwear_kind")     for r in job["results"]]
        result_df["is_valid"]          = [r["is_valid"]           for r in job["results"]]
        result_df["comment"]           = [r["comment"]            for r in job["results"]]
        result_df.to_csv(out_path, index=False)

        job.update({
            "status":        "done",
            "total_rows":    total,
            "total_valid":   total_valid,
            "total_invalid": total_invalid,
            "elapsed_ms":    elapsed_ms,
            "out_path":      out_path,
            "out_name":      out_name,
        })
        logger.info(
            f"job={job_id} complete | rows={total} | "
            f"valid={total_valid} | invalid={total_invalid} | {elapsed_ms}ms"
        )

    except Exception as exc:
        executor.shutdown(wait=False)
        job["status"] = "error"
        job["error"]  = str(exc)
        logger.error(f"job={job_id} failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# Endpoint 1 — Upload CSV → get job_id immediately
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/validate-csv",
    summary="Bulk CSV face validation — start job",
    description=(
        "Upload a CSV file containing `photograph` and `revice_photograph` columns. "
        "Returns a `job_id` immediately — no waiting. "
        "Poll `/job-status/{job_id}` for live progress and per-row results. "
        "Download the final CSV from `/job-download/{job_id}` when done."
    ),
    tags=["Batch"],
)
async def validate_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Upload your CSV file here"),
):
    # ── 1. Validate file type ─────────────────────────────────
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a .csv file, got: '{file.filename}'",
        )

    # ── 2. Read and parse ─────────────────────────────────────
    raw_bytes = await file.read()
    await file.close()

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}")

    # ── 3. Auto-detect image URL columns ─────────────────────
    url_cols = _detect_image_columns(df)
    if not url_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                "No image URL or filename column detected. "
                "Make sure at least one column contains image URLs (http://...) "
                "or image filenames (.jpg / .png / .webp etc.). "
                f"Columns found: {list(df.columns)}"
            ),
        )

    # ── 4. Register job and kick off background task ──────────
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "status":    "queued",
        "filename":  file.filename,
        "total":     len(df),
        "processed": 0,
        "progress":  0.0,
        "results":   [],
        "url_cols":  url_cols,
    }

    background_tasks.add_task(_run_validation_job, job_id, df)
    logger.info(f"validate-csv job queued | job_id={job_id} | rows={len(df)}")

    return JSONResponse(
        status_code=202,
        content={
            "job_id":       job_id,
            "total_rows":   len(df),
            "status":       "queued",
            "status_url":   f"/job-status/{job_id}",
            "download_url": f"/job-download/{job_id}",
            "message":      "Job started. Poll status_url for live progress.",
        },
    )


# ═══════════════════════════════════════════════════════════════
# Endpoint 2 — Live status + per-row results (JSON)
# ═══════════════════════════════════════════════════════════════

@app.get(
    "/job-status/{job_id}",
    summary="Poll CSV validation job — live progress + results",
    description=(
        "Returns live processing status and all results collected so far. "
        "Keep polling until `status` is `done` or `error`."
    ),
    tags=["Batch"],
)
async def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    response: dict = {
        "job_id":          job_id,
        "status":          job["status"],          # queued | processing | done | error
        "filename":        job["filename"],
        "total_rows":      job["total"],
        "processed":       job["processed"],
        "progress":        f"{job['progress']}%",
        "results_so_far":  job.get("results", []),
    }

    if job["status"] == "done":
        response.update({
            "total_valid":   job["total_valid"],
            "total_invalid": job["total_invalid"],
            "elapsed_ms":    job["elapsed_ms"],
            "download_url":  f"/job-download/{job_id}",
        })

    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")

    return JSONResponse(content=response)


# ═══════════════════════════════════════════════════════════════
# Endpoint 3 — Download final result CSV
# ═══════════════════════════════════════════════════════════════

@app.get(
    "/job-download/{job_id}",
    summary="Download validated CSV result",
    description="Download the final CSV with 4 appended result columns. Only available when job status is `done`.",
    tags=["Batch"],
)
async def job_download(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job["status"] in ("queued", "processing"):
        raise HTTPException(
            status_code=202,
            detail=f"Job still running — {job['progress']}% done. Try again shortly.",
        )
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.get('error')}")

    return FileResponse(
        path=job["out_path"],
        media_type="text/csv",
        filename=job["out_name"],
        headers={
            "X-Total-Rows":    str(job["total_rows"]),
            "X-Total-Valid":   str(job["total_valid"]),
            "X-Total-Invalid": str(job["total_invalid"]),
            "X-Processing-Ms": str(job["elapsed_ms"]),
        },
    )


import asyncio
import csv
import io
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import aiohttp
import pandas as pd
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse


# ---------------------------------------------------------------
# In-memory stream registry  {stream_id: stream_dict}
# ---------------------------------------------------------------
_STREAMS: dict[str, dict] = {}

# CSV result columns written per row
RESULT_COLUMNS = [
    "row",
    "photograph",
    "revice_photograph",
    "source_used",
    "status_code",
    "validation_status",
    "is_valid",
    "comment",
    "row_ms",
]


# ===============================================================
# Endpoint 1 — Stream row by row + write partial CSV as we go
# ===============================================================

@app.post(
    "/validate-csv-stream",
    summary="CSV face validation — live stream + partial CSV saved on disk",
    description=(
        "Upload a CSV file. Results stream back ONE ROW AT A TIME as NDJSON. "
        "Every row is also written to a CSV file on disk immediately. "
        "If you stop/disconnect at any point, all rows processed so far are saved. "
        "The first line of the stream contains `stream_id`. "
        "Use GET /stream-download/{stream_id} to download the partial or full CSV anytime."
    ),
    tags=["Batch"],
    response_class=StreamingResponse,
)
async def validate_csv_stream(
    file: UploadFile = File(..., description="Upload your CSV file here"),
):
    # 1. Validate file type
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a .csv file, got: '{file.filename}'",
        )

    # 2. Read and parse CSV
    raw_bytes = await file.read()
    await file.close()

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}")

    # 3. Validate required columns
    missing = {"photograph", "revice_photograph"} - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required column(s): {', '.join(sorted(missing))}. "
                f"CSV must contain: photograph, revice_photograph"
            ),
        )

    total_rows = len(df)
    rows       = [row for _, row in df.iterrows()]

    # 4. Create stream_id + output CSV file path
    stream_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_name  = f"stream_{timestamp}.csv"
    out_path  = os.path.join(tempfile.gettempdir(), out_name)

    # Register stream in memory
    _STREAMS[stream_id] = {
        "status":        "streaming",
        "filename":      file.filename,
        "out_path":      out_path,
        "out_name":      out_name,
        "total_rows":    total_rows,
        "processed":     0,
        "total_valid":   0,
        "total_invalid": 0,
    }

    # Write CSV header immediately so file is valid even after 0 rows
    original_columns = list(df.columns)
    all_columns      = original_columns + [
        "status_code", "validation_status", "is_valid", "comment"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()

    logger.info(
        f"validate-csv-stream started | "
        f"stream_id={stream_id} | file={file.filename} | rows={total_rows}"
    )

    # 5. Async generator — one row at a time
    async def row_generator():
        t0            = time.time()
        valid_count   = 0
        invalid_count = 0

        # Limit to one concurrent download for gentle streaming load
        executor  = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
        semaphore = asyncio.Semaphore(1)
        connector = aiohttp.TCPConnector(limit=1)

        # First line — send stream_id so client can save it for download later
        yield json.dumps({
            "event":       "start",
            "stream_id":   stream_id,
            "total_rows":  total_rows,
            "filename":    file.filename,
            "download_url": f"/stream-download/{stream_id}",
            "message":     "Save stream_id. Use download_url anytime to get partial/full CSV.",
        }) + "\n"

        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                for row_number, row in enumerate(rows, start=1):
                    row_t0 = time.time()

                    # Build URL and validate
                    url, source_col = _build_image_url(row)

                    if url is None:
                        result = _empty_result("No image address provided in this row")
                    else:
                        result = await _download_and_validate(
                            session, semaphore, executor, url
                        )

                    # Count
                    if result["is_valid"] == 1:
                        valid_count += 1
                    else:
                        invalid_count += 1

                    row_ms = round((time.time() - row_t0) * 1000, 2)

                    # Write this row to CSV immediately (partial save)
                    row_data = dict(row)   # original CSV columns
                    row_data.update({
                        "status_code":       result["status_code"],
                        "validation_status": result["validation_status"],
                        "is_valid":          result["is_valid"],
                        "comment":           result["comment"],
                    })
                    with open(out_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=all_columns, extrasaction="ignore"
                        )
                        writer.writerow(row_data)

                    # Update stream registry
                    _STREAMS[stream_id]["processed"]     = row_number
                    _STREAMS[stream_id]["total_valid"]   = valid_count
                    _STREAMS[stream_id]["total_invalid"] = invalid_count

                    # Yield row result to client immediately
                    yield json.dumps({
                        "event":             "row",
                        "stream_id":         stream_id,
                        "row":               row_number,
                        "total_rows":        total_rows,
                        "progress":          f"{round(row_number / total_rows * 100, 1)}%",
                        "photograph":        str(row.get("photograph",        "") or "").strip(),
                        "revice_photograph": str(row.get("revice_photograph", "") or "").strip(),
                        "source_used":       source_col,
                        "status_code":       result["status_code"],
                        "validation_status": result["validation_status"],
                        "is_valid":          result["is_valid"],
                        "comment":           result["comment"],
                        "row_ms":            row_ms,
                        "download_url":      f"/stream-download/{stream_id}",
                    }) + "\n"

                    logger.info(
                        f"stream={stream_id} | row {row_number}/{total_rows} | "
                        f"{result['validation_status']} | {row_ms}ms"
                    )

        except asyncio.CancelledError:
            # Client disconnected — this is normal, partial CSV is already saved
            logger.warning(
                f"stream={stream_id} | client disconnected at row "
                f"{_STREAMS[stream_id]['processed']}/{total_rows} | "
                f"partial CSV saved at {out_path}"
            )
            _STREAMS[stream_id]["status"] = "partial"
            return   # stop generator cleanly

        finally:
            executor.shutdown(wait=False)

        # All rows done
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        _STREAMS[stream_id]["status"]     = "done"
        _STREAMS[stream_id]["elapsed_ms"] = elapsed_ms

        # Final summary line
        yield json.dumps({
            "event":         "done",
            "stream_id":     stream_id,
            "total_rows":    total_rows,
            "total_valid":   valid_count,
            "total_invalid": invalid_count,
            "elapsed_ms":    elapsed_ms,
            "download_url":  f"/stream-download/{stream_id}",
            "message":       "All rows done. Download your full CSV at download_url.",
        }) + "\n"

        logger.info(
            f"stream={stream_id} complete | rows={total_rows} | "
            f"valid={valid_count} | invalid={invalid_count} | {elapsed_ms}ms"
        )

    # 6. Return streaming response
    return StreamingResponse(
        row_generator(),
        media_type="application/x-ndjson",
        headers={
            "X-Stream-Id":       stream_id,
            "X-Total-Rows":      str(total_rows),
            "X-Download-Url":    f"/stream-download/{stream_id}",
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disables NGINX buffering on Render
        },
    )


# ===============================================================
# Endpoint 2 — Download partial or full CSV anytime
# ===============================================================

@app.get(
    "/stream-download/{stream_id}",
    summary="Download partial or full stream result CSV",
    description=(
        "Download the CSV file for a stream — works anytime, even mid-stream. "
        "If the stream is still running or was stopped early, "
        "you get all rows processed so far. "
        "If the stream is done, you get the complete validated CSV."
    ),
    tags=["Batch"],
)
async def stream_download(stream_id: str):
    stream = _STREAMS.get(stream_id)

    if not stream:
        raise HTTPException(
            status_code=404,
            detail=f"Stream not found: {stream_id}. It may have expired or never started.",
        )

    out_path = stream["out_path"]

    if not os.path.exists(out_path):
        raise HTTPException(
            status_code=404,
            detail="CSV file not found on disk. It may have been cleaned up.",
        )

    processed    = stream["processed"]
    total_rows   = stream["total_rows"]
    status       = stream["status"]   # streaming | partial | done

    # Label the filename clearly so user knows if it is partial or full
    if status == "done":
        download_name = stream["out_name"].replace("stream_", "validated_full_")
    else:
        download_name = stream["out_name"].replace(
            "stream_", f"validated_partial_{processed}of{total_rows}_"
        )

    logger.info(
        f"stream-download | stream_id={stream_id} | "
        f"status={status} | rows={processed}/{total_rows}"
    )

    return FileResponse(
        path=out_path,
        media_type="text/csv",
        filename=download_name,
        headers={
            "X-Stream-Status":   status,
            "X-Rows-Processed":  str(processed),
            "X-Total-Rows":      str(total_rows),
            "X-Total-Valid":     str(stream.get("total_valid",   0)),
            "X-Total-Invalid":   str(stream.get("total_invalid", 0)),
        },
    )


# ===============================================================
# Endpoint 3 — Check stream status (without downloading)
# ===============================================================

@app.get(
    "/stream-status/{stream_id}",
    summary="Check stream progress and download availability",
    tags=["Batch"],
)
async def stream_status(stream_id: str):
    stream = _STREAMS.get(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail=f"Stream not found: {stream_id}")

    processed  = stream["processed"]
    total_rows = stream["total_rows"]

    return {
        "stream_id":     stream_id,
        "status":        stream["status"],    # streaming | partial | done
        "filename":      stream["filename"],
        "processed":     processed,
        "total_rows":    total_rows,
        "progress":      f"{round(processed / total_rows * 100, 1)}%" if total_rows else "0%",
        "total_valid":   stream.get("total_valid",   0),
        "total_invalid": stream.get("total_invalid", 0),
        "download_url":  f"/stream-download/{stream_id}",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║              FACE VALIDATION ENGINE  —  main.py  —  v2.5.0                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

OVERVIEW
────────
A production-ready FastAPI service that performs multi-layer facial validation on
uploaded images and CSV bulk inputs. Every image passes through a strict priority
chain of checks and is assigned exactly one category, mapped to a meaningful HTTP
status code so callers can act on the result directly.

VALIDATION PRIORITY CHAIN
──────────────────────────
Each image is evaluated in this order — the first failing check wins:

  1. Dimension Guard     → image too small or too large                → irrelevant
  2. No Human           → MTCNN found no face above confidence 0.90   → no_human
  3. Irrelevant         → no face AND skin coverage < 3 %             → irrelevant
  4. Multiple Person    → MediaPipe or FaceMesh detected > 1 face     → multiple_person
  5. No Landmarks       → MTCNN found faces but FaceMesh returned none → no_landmarks
  6. Sunglasses         → dark goggles/sunglasses detected on any face → sunglasses
  7. Eyes Closed        → both eyes fully shut (EAR two-gate check)   → eyes_closed
  8. Pose               → all remaining faces classified as Frontal or Side
                           all frontal          → front   (HTTP 200 ✅)
                           all side             → side    (HTTP 400)
                           mix of both          → mixed   (HTTP 400)

OUTPUT CATEGORIES & HTTP STATUS CODES
──────────────────────────────────────
  front           200   Valid frontal face — the only passing result
  side            400   Face is turned to the side
  mixed           400   Some faces frontal, some side
  eyes_closed     422   Both eyes fully shut
  sunglasses      423   Dark goggles or sunglasses detected
  multiple_person 409   More than one person in the image
  no_human        404   No human face detected
  irrelevant      415   Image contains no human content at all
  no_landmarks    422   MTCNN detected a face but MediaPipe could not extract landmarks

MODULE BREAKDOWN
────────────────
  MODULE 0 — Image Pre-processing
      validate_image_dimensions()   Rejects images outside 1×1 to 8000×8000 px.
      _resize_for_processing()      Proportionally downscales the longest side to
                                    1600 px for all MediaPipe operations, returning
                                    the frame and a scale factor for coordinate mapping.

  MODULE 1 — No Human / Irrelevant Detection
      _skin_ratio()                 Computes the fraction of pixels that fall within
                                    HSV skin-tone ranges covering light and dark skin.
      detect_no_human()             Returns is_human=False when no MTCNN bbox passes
                                    the confidence + size filter chain.
      detect_irrelevant_image()     Distinguishes no_human (skin present but no face)
                                    from irrelevant (no face AND very low skin coverage).

  MODULE 2 — Multiple Person Detection
      detect_persons_and_size()     Runs MediaPipe FaceDetection on the downscaled frame.
                                    Returns multiple_person if more than one face is found.

  MODULE 3 — Sunglasses / Dark-Goggle Detection
      _classical_sunglasses_crop_evidence() Detects aligned dark lens regions in an enlarged eye band.
      _extract_eye_roi()            Extracts a tight BGR/HSV crop around each eye polygon.
      detect_sunglasses()           YOLO on face + enlarged eye-band, paired-lens fallback,
                                    then per-eye and upper-face-band corroboration:
                                      S5 — iris guard (hard veto if both eyes show iris)
                                      S1 — mean grey brightness  < SG_BRIGHTNESS_MAX
                                      S2 — grey std-dev contrast < SG_CONTRAST_MAX
                                      S3 — HSV-V channel mean    < SG_HSV_V_MAX
                                      S4 — dark pixel coverage   >= SG_DARK_COVERAGE_MIN
                                    All 4 votes (S1–S4) must pass for detection.
                                    Both eyes must be measurable (SG_REQUIRE_BOTH_EYES).

  MODULE 4 — Closed-Eye Detection (EAR)
      _compute_ear()                Eye Aspect Ratio = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
                                    using 6 Soukupova-Cech landmarks per eye.
      _estimate_open_ear_baseline() Estimates the expected open-eye EAR from horizontal
                                    eye width × EAR_OPEN_BASELINE_FACTOR. Person-neutral,
                                    no prior frame required.
      detect_closed_eyes()          Two-gate system (both must pass):
                                      Gate 1 — absolute: both EARs < 0.15 (deeply shut)
                                      Gate 2 — relative: both EARs <= 35% of their baseline
                                    Skips faces smaller than EAR_MIN_RESOLUTION (80 px) to
                                    avoid unreliable measurements on tiny detections.

  MODULE 5 — Head-Pose Estimation
      _np_angle()                   Angle in degrees at vertex b between rays ba and bc.
      _estimate_focal_length()      Focal length via 70-degree diagonal-FOV heuristic,
                                    rotation-invariant across portrait/landscape images.
      _get_mediapipe_angles()       Full SolvePnP head-pose: maps 6 FaceMesh landmarks
                                    to 3-D reference model, runs SOLVEPNP_SQPNP,
                                    applies 180-degree flip guard, returns yaw/pitch/roll.
      _get_nose_visibility_ratio()  Nose-wing width / cheek-to-cheek width ratio.
                                    ~0.25–0.30 for frontal, < 0.08 for full side profile.
      classify_pose_strong()        Weighted 4-signal pose classifier:
                                      S1 (×1.0) — MTCNN eye-to-nose angle range
                                      S2 (×1.5) — left/right angle asymmetry
                                      S3 (×2.5) — MediaPipe yaw (abstains if unavailable)
                                      S4 (×1.5) — nose visibility ratio
                                    total_score >= 0 → Frontal, else → Side.
                                    Confidence = |score| / dynamic_max × 100.

  MODULE 6 — MTCNN ↔ MediaPipe Face Matching
      _match_landmarks_to_faces()   For each MTCNN bbox, finds the nearest FaceMesh face
                                    by Euclidean distance from bbox centre to nose-tip
                                    landmark, with coordinate scaling to proc-frame space.

  CORE PIPELINE
      run_full_validation()         Orchestrates all modules in priority order.
                                    Accepts a PIL image + OpenCV BGR frame.
                                    Returns a structured dict with category, face list,
                                    per-face pose/eye/sunglasses detail, and timing.

  ANNOTATION
      draw_annotations()            Draws bounding boxes, landmark dots, triangle lines,
                                    pose score, EAR value, yaw/pitch/roll, and a summary
                                    banner onto the frame using colour-coded overlays.

API ENDPOINTS
─────────────
  Single-Image
    POST /validate-face             Validate one uploaded image (JPEG/PNG).
                                    Optional ?save=true saves annotated output to disk.
                                    Returns JSON with category, faces, and processing_ms.
    GET  /get-image                 Retrieve a saved output image by filename + category.
    GET  /get-latest                Return the most recently saved image for a category.
    GET  /list-outputs              List all saved images grouped by category with counts.
    GET  /health                    Service health check — version, device, GPU status.

  Folder Batch
    POST /validate-folder           Validate every JPEG/PNG in a local folder path.
                                    Skips files > 25 MB. Optional ?save=true.
                                    Returns a JSON summary with per-image results.

  CSV Batch — Background Job (no timeout risk)
    POST /validate-csv              Upload a CSV (must have `photograph` + `revice_photograph`
                                    columns). Returns a job_id immediately; validation
                                    runs in the background.
    GET  /job-status/{job_id}       Poll for live progress and per-row results so far.
                                    Status values: queued → processing → done | error.
    GET  /job-download/{job_id}     Download the final validated CSV (only when done).

  CSV Batch — Live Stream (row-by-row NDJSON)
    POST /validate-csv-stream       Upload a CSV. Streams one NDJSON line per row as
                                    each result is produced. Also writes every row to a
                                    partial CSV on disk immediately — safe to disconnect.
                                    First line contains stream_id for download tracking.
    GET  /stream-download/{id}      Download the partial or complete result CSV at any
                                    time, even while the stream is still running.
    GET  /stream-status/{id}        Check stream progress and download availability.

IMAGE URL RESOLUTION (CSV endpoints)
──────────────────────────────────────
  Each CSV row is resolved in this priority:
    1. revice_photograph  → PREFIX_REVISE   + filename
    2. photograph         → PREFIX_ORIGINAL + filename
  If both columns are empty/null the row is marked no_human with a comment.

CONCURRENCY & PERFORMANCE
──────────────────────────
  - MTCNN runs on GPU if CUDA is available, otherwise CPU.
  - MediaPipe instances are thread-local (one per worker thread) to avoid
    contention across concurrent requests.
  - CSV batch processing uses asyncio semaphores + ThreadPoolExecutor to
    download and validate up to MAX_CONCURRENT_DOWNLOADS images in parallel.
  - Streaming endpoint uses a single semaphore (concurrency=1) for low load.
  - All MediaPipe operations run on images downscaled to 1600 px max dimension;
    MTCNN runs on the original full-resolution image for maximum accuracy.

DEPENDENCIES
────────────
  fastapi, uvicorn, opencv-python (cv2), mediapipe, facenet-pytorch (MTCNN),
  torch, Pillow, numpy, aiohttp, pandas

REQUIRED CSV COLUMNS
─────────────────────
  photograph          filename of the original doctor photo
  revice_photograph   filename of the revised/updated photo (takes priority)
"""

import os
import io
import base64
import cv2
import math
import time
import logging
import threading
import numpy as np
import torch
import uvicorn
import mediapipe as mp

from contextlib import asynccontextmanager
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional

import asyncio, io, os, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import aiohttp, pandas as pd

# ─────────────────────────────────────────────────────────────
# Optional ML packages — graceful fallback when not installed
# ─────────────────────────────────────────────────────────────
try:
    from skimage.measure import blur_effect as _skimage_blur_effect
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False

try:
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import YOLO as _UltralyticsYOLO
    _ULTRALYTICS_OK = True
except ImportError:
    _ULTRALYTICS_OK = False

# YOLOv8n Open-Images-V7 eyewear class IDs
_OIV7_SUNGLASSES_CLS = 505   # dark/tinted sunglasses
_OIV7_GOGGLES_CLS    = 220   # opaque goggles (ski, swim, safety)
_OIV7_GLASSES_CLS    = 217   # clear prescription glasses — NOT flagged

_OIV7_HAT_CLS        = 243   # Hat (baseball cap, sports cap, bucket hat …)
_OIV7_HELMET_CLS     = 248   # Helmet (safety/hard hat, military helmet …)
_OIV7_BIKE_HELMET    = 43    # Bicycle helmet
_OIV7_FOOTBALL_HELM  = 202   # Football helmet
_OIV7_SWIM_CAP       = 509   # Swim cap

_yolo_oiv7_model  = None   # lazy-loaded on first request
_yolo_load_failed = False
_yolo_warned: set = set()

_sg_clf        = None   # lazy-loaded SunglassesClassifier (glasses-detector library)
_sg_clf_failed = False

_gender_model        = None   # lazy-loaded ConvNeXt-Large (gender + cap/hijab/turban)
_gender_model_failed = False
_gender_model_error  = None


def _yolo_warn_once(key: str, msg: str):
    if key not in _yolo_warned:
        _yolo_warned.add(key)
        logger.warning(f"[sunglasses/YOLO] {msg}")


def _eyewear_kind(model, class_id: int) -> str:
    """Resolve eyewear class by the model's actual label names, not guessed IDs."""
    try:
        names = model.names
        label = names.get(class_id, "") if isinstance(names, dict) else names[class_id]
        label = str(label).strip().lower().replace("_", " ").replace("-", " ")
        if "sunglass" in label or "sun glasses" in label:
            return "sunglasses"
        if "goggle" in label:
            return "goggles"
        if "glass" in label or "eyeglass" in label or "spectacle" in label:
            return "clear_glasses"
    except Exception:
        pass
    # Fallback for known exported Open Images label maps only.
    if class_id == _OIV7_SUNGLASSES_CLS:
        return "sunglasses"
    if class_id == _OIV7_GOGGLES_CLS:
        return "goggles"
    if class_id == _OIV7_GLASSES_CLS:
        return "clear_glasses"
    return "other"


def _headwear_kind(model, class_id: int) -> str:
    try:
        names = model.names
        label = names.get(class_id, "") if isinstance(names, dict) else names[class_id]
        label = str(label).strip().lower().replace("_", " ").replace("-", " ")
        if "helmet" in label:
            return "helmet"
        if any(k in label for k in ("hat", " cap", "beanie", "fedora", "sombrero", "beret", "turban", "headband")):
            return "cap"
    except Exception:
        pass
    if class_id in (_OIV7_HELMET_CLS, _OIV7_BIKE_HELMET, _OIV7_FOOTBALL_HELM):
        return "helmet"
    if class_id in (_OIV7_HAT_CLS, _OIV7_SWIM_CAP):
        return "cap"
    return "other"


def _load_yolo_oiv7():
    global _yolo_oiv7_model, _yolo_load_failed
    if _yolo_oiv7_model is None and _ULTRALYTICS_OK and not _yolo_load_failed:
        try:
            _yolo_oiv7_model = _UltralyticsYOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolov8m-oiv7.pt"))
        except Exception as exc:
            _yolo_load_failed = True
            _yolo_warn_once("load", f"model load failed, classical fallback only: {exc!r}")
    return _yolo_oiv7_model


def _load_sg_clf():
    global _sg_clf, _sg_clf_failed
    if _sg_clf is None and not _sg_clf_failed:
        try:
            import sys as _sys
            if "albumentations" not in _sys.modules:
                import os as _os
                _audit_dir = _os.path.dirname(_os.path.abspath(__file__))
                if _audit_dir not in _sys.path:
                    _sys.path.insert(0, _audit_dir)
                import fake_albumentations as _fa
                _sys.modules["albumentations"] = _fa
                _sys.modules["albumentations.pytorch"] = _fa.pytorch
            from glasses_detector import SunglassesClassifier as _SGC
            _sg_clf = _SGC()
            logger.info("[sunglasses/CLF] SunglassesClassifier loaded")
        except Exception as _exc:
            _sg_clf_failed = True
            logger.warning(f"[sunglasses/CLF] load failed, will skip: {_exc!r}")
    return _sg_clf


def _detect_sg_classifier(frame_cv2_orig, orig_bbox) -> Optional[float]:
    """Face-crop probability from SunglassesClassifier; returns None on any failure."""
    try:
        clf = _load_sg_clf()
        if clf is None or frame_cv2_orig is None or orig_bbox is None:
            return None
        img_h, img_w = frame_cv2_orig.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        pad = int(round(max(fx2 - fx1, fy2 - fy1, 1.0) * 0.15))
        x1, y1 = max(0, int(fx1) - pad), max(0, int(fy1) - pad)
        x2, y2 = min(img_w, int(fx2) + pad), min(img_h, int(fy2) + pad)
        face_crop = frame_cv2_orig[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        # Upscale tiny crops with Lanczos before handing to PIL — the classifier
        # input is 224×224 anyway; explicit high-quality upscale beats the model's
        # own bilinear resize from a 20px source.
        ch, cw = face_crop.shape[:2]
        if min(ch, cw) < 50:
            scale = max(224 / max(ch, cw), 2.0)
            new_w, new_h = int(cw * scale), int(ch * scale)
            face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        return float(clf.predict(pil_img, label_type="proba"))
    except Exception as _exc:
        logger.warning(f"[sunglasses/CLF] inference error: {_exc!r}")
        return None


# ─────────────────────────────────────────────────────────────
# Gender + headwear detector: ConvNeXt-Large (timm), 5 classes:
#   cap, female, hijab, male, turban  (sigmoid outputs, one per class)
# Local weights: models/best_convnext_large.pth
# The cap/hijab/turban probabilities are also used by the appearance classifier.
# ─────────────────────────────────────────────────────────────

_gender_model        = None
_gender_model_failed = False
_gender_model_error  = None
_gender_model_arch   = "ConvNeXt-Large"
_GENDER_MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_convnext_large.pth")
_CONVNEXT_CLASSES    = ["cap", "female", "hijab", "male", "turban"]
_CONVNEXT_DEVICE     = None  # set lazily on first load


def _load_gender_model():
    global _gender_model, _gender_model_failed, _gender_model_error, _CONVNEXT_DEVICE
    if _gender_model is not None:
        return _gender_model
    if _gender_model_failed:
        return None
    if not os.path.isfile(_GENDER_MODEL_PATH):
        _gender_model_failed = True
        _gender_model_error = f"model_file_not_found: {_GENDER_MODEL_PATH}"
        logger.error("[gender] %s", _gender_model_error)
        return None
    try:
        import torch, timm
        from torchvision import transforms as _tv_transforms
        _CONVNEXT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model("convnext_large", pretrained=False, num_classes=len(_CONVNEXT_CLASSES))
        checkpoint = torch.load(_GENDER_MODEL_PATH, map_location=_CONVNEXT_DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(_CONVNEXT_DEVICE)
        model.eval()
        _transform = _tv_transforms.Compose([
            _tv_transforms.Resize((224, 224)),
            _tv_transforms.ToTensor(),
            _tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _gender_model = (model, _transform)
        _gender_model_error = None
        logger.info("[gender] loaded ConvNeXt-Large from %s on %s", _GENDER_MODEL_PATH, _CONVNEXT_DEVICE)
        return _gender_model
    except Exception as exc:
        _gender_model_failed = True
        _gender_model_error = f"{type(exc).__name__}: {exc}"
        logger.exception("[gender] model load failed: %s", _gender_model_error)
        return None


def _gender_unavailable(status, reason, **extra):
    return {"gender": None, "status": status, "reason": str(reason),
            "confidence": None, "probs": None, **extra}


def detect_gender(frame_cv2_orig: np.ndarray, orig_bbox) -> dict:
    loaded = _load_gender_model()
    if loaded is None:
        return _gender_unavailable("model_unavailable", _gender_model_error or "Gender model unavailable")
    model, transform = loaded
    try:
        import torch
        if not isinstance(frame_cv2_orig, np.ndarray) or frame_cv2_orig.ndim != 3:
            raise ValueError("Expected HxWx3 BGR image")
        if orig_bbox is None or len(orig_bbox) < 4:
            raise ValueError("Bounding box must contain x1,y1,x2,y2")
        ih, iw = frame_cv2_orig.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in orig_bbox[:4]]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid face bounding box")
        pad = max(x2 - x1, y2 - y1, 1.0) * 0.10
        xa = max(0, min(iw, int(np.floor(x1 - pad))))
        ya = max(0, min(ih, int(np.floor(y1 - pad))))
        xb = max(0, min(iw, int(np.ceil(x2 + pad))))
        yb = max(0, min(ih, int(np.ceil(y2 + pad))))
        if xb <= xa or yb <= ya:
            raise ValueError("Face crop is empty")
        crop = frame_cv2_orig[ya:yb, xa:xb]
        if crop.size == 0:
            raise ValueError("Face crop is empty")
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_crop).unsqueeze(0).to(_CONVNEXT_DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output[0]).tolist()
        prob_dict = {cls: round(probs[i], 4) for i, cls in enumerate(_CONVNEXT_CLASSES)}
        male_p   = prob_dict["male"]
        female_p = prob_dict["female"]
        # gender: use whichever is higher; flag manual_review if neither is > 0.9
        if male_p > female_p:
            gender, confidence = "male", male_p
        else:
            gender, confidence = "female", female_p
        manual_review = 0 if confidence > 0.9 else 1
        # headwear: cap / hijab / turban if max > 0.85
        hw_names  = ["cap", "hijab", "turban"]
        hw_probs  = [prob_dict[h] for h in hw_names]
        max_hw    = max(hw_probs)
        headwear  = hw_names[hw_probs.index(max_hw)] if max_hw > 0.40 else None
        return {
            "gender":        gender,
            "status":        "predicted",
            "reason":        None,
            "confidence":    round(confidence, 4),
            "probs":         {"male": round(male_p, 4), "female": round(female_p, 4)},
            "all_probs":     prob_dict,
            "manual_review": manual_review,
            "headwear":      headwear,
            "model":         _gender_model_arch,
            "crop_bbox":     [xa, ya, xb, yb],
            "crop_size":     [int(crop.shape[1]), int(crop.shape[0])],
        }
    except Exception as exc:
        logger.exception("[gender] inference failed")
        return _gender_unavailable("inference_error", f"{type(exc).__name__}: {exc}")


def _detect_sunglasses_yolo(frame_cv2: np.ndarray, orig_bbox) -> Optional[dict]:
    """Detect sunglasses on both a face crop and an enlarged eye-band crop.

    The second pass is important for distant/small faces: sunglasses may occupy
    only a few pixels in the full face crop. We upscale the upper-face region
    before inference; this improves detector input scale but cannot recreate
    detail absent from the original image.
    """
    try:
        if not _ULTRALYTICS_OK:
            _yolo_warn_once("missing", "ultralytics not importable — classical fallback only")
            return None
        if orig_bbox is None or frame_cv2 is None:
            return None
        model = _load_yolo_oiv7()
        if model is None:
            return None

        img_h, img_w = frame_cv2.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
        fx2, fy2 = min(float(img_w), fx2), min(float(img_h), fy2)
        fw, fh = max(fx2 - fx1, 1.0), max(fy2 - fy1, 1.0)
        pad = int(round(max(fw, fh) * YOLO_CROP_PAD_FRAC))
        x1, y1 = max(0, int(fx1) - pad), max(0, int(fy1) - pad)
        x2, y2 = min(img_w, int(fx2) + pad), min(img_h, int(fy2) + pad)
        face_crop = frame_cv2[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        # Build multiple upper-face crops. Small sunglasses can be only 10-20 px
        # wide in the source, so a single eye-band crop is not reliable enough.
        rel_fx1 = max(0, int(round(fx1 - x1)))
        rel_fx2 = min(face_crop.shape[1], int(round(fx2 - x1)))
        rel_fy1 = max(0, int(round(fy1 - y1)))
        rel_fy2 = min(face_crop.shape[0], int(round(fy2 - y1)))
        eye_x_pad = int(round(0.22 * fw))
        eye_x1 = max(0, rel_fx1 - eye_x_pad)
        eye_x2 = min(face_crop.shape[1], rel_fx2 + eye_x_pad)
        # Two slightly different vertical windows tolerate face-box variation.
        eye_crops = []
        bbox_touches_top = fy1 <= max(2.0, 0.015 * img_h)
        eye_windows = (
            ((0.02, 0.36, "zoomed_eye_band_top"),
             (0.06, 0.44, "zoomed_eye_band_top_wide"),
             (0.12, 0.50, "zoomed_eye_band"),
             (0.20, 0.60, "zoomed_eye_band_wide"),
             (0.28, 0.68, "zoomed_eye_band_low"))
            if bbox_touches_top else
            ((0.08, 0.44, "zoomed_eye_band_top"),
             (0.12, 0.50, "zoomed_eye_band"),
             (0.18, 0.58, "zoomed_eye_band_wide"),
             (0.26, 0.66, "zoomed_eye_band_low"))
        )
        for top_frac, bottom_frac, crop_name in eye_windows:
            cy1 = max(0, int(round(rel_fy1 + top_frac * fh)))
            cy2 = min(face_crop.shape[0], int(round(rel_fy1 + bottom_frac * fh)))
            crop = face_crop[cy1:cy2, eye_x1:eye_x2]
            if crop.size:
                eh, ew = crop.shape[:2]
                # Upscale aggressively for small/distant faces; interpolation does
                # not create detail, but lets the detector use more input pixels.
                scale = min(5.0, 1024.0 / max(eh, ew, 1))
                if scale > 1.0:
                    crop = cv2.resize(crop, (max(1, int(ew * scale)), max(1, int(eh * scale))),
                                      interpolation=cv2.INTER_CUBIC)
                eye_crops.append((crop_name, crop, 1024))
        if not eye_crops:
            eye_crops = [("zoomed_eye_band", face_crop, 768)]

        # Run the full face plus multiple zoomed eye-region passes.
        passes = [("face", face_crop, 640)] + eye_crops
        scale = max((max(crop.shape[:2]) / max(1, max(face_crop.shape[:2])) for _, crop, _ in eye_crops), default=1.0)
        sg = gg = cl = 0.0
        best_source = {"sunglasses": None, "goggles": None, "clear_glasses": None}
        for source, crop, imgsz in passes:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            results = model(pil_img, verbose=False, conf=0.05, iou=0.45, imgsz=imgsz)
            crop_h, crop_w = crop.shape[:2]
            for r in results:
                if r.boxes is None:
                    continue
                boxes_xyxy = r.boxes.xyxy.tolist() if r.boxes.xyxy is not None else []
                for idx, (cid_raw, cf_raw) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
                    cid, cf = int(cid_raw), float(cf_raw)
                    kind = _eyewear_kind(model, cid)
                    if kind not in ("sunglasses", "goggles", "clear_glasses"):
                        continue
                    # Box-position validation: know WHERE the box falls (ModaNet lesson —
                    # each class has an expected anatomical region; detections outside it are noise).
                    # High-confidence hits (≥ YOLO_SG_CONF_HARD) bypass this — YOLO is trusted.
                    if idx < len(boxes_xyxy) and cf < YOLO_SG_CONF_HARD:
                        bx1_b, by1_b, bx2_b, by2_b = boxes_xyxy[idx]
                        bw_box = bx2_b - bx1_b
                        bcy = (by1_b + by2_b) / 2.0
                        if source == "face" and kind in ("sunglasses", "goggles"):
                            # Sunglasses box center must be roughly in the eye zone.
                            # Allow -15% above face top (glasses can sit on forehead)
                            # and up to 75% down (wrap-around frames, large lenses).
                            face_h_crop = max(1.0, float(rel_fy2 - rel_fy1))
                            face_w_crop = max(1.0, float(rel_fx2 - rel_fx1))
                            box_face_frac_y = (bcy - rel_fy1) / face_h_crop
                            box_face_frac_w = bw_box / face_w_crop
                            if not (-0.15 <= box_face_frac_y <= 0.75 and box_face_frac_w >= 0.10):
                                continue
                        elif source.startswith("zoomed") and kind in ("sunglasses", "goggles"):
                            # Eye band crops: reject detections narrower than 6% of crop — noise
                            if bw_box < 0.06 * max(1, crop_w):
                                continue
                    if kind == "sunglasses" and cf > sg:
                        sg, best_source["sunglasses"] = cf, source
                    elif kind == "goggles" and cf > gg:
                        gg, best_source["goggles"] = cf, source
                    elif kind == "clear_glasses" and cf > cl:
                        cl, best_source["clear_glasses"] = cf, source

        dark = max(sg, gg)
        # A positive on a tight, enlarged eye crop is more informative than the
        # same low confidence on the whole image. Keep a separate calibrated floor.
        zoom_hit = any(best_source.get(k) and best_source[k].startswith("zoomed_eye_band")
                       for k in ("sunglasses", "goggles"))
        # A zoomed crop is not independent evidence: interpolation can amplify
        # noise. Require the same calibrated confidence floor on every crop.
        zoom_hard_floor = YOLO_SG_CONF_HARD
        hard = dark >= YOLO_SG_CONF_HARD
        soft = (not hard) and dark >= YOLO_SG_CONF_SOFT
        veto = "clear_glasses" if (cl >= YOLO_CLEAR_VETO and not hard and not soft) else None
        return {
            "yolo_detected": bool(hard),
            "yolo_soft": bool(soft),
            "yolo_sunglasses": round(sg, 3),
            "yolo_goggles": round(gg, 3),
            "yolo_clear_glasses": round(cl, 3),
            "yolo_veto": veto,
            "yolo_best_source": best_source,
            "yolo_zoom_hit": bool(zoom_hit),
            "yolo_effective_hard_threshold": YOLO_SG_CONF_HARD,
            "yolo_class_labels": (list(model.names.values()) if isinstance(getattr(model, "names", None), dict)
                                  else list(getattr(model, "names", []) or [])),
            "zoomed_eye_band_scale": round(float(scale), 2),
            "zoomed_eye_band_sizes": [
                [int(crop.shape[1]), int(crop.shape[0])]
                for source, crop, _ in passes if source.startswith("zoomed_eye_band")
            ],
        }
    except Exception as exc:
        _yolo_warn_once("infer", f"inference failed, classical fallback only: {exc!r}")
        return None


def _detect_headwear_yolo(frame_cv2: np.ndarray, orig_bbox) -> Optional[dict]:
    try:
        if not _ULTRALYTICS_OK or orig_bbox is None or frame_cv2 is None:
            return None
        model = _load_yolo_oiv7()
        if model is None:
            return None

        img_h, img_w = frame_cv2.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
        fx2, fy2 = min(float(img_w), fx2), min(float(img_h), fy2)
        fw, fh = max(fx2 - fx1, 1.0), max(fy2 - fy1, 1.0)

        # Hats/helmets sit ABOVE the face bbox — extend crop upward aggressively.
        top_ext  = int(fh * HW_CROP_PAD_TOP_FRAC)
        side_pad = int(fw * HW_CROP_PAD_SIDE)
        x1 = max(0, int(fx1) - side_pad)
        y1 = max(0, int(fy1) - top_ext)
        x2 = min(img_w, int(fx2) + side_pad)
        y2 = min(img_h, int(fy2) + int(fh * 0.15))

        head_crop = frame_cv2[y1:y2, x1:x2]
        if head_crop.size == 0:
            return None

        ch, cw = head_crop.shape[:2]
        scale = min(4.0, 1024.0 / max(ch, cw, 1))
        if scale > 1.1:
            head_crop = cv2.resize(head_crop,
                                   (max(1, int(cw * scale)), max(1, int(ch * scale))),
                                   interpolation=cv2.INTER_CUBIC)

        pil_img = Image.fromarray(cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB))
        results = model(pil_img, verbose=False, conf=0.05, iou=0.45, imgsz=640)

        best_helmet = 0.0
        best_cap    = 0.0
        for r in results:
            if r.boxes is None:
                continue
            for cid_raw, cf_raw in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                cid, cf = int(cid_raw), float(cf_raw)
                kind = _headwear_kind(model, cid)
                if kind == "helmet" and cf > best_helmet:
                    best_helmet = cf
                elif kind == "cap" and cf > best_cap:
                    best_cap = cf

        best = max(best_helmet, best_cap)
        kind_best = "helmet" if best_helmet >= best_cap else "cap"
        hard = best >= HW_YOLO_CONF_HARD
        soft = (not hard) and best >= HW_YOLO_CONF_SOFT
        return {
            "yolo_detected":    bool(hard),
            "yolo_soft":        bool(soft),
            "yolo_helmet_conf": round(best_helmet, 3),
            "yolo_cap_conf":    round(best_cap, 3),
            "yolo_best_kind":   kind_best if (hard or soft) else None,
        }
    except Exception as exc:
        _yolo_warn_once("hw_infer", f"headwear YOLO failed: {exc!r}")
        return None


def _detect_headwear_yolo_fullimage(frame_cv2: np.ndarray) -> Optional[dict]:
    """Run OIV7 YOLO on the full image to find helmets/caps when no face is detected.
    Used for full-face helmets (motorcycle, football) that completely hide the face."""
    try:
        if not _ULTRALYTICS_OK or frame_cv2 is None:
            return None
        model = _load_yolo_oiv7()
        if model is None:
            return None

        img_h, img_w = frame_cv2.shape[:2]
        imgsz = min(1280, max(640, max(img_h, img_w)))
        pil_img = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        results = model(pil_img, verbose=False, conf=0.08, iou=0.45, imgsz=imgsz)

        best_helmet = 0.0
        best_cap    = 0.0
        helmet_box  = None
        cap_box     = None
        for r in results:
            if r.boxes is None:
                continue
            for box, cid_raw, cf_raw in zip(r.boxes.xyxy.tolist(),
                                             r.boxes.cls.tolist(),
                                             r.boxes.conf.tolist()):
                cid, cf = int(cid_raw), float(cf_raw)
                kind = _headwear_kind(model, cid)
                if kind == "helmet" and cf > best_helmet:
                    best_helmet = cf
                    helmet_box  = [round(v, 1) for v in box]
                elif kind == "cap" and cf > best_cap:
                    best_cap = cf
                    cap_box  = [round(v, 1) for v in box]

        best = max(best_helmet, best_cap)
        kind_best = "helmet" if best_helmet >= best_cap else "cap"
        hard = best >= HW_YOLO_CONF_HARD
        soft = (not hard) and best >= HW_YOLO_CONF_SOFT
        return {
            "yolo_detected":    bool(hard),
            "yolo_soft":        bool(soft),
            "yolo_helmet_conf": round(best_helmet, 3),
            "yolo_cap_conf":    round(best_cap, 3),
            "yolo_best_kind":   kind_best if (hard or soft) else None,
            "helmet_box":       helmet_box,
            "cap_box":          cap_box,
        }
    except Exception as exc:
        _yolo_warn_once("hw_full_infer", f"full-image headwear YOLO failed: {exc!r}")
        return None


def _detect_face_mask_classical(frame_cv2: np.ndarray, orig_bbox) -> Optional[dict]:
    """Multi-signal face mask detector.
    Compares nose/cheek region (skin) against mouth/chin region (mask if present).
    Uses saturation, texture, uniformity, and hue — works for any mask color.
    """
    try:
        if frame_cv2 is None or orig_bbox is None:
            return None
        img_h, img_w = frame_cv2.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
        fx2, fy2 = min(float(img_w), fx2), min(float(img_h), fy2)
        fw, fh = max(fx2 - fx1, 1.0), max(fy2 - fy1, 1.0)

        # mid = nose-tip / cheek area (always skin)
        # low = mouth + chin (covered by mask if present)
        mid_y1 = int(fy1 + fh * 0.28)
        mid_y2 = int(fy1 + fh * 0.54)
        low_y1 = int(fy1 + fh * 0.54)
        low_y2 = int(min(float(img_h), fy2))
        x1 = max(0, int(fx1 + fw * 0.08))
        x2 = min(img_w, int(fx2 - fw * 0.08))

        if mid_y2 <= mid_y1 or low_y2 <= low_y1 or x2 <= x1:
            return None

        mid_bgr = frame_cv2[mid_y1:mid_y2, x1:x2]
        low_bgr = frame_cv2[low_y1:low_y2, x1:x2]
        if mid_bgr.size == 0 or low_bgr.size == 0:
            return None

        mid_hsv = cv2.cvtColor(mid_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        low_hsv = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        mid_gray = cv2.cvtColor(mid_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        low_gray = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        mid_sat  = float(np.mean(mid_hsv[:, :, 1]))
        low_sat  = float(np.mean(low_hsv[:, :, 1]))
        mid_val  = float(np.mean(mid_hsv[:, :, 2]))
        low_val  = float(np.mean(low_hsv[:, :, 2]))
        low_std  = float(np.std(low_gray))
        mid_std  = float(np.std(mid_gray))
        brightness_diff = abs(mid_val - low_val)

        # Signal 1: saturation drop — skin is warm/saturated; masks are desaturated
        sat_drop = mid_sat - low_sat          # positive = lower face less saturated
        s1_sat   = sat_drop > 18             # surgical/cloth masks: typically large drop

        # Signal 2: texture contrast — skin has fine texture, mask is smooth
        mid_lap = float(cv2.Laplacian(mid_gray.astype(np.uint8), cv2.CV_64F).var())
        low_lap = float(cv2.Laplacian(low_gray.astype(np.uint8), cv2.CV_64F).var())
        tex_ratio = mid_lap / max(low_lap, 0.5)
        s2_tex   = tex_ratio > 2.0            # mid has 2× more texture than lower

        # Signal 3: lower face is uniformly smooth (low std)
        s3_unif  = low_std < 32 and low_std < mid_std * 0.75

        # Signal 4: brightness jump (any mask that differs from skin tone)
        s4_bright = brightness_diff > 22

        # Signal 5: lower-face hue is outside skin-tone range
        # Skin hue in HSV (OpenCV scale 0-180): ~0-25 for warm tones
        low_hue = float(np.mean(low_hsv[:, :, 0]))
        low_sat_val = low_sat
        skin_hue_ok = (low_hue < 28) and (low_sat_val > 28)  # warm+saturated = skin
        s5_nonskin  = not skin_hue_ok                          # True when lower face ≠ skin

        score = sum([s1_sat, s2_tex, s3_unif, s4_bright, s5_nonskin])
        # 2 signals = soft, 3+ = confident mask
        mask_signal = score >= 2

        return {
            "mask_signal":    bool(mask_signal),
            "mask_score":     int(score),
            "s1_sat_drop":    bool(s1_sat),
            "s2_texture":     bool(s2_tex),
            "s3_uniform":     bool(s3_unif),
            "s4_bright_diff": bool(s4_bright),
            "s5_nonskin":     bool(s5_nonskin),
            "sat_drop":       round(sat_drop, 1),
            "tex_ratio":      round(tex_ratio, 2),
            "lower_face_std": round(low_std, 1),
            "brightness_diff":round(brightness_diff, 1),
            "lower_hue":      round(low_hue, 1),
        }
    except Exception:
        return None


def _detect_headwear_classical(frame_cv2: np.ndarray, orig_bbox) -> Optional[dict]:
    try:
        if frame_cv2 is None or orig_bbox is None:
            return None
        img_h, img_w = frame_cv2.shape[:2]
        fx1, fy1, fx2, fy2 = [float(v) for v in orig_bbox[:4]]
        fx1, fy1 = max(0.0, fx1), max(0.0, fy1)
        fx2, fy2 = min(float(img_w), fx2), min(float(img_h), fy2)
        fw, fh = max(fx2 - fx1, 1.0), max(fy2 - fy1, 1.0)

        # Analyse two regions:
        #  above_region  → where helmet dome / cap body sits (above face top)
        #  forehead_region → top 20% of face bbox (helmet rim sits here)
        above_y1 = max(0, int(fy1 - fh * 0.45))
        above_y2 = max(above_y1 + 1, int(fy1))
        fore_y1  = int(fy1)
        fore_y2  = max(fore_y1 + 1, int(fy1 + fh * 0.22))
        mid_y1   = int(fy1 + fh * 0.30)
        mid_y2   = max(mid_y1 + 1, int(fy1 + fh * 0.60))
        x1 = max(0, int(fx1))
        x2 = min(img_w, int(fx2))

        if x2 <= x1:
            return None

        def _region_stats(y1, y2):
            if y2 <= y1:
                return None
            r = frame_cv2[y1:y2, x1:x2]
            if r.size == 0:
                return None
            g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            return {"mean": float(np.mean(g)), "std": float(np.std(g))}

        above_s = _region_stats(above_y1, above_y2)
        fore_s  = _region_stats(fore_y1,  fore_y2)
        mid_s   = _region_stats(mid_y1,   mid_y2)

        if above_s is None or fore_s is None or mid_s is None:
            return None

        # Signal 1 (dark brim): above/forehead region is dark — classic dark helmet or cap brim
        dark_brim = (fore_s["mean"] < HW_BRIM_DARK_THR and
                     float(np.mean(cv2.cvtColor(
                         frame_cv2[fore_y1:fore_y2, x1:x2], cv2.COLOR_BGR2GRAY
                     ) < HW_BRIM_DARK_THR)) >= HW_BRIM_COVERAGE_MIN)

        # Signal 2 (any-color helmet): above region is notably DIFFERENT from mid-face
        # (skin is consistent; a helmet introduces a new texture/color above the face)
        brightness_diff = abs(above_s["mean"] - mid_s["mean"])
        above_uniform   = above_s["std"] < 35        # helmets are smoother than background
        color_diff_signal = (brightness_diff > 22 and above_uniform)

        # Signal 3: forehead is cut off (helmet rim covers part of forehead — low std)
        fore_cut = fore_s["std"] < 28 and abs(fore_s["mean"] - mid_s["mean"]) > 18

        brim_signal = dark_brim or color_diff_signal or fore_cut

        return {
            "brim_signal":       bool(brim_signal),
            "dark_brim":         bool(dark_brim),
            "color_diff_signal": bool(color_diff_signal),
            "fore_cut":          bool(fore_cut),
            "brim_mean":         round(fore_s["mean"], 1),
            "above_mean":        round(above_s["mean"], 1),
            "mid_mean":          round(mid_s["mean"], 1),
            "brightness_diff":   round(brightness_diff, 1),
            "above_std":         round(above_s["std"], 1),
            "brim_dark_cov":     round(float(np.mean(
                cv2.cvtColor(frame_cv2[fore_y1:fore_y2, x1:x2], cv2.COLOR_BGR2GRAY)
                < HW_BRIM_DARK_THR)) if fore_y2 > fore_y1 else 0.0, 3),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("FaceValidationEngine")


def _exif_upright(img):
    """Apply the EXIF orientation tag so phone photos are not processed sideways
    (a rotated face breaks MTCNN/pose and is wrongly reported as side/no_human)."""
    try:
        from PIL import ImageOps as _ImageOps
        return _ImageOps.exif_transpose(img)
    except Exception:
        return img
# Keep the terminal clean: silence chatty third-party libraries and per-request access lines.
for _noisy in ("ultralytics", "transformers", "PIL", "urllib3", "httpx", "asyncio",
               "matplotlib", "numba", "uvicorn.access", "timm", "filelock", "absl"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {device}")

# ─────────────────────────────────────────────────────────────
# Output folders
# ─────────────────────────────────────────────────────────────
BASE_OUTPUT = "detection_outputs"
CATEGORY_FOLDERS = {
    "front":           os.path.join(BASE_OUTPUT, "front"),
    "side":            os.path.join(BASE_OUTPUT, "side"),
    "mixed":           os.path.join(BASE_OUTPUT, "mixed"),
    "eyes_closed":     os.path.join(BASE_OUTPUT, "eyes_closed"),
    "multiple_person": os.path.join(BASE_OUTPUT, "multiple_person"),
    "sunglasses":      os.path.join(BASE_OUTPUT, "sunglasses"),
    "no_human":        os.path.join(BASE_OUTPUT, "no_human"),
    "irrelevant":      os.path.join(BASE_OUTPUT, "irrelevant"),
    "no_landmarks":    os.path.join(BASE_OUTPUT, "no_landmarks"),
    "blurred":                   os.path.join(BASE_OUTPUT, "blurred"),
    "blur_uncertain":            os.path.join(BASE_OUTPUT, "blur_uncertain"),
    "blur_sunglasses_uncertain": os.path.join(BASE_OUTPUT, "blur_sunglasses_uncertain"),
    "low_resolution":            os.path.join(BASE_OUTPUT, "low_resolution"),
    "focused":                   os.path.join(BASE_OUTPUT, "focused"),
}
for _folder in CATEGORY_FOLDERS.values():
    os.makedirs(_folder, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Constants & thresholds
# ─────────────────────────────────────────────────────────────

# SolvePnP 3-D reference model (generic head, millimetre scale)
# Order: nose-tip, chin, left-eye-corner, right-eye-corner,
#        left-mouth-corner, right-mouth-corner
MODEL_POINTS_3D = np.array([
    (   0.0,    0.0,    0.0),
    (   0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)

# MediaPipe FaceMesh landmark indices
NOSE_TIP     = 1
CHIN         = 152
LEFT_EYE_MP  = 263
RIGHT_EYE_MP = 33
LEFT_MOUTH   = 287
RIGHT_MOUTH  = 57
NOSE_LEFT    = 279
NOSE_RIGHT   = 49
LEFT_CHEEK   = 234
RIGHT_CHEEK  = 454

# EAR landmarks — Soukupova-Cech order:
# [outer, upper-outer, upper-inner, inner, lower-inner, lower-outer]
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

# Sunglasses eye-polygon landmarks
SG_LEFT_EYE_POLY  = [33,  133, 160, 159, 158, 144, 145, 153]
SG_RIGHT_EYE_POLY = [362, 263, 387, 386, 385, 380, 381, 373]

# Detection / filtering
MTCNN_CONFIDENCE_THRESHOLD = 0.90
MIN_FACE_SIZE_PX           = 1     # minimum bbox side (original px space)

# Image dimension limits
IMG_MIN_DIM            = 1
IMG_MAX_DIM            = 8000
IMG_PROCESSING_MAX_DIM = 1600   # downscale target for all MediaPipe ops

# Batch limit
BATCH_MAX_FILE_BYTES = 25 * 1024 * 1024   # 25 MB

# Pose thresholds
YAW_FRONTAL_MAX       = 20.0
YAW_SLIGHTLY_SIDE_MAX = 35.0
ASYM_FRONTAL_MAX      = 12.0
ASYM_SLIGHTLY_MAX     = 22.0
MTCNN_FRONTAL_R_MIN   = 30
MTCNN_FRONTAL_R_MAX   = 62
MTCNN_FRONTAL_L_MIN   = 30
MTCNN_FRONTAL_L_MAX   = 63
NOSE_VISIBILITY_MIN   = 0.08

# EAR  (HIGH-SEVERITY: full/deep closure only, small-eye safe)
EAR_CLOSED_THRESHOLD     = 0.15   # only deeply shut eyes trigger this
EAR_MIN_RESOLUTION       = 80     # min face height in ORIGINAL px
EAR_RELATIVE_DROP_MIN    = 0.35   # EAR must be <= 35% of the eye's own open baseline
EAR_OPEN_BASELINE_FACTOR = 0.85   # eye width * this = estimated open EAR baseline 

# Sunglasses
SG_BRIGHTNESS_MAX    = 62   # dark + tinted lenses: black~10, gray~45, brown~55, amber~65
SG_CONTRAST_MAX      = 22
SG_HSV_V_MAX         = 65   # raised: tinted/mirrored lenses have HSV-V up to ~65
SG_DARK_COVERAGE_MIN = 0.48  # MediaPipe eye ROI is small; 48% coverage is enough
SG_DARK_PIXEL_THR    = 60   # count pixels up to brightness 60 as "dark" for tinted lenses
SG_IRIS_CONTRAST_MIN = 65
SG_IRIS_BRIGHT_MIN   = 70
SG_VOTE_THRESHOLD    = 4    # require all 4 signals — dark circles can't pass all 4
SG_REQUIRE_BOTH_EYES = False

YOLO_SG_CONF_HARD   = 0.50   # real goggles score 0.70+; 0.30–0.49 are dark-frame false positives
YOLO_SG_CONF_SOFT   = 0.25   # raised: marginal scores no longer feed evidence accumulator
YOLO_CLEAR_VETO     = 0.38
YOLO_CROP_PAD_FRAC  = 0.25
YOLO_EYE_BAND_Y     = (0.0, 0.65)
YOLO_EYE_BAND_X     = (-0.10, 1.10)
SG_EYE_VOTES        = 4    # require ALL 4 classical signals per eye
SG_EYE_VOTES_SOFT   = 3    # even with weak YOLO assist, still need 3/4

HW_YOLO_CONF_HARD    = 0.28   # headwear: YOLO hard-detect confidence floor
HW_YOLO_CONF_SOFT    = 0.13   # headwear: YOLO soft-assist floor
HW_VOTE_THRESHOLD    = 3      # votes needed to declare headwear detected
HW_CROP_PAD_TOP_FRAC = 0.60   # extend crop THIS much above face top (hat sits above face)
HW_CROP_PAD_SIDE     = 0.25   # side padding fraction
HW_FOREHEAD_TOP_FRAC = 0.22   # top portion of face bbox used for brim analysis
HW_BRIM_DARK_THR     = 80     # mean brightness below this in forehead → dark brim
HW_BRIM_COVERAGE_MIN = 0.55   # fraction of forehead pixels darker than threshold
HW_MASK_LOWER_FRAC   = 0.45   # lower face portion analysed for mask (from nose down)
HW_MASK_SKIN_DIFF    = 28     # lower-face brightness must differ from mid-face by this
HW_MASK_UNIFORM_THR  = 22     # std-dev of lower face below this → mask-like uniform cover

# Blur detection
BLUR_NORMALIZE_SIZE        = 256   # longest side used for consistent metric extraction
BLUR_SCORE_THRESHOLD       = 78.0  # raised further — valid doctor thumbnails score 44-103
BLUR_WHOLE_IMAGE_THRESHOLD = 65.0  # retained for compatibility; regional consensus is used
BLUR_MIN_FACE_PX           = 40    # below this, face sharpness is not reliable
BLUR_FACE_LOW_RES          = 50    # below this, report low_resolution rather than falsely claiming blur

# Framing / passport composition
FRAMING_TOP_MARGIN_MIN    = 0.02   # space above face top (for hair)
FRAMING_BOTTOM_MARGIN_MIN = 0.005  # space below face bottom (for neck)
FRAMING_FACE_H_MAX        = 0.85   # face_h/img_h > this → too zoomed in
FRAMING_FACE_W_MAX        = 0.80   # face_w/img_w > this → too zoomed in

# Drawing
LINE_COLOR = (255, 255, 0)
FONT       = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1
FONT_THICK = 2

# HTTP status codes
CATEGORY_STATUS_MAP = {
    "front":           200,
    "side":            400,
    "mixed":           400,
    "eyes_closed":     422,
    "sunglasses":      423,
    "blurred":                   422,
    "blur_uncertain":            422,
    "blur_sunglasses_uncertain": 422,
    "low_resolution":            422,
    "focused":                   422,
    "multiple_person": 409,
    "no_human":        404,
    "irrelevant":      415,
    "no_landmarks":    422,
}

# ─────────────────────────────────────────────────────────────
# Thread-safe MediaPipe instances
# ─────────────────────────────────────────────────────────────
_thread_local            = threading.local()
mp_face_detection_module = mp.solutions.face_detection
mp_face_mesh_module      = mp.solutions.face_mesh


def _get_face_detector():
    if not hasattr(_thread_local, "face_detector"):
        _thread_local.face_detector = mp_face_detection_module.FaceDetection(
            model_selection=1, min_detection_confidence=0.70
        )
    return _thread_local.face_detector


def _get_face_mesh():
    if not hasattr(_thread_local, "face_mesh"):
        _thread_local.face_mesh = mp_face_mesh_module.FaceMesh(
            static_image_mode=True, max_num_faces=10, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
    return _thread_local.face_mesh


# ─────────────────────────────────────────────────────────────
# MTCNN global
# ─────────────────────────────────────────────────────────────
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=True, keep_all=True, device=device,
)

# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app_: "FastAPI"):
    logger.info("Face Validation Engine v2.5.0 starting.")
    yield
    logger.info("Face Validation Engine shut down.")

app = FastAPI(
    title="Production Face Validation Engine",
    description="Multi-layer facial validation: pose · eyes · sunglasses · liveness",
    version="2.5.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# MODULE 0 — Image pre-processing
# ═══════════════════════════════════════════════════════════════

def validate_image_dimensions(frame_cv2: np.ndarray) -> Optional[str]:
    h, w = frame_cv2.shape[:2]
    if h < IMG_MIN_DIM or w < IMG_MIN_DIM:
        return f"Image too small: {w}x{h}px (min {IMG_MIN_DIM}px per side)"
    if h > IMG_MAX_DIM or w > IMG_MAX_DIM:
        return f"Image too large: {w}x{h}px (max {IMG_MAX_DIM}px per side)"
    return None


def _resize_for_processing(frame_cv2: np.ndarray) -> tuple:
    """
    Proportionally downscale so longest side <= IMG_PROCESSING_MAX_DIM.
    Returns (proc_frame, proc_scale).   proc_scale=1.0 means no resize.

    Coordinate contract:
      original_coord * proc_scale  = proc_coord
      proc_coord    / proc_scale   = original_coord
    """
    h, w    = frame_cv2.shape[:2]
    max_dim = max(h, w)
    if max_dim <= IMG_PROCESSING_MAX_DIM:
        return frame_cv2, 1.0
    scale   = IMG_PROCESSING_MAX_DIM / max_dim
    new_w   = max(1, int(w * scale))
    new_h   = max(1, int(h * scale))
    resized = cv2.resize(frame_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# ═══════════════════════════════════════════════════════════════
# MODULE 1 — No Human / Irrelevant
# ═══════════════════════════════════════════════════════════════

def _skin_ratio(frame_cv2: np.ndarray) -> float:
    """Fraction of pixels that match skin-tone HSV ranges (light and dark skin)."""
    hsv      = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2HSV)
    total_px = frame_cv2.shape[0] * frame_cv2.shape[1]
    mask1    = cv2.inRange(hsv, np.array([0, 20, 60], np.uint8),
                                np.array([25, 255, 255], np.uint8))
    mask2    = cv2.inRange(hsv, np.array([0, 10, 30], np.uint8),
                                np.array([30, 180, 180], np.uint8))
    return float(np.sum(cv2.bitwise_or(mask1, mask2) > 0)) / (total_px + 1e-8)


def detect_no_human(valid_bboxes: list, valid_probs: list) -> dict:
    """Called AFTER size+confidence filtering (single filter chain)."""
    if not valid_bboxes:
        return {"is_human": False, "reason": "No face passed filter",
                "max_confidence": 0.0}
    probs = [float(p) for p in valid_probs if p is not None]
    if not probs:
        return {"is_human": False, "reason": "All detections returned null confidence",
                "max_confidence": 0.0}
    return {"is_human": True, "reason": "Human face detected",
            "max_confidence": round(max(probs), 4)}


def detect_irrelevant_image(valid_bboxes: list, frame_cv2: np.ndarray) -> dict:
    """
    irrelevant : no face AND skin < 3%
    no_human   : no face BUT skin >= 3%
    """
    if valid_bboxes:
        return {"is_irrelevant": False, "reason": "Face detected"}
    skin = _skin_ratio(frame_cv2)
    if skin < 0.03:
        return {"is_irrelevant": True,
                "reason": f"No face + very low skin coverage ({skin:.3f})",
                "skin_ratio": round(skin, 4)}
    return {"is_irrelevant": False,
            "reason": f"No face but skin present ({skin:.3f}) -> no_human",
            "skin_ratio": round(skin, 4)}


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — Multiple person detection
# ═══════════════════════════════════════════════════════════════

def detect_persons_and_size(proc_frame: np.ndarray) -> dict:
    """Uses MediaPipe FaceDetection on the downscaled frame to count faces. Triggers multiple_person only when more than one face is found."""
    face_detector = _get_face_detector()
    rgb           = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    result        = face_detector.process(rgb)
    if not result.detections:
        return {"valid": True, "total_faces": 0,
                "category": "ok", "reason": "No detections — deferred to MTCNN"}
    total = len(result.detections)
    if total > 1:
        return {"valid": False, "total_faces": total,
                "category": "multiple_person",
                "reason": f"{total} persons detected. Only one allowed."}
    return {"valid": True, "total_faces": 1,
            "category": "ok", "reason": "Single face detected."}


# ═══════════════════════════════════════════════════════════════
# MODULE 3 — Sunglasses / dark-goggle detection
# ═══════════════════════════════════════════════════════════════

def _extract_eye_roi(proc_frame: np.ndarray, face_lm, indices: list):
    """Returns (gray, hsv) for the eye polygon, or (None, None) if too small."""
    h, w = proc_frame.shape[:2]
    lm   = face_lm.landmark
    xs   = [int(lm[i].x * w) for i in indices]
    ys   = [int(lm[i].y * h) for i in indices]
    x1, x2 = max(0, min(xs) - 4), min(w, max(xs) + 4)
    y1, y2 = max(0, min(ys) - 4), min(h, max(ys) + 4)
    roi    = proc_frame[y1:y2, x1:x2]
    if roi.size < 16:
        return None, None
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


def detect_sunglasses(proc_frame: np.ndarray, face_landmarks,
                      frame_cv2_orig: np.ndarray = None,
                      orig_bbox=None) -> dict:
    """
    Stage 1  YOLO OIV7: confident hit -> flag; clear-glasses -> veto; weak hit -> 'soft'.
    Stage 2  classical, evaluated PER EYE:
             an eye is dark when >= SG_EYE_VOTES of 4 signals fire (2 when YOLO was soft)
             sunglasses  <=>  EVERY measurable eye is dark
    """
    yolo = _detect_sunglasses_yolo(frame_cv2_orig, orig_bbox)

    # YOLO hard hit → immediate flag
    if yolo is not None and yolo["yolo_detected"]:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": (f"YOLO: sunglasses={yolo['yolo_sunglasses']:.2f} "
                           f"goggles={yolo['yolo_goggles']:.2f}"),
                "details": yolo}

    # Clear-glasses veto → immediate pass
    if yolo is not None and yolo["yolo_veto"] == "clear_glasses":
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": f"YOLO: clear prescription glasses (conf={yolo['yolo_clear_glasses']:.2f})",
                "details": yolo}

    # Hard uncertain floor: face < 20 px is too small for any detector to be reliable.
    if frame_cv2_orig is not None and orig_bbox is not None:
        try:
            _ih0, _iw0 = frame_cv2_orig.shape[:2]
            _fx1, _fy1, _fx2, _fy2 = [float(v) for v in orig_bbox[:4]]
            _fw0 = max(0.0, min(_iw0, _fx2) - max(0.0, _fx1))
            _fh0 = max(0.0, min(_ih0, _fy2) - max(0.0, _fy1))
            if min(_fw0, _fh0) < 20:
                return {"detected": False, "status": "low_resolution_uncertain", "votes": 0,
                        "reason": f"Face too small for any reliable detection ({int(_fw0)}x{int(_fh0)} px)",
                        "details": {"face_size": [int(_fw0), int(_fh0)]}}
        except Exception:
            pass

    soft = bool(yolo and yolo.get("yolo_soft"))

    # glasses-detector classifier — purpose-trained model (F1 ~0.95), run before gate decisions.
    clf_prob  = _detect_sg_classifier(frame_cv2_orig, orig_bbox)
    yolo_info = dict(yolo) if yolo else {}
    clf_info  = {"classifier_prob": round(clf_prob, 3) if clf_prob is not None else None}

    # STAGE 0 — Classifier is now the primary detector (threshold = 0.50, majority vote).
    if clf_prob is not None and clf_prob >= 0.65:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": f"Classifier: sunglasses prob={clf_prob:.2f}",
                "details": {**yolo_info, **clf_info}}

    # Classifier very confident no sunglasses → requires strong corroborating evidence.
    clf_soft_veto = (clf_prob is not None and clf_prob <= 0.20)

    # STAGE 0c — CLIP zero-shot (tiebreaker only — not standalone).
    # CLIP softmax is miscalibrated for binary face classification; it cannot
    # independently declare sunglasses. Use only to break ties when other signals
    # already show partial evidence.
    clip_prob = _sg_clip_score(frame_cv2_orig, orig_bbox)
    clip_info = {"clip_prob": round(clip_prob, 3) if clip_prob is not None else None}

    # YOLO soft + classifier partially confirms → flag.
    if soft and clf_prob is not None and clf_prob >= 0.55:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": f"YOLO soft + Classifier: prob={clf_prob:.2f}",
                "details": {**yolo_info, **clf_info, **clip_info}}
    # YOLO soft + CLIP strongly confirms (threshold raised to avoid false positives).
    if soft and clip_prob is not None and clip_prob >= 0.80:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": f"YOLO soft + CLIP: prob={clip_prob:.2f}",
                "details": {**yolo_info, **clf_info, **clip_info}}

    # Evidence accumulator — replaces the old YOLO hard gate.
    # Accumulates signals; score >= 3 -> detected, independent of YOLO verdict.
    # This handles tinted/amber/low-res cases where YOLO is uncertain.
    ev = {}; ev_score = 0
    if soft:
        ev["yolo_soft"] = True; ev_score += 2

    _rb = _sg_relative_brightness(frame_cv2_orig, orig_bbox)
    ev["relative_brightness"] = _rb
    if _rb is not None:
        if _rb.get("ratio", 1.0) < 0.65:
            ev["rb_dark"] = True; ev_score += 2
        if _rb.get("sat_drop", 0.0) > 30:
            ev["sat_drop_signal"] = True; ev_score += 1

    # Small-face check: if face is tiny and evidence is weak, return uncertain.
    if frame_cv2_orig is not None and orig_bbox is not None:
        try:
            _ih, _iw = frame_cv2_orig.shape[:2]
            _bx1, _by1, _bx2, _by2 = [float(v) for v in orig_bbox[:4]]
            _face_w = max(0.0, min(_iw, _bx2) - max(0.0, _bx1))
            _face_h = max(0.0, min(_ih, _by2) - max(0.0, _by1))
            if min(_face_w, _face_h) < 40 and ev_score < 3:
                _details = {**yolo_info, **clf_info, "evidence": ev, "ev_score": ev_score}
                _details["face_size"] = [int(_face_w), int(_face_h)]
                return {"detected": False, "status": "uncertain", "votes": 0,
                        "reason": "Face too small; insufficient evidence for sunglasses",
                        "details": _details}
        except Exception:
            pass

    # Early evidence decision before landmark extraction.
    if ev_score >= 4 and not clf_soft_veto:
        return {"detected": True, "status": "detected", "votes": ev_score,
                "reason": f"Evidence accumulator: score={ev_score} ({', '.join(k for k in ev if ev[k] is True)})",
                "details": {**yolo_info, **clf_info, "evidence": ev}}

    # Classifier-only path when YOLO is unavailable.
    classical_pair = (
        _classical_sunglasses_crop_evidence(frame_cv2_orig, orig_bbox)
        if yolo is None else {"detected": False, "reason": "skipped — using evidence accumulator"}
    )
    if yolo is None and clf_prob is not None and clf_prob >= 0.65:
        return {"detected": True, "status": "detected", "votes": 4,
                "reason": f"Classifier: sunglasses prob={clf_prob:.2f} (YOLO unavailable)",
                "details": clf_info}
        # Keep classical crop evidence for diagnostics only. Dark hair,
        # shadows and eyebrows can create paired components; it must not
        # independently declare sunglasses present.

    if face_landmarks is None:
        details = dict(yolo) if yolo else {}
        details["classical_crop"] = classical_pair
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": "No landmarks; crop-based detector inconclusive",
                "details": details}

    details = dict(yolo) if yolo else {}

    # Analyze the complete upper-face band as well as the tiny eye polygons.
    # Eye polygons alone miss wide sunglasses whose frame spans the bridge and
    # temples. Use original-resolution crop when available; otherwise proc frame.
    band_stats = None
    try:
        source_frame = frame_cv2_orig if frame_cv2_orig is not None else proc_frame
        if orig_bbox is not None and frame_cv2_orig is not None:
            ih, iw = source_frame.shape[:2]
            bx1, by1, bx2, by2 = [int(round(float(v))) for v in orig_bbox[:4]]
            bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
            px = int(0.12 * bw)
            cx1, cx2 = max(0, bx1 - px), min(iw, bx2 + px)
            cy1, cy2 = max(0, by1 + int(0.16 * bh)), min(ih, by1 + int(0.58 * bh))
            band = source_frame[cy1:cy2, cx1:cx2]
        elif face_landmarks is not None:
            fh, fw = proc_frame.shape[:2]
            lm = face_landmarks.landmark
            xs = [lm[i].x * fw for i in (33, 133, 263, 362)]
            ys = [lm[i].y * fh for i in (1, 168, 6, 197)]
            x1 = max(0, int(min(xs) - 0.20 * (max(xs)-min(xs))) )
            x2 = min(fw, int(max(xs) + 0.20 * (max(xs)-min(xs))) )
            y1 = max(0, int(min(ys) - 0.25 * (max(ys)-min(ys))) )
            y2 = min(fh, int(max(ys) + 1.80 * (max(ys)-min(ys))) )
            band = proc_frame[y1:y2, x1:x2]
        else:
            band = None
        if band is not None and band.size:
            bhh, bww = band.shape[:2]
            # Upscaling is for measurement consistency only, not added detail.
            if bww < 160:
                band = cv2.resize(band, (160, max(1, int(bhh * 160 / bww))), interpolation=cv2.INTER_CUBIC)
            gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            hsv_band = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            dark_mask = gray_band < SG_DARK_PIXEL_THR
            row_coverage = np.mean(dark_mask, axis=1)
            # Long, connected dark rows across the eye band are characteristic
            # of tinted lenses; require low luminance and broad horizontal coverage.
            band_stats = {
                "brightness": float(np.mean(gray_band)),
                "contrast": float(np.std(gray_band)),
                "dark_coverage": float(np.mean(dark_mask)),
                "max_row_coverage": float(np.max(row_coverage)) if row_coverage.size else 0.0,
                "hsv_value": float(np.mean(hsv_band[:, :, 2])),
            }
            # Dark sunglasses often form a broad horizontal strip even when
            # MediaPipe's eye polygons are tiny or partially misplaced. The
            # row-coverage test helps distinguish the strip from dark hair.
            band_dark = (
                band_stats["brightness"] < 105 and
                band_stats["dark_coverage"] >= 0.22 and
                band_stats["max_row_coverage"] >= 0.62 and
                band_stats["hsv_value"] < 115
            )
            details["upper_face_band"] = {k: round(v, 3) for k, v in band_stats.items()}
            details["upper_face_band_dark_candidate"] = bool(band_dark)
        else:
            band_dark = False
            details["upper_face_band"] = "inconclusive"
    except Exception as exc:
        band_dark = False
        details["upper_face_band_error"] = str(exc)

    l_gray, l_hsv = _extract_eye_roi(proc_frame, face_landmarks, SG_LEFT_EYE_POLY)
    r_gray, r_hsv = _extract_eye_roi(proc_frame, face_landmarks, SG_RIGHT_EYE_POLY)

    # Add texture + iris suppression signals to evidence score now that we have eye ROIs.
    for _gr in (l_gray, r_gray):
        _tv = _sg_lens_texture(_gr)
        if _tv is not None and _tv < 30:
            ev_score += 1; ev["lens_smooth"] = True; break
    _iri_l = _sg_iris_suppression(l_gray)
    _iri_r = _sg_iris_suppression(r_gray)
    if _iri_l or _iri_r:
        ev_score += 1; ev["iris_suppressed"] = True

    # Eye-openness (EAR) from FaceMesh: sunglasses suppress the detected eye
    # slit even when eyes are open — both EARs drop below 0.20.
    _ear_res = _sg_eye_openness(face_landmarks, proc_frame)
    if _ear_res is not None:
        ev["ear"] = _ear_res
        if _ear_res.get("both_suppressed"):
            ev_score += 1; ev["ear_suppressed"] = True

    details["ev_score"] = ev_score
    details["evidence"] = ev

    # Dark-circle guard for the post-ROI evidence accumulator.
    # Sunglasses cover the FULL eye opening uniformly; under-eye dark circles
    # concentrate brightness BELOW the eyelid, leaving the top of the ROI brighter.
    # If both measured eye ROIs show that pattern (top_mean - bottom_mean > 18),
    # reduce the evidence score so the accumulator does not fire on dark circles.
    if not soft and ev_score >= 4 and (l_gray is not None or r_gray is not None):
        _dc_grays = [g for g in (l_gray, r_gray) if g is not None and g.shape[0] >= 2]
        if _dc_grays:
            _dc_per_eye = []
            for _gr in _dc_grays:
                _mid2 = max(1, _gr.shape[0] // 2)
                _dc_per_eye.append(float(np.mean(_gr[:_mid2])) - float(np.mean(_gr[_mid2:])) > 18)
            if all(_dc_per_eye):
                ev["dark_circle_precheck"] = True
                ev_score = max(0, ev_score - 2)
                details["ev_score"] = ev_score
                details["evidence"] = ev

    if ev_score >= 4 and not clf_soft_veto:
        return {"detected": True, "status": "detected", "votes": ev_score,
                "reason": f"Evidence accumulator (post-ROI): score={ev_score}",
                "details": {**details, **clf_info, **clip_info}}
    if ev_score >= 3 and clf_prob is not None and clf_prob >= 0.60 and not clf_soft_veto:
        return {"detected": True, "status": "detected", "votes": ev_score,
                "reason": f"Evidence+classifier: score={ev_score}, clf={clf_prob:.2f}",
                "details": {**details, **clf_info, **clip_info}}

    def _stats(gray, hsv, side):
        if gray is None:
            details[f"{side}_roi"] = "inconclusive"
            return None
        s = {
            "b":  round(float(np.mean(gray)), 2),
            "c":  round(float(np.std(gray)),  2),
            "v":  round(float(np.mean(hsv[:, :, 2])), 2),
            "dc": round(float(np.sum(gray < SG_DARK_PIXEL_THR)) / (gray.size + 1e-8), 4),
        }
        # Vertical uniformity: sunglasses darken the full ROI top-to-bottom.
        # Dark circles concentrate below the eyelid → top half is brighter than bottom.
        mid = max(1, gray.shape[0] // 2)
        s["vert_diff"] = round(float(np.mean(gray[:mid])) - float(np.mean(gray[mid:])), 2)
        details[f"{side}_brightness"], details[f"{side}_contrast"] = s["b"], s["c"]
        details[f"{side}_v_channel"],  details[f"{side}_dark_coverage"] = s["v"], s["dc"]
        details[f"{side}_vert_diff"] = s["vert_diff"]
        return s

    L, R = _stats(l_gray, l_hsv, "left"), _stats(r_gray, r_hsv, "right")
    eyes = {k: v for k, v in (("left", L), ("right", R)) if v is not None}
    if not eyes:
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": "Both eye ROIs inconclusive", "details": details}

    def _iris(s):
        return s is not None and s["c"] > SG_IRIS_CONTRAST_MIN and s["b"] > SG_IRIS_BRIGHT_MIN
    iris_l, iris_r = _iris(L), _iris(R)
    if iris_l and iris_r and not soft:
        details["s5_iris_guard"] = "iris_visible_both_eyes — hard veto"
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": "Iris visible in both eyes (clear/prescription glass or bare)",
                "details": details}
    details["s5_iris_guard"] = f"iris NOT confirmed both eyes (L={iris_l}, R={iris_r})"

    # Vertical-uniformity veto: if in EVERY measurable eye the top half is
    # significantly brighter than the bottom, the darkness is under-eye skin
    # (dark circles), not a lens covering the full eye opening.
    _VERT_DC_THR = 18  # top_b - bot_b > 18 → darkness concentrated below eyelid
    dc_veto = {side: s["vert_diff"] > _VERT_DC_THR for side, s in eyes.items()}
    details["dark_circle_vert_check"] = dc_veto
    if all(dc_veto.values()) and not soft:
        details["dark_circle_veto"] = "top half brighter in all eyes — darkness below eyelid"
        return {"detected": False, "status": "not_detected", "votes": 0,
                "reason": "Dark-circle veto: brightness concentrated below eyelid, not a lens",
                "details": details}

    need = SG_EYE_VOTES_SOFT if soft else SG_EYE_VOTES
    per_eye = {}
    for side, s in eyes.items():
        sig = {
            "brightness": s["b"]  <  SG_BRIGHTNESS_MAX,
            "contrast":   s["c"]  <  SG_CONTRAST_MAX,
            "hsv_v":      s["v"]  <  SG_HSV_V_MAX,
            "dark_cov":   s["dc"] >= SG_DARK_COVERAGE_MIN,
        }
        per_eye[side] = sum(sig.values())
        details[f"{side}_signals"] = sig
        details[f"{side}_votes"]   = per_eye[side]

    single = len(eyes) == 1
    eye_detected = all(v >= need for v in per_eye.values())
    # Raised from 2→3: dark circles routinely score 2 (low brightness + low HSV)
    # but rarely pass all 3 classical signals on the top half of the ROI.
    asymmetric_eye_detected = False  # one-eye darkness is not sufficient proof
    # The broad band helps when landmark eye polygons are too tight. Treat it
    # as supporting evidence, not a standalone trigger, to reduce dark-room FPs.
    any_eye_dark = asymmetric_eye_detected
    # The wide-band cue can stand alone only when it has a strong, continuous
    # horizontal dark row. Otherwise require eye votes or a soft model hit.
    band_strong = bool(
        band_dark and band_stats is not None and
        band_stats.get("max_row_coverage", 0.0) >= 0.82 and
        band_stats.get("dark_coverage", 0.0) >= 0.32 and
        band_stats.get("brightness", 255.0) < 75
    )
    # Re-enable band path when YOLO already gave a soft signal (combined evidence)
    # or when the band signal itself is very strong. Stand-alone band is still disabled.
    band_supported = bool(band_dark and (soft or band_strong))
    # Require face-relative brightness to confirm goggles for per-eye and band paths.
    # Dark goggles/sunglasses make the eye-band noticeably darker than the cheeks (ratio < 0.68).
    # Dark-framed clear glasses, dark eyes, heavy eyeliner, shadows, and caps do not.
    # This applies even when YOLO gave a soft signal — the YOLO model can fire on
    # dark frames without the actual lenses being dark.
    rb_confirms = _rb is not None and _rb.get("ratio", 1.0) < 0.68
    if not rb_confirms:
        eye_detected = False
        band_supported = False
    # A two-signal hit in either eye is intentionally accepted for asymmetric
    # views. This addresses side profiles where the far lens is occluded or the
    # landmark ROI contains mostly skin, while the visible lens is clearly dark.
    detected = bool(eye_detected or band_supported)
    votes = max(per_eye.values())
    details.update({"votes": votes, "votes_needed_per_eye": need,
                    "measurable_eyes": len(eyes), "single_eye_only": single,
                    "yolo_soft_assist": soft, "eye_vote_detection": bool(eye_detected),
                    "asymmetric_eye_detection": bool(asymmetric_eye_detected),
                    "band_supported_detection": band_supported})
    suffix = " [asymmetric-view: one eye has >=2 signals]" if (detected and asymmetric_eye_detected and not eye_detected) else (" [single-eye — low confidence]" if (detected and single) else "")
    reason = (f"Dark goggles detected (per-eye votes {per_eye}; asymmetric 2-vote rule enabled){suffix}"
              if detected else f"No dark goggles (per-eye votes {per_eye}, need {need})")
    return {"detected": detected, "votes": votes, "reason": reason, "details": details}


# ═══════════════════════════════════════════════════════════════
# MODULE 3b — Headwear detection (cap / helmet / face mask)
# ═══════════════════════════════════════════════════════════════

def detect_headwear(frame_cv2_orig: np.ndarray, orig_bbox) -> dict:
    votes = 0
    vote_log = {}
    details = {}

    # ── Detector 1a: YOLO on face-extended crop (hat rim area) ──
    hw = _detect_headwear_yolo(frame_cv2_orig, orig_bbox)
    if hw is not None:
        details["crop_yolo"] = hw
        if hw.get("yolo_detected"):
            v = 4; votes += v
            vote_log["crop_yolo_hard"] = f"+{v}"
        elif hw.get("yolo_soft"):
            v = 2; votes += v
            vote_log["crop_yolo_soft"] = f"+{v}"
        # No penalty for no-detect on crop — full-image pass below compensates

    # ── Detector 1b: YOLO on full image (catches helmets that dwarf face bbox) ──
    fi = _detect_headwear_yolo_fullimage(frame_cv2_orig)
    if fi is not None:
        details["full_yolo"] = fi
        if fi.get("yolo_detected"):
            # Only add votes if crop didn't already give a hard hit (avoid double-counting)
            if not (hw or {}).get("yolo_detected"):
                v = 4; votes += v
                vote_log["full_yolo_hard"] = f"+{v}"
        elif fi.get("yolo_soft"):
            if not (hw or {}).get("yolo_detected") and not (hw or {}).get("yolo_soft"):
                v = 2; votes += v
                vote_log["full_yolo_soft"] = f"+{v}"
        elif not (hw or {}).get("yolo_detected") and not (hw or {}).get("yolo_soft"):
            v = -1; votes += v
            vote_log["yolo_no_detect"] = f"{v}"

    # Best YOLO result across both passes
    best_yolo = hw if (hw or {}).get("yolo_detected") else (fi if (fi or {}).get("yolo_detected") else
                hw if (hw or {}).get("yolo_soft")     else (fi if (fi or {}).get("yolo_soft")     else hw))

    # ── Detector 2: classical above-face / brim analysis ──────
    cl = _detect_headwear_classical(frame_cv2_orig, orig_bbox)
    if cl is not None:
        details["classical"] = cl
        if cl.get("brim_signal"):
            v = 2; votes += v
            vote_log["classical_brim"] = (f"+{v} dark={cl.get('dark_brim')} "
                                          f"diff={cl.get('color_diff_signal')} fore={cl.get('fore_cut')}")
        else:
            v = -1; votes += v
            vote_log["classical_no_brim"] = f"{v}"

    # ── Detector 3: face mask (multi-signal classical) ────────
    mk = _detect_face_mask_classical(frame_cv2_orig, orig_bbox)
    mask_detected = False
    if mk is not None:
        details["mask"] = mk
        mask_score = mk.get("mask_score", 0)
        if mk.get("mask_signal"):
            mask_detected = True
            # Confident mask (3+ signals) → early return; YOLO noise can't veto it
            if mask_score >= 3:
                details["vote_log"] = {"mask_early_return": f"score={mask_score}/5"}
                details["total_votes"] = 99
                return {
                    "detected": True, "votes": 99, "kind": "mask", "mask": True,
                    "reason":   f"Face mask detected (score={mask_score}/5 signals)",
                    "details":  details,
                }
            # Soft mask (2 signals) — add to vote pool but needs corroboration
            v = 4; votes += v
            vote_log["mask_signal"] = f"+{v} (score={mask_score}/5 std={mk['lower_face_std']:.1f})"

    detected = votes >= HW_VOTE_THRESHOLD
    yolo_kind = (best_yolo or {}).get("yolo_best_kind") or "cap_or_helmet"
    kind = "mask" if mask_detected else yolo_kind
    details["vote_log"] = vote_log
    details["total_votes"] = votes
    return {
        "detected":    detected,
        "votes":       votes,
        "kind":        kind if detected else None,
        "mask":        mask_detected,
        "reason":      f"Headwear votes={votes}/{HW_VOTE_THRESHOLD} kind={kind}",
        "details":     details,
    }


def _clean_headwear(hw: dict) -> dict:
    if not hw:
        return hw
    d = hw.get("details") or {}
    return {
        "detected": hw.get("detected"),
        "kind":     hw.get("kind"),
        "mask":     hw.get("mask"),
        "reason":   hw.get("reason"),
        "details": {
            "yolo_detected":    d.get("yolo_detected"),
            "yolo_helmet_conf": d.get("yolo_helmet_conf"),
            "yolo_cap_conf":    d.get("yolo_cap_conf"),
            "yolo_best_kind":   d.get("yolo_best_kind"),
            "brim_signal":      d.get("brim_signal"),
            "brim_dark_cov":    d.get("brim_dark_cov"),
            "mask_signal":      (d.get("mask") or {}).get("mask_signal"),
            "lower_face_std":   (d.get("mask") or {}).get("lower_face_std"),
        },
    }


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — Closed-eye detection  (EAR)
# ═══════════════════════════════════════════════════════════════

def _compute_ear(landmarks_2d: list, eye_idx: list) -> Optional[float]:
    """
    EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
    Returns None on failure (callers must handle None explicitly).
    """
    try:
        pts = [landmarks_2d[i] for i in eye_idx]
        v1  = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2  = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h   = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return round(float((v1 + v2) / (2.0 * h)), 4) if h > 0 else None
    except Exception:
        return None


def _estimate_open_ear_baseline(landmarks_2d: list, eye_idx: list) -> Optional[float]:
    """
    Estimates what this person's EAR would be when eyes are open,
    using only the horizontal eye width (pts[0] to pts[3]).
    This is self-contained per image — no prior frame needed.

    Formula:  baseline = eye_width * EAR_OPEN_BASELINE_FACTOR / (2 * eye_width)
                       = EAR_OPEN_BASELINE_FACTOR / 2

    This gives a person-neutral open baseline (~0.42 for normal eyes).
    Small-eyed people have a smaller eye_width, so their EAR is already
    naturally lower — the relative drop check catches only TRUE closure.
    """
    try:
        pts       = [landmarks_2d[i] for i in eye_idx]
        eye_width = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        if eye_width < 1e-8:
            return None
        # Vertical open height estimated as fraction of eye width
        estimated_open_v = eye_width * EAR_OPEN_BASELINE_FACTOR
        return round(float(estimated_open_v / (2.0 * eye_width)), 4)
    except Exception:
        return None


def detect_closed_eyes(
    face_landmarks,
    proc_w: int,
    proc_h: int,
    orig_bbox,     # original-space bbox — for resolution guard only
    proc_bbox,     # unused, kept for call-site compatibility
) -> dict:
    """
    HIGH-SEVERITY closed-eye detection — small-eye safe.

    Two-gate system:
      Gate 1 (absolute): both EARs < EAR_CLOSED_THRESHOLD (0.15 — deeply shut)
      Gate 2 (relative): both EARs <= EAR_RELATIVE_DROP_MIN * their open baseline
                         Protects naturally small/narrow eyes from false positives.

    Both gates must pass for eyes_closed = True.
    """
    result = {"eyes_closed": False, "left_ear": None, "right_ear": None,
              "avg_ear": None, "reason": "Not checked", "checked": False}

    if face_landmarks is None:
        result["reason"] = "MediaPipe landmarks unavailable"
        return result

    # Resolution guard — skip on tiny faces (unreliable EAR)
    if orig_bbox is None:
        result["reason"] = "No bounding box — skipping EAR (unreliable without size guard)"
        return result

    x1, y1, x2, y2 = orig_bbox
    orig_face_h = abs(y2 - y1)
    if orig_face_h < EAR_MIN_RESOLUTION:
        result["reason"] = (
            f"Face too small ({orig_face_h:.0f}px < {EAR_MIN_RESOLUTION}px) — skipped"
        )
        return result

    lm           = face_landmarks.landmark
    landmarks_2d = [(lm[i].x * proc_w, lm[i].y * proc_h) for i in range(len(lm))]

    left_ear  = _compute_ear(landmarks_2d, LEFT_EYE_EAR)
    right_ear = _compute_ear(landmarks_2d, RIGHT_EYE_EAR)

    result.update({"left_ear": left_ear, "right_ear": right_ear, "checked": True})

    if left_ear is None or right_ear is None:
        result["reason"] = (
            f"EAR computation failed (L={left_ear}, R={right_ear}) — treating as open"
        )
        return result

    avg_ear           = round((left_ear + right_ear) / 2.0, 4)
    result["avg_ear"] = avg_ear

    # Gate 1: absolute threshold — must be deeply shut
    gate1_l = left_ear  < EAR_CLOSED_THRESHOLD
    gate1_r = right_ear < EAR_CLOSED_THRESHOLD
    gate1   = gate1_l and gate1_r

    # Gate 2: relative drop — EAR must be <= 35% of estimated open baseline
    # Protects small-eyed people: their baseline is also small, so ratio is fair
    base_l = _estimate_open_ear_baseline(landmarks_2d, LEFT_EYE_EAR)
    base_r = _estimate_open_ear_baseline(landmarks_2d, RIGHT_EYE_EAR)

    if base_l is not None and base_r is not None:
        ratio_l = round(left_ear  / base_l, 4) if base_l > 0 else 1.0
        ratio_r = round(right_ear / base_r, 4) if base_r > 0 else 1.0
        gate2   = (ratio_l <= EAR_RELATIVE_DROP_MIN and
                   ratio_r <= EAR_RELATIVE_DROP_MIN)
        result.update({
            "left_baseline":  base_l, "right_baseline":  base_r,
            "left_ratio":     ratio_l, "right_ratio":    ratio_r,
        })
    else:
        # Baseline unavailable — fall back to gate1 only
        gate2   = True
        ratio_l = ratio_r = None
        result["baseline_note"] = "Baseline unavailable — gate2 bypassed"

    if gate1 and gate2:
        result["eyes_closed"] = True
        result["reason"] = (
            f"Both eyes fully closed: L={left_ear:.3f}, R={right_ear:.3f} "
            f"(threshold={EAR_CLOSED_THRESHOLD}, "
            f"relative L={ratio_l}, R={ratio_r} <= {EAR_RELATIVE_DROP_MIN})"
        )
    elif not gate1:
        open_eyes = []
        if left_ear  >= EAR_CLOSED_THRESHOLD: open_eyes.append(f"L={left_ear:.3f}")
        if right_ear >= EAR_CLOSED_THRESHOLD: open_eyes.append(f"R={right_ear:.3f}")
        result["reason"] = (
            f"Gate1 failed — not deeply closed ({', '.join(open_eyes)}, "
            f"threshold={EAR_CLOSED_THRESHOLD})"
        )
    else:
        result["reason"] = (
            f"Gate2 failed — relative drop insufficient "
            f"(L ratio={ratio_l}, R ratio={ratio_r}, min drop={EAR_RELATIVE_DROP_MIN}): "
            f"likely small-eyed person with open eyes"
        )
    return result


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — Head-pose estimation
# ═══════════════════════════════════════════════════════════════

def _np_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b between rays ba and bc."""
    ba    = a - b;  bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def _estimate_focal_length(img_w: int, img_h: int) -> float:
    """
    Focal length in pixels via 70-degree diagonal-FOV heuristic.
      f = sqrt(w^2+h^2) / (2 * tan(FOV_diag/2))
    Diagonal is rotation-invariant: same f for portrait and landscape.
    """
    diag = math.sqrt(img_w ** 2 + img_h ** 2)
    return diag / (2.0 * math.tan(math.radians(70.0) / 2.0))


def _get_mediapipe_angles(face_landmarks, img_w: int, img_h: int) -> Optional[dict]:
    """
    Computes head yaw/pitch/roll via SolvePnP + ZYX Euler decomposition.

    Camera model: diagonal-FOV focal length, principal point at image centre,
    zero distortion.

    180-degree flip guard: rm[:,2] is the face forward-Z in camera space.
    Normal orientation has forward[2] < 0 (Z points into scene, away from camera).
    If forward[2] > 0 (flipped), negate rm to correct orientation.

    ZYX Euler decomposition:
      sy  = sqrt(R[0,0]^2 + R[1,0]^2)
      if sy > eps:
        pitch = atan2( R[2,1], R[2,2])
        yaw   = atan2(-R[2,0], sy)
        roll  = atan2( R[1,0], R[0,0])
      else (gimbal lock, pitch ~ +-90 deg):
        pitch = atan2(-R[1,2], R[1,1])
        yaw   = atan2(-R[2,0], sy)
        roll  = 0  (lost at gimbal lock — standard behaviour)
    """
    try:
        lm  = face_landmarks.landmark
        pts = np.array([
            (lm[NOSE_TIP].x     * img_w, lm[NOSE_TIP].y     * img_h),
            (lm[CHIN].x         * img_w, lm[CHIN].y          * img_h),
            (lm[LEFT_EYE_MP].x  * img_w, lm[LEFT_EYE_MP].y  * img_h),
            (lm[RIGHT_EYE_MP].x * img_w, lm[RIGHT_EYE_MP].y * img_h),
            (lm[LEFT_MOUTH].x   * img_w, lm[LEFT_MOUTH].y   * img_h),
            (lm[RIGHT_MOUTH].x  * img_w, lm[RIGHT_MOUTH].y  * img_h),
        ], dtype=np.float64)

        fl  = _estimate_focal_length(img_w, img_h)
        cm  = np.array([[fl, 0,  img_w / 2],
                        [0,  fl, img_h / 2],
                        [0,  0,  1.0      ]], dtype=np.float64)

        # Use SOLVEPNP_SQPNP (globally convergent) for reliable pose estimation
        ok, rvec, _tvec = cv2.solvePnP(
            MODEL_POINTS_3D, pts, cm, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            return None

        rm, _ = cv2.Rodrigues(rvec)

        # Do not negate the full rotation matrix: -R is not a valid SO(3)
        # rotation and corrupts the Euler angles. Keep the proper Rodrigues
        # rotation returned by solvePnP and classify using absolute yaw.

        sy = math.sqrt(rm[0, 0] ** 2 + rm[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.atan2( rm[2, 1], rm[2, 2])
            yaw   = math.atan2(-rm[2, 0], sy)
            roll  = math.atan2( rm[1, 0], rm[0, 0])
        else:   # gimbal lock — pitch is near ±90 degrees, roll is undefined
            pitch = math.atan2(-rm[1, 2], rm[1, 1])
            yaw   = math.atan2(-rm[2, 0], sy)
            roll  = 0.0

        return {
            "yaw":   round(math.degrees(yaw),   2),
            "pitch": round(math.degrees(pitch),  2),
            "roll":  round(math.degrees(roll),   2),
        }
    except Exception:
        return None


def _get_nose_visibility_ratio(face_landmarks) -> float:
    """
    Nose-wing width / cheek-to-cheek width — purely normalised ratio.
    ~0.25–0.30 frontal; < 0.08 full side.
    """
    try:
        lm = face_landmarks.landmark
        nw = abs(lm[NOSE_RIGHT].x - lm[NOSE_LEFT].x)
        fw = abs(lm[RIGHT_CHEEK].x - lm[LEFT_CHEEK].x)
        return round(nw / fw, 4) if fw > 1e-8 else 1.0
    except Exception:
        return 1.0


def classify_pose_strong(
    angR: float,
    angL: float,
    yaw_data: Optional[dict],
    nose_ratio: float,
) -> tuple:
    """
    Weighted 4-signal pose classifier.

    Weights:  S1 x1.0  S2 x1.5  S3 x2.5  S4 x1.5   (theoretical max ±6.5)

    S3 abstains when yaw unavailable (was s1*0.5 phantom penalty).
    unavailable S3 slot excluded from dynamic_max.
    classification boundary score >= 0 -> Frontal.

    confidence = |total_score| / dynamic_max * 100
    """
    total_score = 0.0
    dynamic_max = 0.0
    signals     = {}

    # S1 — MTCNN eye-to-nose angle range
    ri, li = int(angR), int(angL)
    if (MTCNN_FRONTAL_R_MIN <= ri <= MTCNN_FRONTAL_R_MAX and
            MTCNN_FRONTAL_L_MIN <= li <= MTCNN_FRONTAL_L_MAX):
        s1 = 1.0;  signals["mtcnn_angle"] = "Frontal"
    elif 25 <= ri <= 68 and 25 <= li <= 70:
        s1 = 0.3;  signals["mtcnn_angle"] = "Slightly Side"
    else:
        s1 = -1.0; signals["mtcnn_angle"] = "Side"
    total_score += s1 * 1.0
    dynamic_max += 1.0

    # S2 — Eye-angle asymmetry
    asym = abs(angR - angL)
    if asym <= ASYM_FRONTAL_MAX:
        s2 = 1.0;  signals["asymmetry"] = f"Frontal ({asym:.1f}deg)"
    elif asym <= ASYM_SLIGHTLY_MAX:
        s2 = 0.3;  signals["asymmetry"] = f"Slightly Side ({asym:.1f}deg)"
    else:
        s2 = -1.0; signals["asymmetry"] = f"Side ({asym:.1f}deg)"
    total_score += s2 * 1.5
    dynamic_max += 1.5

    # S3 — MediaPipe yaw  (highest weight; abstains when unavailable or roll≈180° flip)
    # When SolvePnP returns |roll| > 150°, the solution is in a flipped orientation
    # and the extracted yaw is unreliable — abstain rather than penalise wrongly.
    _roll_ok = yaw_data is not None and abs(yaw_data.get("roll", 0.0)) < 150.0
    if _roll_ok:
        yaw_abs = abs(yaw_data["yaw"])
        signals.update(yaw_data)
        if yaw_abs <= YAW_FRONTAL_MAX:
            s3 = 1.0;  signals["mediapipe_yaw"] = f"Frontal ({yaw_abs}deg)"
        elif yaw_abs <= YAW_SLIGHTLY_SIDE_MAX:
            decay = 1.0 - ((yaw_abs - YAW_FRONTAL_MAX) /
                            (YAW_SLIGHTLY_SIDE_MAX - YAW_FRONTAL_MAX))
            s3 = max(0.1, decay * 0.8)
            signals["mediapipe_yaw"] = f"Slightly Side ({yaw_abs}deg, s3={s3:.3f})"
        else:
            s3 = -1.0; signals["mediapipe_yaw"] = f"Full Side ({yaw_abs}deg)"
        total_score += s3 * 2.5
        dynamic_max += abs(s3) * 2.5
    else:
        if yaw_data is not None:
            signals.update(yaw_data)
        # abstain — no phantom penalty, no dynamic_max contribution
        _roll_val = abs(yaw_data.get("roll", 0.0)) if yaw_data is not None else None
        reason = "unavailable" if yaw_data is None else f"roll={_roll_val:.1f}deg flip"
        signals["mediapipe_yaw"] = f"{reason} — S3 abstained"

    # S4 — Nose visibility ratio
    signals["nose_visibility_ratio"] = nose_ratio
    if nose_ratio >= 0.18:
        s4 = 1.0;  signals["nose_visibility"] = f"Frontal ({nose_ratio:.4f})"
    elif nose_ratio >= NOSE_VISIBILITY_MIN:
        s4 = 0.2;  signals["nose_visibility"] = f"Slightly Side ({nose_ratio:.4f})"
    else:
        s4 = -1.0; signals["nose_visibility"] = f"Full Side ({nose_ratio:.4f})"
    total_score += s4 * 1.5
    dynamic_max += 1.5

    # High-confidence side gates. A true profile typically has a large yaw
    # and/or very little visible nose width. These should not be outvoted by
    # noisy eye-angle measurements from partial occlusion or sunglasses.
    yaw_abs  = abs(float(yaw_data.get("yaw",  0.0))) if yaw_data is not None else None
    roll_abs = abs(float(yaw_data.get("roll", 0.0))) if yaw_data is not None else None
    # Only trust yaw-based gates when SolvePnP roll is in a normal range
    yaw_trust = roll_abs is None or roll_abs < 150.0
    strong_side_reasons = []
    # Three-quarter/profile photos often have yaw around 30-40 degrees.
    # Treat moderate yaw as side when corroborated by reduced visible nose width;
    # this avoids requiring a near-perfect 90-degree profile.
    if yaw_abs is not None and yaw_trust and yaw_abs >= 38.0:
        strong_side_reasons.append(f"yaw={yaw_abs:.1f}deg")
    elif yaw_abs is not None and yaw_trust and yaw_abs >= 30.0 and nose_ratio < 0.16:
        strong_side_reasons.append(f"moderate_yaw={yaw_abs:.1f}deg+nose_ratio={nose_ratio:.4f}")
    if nose_ratio < 0.075:
        strong_side_reasons.append(f"nose_ratio={nose_ratio:.4f}")
    if asym >= 24.0 and nose_ratio < 0.16:
        strong_side_reasons.append(f"eye_asymmetry={asym:.1f}deg+narrow_nose")

    if strong_side_reasons:
        final = "Side"
        signals["strong_side_gate"] = strong_side_reasons
    else:
        final = "Frontal" if total_score >= 0 else "Side"
        signals["strong_side_gate"] = []

    conf_pct = (abs(total_score) / dynamic_max * 100.0) if dynamic_max > 0 else 0.0
    signals.update({
        "total_score": round(total_score, 4),
        "max_score":   round(dynamic_max, 4),
        "confidence":  f"{round(conf_pct, 1)}%",
        "pose_decision": final,
    })
    return final, signals


# ═══════════════════════════════════════════════════════════════
# MODULE 6 — MTCNN <-> MediaPipe face matching
# ═══════════════════════════════════════════════════════════════

def _match_landmarks_to_faces(
    orig_bboxes: list,
    mp_faces:    list,
    proc_w:      int,
    proc_h:      int,
    proc_scale:  float,
) -> list:
    """
    Matches each MTCNN bbox (original space) to the closest MediaPipe face
    by Euclidean distance to the nose-tip landmark.

    bbox centre is multiplied by proc_scale to convert it to
    proc-frame space before computing distance.  Without this, on a 4K image
    (scale=0.5) the centre was ~1000 px from the correct nose-tip.

    Distance formula:
      d = sqrt( (lm.x*proc_w  -  cx_orig*proc_scale)^2
              + (lm.y*proc_h  -  cy_orig*proc_scale)^2 )
    """
    matched = []
    for bbox in orig_bboxes:
        if bbox is None:
            matched.append(None)
            continue
        bx1, by1, bx2, by2 = bbox
        # convert original-space centre -> proc-frame space
        cx_proc = ((bx1 + bx2) / 2.0) * proc_scale
        cy_proc = ((by1 + by2) / 2.0) * proc_scale

        best_mp, best_dist = None, float("inf")
        for mp_face in mp_faces:
            lm   = mp_face.landmark
            dist = math.sqrt(
                (lm[NOSE_TIP].x * proc_w - cx_proc) ** 2 +
                (lm[NOSE_TIP].y * proc_h - cy_proc) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_mp   = mp_face
        matched.append(best_mp)
    return matched


# ═══════════════════════════════════════════════════════════════
# BLUR SCORING
# ═══════════════════════════════════════════════════════════════

def compute_blur_score(crop_bgr: np.ndarray):
    """
    Multi-signal no-reference blur assessment.

    Returns (blur_score, details). blur_score is a HEURISTIC 0..100 index,
    not a probability. A calibrated probability requires a labelled dataset
    and a separately trained/calibrated model.

    Signals: skimage blur-effect (when available), Laplacian variance,
    Tenengrad, Canny edge density, high-frequency energy, regional sharpness,
    luminance, and contrast. Scores are resolution-normalized but thresholds
    still require validation on the deployment dataset.
    """
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return 100.0, {"reason": "empty_crop", "decision": "uncertain",
                       "score_kind": "heuristic_not_probability"}

    # Preserve aspect ratio. Stretching every crop to a square distorts edges
    # and changes sharpness statistics.
    src_h, src_w = crop_bgr.shape[:2]
    if src_h < 2 or src_w < 2:
        return 100.0, {"reason": "crop_too_small", "decision": "uncertain",
                       "score_kind": "heuristic_not_probability"}
    target = max(64, int(BLUR_NORMALIZE_SIZE))
    scale = min(target / float(max(src_h, src_w)), 1.0)
    # Upscale small crops only to stabilize metric dimensions; upscaling does
    # not create real detail and is explicitly recorded below.
    if max(src_h, src_w) < target:
        scale = target / float(max(src_h, src_w))
    nw, nh = max(2, int(round(src_w * scale))), max(2, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    norm = cv2.resize(crop_bgr, (nw, nh), interpolation=interpolation)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY) if norm.ndim == 3 else norm.copy()
    gray_f = gray.astype(np.float32) / 255.0

    # Light denoising limits high-frequency noise, while metrics are also
    # calculated on the un-denoised image to avoid erasing useful fine detail.
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape
    details = {
        "source_dimensions": [int(src_w), int(src_h)],
        "metric_dimensions": [int(w), int(h)],
        "metric_upscaled": bool(scale > 1.0),
        "score_kind": "heuristic_0_100_not_probability",
    }

    # Signal 1: Laplacian variance (more edge detail generally means sharper).
    lap_var = float(cv2.Laplacian(denoised, cv2.CV_64F, ksize=3).var())

    # Signal 2: Tenengrad / Sobel gradient energy.
    sx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    grad_energy = sx * sx + sy * sy
    tenengrad = float(np.mean(grad_energy))

    # Signal 3: edge density, using thresholds adapted to local contrast.
    med = float(np.median(denoised))
    sigma = float(np.std(denoised))
    lower = int(max(10, min(80, 0.66 * med)))
    upper = int(max(lower + 10, min(220, 1.33 * med)))
    edges = cv2.Canny(denoised, lower, upper)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    # Signal 4: high-frequency energy ratio in the Fourier domain.
    f = np.fft.fftshift(np.fft.fft2(gray_f))
    mag = np.abs(f)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    cutoff = 0.18 * min(h, w)
    total_energy = float(np.sum(mag ** 2)) + 1e-12
    high_freq_ratio = float(np.sum((mag ** 2)[radius >= cutoff]) / total_energy)

    # Signal 5: patch-level detail. Median is robust to a few textured patches;
    # 75th percentile indicates whether any regions contain real edges.
    patch_laps = []
    rows = np.array_split(denoised, min(4, h), axis=0)
    for row in rows:
        for patch in np.array_split(row, min(4, w), axis=1):
            if patch.size >= 16:
                patch_laps.append(float(cv2.Laplacian(patch, cv2.CV_64F, ksize=3).var()))
    patch_median = float(np.median(patch_laps)) if patch_laps else 0.0
    patch_p75 = float(np.percentile(patch_laps, 75)) if patch_laps else 0.0

    # Signal 6: optional no-reference blur-effect metric (0 sharp .. 1 blurred).
    skimage_effect = None
    if _SKIMAGE_OK:
        try:
            skimage_effect = float(np.clip(_skimage_blur_effect(gray_f, h_size=11), 0.0, 1.0))
        except Exception:
            skimage_effect = None

    # Exposure/contrast indicators matter because very dark, flat, or washed-out
    # images can look "blurry" to edge metrics even when the focus is unknown.
    mean_luma = float(np.mean(gray))
    p05, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p05)
    clipped_dark = float(np.mean(gray <= 5))
    clipped_bright = float(np.mean(gray >= 250))

    # Convert raw metrics into conservative, interpretable heuristic sharpness
    # signals. These anchors are starting values, not universal thresholds.
    lap_sharp = float(np.clip(lap_var / 900.0, 0.0, 1.0))
    ten_sharp = float(np.clip(tenengrad / 1800.0, 0.0, 1.0))
    edge_sharp = float(np.clip(edge_density / 0.045, 0.0, 1.0))
    hf_sharp = float(np.clip(high_freq_ratio / 0.12, 0.0, 1.0))
    patch_sharp = float(np.clip(patch_median / 900.0, 0.0, 1.0))
    cv_sharp = (0.27 * lap_sharp + 0.23 * ten_sharp +
                0.17 * edge_sharp + 0.18 * hf_sharp + 0.15 * patch_sharp)
    cv_blur = 100.0 * (1.0 - cv_sharp)

    if skimage_effect is not None:
        # Keep the optional signal supplementary so one metric cannot dominate.
        blur_score = float(np.clip(0.75 * cv_blur + 0.25 * skimage_effect * 100.0, 0, 100))
    else:
        blur_score = float(np.clip(cv_blur, 0, 100))

    # Agreement / reliability flags.
    low_contrast = dynamic_range < 25 or sigma < 12
    extreme_exposure = mean_luma < 12 or mean_luma > 245
    noisy_or_textured = lap_var > 2500 and tenengrad > 5000 and high_freq_ratio > 0.10
    low_detail_votes = sum([
        lap_sharp < 0.20,
        ten_sharp < 0.20,
        edge_sharp < 0.20,
        hf_sharp < 0.20,
        patch_sharp < 0.20,
    ])
    high_detail_votes = sum([
        lap_sharp > 0.45,
        ten_sharp > 0.45,
        edge_sharp > 0.35,
        hf_sharp > 0.30,
        patch_sharp > 0.40,
    ])

    if extreme_exposure:
        decision = "uncertain"
        reason = "extreme_exposure"
    # Strong, consistent absence of detail can itself be evidence of blur.
    # Keep truly flat/featureless images uncertain by requiring some tonal range.
    elif blur_score >= 85 and low_detail_votes >= 5 and dynamic_range >= 12 and sigma >= 4:
        decision = "blurred"
        reason = "strong_multi_signal_blur_despite_reduced_contrast"
    elif low_contrast:
        decision = "uncertain"
        reason = "low_contrast_or_featureless_image"
    elif blur_score >= 78 and low_detail_votes >= 5:
        decision = "blurred"
        reason = "multiple_sharpness_signals_low"
    elif blur_score <= 42 and high_detail_votes >= 3:
        decision = "not_blurry"
        reason = "multiple_sharpness_signals_high"
    else:
        decision = "uncertain"
        reason = "mixed_or_borderline_sharpness_signals"

    # Blur subtype is a heuristic hint only, not a guaranteed physical diagnosis.
    gx_energy = float(np.mean(sx * sx)) + 1e-8
    gy_energy = float(np.mean(sy * sy)) + 1e-8
    anisotropy = max(gx_energy, gy_energy) / min(gx_energy, gy_energy)
    if decision == "blurred" and anisotropy >= 2.5:
        blur_type = "possible_motion_blur"
    elif decision == "blurred":
        blur_type = "possible_defocus_or_general_blur"
    else:
        blur_type = None

    details.update({
        "laplacian_var": round(lap_var, 3),
        "tenengrad": round(tenengrad, 3),
        "edge_density": round(edge_density, 5),
        "high_frequency_ratio": round(high_freq_ratio, 5),
        "patch_median_var": round(patch_median, 3),
        "patch_p75_var": round(patch_p75, 3),
        "mean_luminance": round(mean_luma, 2),
        "luminance_dynamic_range_p05_p95": round(dynamic_range, 2),
        "contrast_std": round(sigma, 2),
        "clipped_dark_fraction": round(clipped_dark, 4),
        "clipped_bright_fraction": round(clipped_bright, 4),
        "skimage_blur_effect": round(skimage_effect, 4) if skimage_effect is not None else None,
        "cv_blur_score": round(cv_blur, 2),
        "blur_score": round(blur_score, 2),
        "sharpness_index": round(1.0 - blur_score / 100.0, 4),
        "decision": decision,
        "reason": reason,
        "blur_type_hint": blur_type,
        "low_detail_signal_count": int(low_detail_votes),
        "high_detail_signal_count": int(high_detail_votes),
        "anisotropy": round(anisotropy, 3),
        "low_contrast": bool(low_contrast),
        "extreme_exposure": bool(extreme_exposure),
        "noisy_or_textured": bool(noisy_or_textured),
        "probability_note": "blur_score is heuristic, not a calibrated probability",
    })
    return round(blur_score, 2), details


def _detect_blur_on_face(frame_cv2: np.ndarray, bbox_, valid_indices: list) -> dict:
    """Assess blur severity, face resolution, and source-image pixel dimensions."""
    if frame_cv2 is None or not isinstance(frame_cv2, np.ndarray) or frame_cv2.size == 0:
        return {"is_blurred": False, "category": "blur_uncertain", "blur_score": None,
                "blur_level_percent": None, "blur_level_label": "unavailable",
                "blur_description": "No image pixels were available for analysis.",
                "image_resolution": _resolution_metadata(0, 0),
                "details": {"reason": "empty_image"}, "faces": [], "uncertain": True}

    img_h, img_w = frame_cv2.shape[:2]
    image_resolution = _resolution_metadata(img_w, img_h)
    if not valid_indices or bbox_ is None:
        fallback = _check_whole_image_blur(frame_cv2)
        fallback.setdefault("details", {})["reason"] = "no_face_bbox_whole_image_fallback"
        return fallback
    global_score, global_details = compute_blur_score(frame_cv2)
    face_results = []
    for face_number, idx in enumerate(valid_indices or [], start=1):
        try:
            b = bbox_[idx]
            x1 = max(0, min(img_w, int(round(float(b[0])))))
            y1 = max(0, min(img_h, int(round(float(b[1])))))
            x2 = max(0, min(img_w, int(round(float(b[2])))))
            y2 = max(0, min(img_h, int(round(float(b[3])))))
            face_w, face_h = max(0, x2-x1), max(0, y2-y1)
            face_pixels = face_w * face_h
            face_resolution = _resolution_metadata(face_w, face_h)
            item = {"face_id": face_number, "bbox": [x1, y1, x2, y2],
                    "face_width_px": face_w, "face_height_px": face_h,
                    "face_pixel_count": face_pixels,
                    "face_megapixels": round(face_pixels / 1_000_000.0, 5),
                    "face_resolution": face_resolution}
            if face_w < BLUR_MIN_FACE_PX or face_h < BLUR_MIN_FACE_PX:
                item.update({"category": "low_resolution", "is_blurred": False,
                             "blur_score": None, "blur_level_percent": None,
                             "blur_level_label": "unreliable_due_to_low_resolution",
                             "blur_description": "Face crop is too small for a reliable blur estimate. Upload a larger image or crop closer to the face.",
                             "details": {"reason": "face_too_small_for_reliable_blur_measurement"}})
            else:
                crop = frame_cv2[y1:y2, x1:x2]
                score, details = compute_blur_score(crop)
                ey1, ey2 = int(round(face_h*.18)), int(round(face_h*.62))
                ex1, ex2 = int(round(face_w*.08)), int(round(face_w*.92))
                eye_crop = crop[max(0,ey1):min(face_h,ey2), max(0,ex1):min(face_w,ex2)]
                eye_score, eye_details = (compute_blur_score(eye_crop)
                    if eye_crop.size and min(eye_crop.shape[:2]) >= 16
                    else (None, {"decision":"uncertain", "reason":"eye_region_too_small"}))
                eye_w = max(0, min(face_w,ex2)-max(0,ex1))
                eye_h = max(0, min(face_h,ey2)-max(0,ey1))
                eye_resolution = _resolution_metadata(eye_w, eye_h)
                face_is_blurred = (details.get("decision") == "blurred"
                                   and isinstance(score,(int,float)) and score >= BLUR_SCORE_THRESHOLD)
                if min(face_w, face_h) < BLUR_FACE_LOW_RES:
                    category, is_blurred = "low_resolution", False
                    reason = f"Detected face is only {face_w}x{face_h}px; blur confidence is limited."
                elif face_is_blurred:
                    category, is_blurred = "blurred", True
                    reason = "Multiple sharpness signals indicate blur in the face crop."
                elif details.get("decision") == "not_blurry":
                    category, is_blurred = "not_blurry", False
                    reason = "Face crop contains sufficient sharpness evidence."
                else:
                    category, is_blurred = "blur_uncertain", False
                    reason = "Sharpness signals disagree or image contrast is insufficient."
                if category == "low_resolution":
                    level = {"blur_level_percent": None,
                             "blur_level_label": "unreliable_due_to_low_resolution",
                             "blur_description": "Blur percentage is withheld because the face crop is below the recommended pixel size."}
                else:
                    level = _blur_level_metadata(score, details.get("decision"), face_resolution)
                item.update({"category": category, "is_blurred": is_blurred,
                             "blur_score": score, **level, "reason": reason,
                             "details": {**details,
                                 "source_image_resolution": image_resolution,
                                 "face_resolution": face_resolution,
                                 "eye_region_resolution": eye_resolution,
                                 "eye_region_pixel_dimensions": {"width_px": eye_w, "height_px": eye_h},
                                 "resolution_warning": reason if category == "low_resolution" else None},
                             "eye_region_blur_score": eye_score,
                             "eye_region_details": eye_details})
            face_results.append(item)
        except Exception as exc:
            logger.warning(f"Blur analysis failed for face #{face_number}: {exc!r}")
            face_results.append({"face_id": face_number, "category":"blur_uncertain",
                                 "is_blurred":False, "blur_score":None,
                                 "blur_level_percent":None,
                                 "blur_description":"Blur could not be measured for this face.",
                                 "details":{"reason":"blur_analysis_exception", "error":str(exc)}})

    blurred_faces = [f for f in face_results if f.get("category") == "blurred"]
    low_res_faces = [f for f in face_results if f.get("category") == "low_resolution"]
    uncertain_faces = [f for f in face_results if f.get("category") == "blur_uncertain"]
    scores = [f["blur_score"] for f in face_results if isinstance(f.get("blur_score"),(int,float))]
    if blurred_faces: category = "blurred"
    elif low_res_faces: category = "low_resolution"
    elif uncertain_faces: category = "blur_uncertain"
    else: category = "not_blurry"
    primary_score = max(scores) if scores else None
    if category == "low_resolution":
        level = {"blur_level_percent": None, "blur_level_label":"unreliable_due_to_low_resolution",
                 "blur_description":"Blur percentage is withheld because the detected face has too few original pixels. See face_resolution for exact dimensions."}
    else:
        decision = "blurred" if category == "blurred" else ("not_blurry" if category == "not_blurry" else "uncertain")
        level = _blur_level_metadata(primary_score, decision, image_resolution)
    return {"is_blurred": bool(blurred_faces), "category": category,
            "blur_score": primary_score, **level,
            "image_resolution": image_resolution,
            "details": {"global_image_blur_score": global_score,
                        "global_image_metrics": global_details,
                        "source_image_resolution": image_resolution,
                        "faces_analyzed": len(face_results),
                        "blurred_face_count": len(blurred_faces),
                        "low_resolution_face_count": len(low_res_faces),
                        "uncertain_face_count": len(uncertain_faces),
                        "blur_levels": {"0": "Sharp / no strong blur evidence",
                                        "20": "Very slight blur", "40": "Mild blur",
                                        "60": "Moderate blur", "80": "Heavy blur",
                                        "100": "Very heavy blur"},
                        "aggregation": "any clearly blurred face marks image blurred; low resolution is not treated as blur"},
            "faces": face_results, "uncertain": bool(low_res_faces or uncertain_faces),
            "score_kind": "heuristic_0_100_not_probability", "blur_probability": None,
            "blur_level_scale_note": "20/40/60/80/100 are heuristic severity bands; they are not the percentage of pixels blurred or a probability.",
            "probability_note": "Blur level is a heuristic severity estimate, not a calibrated probability."}


def _check_whole_image_blur(frame_cv2: np.ndarray) -> dict:
    """Whole-image blur fallback with regional consensus and pixel diagnostics."""
    if frame_cv2 is None or not isinstance(frame_cv2, np.ndarray) or frame_cv2.size == 0:
        return {"is_blurred":False, "category":"blur_uncertain", "blur_score":None,
                "blur_level_percent":None, "blur_level_label":"unavailable",
                "blur_description":"No image pixels were available for analysis.",
                "image_resolution":_resolution_metadata(0,0),
                "details":{"reason":"empty_image"}, "source":"whole_image_fallback", "uncertain":True}
    h,w = frame_cv2.shape[:2]
    image_resolution = _resolution_metadata(w,h)
    regions = [("center", frame_cv2[int(h*.15):max(int(h*.85),1), int(w*.15):max(int(w*.85),1)])]
    for name,(ya,yb,xa,xb) in {"top_left":(0,.65,0,.65), "top_right":(0,.65,.35,1),
                               "bottom_left":(.35,1,0,.65), "bottom_right":(.35,1,.35,1)}.items():
        regions.append((name,frame_cv2[int(h*ya):max(int(h*yb),1), int(w*xa):max(int(w*xb),1)]))
    measured=[]
    for name,region in regions:
        if region is None or region.size == 0 or min(region.shape[:2]) < 16: continue
        score,details=compute_blur_score(region)
        measured.append({"region":name,"pixel_dimensions":{"width_px":int(region.shape[1]),"height_px":int(region.shape[0]),"pixel_count":int(region.shape[0]*region.shape[1])},
                         "blur_score":score,"decision":details.get("decision"),"details":details})
    blurred=[r for r in measured if r["decision"]=="blurred"]
    sharp=[r for r in measured if r["decision"]=="not_blurry"]
    if len(blurred)>=2 and len(blurred)>len(sharp): category,is_blurred="blurred",True
    elif len(sharp)>=2 and len(sharp)>=len(blurred): category,is_blurred="not_blurry",False
    else: category,is_blurred="blur_uncertain",False
    scores=[r["blur_score"] for r in measured]
    score=round(float(np.median(scores)),2) if scores else None
    level=_blur_level_metadata(score, "blurred" if category=="blurred" else ("not_blurry" if category=="not_blurry" else "uncertain"), image_resolution)
    if category=="not_blurry" and score is not None and score<=42:
        level=_blur_level_metadata(score,"not_blurry",image_resolution)
    return {"is_blurred":is_blurred,"category":category,"blur_score":score,**level,
            "image_resolution":image_resolution,
            "details":{"source_image_resolution":image_resolution,"region_count":len(measured),
                       "blurred_region_count":len(blurred),"sharp_region_count":len(sharp),
                       "regions":measured,"decision_rule":"regional consensus; uncertain when signals disagree"},
            "source":"whole_image_fallback","uncertain":category=="blur_uncertain",
            "score_kind":"heuristic_0_100_not_probability","blur_probability":None,
            "blur_level_scale_note":"20/40/60/80/100 are heuristic severity bands; they are not the percentage of pixels blurred or a probability.",
            "probability_note":"Blur level is a heuristic severity estimate, not a calibrated probability."}


def detect_framing(frame_cv2: np.ndarray, bbox_, valid_indices: list) -> dict:
    """
    Passport-photo framing / composition check.
    MTCNN bbox spans approximately forehead→chin. We check:
      - Top margin  < 6%  → hair/forehead cut off
      - Bottom margin < 8% → neck/chin cut off
      - Face height > 72% of image → too zoomed in
      - Face width  > 65% of image → too zoomed in
    """
    if not valid_indices or bbox_ is None:
        return {"is_focused": False, "issues": [], "details": {}}

    img_h, img_w = frame_cv2.shape[:2]
    b = bbox_[valid_indices[0]]
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

    top_margin    = y1 / img_h
    bottom_margin = (img_h - y2) / img_h
    face_h_ratio  = (y2 - y1) / img_h
    face_w_ratio  = (x2 - x1) / img_w

    issues = []
    if top_margin < FRAMING_TOP_MARGIN_MIN:
        issues.append("forehead or hair cut off — more space needed above head")
    if bottom_margin < FRAMING_BOTTOM_MARGIN_MIN:
        issues.append("neck or chin cut off — more space needed below face")
    if face_h_ratio > FRAMING_FACE_H_MAX:
        issues.append(f"face fills {face_h_ratio:.0%} of image height — too zoomed in")
    if face_w_ratio > FRAMING_FACE_W_MAX:
        issues.append(f"face fills {face_w_ratio:.0%} of image width — too zoomed in")

    return {
        "is_focused": len(issues) > 0,
        "issues":     issues,
        "details": {
            "top_margin":    round(top_margin,    4),
            "bottom_margin": round(bottom_margin, 4),
            "face_h_ratio":  round(face_h_ratio,  4),
            "face_w_ratio":  round(face_w_ratio,  4),
        },
    }


# ═══════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════════════════════════════

def _get_upscaled_face_ctx(frame_cv2: np.ndarray, bbox, min_face_dim: int = 80):
    """For small detected faces, extract and Lanczos-upscale the face region.

    Returns (frame_to_use, bbox_to_use): either the original frame+bbox (face is
    already large enough) or an upscaled face-crop with remapped bbox coordinates.
    All subsequent detectors (YOLO, classifier, blur) operate on the returned pair,
    getting a larger, cleaner input without changing the pipeline structure.
    """
    try:
        ih, iw = frame_cv2.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        if min(fw, fh) >= min_face_dim:
            return frame_cv2, bbox
        pad = int(max(fw, fh) * 0.25)
        cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
        cx2 = min(iw, x2 + pad); cy2 = min(ih, y2 + pad)
        crop = frame_cv2[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return frame_cv2, bbox
        ch, cw = crop.shape[:2]
        scale = max(min_face_dim / max(min(ch, cw), 1), 2.0)
        new_w, new_h = int(round(cw * scale)), int(round(ch * scale))
        upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        new_bbox = [
            int(round((x1 - cx1) * scale)),
            int(round((y1 - cy1) * scale)),
            int(round((x2 - cx1) * scale)),
            int(round((y2 - cy1) * scale)),
        ]
        return upscaled, new_bbox
    except Exception:
        return frame_cv2, bbox


def _upscale_if_needed(frame_cv2: np.ndarray, min_dim: int = 300) -> tuple:
    """Upscale images whose shortest side is below min_dim using Lanczos.
    Returns (upscaled_frame, pil_image, was_upscaled, scale_factor)."""
    h, w = frame_cv2.shape[:2]
    short = min(h, w)
    if short >= min_dim:
        return frame_cv2, None, False, 1.0
    scale = min_dim / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    upscaled = cv2.resize(frame_cv2, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    pil_up = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
    return upscaled, pil_up, True, scale


def run_full_validation(
    pil_image: Image.Image,
    frame_cv2: np.ndarray,
) -> dict:
    """
    Priority chain:
      dimension_guard -> upscale_if_needed -> full_image_prescreen
      -> no_human -> irrelevant -> multiple_person
      -> no_landmarks -> sunglasses -> eyes_closed -> pose
    """
    img_h, img_w = frame_cv2.shape[:2]
    t0           = time.time()

    # Step 0: dimension guard
    dim_error = validate_image_dimensions(frame_cv2)
    if dim_error:
        return {"category": "irrelevant", "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [],
                "validation": {"dimension_error": dim_error},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 0b: upscale tiny images before ALL detection.
    # Detectors (MTCNN, YOLO, MediaPipe) all perform better on larger inputs.
    frame_cv2_up, pil_up, _was_upscaled, _up_scale = _upscale_if_needed(frame_cv2, min_dim=300)
    if _was_upscaled:
        frame_cv2 = frame_cv2_up
        pil_image = pil_up
        img_h, img_w = frame_cv2.shape[:2]
        logger.debug("Pre-screen: upscaled image to %dx%d (scale=%.2f)", img_w, img_h, _up_scale)

    # Step 1: downscale for all MediaPipe operations
    proc_frame, proc_scale = _resize_for_processing(frame_cv2)
    proc_h, proc_w         = proc_frame.shape[:2]

    # Step 2: MTCNN on original full-resolution image
    bbox_, prob_, mtcnn_lms_ = mtcnn.detect(pil_image, landmarks=True)

    # Step 3: confidence + size filter
    valid_indices: list = []
    if bbox_ is not None and prob_ is not None:
        for i, p in enumerate(prob_):
            if p is None or float(p) < MTCNN_CONFIDENCE_THRESHOLD:
                continue
            fw = abs(float(bbox_[i][2]) - float(bbox_[i][0]))
            fh = abs(float(bbox_[i][3]) - float(bbox_[i][1]))
            if fw < MIN_FACE_SIZE_PX or fh < MIN_FACE_SIZE_PX:
                logger.debug(f"Skipping small face #{i+1}: {fw:.0f}x{fh:.0f}px")
                continue
            valid_indices.append(i)

    valid_bboxes = [bbox_[i] for i in valid_indices] if valid_indices else []
    valid_probs  = [prob_[i]  for i in valid_indices] if valid_indices else []

    # Early, landmark-independent sunglasses pass. This runs before framing and
    # FaceMesh early returns so clipped/occluded faces can still be categorized.
    # For small faces, use an upscaled face crop so all detectors get a larger input.
    pre_sunglasses = []
    for pre_idx, pre_bbox in enumerate(valid_bboxes):
        _sg_frame, _sg_bbox = _get_upscaled_face_ctx(frame_cv2, pre_bbox)
        pre_result = detect_sunglasses(
            proc_frame, None, frame_cv2_orig=_sg_frame, orig_bbox=_sg_bbox
        )
        pre_sunglasses.append(pre_result)
        logger.debug(
            "Sunglasses precheck | face=%s detected=%s reason=%s details=%s",
            pre_idx + 1, pre_result.get("detected"), pre_result.get("reason"),
            pre_result.get("details", {})
        )
    pre_sunglasses_any = any(r.get("detected", False) for r in pre_sunglasses)

    pre_headwear = []
    for _hw_idx, _hw_bbox in enumerate(valid_bboxes):
        _hw_frame, _hw_up_bbox = _get_upscaled_face_ctx(frame_cv2, _hw_bbox)
        _hw_result = detect_headwear(_hw_frame, _hw_up_bbox)
        pre_headwear.append(_hw_result)
        logger.debug("Headwear precheck | face=%s detected=%s kind=%s",
                    _hw_idx + 1, _hw_result.get("detected"), _hw_result.get("kind"))
    pre_headwear_any = any(r.get("detected", False) for r in pre_headwear)

    # Pre-gender: run on every detected face bbox so gender is available
    # even in early-return paths (multiple_person, blurry, etc.)
    pre_gender = [detect_gender(frame_cv2, b) for b in valid_bboxes]

    # ConvNeXt also predicts headwear (cap/hijab/turban) — supplement YOLO results
    for _gi, _gres in enumerate(pre_gender):
        _cnx_hw = _gres.get("headwear")
        if _cnx_hw and _gi < len(pre_headwear):
            if not pre_headwear[_gi].get("detected"):
                pre_headwear[_gi]["detected"]     = True
                pre_headwear[_gi]["kind"]          = _cnx_hw
                pre_headwear[_gi]["source"]        = "convnext"
                pre_headwear[_gi]["convnext_prob"] = (_gres.get("all_probs") or {}).get(_cnx_hw)
            else:
                # YOLO only knows generic hat/helmet classes, so it cannot tell a doctor cap
                # from a turban. When ConvNeXt (trained on cap/hijab/turban) is confident and
                # disagrees, trust ConvNeXt for the *kind*.
                _p = float((_gres.get("all_probs") or {}).get(_cnx_hw) or 0.0)
                _yk = (pre_headwear[_gi].get("kind") or "").lower()
                if _p >= 0.60 and _yk != _cnx_hw:
                    pre_headwear[_gi]["yolo_kind"]     = _yk
                    pre_headwear[_gi]["kind"]          = _cnx_hw
                    pre_headwear[_gi]["source"]        = "convnext"
                    pre_headwear[_gi]["convnext_prob"] = _p
    pre_headwear_any = any(r.get("detected", False) for r in pre_headwear)

    # Module A: hair color per face
    pre_hair = [detect_hair_color(frame_cv2, b) for b in valid_bboxes]

    # Module B: spectacles per face
    pre_specs = [detect_spectacles(frame_cv2, b) for b in valid_bboxes]

    # Module C: combine into appearance label per face
    pre_appearance = [
        _appearance_for_face(
            frame_cv2, valid_bboxes[_ai],
            pre_gender[_ai]   if _ai < len(pre_gender)   else {},
            pre_hair[_ai]     if _ai < len(pre_hair)     else {},
            pre_specs[_ai]    if _ai < len(pre_specs)    else {},
            pre_headwear[_ai] if _ai < len(pre_headwear) else {},
        )
        for _ai in range(len(valid_bboxes))
    ]
    logger.debug("Appearance labels: %s", [a.get("appearance_label") for a in pre_appearance])

    # Step 4: no-human / irrelevant
    no_human = detect_no_human(valid_bboxes, valid_probs)
    if not no_human["is_human"]:
        # Blurry images can prevent face detection entirely — check before no_human/irrelevant
        whole_blur = _check_whole_image_blur(frame_cv2)
        if whole_blur["is_blurred"]:
            return {"category": "blurred", "primary_category": "blurred",
                    "categories": ["blurred"], "is_valid": False,
                    "total_faces": 0, "frontal_count": 0, "side_count": 0, "faces": [],
                    "blur": whole_blur,
                    "validation": {"no_human": no_human, "blur": whole_blur,
                                   "pose_unavailable": "No face landmarks were detected; pose cannot be inferred reliably."},
                    "processing_ms": round((time.time() - t0) * 1000, 2)}
        # Full-image pre-screen: check ALL categories on the whole image before
        # giving up with no_human. Catches helmets, goggles and sunglasses that
        # completely hide or are cropped away from the face.
        fi_hw = _detect_headwear_yolo_fullimage(frame_cv2)

        # Full-image sunglasses/goggles YOLO scan (pass whole image as bbox).
        _full_bbox = [0, 0, img_w, img_h]
        fi_sg = _detect_sunglasses_yolo(frame_cv2, _full_bbox)
        fi_sg_hit = bool(fi_sg and (fi_sg.get("yolo_detected") or fi_sg.get("yolo_soft")))

        if fi_sg_hit and fi_sg.get("yolo_detected"):
            sg_kind = "goggles" if (fi_sg.get("yolo_goggles", 0) > fi_sg.get("yolo_sunglasses", 0)) else "sunglasses"
            logger.debug("Full-image sunglasses scan | kind=%s sg=%.3f goggles=%.3f",
                        sg_kind, fi_sg.get("yolo_sunglasses", 0), fi_sg.get("yolo_goggles", 0))
            return {
                "category":          "sunglasses",
                "primary_category":  "sunglasses",
                "categories":        ["sunglasses"],
                "is_valid":          False,
                "total_faces":       0,
                "frontal_count":     0,
                "side_count":        0,
                "faces":             [],
                "sunglasses_detected": True,
                "headwear_detected": False,
                "blur":              whole_blur,
                "validation":        {"no_human": no_human, "full_image_sunglasses": fi_sg},
                "processing_ms":     round((time.time() - t0) * 1000, 2),
            }

        if fi_hw and (fi_hw.get("yolo_detected") or fi_hw.get("yolo_soft")):
            hw_kind = fi_hw.get("yolo_best_kind") or "helmet"
            logger.debug("Full-image headwear scan | kind=%s helmet=%.3f cap=%.3f",
                        hw_kind, fi_hw.get("yolo_helmet_conf", 0), fi_hw.get("yolo_cap_conf", 0))
            return {
                "category":         "headwear",
                "primary_category": "headwear",
                "categories":       ["headwear"],
                "is_valid":         False,
                "total_faces":      0,
                "frontal_count":    0,
                "side_count":       0,
                "faces":            [],
                "headwear_detected": True,
                "headwear_kind":    hw_kind,
                "headwear_detail":  fi_hw,
                "blur":             whole_blur,
                "sunglasses_detected": False,
                "validation":       {"no_human": no_human, "full_image_headwear": fi_hw},
                "processing_ms":    round((time.time() - t0) * 1000, 2),
            }

        irr      = detect_irrelevant_image(valid_bboxes, frame_cv2)
        category = "irrelevant" if irr["is_irrelevant"] else "no_human"
        # If the whole image is inconclusive, preserve that fact as a secondary
        # quality label. This makes API responses consistent with the face path.
        secondary_categories = (["blur_uncertain"] if whole_blur.get("category") == "blur_uncertain" else [])
        return {"category": category, "primary_category": category,
                "categories": secondary_categories + [category],
                "is_valid": False, "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [],
                "blur": whole_blur,
                "validation": {"no_human": no_human, "irrelevant": irr, "blur": whole_blur},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 5: quality check is an attribute, not an early-exit classification.
    # Continue to pose / person classification even when the face is blurry.
    blur_result = _detect_blur_on_face(frame_cv2, bbox_, valid_indices)
    quality_categories = []
    if blur_result.get("is_blurred"):
        quality_categories.append("blurred")
    elif blur_result.get("category") in ("low_resolution", "blur_uncertain"):
        quality_categories.append(blur_result["category"])
    logger.debug(f"Blur | score={blur_result.get('blur_score', '?')} "
                f"category={blur_result.get('category')} "
                f"is_blurred={blur_result.get('is_blurred')} "
                f"details={blur_result.get('details', {})}")

    # Step 5b: multiple-person check (operates on downscaled proc_frame)
    person_check = detect_persons_and_size(proc_frame)
    if not person_check["valid"]:
        categories = quality_categories + [person_check["category"]]
        if pre_sunglasses_any:
            categories.append("sunglasses")
        if pre_headwear_any:
            categories.append("headwear")
        _pc_gender0 = pre_gender[0] if pre_gender else {}
        if _pc_gender0.get("gender"):
            categories.append(_pc_gender0["gender"])
        return {"category": ("sunglasses" if pre_sunglasses_any else person_check["category"]),
                "categories": list(dict.fromkeys(categories)),
                "primary_category": ("sunglasses" if pre_sunglasses_any else person_check["category"]),
                "sunglasses_detected": pre_sunglasses_any,
                "gender": _pc_gender0.get("gender"),
                "gender_confidence": _pc_gender0.get("confidence"),
                "faces": [{"face_id": i + 1, "bbox": [round(float(v), 2) for v in b],
                           "sunglasses": r, "gender": pre_gender[i] if i < len(pre_gender) else {}, "pose": "Unknown",
                           "appearance": pre_appearance[i] if i < len(pre_appearance) else {},
                           "headwear": pre_headwear[i] if i < len(pre_headwear) else {},
                           "hair": pre_hair[i] if i < len(pre_hair) else {},
                           "specs": pre_specs[i] if i < len(pre_specs) else {}}
                          for i, (b, r) in enumerate(zip(valid_bboxes, pre_sunglasses))],
                "total_faces": person_check["total_faces"],
                "frontal_count": 0, "side_count": 0,
                "blur": blur_result,
                "validation": {"no_human": no_human, "person_check": person_check,
                               "blur": blur_result},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 5c: framing check — passport composition (hair, neck, zoom)
    framing_result = detect_framing(frame_cv2, bbox_, valid_indices)
    if framing_result["is_focused"]:
        categories = quality_categories + ["focused"]
        if pre_sunglasses_any:
            categories.append("sunglasses")
        if pre_headwear_any:
            categories.append("headwear")
        _fr_g = pre_gender[0].get("gender") if pre_gender else None
        if _fr_g:
            categories.append(_fr_g)
        return {
            "category":      ("sunglasses" if pre_sunglasses_any else "focused"),
            "primary_category": ("sunglasses" if pre_sunglasses_any else "focused"),
            "categories":    list(dict.fromkeys(categories)),
            "sunglasses_detected": pre_sunglasses_any,
            "is_valid":      False,
            "total_faces":   len(valid_indices),
            "frontal_count": 0, "side_count": 0,
            "gender": pre_gender[0].get("gender") if pre_gender else None,
            "gender_confidence": pre_gender[0].get("confidence") if pre_gender else None,
            "faces": [{"face_id": i + 1, "bbox": [round(float(v), 2) for v in b],
                       "sunglasses": r, "gender": pre_gender[i] if i < len(pre_gender) else {}, "pose": "Unknown",
                           "appearance": pre_appearance[i] if i < len(pre_appearance) else {},
                           "headwear": pre_headwear[i] if i < len(pre_headwear) else {},
                           "hair": pre_hair[i] if i < len(pre_hair) else {},
                           "specs": pre_specs[i] if i < len(pre_specs) else {}}
                      for i, (b, r) in enumerate(zip(valid_bboxes, pre_sunglasses))],
            "blur":          blur_result,
            "blur_debug":    blur_result,
            "framing":       framing_result,
            "validation":    {"no_human": no_human, "person_check": person_check,
                              "framing": framing_result, "blur": blur_result},
            "processing_ms": round((time.time() - t0) * 1000, 2),
        }

    # Step 6: FaceMesh (proc_frame)
    face_mesh  = _get_face_mesh()
    frame_rgb  = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    mp_results = face_mesh.process(frame_rgb)
    mp_faces   = mp_results.multi_face_landmarks or []
    mp_count   = len(mp_faces)
    mtcnn_count = len(valid_indices)

    if mtcnn_count > 0 and mp_count == 0:
        categories = quality_categories + ["no_landmarks"]
        if pre_sunglasses_any:
            categories.append("sunglasses")
        _nl_g = pre_gender[0].get("gender") if pre_gender else None
        if _nl_g:
            categories.append(_nl_g)
        return {"category": ("sunglasses" if pre_sunglasses_any else "no_landmarks"),
                "primary_category": ("sunglasses" if pre_sunglasses_any else "no_landmarks"),
                "categories": list(dict.fromkeys(categories)),
                "sunglasses_detected": pre_sunglasses_any, "is_valid": False,
                "total_faces": len(valid_indices), "frontal_count": 0, "side_count": 0,
                "gender": pre_gender[0].get("gender") if pre_gender else None,
                "gender_confidence": pre_gender[0].get("confidence") if pre_gender else None,
                "faces": [{"face_id": i + 1, "bbox": [round(float(v), 2) for v in b],
                           "sunglasses": r, "gender": pre_gender[i] if i < len(pre_gender) else {}, "pose": "Unknown",
                           "appearance": pre_appearance[i] if i < len(pre_appearance) else {},
                           "headwear": pre_headwear[i] if i < len(pre_headwear) else {},
                           "hair": pre_hair[i] if i < len(pre_hair) else {},
                           "specs": pre_specs[i] if i < len(pre_specs) else {}}
                          for i, (b, r) in enumerate(zip(valid_bboxes, pre_sunglasses))],
                "blur": blur_result,
                "validation": {"no_human": no_human,
                               "blur": blur_result,
                               "reason": "FaceMesh failed (lighting/occlusion/resolution)"},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    if mp_count > 1:
        categories = quality_categories + ["multiple_person"]
        if pre_sunglasses_any:
            categories.append("sunglasses")
        if pre_headwear_any:
            categories.append("headwear")
        for _mg in pre_gender:
            _g_val = _mg.get("gender")
            if _g_val:
                categories.append(_g_val)
        _mp_primary = ("sunglasses" if pre_sunglasses_any else
                       ("headwear" if pre_headwear_any else "multiple_person"))
        return {"category": _mp_primary,
                "primary_category": _mp_primary,
                "categories": list(dict.fromkeys(categories)),
                "sunglasses_detected": pre_sunglasses_any, "is_valid": False,
                "total_faces": mp_count, "frontal_count": 0, "side_count": 0,
                "gender": pre_gender[0].get("gender") if pre_gender else None,
                "gender_confidence": pre_gender[0].get("confidence") if pre_gender else None,
                "faces": [{"face_id": i + 1, "bbox": [round(float(v), 2) for v in b],
                           "sunglasses": r, "gender": pre_gender[i] if i < len(pre_gender) else {}, "pose": "Unknown",
                           "appearance": pre_appearance[i] if i < len(pre_appearance) else {},
                           "headwear": pre_headwear[i] if i < len(pre_headwear) else {},
                           "hair": pre_hair[i] if i < len(pre_hair) else {},
                           "specs": pre_specs[i] if i < len(pre_specs) else {}}
                          for i, (b, r) in enumerate(zip(valid_bboxes, pre_sunglasses))],
                "blur": blur_result,
                "validation": {"no_human": no_human, "blur": blur_result,
                               "multi_person": {"mtcnn_faces": mtcnn_count,
                                                "mediapipe_faces": mp_count,
                                                "reason": "Multiple faces by FaceMesh"}},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 7: match bboxes -> landmarks
    matched_mp = _match_landmarks_to_faces(
        valid_bboxes, mp_faces, proc_w, proc_h, proc_scale
    )

    # Step 8: per-face analysis
    face_results: list = []
    all_poses:    list = []
    face_counter = 0   

    for local_idx, global_idx in enumerate(valid_indices):
        bbox      = bbox_[global_idx]
        prob      = float(prob_[global_idx])
        mtcnn_lms = mtcnn_lms_[global_idx]
        mp_face   = matched_mp[local_idx]

        if mtcnn_lms is None:
            logger.warning(f"MTCNN landmarks None for detection #{local_idx+1} — skip")
            continue

        face_counter += 1
        face_data = {
            "face_id":    face_counter,
            "confidence": round(prob, 4),
            "bbox":       [round(float(v), 2) for v in bbox],
        }

        angR = _np_angle(np.array(mtcnn_lms[0]), np.array(mtcnn_lms[1]),
                         np.array(mtcnn_lms[2]))
        angL = _np_angle(np.array(mtcnn_lms[1]), np.array(mtcnn_lms[0]),
                         np.array(mtcnn_lms[2]))
        face_data["angle_right_eye"] = round(angR, 2)
        face_data["angle_left_eye"]  = round(angL, 2)
        face_data["landmarks"] = {
            "left_eye":    [float(mtcnn_lms[0][0]), float(mtcnn_lms[0][1])],
            "right_eye":   [float(mtcnn_lms[1][0]), float(mtcnn_lms[1][1])],
            "nose":        [float(mtcnn_lms[2][0]), float(mtcnn_lms[2][1])],
            "left_mouth":  [float(mtcnn_lms[3][0]), float(mtcnn_lms[3][1])],
            "right_mouth": [float(mtcnn_lms[4][0]), float(mtcnn_lms[4][1])],
        }

        # Sunglasses: YOLO on original-res face crop, fallback to signal voting on proc_frame.
        # Use upscaled face context for small faces so all detectors get a larger input.
        _lm_sg_frame, _lm_sg_bbox = _get_upscaled_face_ctx(frame_cv2, bbox)
        landmark_sunglasses = detect_sunglasses(
            proc_frame, mp_face, frame_cv2_orig=_lm_sg_frame, orig_bbox=_lm_sg_bbox
        )
        # OR-combine the landmark-independent crop pass with the landmark-aware
        # pass; a later weak/negative eye-polygon result must not erase a crop hit.
        pre_sg = pre_sunglasses[local_idx] if local_idx < len(pre_sunglasses) else {}
        if pre_sg.get("detected") or landmark_sunglasses.get("detected"):
            chosen = landmark_sunglasses if landmark_sunglasses.get("detected") else pre_sg
            merged_details = dict(chosen.get("details") or {})
            merged_details["pre_landmark_crop_pass"] = pre_sg
            merged_details["landmark_aware_pass"] = landmark_sunglasses
            face_data["sunglasses"] = {
                "detected": True, "votes": max(int(pre_sg.get("votes", 0) or 0),
                                                 int(landmark_sunglasses.get("votes", 0) or 0)),
                "reason": "Detected by independent crop or landmark-aware pass",
                "details": merged_details,
            }
        else:
            face_data["sunglasses"] = landmark_sunglasses

        # Headwear: use pre-pass result (no landmarks needed — crops above face bbox)
        face_data["headwear"] = pre_headwear[local_idx] if local_idx < len(pre_headwear) else {}

        # Closed eyes: pass original bbox for size guard
        proc_bbox = [v * proc_scale for v in bbox] if bbox is not None else None
        face_data["eyes"] = detect_closed_eyes(
            mp_face, proc_w, proc_h,
            orig_bbox=bbox,
            proc_bbox=proc_bbox,
        )

        # Pose: proc-space dims for camera matrix
        yaw_data   = _get_mediapipe_angles(mp_face, proc_w, proc_h) if mp_face else None
        nose_ratio = _get_nose_visibility_ratio(mp_face)             if mp_face else 1.0
        pose, pose_signals = classify_pose_strong(angR, angL, yaw_data, nose_ratio)

        face_data["pose"]         = pose
        face_data["pose_signals"] = pose_signals
        face_data["gender"]       = detect_gender(frame_cv2, bbox)
        face_data["hair"]         = pre_hair[local_idx]       if local_idx < len(pre_hair)       else {}
        face_data["specs"]        = pre_specs[local_idx]      if local_idx < len(pre_specs)      else {}
        face_data["appearance"]   = pre_appearance[local_idx] if local_idx < len(pre_appearance) else {}
        all_poses.append(pose)
        face_results.append(face_data)

    # Step 9: guard
    if not face_results:
        categories = quality_categories + ["no_human"]
        return {"category": "no_human", "primary_category": "no_human",
                "categories": categories, "is_valid": False, "total_faces": 0,
                "frontal_count": 0, "side_count": 0, "faces": [], "blur": blur_result,
                "validation": {"no_human": no_human, "person_check": person_check,
                               "blur": blur_result},
                "processing_ms": round((time.time() - t0) * 1000, 2)}

    # Step 10: priority decision
    sunglasses_any  = any(f["sunglasses"].get("detected", False) for f in face_results)
    headwear_any    = any((f.get("headwear") or {}).get("detected", False) for f in face_results)
    eyes_closed_any = any(f["eyes"].get("eyes_closed",   False) for f in face_results)

    # Pose is an independent result: sunglasses must not hide the fact that
    # the head is turned sideways. Keep `category` as the legacy primary label,
    # but expose every applicable condition in `categories`.
    fc = all_poses.count("Frontal")
    sc = all_poses.count("Side")
    if fc > 0 and sc == 0:
        pose_category = "front"
    elif sc > 0 and fc == 0:
        pose_category = "side"
    else:
        pose_category = "mixed"

    if sunglasses_any:
        final_category = "sunglasses"
    elif eyes_closed_any:
        final_category = "eyes_closed"
    else:
        final_category = pose_category

    # blur_sunglasses_uncertain: image quality is degraded AND sunglasses not confirmed
    # AND there is actual evidence of sunglasses that blur may have suppressed.
    # Without the evidence check, every blurry image without sunglasses would match.
    def _sg_has_evidence(sg):
        if not sg or sg.get("detected"):
            return False
        if sg.get("status") == "low_resolution_uncertain":
            return True
        d = sg.get("details") or {}
        clf = d.get("classifier_prob")
        if clf is not None and clf >= 0.30:
            return True
        if d.get("yolo_soft"):
            return True
        ev = d.get("evidence") or {}
        if d.get("ev_score", 0) >= 2:
            return True
        return False

    blur_sg_uncertain = bool(
        quality_categories and not sunglasses_any and not headwear_any
        and any(_sg_has_evidence(f.get("sunglasses")) for f in face_results)
    )

    categories = list(quality_categories)
    categories.append(pose_category)
    if sunglasses_any:
        categories.append("sunglasses")
    if headwear_any:
        categories.append("headwear")
    if eyes_closed_any:
        categories.append("eyes_closed")
    if blur_sg_uncertain:
        categories.append("blur_sunglasses_uncertain")
        if final_category in ("blurred", "low_resolution", "blur_uncertain", pose_category):
            final_category = "blur_sunglasses_uncertain"
    # Add all unique genders detected across all faces
    for _fr in face_results:
        _g = (_fr.get("gender") or {}).get("gender")
        if _g and _g not in ("unknown", "uncertain"):
            categories.append(_g)
    # Preserve order while removing duplicates.
    categories = list(dict.fromkeys(categories))
    return {
        "category":      final_category,
        "primary_category": final_category,
        "pose_category": pose_category,
        "sunglasses_detected": sunglasses_any,
        "headwear_detected": headwear_any,
        "display_category": (f"{pose_category.lower()}+sunglasses" if sunglasses_any else
                             (f"{pose_category.lower()}+headwear" if headwear_any else pose_category.lower())),
        "categories":    categories,
        "is_valid":      (final_category == "front" and not quality_categories),
        "total_faces":   len(face_results),
        "frontal_count": all_poses.count("Frontal"),
        "side_count":    all_poses.count("Side"),
        "faces":         face_results,
        "gender":        (face_results[0].get("gender") or {}).get("gender") if face_results else None,
        "gender_confidence": (face_results[0].get("gender") or {}).get("confidence") if face_results else None,
        "gender_detail": face_results[0].get("gender") if face_results else None,
        "blur":          blur_result,
        "blur_debug":    blur_result,
        "validation":    {"no_human": no_human, "person_check": person_check,
                           "blur": blur_result},
        "processing_ms": round((time.time() - t0) * 1000, 2),
    }


# ═══════════════════════════════════════════════════════════════
# ANNOTATION
# ═══════════════════════════════════════════════════════════════

def _annotation_frame(frame_cv2: np.ndarray) -> np.ndarray:
    """Image to draw the debug overlay on.

    run_full_validation() enlarges small photos (short side < 300 px) before detecting, so every
    box / landmark it returns is in that enlarged space. Drawing them on the original small image
    put the overlay off to the side of the face. Draw on the same enlarged image instead."""
    up, _pil, was_up, _scale = _upscale_if_needed(frame_cv2, min_dim=300)
    return (up if was_up else frame_cv2).copy()


def draw_annotations(frame: np.ndarray, result: dict) -> np.ndarray:
    h_f, w_f  = frame.shape[:2]
    category  = result["category"]
    color_map = {
        "front": (0,255,0), "side": (0,0,255), "mixed": (255,165,0),
        "eyes_closed": (0,255,255), "multiple_person": (255,0,255),
        "sunglasses": (255,0,255), "no_human": (128,128,128),
        "irrelevant": (128,128,128), "no_landmarks": (100,100,200),
        "blurred": (0,0,255), "blur_uncertain": (0,165,255),
        "blur_sunglasses_uncertain": (0,100,255), "low_resolution": (0,165,255),
    }
    box_color = color_map.get(category, (200,200,200))

    def _cx(v): return max(0, min(int(v), w_f - 1))
    def _cy(v): return max(0, min(int(v), h_f - 1))

    for face in result.get("faces", []):
        bbox = face.get("bbox"); pose = face.get("pose","Unknown")
        eyes = face.get("eyes",{}); sigs = face.get("pose_signals",{})
        lms  = face.get("landmarks",{}); sg = face.get("sunglasses",{})

        # Defaults for faces without a bbox (landmarks still drawn at fallback size)
        _lm_r = 3; _ln_thick = 1

        if bbox:
            x1,y1,x2,y2 = _cx(bbox[0]),_cy(bbox[1]),_cx(bbox[2]),_cy(bbox[3])
            # Scale every drawing parameter to the face pixel width so small/blurry/
            # distant faces get thin marks instead of oversized blobs.
            # Formula: face_px=200 → box_thick=2, lm_r=5, font=0.57
            #          face_px=80  → box_thick=1, lm_r=2, font=0.35 (minimal, unobtrusive)
            #          face_px=400 → box_thick=3, lm_r=7, font=0.85 (clear on large face)
            _face_px   = max(1, x2 - x1)
            _box_thick = max(1, _face_px // 80)
            _lm_r      = max(1, _face_px // 40)
            _ln_thick  = max(1, _face_px // 90)
            _fscale    = max(0.35, min(0.85, _face_px / 350.0))
            _fthick    = max(1, _face_px // 150)
            _lh        = max(10, int(_fscale * 15))
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, _box_thick)
            # When the face touches the top edge, labels are placed below the
            # face box instead of stacking over the eyes/forehead.
            label_y = (y1 - _lh * 5) if y1 >= _lh * 5 + 5 else min(h_f - 8, y2 + _lh)
            label_x = min(x1, max(0, w_f - 180))
            cv2.putText(frame, f"Conf:{face.get('confidence',0):.2f}",
                        (label_x, max(12, label_y)), FONT, _fscale, (200,200,200), 1, cv2.LINE_AA)
            if sg.get("detected"):
                cv2.putText(frame, "SUNGLASSES", (label_x, min(h_f - 8, max(12, label_y + _lh))),
                            FONT, _fscale, (255,0,255), _fthick, cv2.LINE_AA)
            cv2.putText(frame,
                        f"{pose}|Score:{sigs.get('total_score','?')}|{sigs.get('confidence','?')}",
                        (label_x,min(h_f - 8, max(12, label_y + _lh * 2))), FONT, _fscale, box_color, _fthick, cv2.LINE_AA)
            if "yaw" in sigs:
                cv2.putText(frame,
                            f"Yaw:{sigs['yaw']} Pitch:{sigs['pitch']} Roll:{sigs['roll']}",
                            (label_x,min(h_f - 8, max(12, label_y + _lh * 3))), FONT, _fscale, (200,200,200), 1, cv2.LINE_AA)
            ear_text  = "EYES CLOSED" if eyes.get("eyes_closed") else f"EAR:{eyes.get('avg_ear','?')}"
            ear_color = (0,255,255) if eyes.get("eyes_closed") else (150,255,150)
            cv2.putText(frame, ear_text, (label_x,min(h_f - 8, max(12, label_y + _lh * 4))),
                        FONT, _fscale, ear_color, 1, cv2.LINE_AA)

        for key in ("left_eye","right_eye","nose","left_mouth","right_mouth"):
            pt = lms.get(key)
            if pt: cv2.circle(frame, (_cx(pt[0]),_cy(pt[1])), _lm_r, (0,255,255), -1)
        try:
            le,re,ns = lms["left_eye"],lms["right_eye"],lms["nose"]
            for p1,p2 in [(le,re),(le,ns),(re,ns)]:
                cv2.line(frame, (_cx(p1[0]),_cy(p1[1])), (_cx(p2[0]),_cy(p2[1])),
                         LINE_COLOR, _ln_thick)
        except Exception:
            pass

    # Put the status strip at the bottom: a top banner can physically cover
    # sunglasses/eyes when the source face is cropped near the upper edge.
    banner_h = min(34, h_f)
    banner_y = max(0, h_f - banner_h)
    cv2.rectangle(frame, (0, banner_y), (w_f - 1, h_f - 1), (30,30,30), -1)
    status_text = (
        f"Category:{category.upper()} | Faces:{result['total_faces']} | "
        f"F:{result['frontal_count']} S:{result['side_count']} | "
        f"{result['processing_ms']}ms"
    )
    cv2.putText(frame, status_text,
                (8, min(h_f - 6, banner_y + 23)), FONT, FONT_SCALE,
                (255,255,255), FONT_THICK, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════
# Response cleaning
# ═══════════════════════════════════════════════════════════════

def _clean_sunglasses(sg: dict) -> dict:
    if not sg:
        return sg
    d = sg.get("details") or {}
    return {
        "detected": sg.get("detected"),
        "reason":   sg.get("reason"),
        "details": {
            "yolo_detected":      d.get("yolo_detected"),
            "yolo_sunglasses":    d.get("yolo_sunglasses"),
            "yolo_goggles":       d.get("yolo_goggles"),
            "yolo_clear_glasses": d.get("yolo_clear_glasses"),
            "classifier_prob":    d.get("classifier_prob"),
        },
    }


def _clean_blur(b: dict) -> dict:
    if not b:
        return b
    return {
        "is_blurred":       b.get("is_blurred"),
        "blur_score":       b.get("blur_score"),
        "blur_level":       b.get("blur_level_label"),
        "blur_level_pct":   b.get("blur_level_percent"),
        "blur_description": b.get("blur_description"),
    }


def _clean_face(f: dict) -> dict:
    g = f.get("gender") or {}
    return {
        "face_id":           f.get("face_id"),
        "bbox":              f.get("bbox"),
        "pose":              f.get("pose"),
        "gender":            g.get("gender"),
        "gender_status":     g.get("status", "not_run"),
        "gender_reason":     g.get("reason"),
        "gender_confidence": g.get("confidence"),
        "gender_probabilities": g.get("probs"),
        "gender_all_probs":  g.get("all_probs"),
        "gender_model":      g.get("model"),
        "gender_input_preprocessing": g.get("input_preprocessing"),
        "gender_crop_bbox":  g.get("crop_bbox"),
        "sunglasses":        _clean_sunglasses(f.get("sunglasses") or {}),
        "headwear":          _clean_headwear(f.get("headwear") or {}),
        "blur_score":        f.get("blur_score"),
        "blur_level":        f.get("blur_level_label"),
        "hair":              f.get("hair") or {},
        "specs":             f.get("specs") or {},
        "appearance":        f.get("appearance") or {},
    }


_APPEARANCE_ORDER = ["black_hair", "specs_black_hair", "white_hair", "specs_white_hair",
                     "doctor_cap", "doctor_cap_specs", "turban", "turban_specs"]
_APPEARANCE_TITLES = {
    "black_hair":       "Black hair",
    "specs_black_hair": "Spectacles & black hair",
    "white_hair":       "White hair",
    "specs_white_hair": "Spectacles & white hair",
    "doctor_cap":       "Doctor cap",
    "doctor_cap_specs": "Doctor cap & spectacles",
    "turban":           "Turban",
    "turban_specs":     "Turban & spectacles",
}
# code -> (short title, plain-language explanation shown to the user)
_PROBLEM_INFO = {
    "goggles":         ("Sunglasses / goggles", "Sunglasses or dark goggles are covering the eyes. Please upload a photo without them."),
    "multiple_person": ("More than one person", "More than one face was found. Please upload a photo of one person only."),
    "side_face":       ("Side face", "The face is turned to the side. Please upload a front-facing photo."),
    "blur":            ("Blurred photo", "The photo is blurred. Please upload a sharper photo."),
    "low_resolution":  ("Low resolution", "The face is too small or the quality is too low. Please upload a larger, clearer photo."),
    "no_human":        ("No face found", "No human face was detected in this photo."),
    "irrelevant":      ("Not a person photo", "This image does not look like a photo of a person."),
    "no_landmarks":    ("Face not readable", "A face was found but its features could not be read (lighting, angle or something covering it)."),
    "focused":         ("Poor framing", "The face is too close, too far or cropped. Please retake with the whole head and shoulders visible."),
    "front":           ("Front face", "Front-facing photo of one person."),
}
_WARNING_TEXT = {
    "eyes_closed": "Eyes appear to be closed.",
    "focused":     "Face framing is off (too close or cropped). Head and shoulders should be visible.",
    "blurred":     "The photo looks blurred.",
    "blur_uncertain": "The photo may be slightly blurred.",
    "low_resolution": "The photo resolution is low.",
}
_ISSUE_ORDER = ["no_face", "focused", "goggles", "multiple_person", "blur", "low_resolution", "eyes_closed"]


# ── small helpers: JSON must always be valid and the same shape ──────────────
def _num(v, nd: int = 3):
    try:
        v = float(v)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, nd)
    except Exception:
        return None


def _clamp01(v) -> float:
    return max(0.0, min(1.0, float(v)))


def _json_safe(o):
    """numpy -> python, NaN/inf -> None, so jsonable_encoder / clients never choke."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _num(o, 4)
    return o


# ── SECTION 1 builders: a confidence for EVERY category ─────────────────────
def _scores_appearance(app: dict):
    probs = (app or {}).get("category_probs")
    if probs:
        return {k: _num(probs.get(k, 0.0)) for k in _APPEARANCE_ORDER}, "model_probability"
    lab = (app or {}).get("appearance_label")
    if not lab:                                   # appearance was not evaluated -> unknown, not 0%
        return {k: None for k in _APPEARANCE_ORDER}, "not_evaluated"
    return {k: (1.0 if k == lab else 0.0) for k in _APPEARANCE_ORDER}, "rule_based"


def _scores_gender(g: dict) -> dict:
    p = (g or {}).get("all_probs") or {}
    m, f = p.get("male"), p.get("female")
    try:
        s = float(m) + float(f)
        if s > 0:
            return {"male": _num(float(m) / s), "female": _num(float(f) / s)}
    except Exception:
        pass
    return {"male": None, "female": None}


def _scores_pose(faces_raw: list) -> dict:
    front_vals = []
    for f in faces_raw:
        sig = f.get("pose_signals") or {}
        tot, mx = sig.get("total_score"), sig.get("max_score")
        try:
            raw = 0.5 + 0.5 * max(-1.0, min(1.0, float(tot) / float(mx)))
        except Exception:
            continue
        if sig.get("pose_decision") == "Side":
            raw = min(raw, 0.49)
        elif sig.get("pose_decision") == "Frontal":
            raw = max(raw, 0.5)
        front_vals.append(raw)
    if not front_vals:
        return {"front": None, "side": None}
    front = min(front_vals)                      # the worst face decides
    return {"front": _num(front), "side": _num(1.0 - front)}


def _scores_checks(result: dict, faces_raw: list, blur: dict, cat: str, cats: list):
    """Problem categories. Every score is 0-1 and 0.5 is the decision point, so a score >= 0.5
    means the check fired. Returns (scores, flags)."""
    faces_n = result.get("total_faces", len(faces_raw))
    scores, flags = {}, {}

    # goggles / sunglasses
    best, flag = 0.0, bool(result.get("sunglasses_detected")) or cat == "sunglasses"
    for f in faces_raw:
        sg = f.get("sunglasses") or {}
        d = sg.get("details") or {}
        flag = flag or bool(sg.get("detected"))
        for key in ("yolo_sunglasses", "yolo_goggles", "classifier_prob"):
            try:
                best = max(best, float(d.get(key) or 0.0))
            except Exception:
                pass
    fi = ((result.get("validation") or {}).get("full_image_sunglasses")) or {}
    for key in ("yolo_sunglasses", "yolo_goggles"):
        try:
            best = max(best, float(fi.get(key) or 0.0))
        except Exception:
            pass
    scores["goggles"] = _clamp01(max(best, 0.5) if flag else min(best, 0.49))
    flags["goggles"] = flag

    # multiple people
    flag = faces_n > 1 or cat == "multiple_person"
    scores["multiple_person"] = 1.0 if flag else 0.0
    flags["multiple_person"] = flag

    # blur  (blur_score 0-100; the pipeline decides "blurred" at BLUR_SCORE_THRESHOLD -> 0.5)
    flag = bool(blur.get("is_blurred")) or cat == "blurred" or "blurred" in cats
    try:
        sc = _clamp01(0.5 * float(blur.get("blur_score")) / float(BLUR_SCORE_THRESHOLD))
    except Exception:
        sc = 0.0
    # confidence >= 0.40 → count as blur (show in detected)
    if sc >= 0.40:
        flag = True
    scores["blur"] = max(sc, 0.5) if flag else min(sc, 0.49)
    flags["blur"] = flag

    # low resolution (face size in the ORIGINAL photo; 50 px -> 0.5, 100 px or more -> 0)
    try:
        iw, ih = result.get("image_size") or (None, None)
        up = max(1.0, 300.0 / min(iw, ih)) if (iw and ih and min(iw, ih) < 300) else 1.0
        sides = [min(abs(b[2] - b[0]), abs(b[3] - b[1])) / up
                 for b in (f.get("bbox") for f in faces_raw) if b]
        sc = _clamp01(1.0 - min(sides) / (2.0 * BLUR_FACE_LOW_RES)) if sides else 0.0
    except Exception:
        sc = 0.0
    flag = cat == "low_resolution" or "low_resolution" in cats or (faces_n > 0 and sc >= 0.5)
    scores["low_resolution"] = max(sc, 0.5) if flag else min(sc, 0.49)
    flags["low_resolution"] = flag

    # eyes closed (EAR 0.15 -> 0.5)
    sc, flag = 0.0, False
    for f in faces_raw:
        e = f.get("eyes") or {}
        flag = flag or bool(e.get("eyes_closed"))
        try:
            sc = max(sc, _clamp01(1.0 - float(e.get("avg_ear")) / 0.30))
        except Exception:
            pass
    scores["eyes_closed"] = max(sc, 0.5) if flag else min(sc, 0.49)
    flags["eyes_closed"] = flag

    # poor framing (pipeline category "focused": face too close / cropped / badly placed)
    flag = cat == "focused" or "focused" in cats
    scores["focused"] = 1.0 if flag else 0.0
    flags["focused"] = flag

    # headwear on the head (cap / helmet / turban / hijab) - informational, not a rejection by itself
    hw_best, flag = 0.0, cat == "headwear" or "headwear" in cats
    for f in faces_raw:
        try:
            hp = ((f.get("appearance") or {}).get("head_probs") or {}).get("headwear") or {}
            if hp.get("none") is not None:
                hw_best = max(hw_best, 1.0 - float(hp["none"]))
            pr = (f.get("gender") or {}).get("all_probs") or {}
            for k in ("cap", "hijab", "turban"):
                hw_best = max(hw_best, float(pr.get(k) or 0.0))
            hd = f.get("headwear") or {}
            flag = flag or bool(hd.get("detected"))
            for k in ("yolo_helmet_conf", "yolo_cap_conf"):
                hw_best = max(hw_best, float((hd.get("details") or {}).get(k) or 0.0))
        except Exception:
            pass
    scores["headwear"] = max(hw_best, 0.5) if flag else min(hw_best, 0.49)
    flags["headwear"] = flag

    # no face at all
    flag = faces_n == 0
    scores["no_face"] = 1.0 if flag else 0.0
    flags["no_face"] = flag

    return {k: _num(v) for k, v in scores.items()}, flags


def _format_category_report(out: dict) -> str:
    """Plain-text version of the JSON: section 1 = confidence per category, section 2 = matched."""
    sc, mt = out.get("scores") or {}, out.get("matched") or {}
    on = set(mt.get("categories") or []) | set(mt.get("issues") or [])

    def bar(v):
        n = int(round((v or 0.0) * 20))
        return "#" * n + "." * (20 - n)

    def row(name, v):
        pct = "  n/a" if v is None else f"{v * 100:5.1f}%"
        return f"   {'>' if name in on else ' '} {name:18s} {pct}  {bar(v)}"

    L = ["", "=" * 62, " 1) CONFIDENCE PER CATEGORY   ( > = matched )", "=" * 62]
    for group, title in (("appearance", "Appearance"), ("gender", "Gender"), ("pose", "Pose"), ("checks", "Problem checks")):
        L.append(f" {title}")
        d = sc.get(group) or {}
        items = sorted(d.items(), key=lambda kv: -(kv[1] if kv[1] is not None else -1)) if group == "appearance" else d.items()
        L += [row(k, v) for k, v in items]
    L += ["", "=" * 62, " 2) MATCHED CATEGORY", "=" * 62]
    ap = mt.get("appearance")
    L.append(f"   Appearance : {_APPEARANCE_TITLES.get(ap, ap) or '-'}"
             + (f"  ({((sc.get('appearance') or {}).get(ap) or 0) * 100:.1f}%)" if ap and (sc.get('appearance') or {}).get(ap) is not None else ""))
    L.append(f"   Gender     : {mt.get('gender') or '-'}")
    L.append(f"   Pose       : {mt.get('pose') or '-'}")
    L.append(f"   Problems   : {', '.join(mt.get('issues') or []) or 'none'}")
    L.append(f"   All matched: {', '.join(mt.get('categories') or []) or '-'}")
    L.append(f"   RESULT     : {str(out.get('status', '')).upper()} - {(out.get('result') or {}).get('message', '')}")
    L.append("=" * 62)
    return "\n".join(L)


def _print_category_report(out: dict) -> None:
    if os.environ.get("REPORT_PRINT", "1") == "0":
        return
    try:
        print(_format_category_report(out), flush=True)
    except Exception as exc:                      # printing must never break the API
        logger.debug("report print failed: %r", exc)


def _clean_response(result: dict, detail: bool = False) -> dict:
    """
    User-facing JSON for /validate-face. The shape is always the same; a failing section
    degrades to nulls and is listed in "errors" instead of breaking the response.

      status   : "accepted" | "review" | "rejected"
      result   : code + title + one plain sentence
      scores   : SECTION 1 - a confidence (0-1) for EVERY category
                   appearance (8), gender (male/female), pose (front/side),
                   checks (goggles, multiple_person, blur, low_resolution, eyes_closed, no_face)
      matched  : SECTION 2 - which categories are TRUE for this image
      checks / warnings / appearance / face : readable detail
    Old keys (category, audit_category, is_valid, categories, quality, ...) are kept.
    ?detail=true adds the full per-face technical breakdown ("faces").
    """
    errors = []
    faces_raw = result.get("faces") or []
    f0        = faces_raw[0] if faces_raw else {}
    cat       = result.get("category") or ""
    cats      = list(dict.fromkeys(c for c in result.get("categories", []) if c))
    app       = f0.get("appearance") or {}
    hair_res  = f0.get("hair") or {}
    spec_res  = f0.get("specs") or {}
    hw_res    = f0.get("headwear") or {}
    gender0   = f0.get("gender") or {}
    try:
        blur = _clean_blur(result.get("blur") or {}) or {}
    except Exception:
        blur = {}
    label     = app.get("appearance_label")
    ml        = app.get("source") == "ml"
    faces_n   = result.get("total_faces", len(faces_raw))

    # ── result code (kept identical to the old `audit_category`) ───────────────
    if cat == "sunglasses":
        code = "goggles"
    elif cat in ("multiple_person", "no_human", "irrelevant", "no_landmarks", "low_resolution", "focused"):
        code = cat
    elif cat in ("side", "mixed"):
        code = "side_face"
    elif cat in ("blurred", "blur_uncertain", "blur_sunglasses_uncertain"):
        code = "blur"
    else:
        code = label or "front"

    # ── SECTION 1: confidence for every category ──────────────────────────────
    scores = {"appearance": {}, "gender": {}, "pose": {}, "checks": {}}
    flags, basis = {}, None
    try:
        scores["appearance"], basis = _scores_appearance(app)
    except Exception as exc:
        errors.append(f"appearance_scores: {exc!r}")
        scores["appearance"] = {k: None for k in _APPEARANCE_ORDER}
    try:
        scores["gender"] = _scores_gender(gender0)
    except Exception as exc:
        errors.append(f"gender_scores: {exc!r}"); scores["gender"] = {"male": None, "female": None}
    try:
        scores["pose"] = _scores_pose(faces_raw)
    except Exception as exc:
        errors.append(f"pose_scores: {exc!r}"); scores["pose"] = {"front": None, "side": None}
    try:
        scores["checks"], flags = _scores_checks(result, faces_raw, blur, cat, cats)
    except Exception as exc:
        errors.append(f"check_scores: {exc!r}")
        scores["checks"] = {k: None for k in _ISSUE_ORDER}
        flags = {k: False for k in _ISSUE_ORDER}

    # Override code to "blur" if blur flag fired (score >= 0.40) and no higher-priority issue
    if flags.get("blur") and code not in ("goggles", "multiple_person", "no_human", "irrelevant",
                                           "no_landmarks", "low_resolution", "focused", "side_face"):
        code = "blur"

    # ── SECTION 2: which categories are TRUE ──────────────────────────────────
    g = scores["gender"]
    p = scores["pose"]
    m_gender = None if g.get("male") is None else ("male" if g["male"] >= g["female"] else "female")
    m_pose = None if p.get("front") is None else ("front" if p["front"] >= 0.5 else "side")
    issues = [k for k in _ISSUE_ORDER if flags.get(k)]
    hw_flag = bool(flags.get("headwear"))
    matched_all = [x for x in (label, m_gender, m_pose) if x] + issues + (["headwear"] if hw_flag else [])
    matched = {
        "appearance": label,
        "gender":     m_gender,
        "pose":       m_pose,
        "issues":     issues,
        "categories": matched_all,
    }

    # ── warnings: things that did not decide the result but matter ────────────
    eyes_closed = bool(flags.get("eyes_closed"))
    warnings = []
    for key in ("eyes_closed", "focused", "blurred", "blur_uncertain", "low_resolution"):
        if (key == "eyes_closed" and eyes_closed) or (key != "eyes_closed" and (cat == key or key in cats)):
            if code != {"blurred": "blur", "blur_uncertain": "blur", "low_resolution": "low_resolution"}.get(key):
                warnings.append(_WARNING_TEXT[key])
    warnings = list(dict.fromkeys(warnings))

    # ── status + message ──────────────────────────────────────────────────────
    uncertain = bool(app.get("uncertain"))
    if code in _APPEARANCE_TITLES:
        title = _APPEARANCE_TITLES[code]
        if uncertain or warnings:
            status = "review"
            why = "Low confidence - please check manually." if uncertain else warnings[0]
            message = f"Looks like: {title}. {why}"
        else:
            status = "accepted"
            message = f"Accepted. Photo classified as: {title}."
    else:
        title, why = _PROBLEM_INFO.get(code, (code.replace("_", " ").capitalize(), ""))
        status, message = "rejected", why or title
        warnings = []

    # ── checks (one readable line per rule) ───────────────────────────────────
    has_face = faces_n > 0
    poses    = [f.get("pose") for f in faces_raw if f.get("pose")]
    sg_found = bool(flags.get("goggles"))
    blurred  = bool(flags.get("blur"))
    low_res  = bool(flags.get("low_resolution"))
    checks = {
        "one_person":   {"ok": faces_n == 1,
                         "note": f"{faces_n} face(s) found"},
        "front_facing": {"ok": (result.get("frontal_count", 0) >= 1 and result.get("side_count", 0) == 0) if has_face else None,
                         "note": ", ".join(poses) if poses else "not evaluated"},
        "eyes_open":    {"ok": (not eyes_closed) if has_face else None,
                         "note": "eyes closed" if eyes_closed else ("eyes open" if has_face else "not evaluated")},
        "no_sunglasses": {"ok": (not sg_found) if has_face else None,
                          "note": "sunglasses / goggles found" if sg_found else ("none found" if has_face else "not evaluated")},
        "sharp":        {"ok": not blurred,
                         "note": blur.get("blur_level") or ("blurred" if blurred else "ok")},
        "resolution":   {"ok": not low_res,
                         "note": "low resolution" if low_res else "ok"},
        "framing":      {"ok": not (cat == "focused" or "focused" in cats),
                         "note": "poorly framed" if (cat == "focused" or "focused" in cats) else "ok"},
    }

    # ── FIRST in the JSON: everything detected, each with its confidence ──────
    #    e.g. {"male": 0.999, "front": 1.0, "black_hair": 0.67}
    detected = {}
    if m_gender:
        detected[m_gender] = g.get(m_gender)
    if m_pose:
        detected[m_pose] = p.get(m_pose)
    if label:
        detected[label] = scores["appearance"].get(label)
    for k in issues:
        detected[k] = scores["checks"].get(k)
    if hw_flag:
        detected["headwear"] = scores["checks"].get("headwear")
    # checks that were measured (score > 0) but did NOT fire, e.g. {"blur": 0.245}
    not_triggered = {k: v for k, v in scores["checks"].items() if k not in detected and v}

    out = {
        "detected":            detected,
        "not_triggered":       not_triggered,
        "status":              status,
        "result":              {"code": code, "title": title, "message": message},
        "needs_manual_review": status == "review",
        "warnings":            warnings,

        # SECTION 1 - confidence of every category (0-1)
        "scores":  {**scores, "basis": {"appearance": basis,
                                        "gender": "model_probability",
                                        "pose": "rule_score",
                                        "checks": "rule_score (0.5 = decision point)"}},
        # SECTION 2 - which categories are true for this image
        "matched": matched,

        "appearance": {
            "label":       label,
            "title":       _APPEARANCE_TITLES.get(label),
            "hair_color":  (app.get("hair_color") if ml else hair_res.get("hair_color")),
            "specs":       (bool(app.get("specs_detected")) if ml else bool(spec_res.get("detected"))),
            "headwear":    (app.get("headwear_kind") if ml else (hw_res.get("kind") if hw_res.get("detected") else None)),
            "confidence":  app.get("confidence"),
            "uncertain":   app.get("uncertain"),
            "source":      app.get("source"),
        },
        "face": {
            "total":             faces_n,
            "frontal":           result.get("frontal_count", 0),
            "side":              result.get("side_count", 0),
            "pose":              poses[0] if len(poses) == 1 else (poses or None),
            "gender":            m_gender,
            "gender_confidence": (g.get(m_gender) if m_gender else None),
        },
        "checks": checks,
        "quality": {
            "blur_score": blur.get("blur_score"),
            "is_blurred": blurred,
            "low_res":    low_res,
            "sunglasses": sg_found,
        },

        # ── kept from the old format so existing clients keep working ─────────
        "category":       result.get("category"),
        "audit_category": code,
        "is_valid":       result.get("is_valid"),
        "categories":     cats,
        "processing_ms":  result.get("processing_ms"),
        "output_image":   result.get("output_image"),
        "filename":       result.get("filename"),
        "saved":          result.get("saved", False),
        "errors":         errors,
    }
    if detail:
        try:
            out["faces"] = [_clean_face(f) for f in faces_raw]
        except Exception as exc:
            out["errors"].append(f"faces_detail: {exc!r}")
    return _json_safe(out)

# ═══════════════════════════════════════════════════════════════
# API endpoints
# ═══════════════════════════════════════════════════════════════

# Allowed MIME types for /validate-face
_ALLOWED_IMAGE_MIMES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/bmp", "image/gif", "image/tiff", "image/tif",
    "image/x-bmp", "image/x-ms-bmp", "image/avif", "image/heic",
    "image/heif", "image/x-png", "image/pjpeg",
    "application/octet-stream",  # some clients send this for images
}
_ALLOWED_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
    ".tiff", ".tif", ".avif", ".heic", ".heif",
}

@app.post("/validate-face")
async def validate_face(
    file: UploadFile = File(...),
    save: bool       = Query(False),
    detail: bool     = Query(False, description="Include the full per-face technical breakdown ('faces')."),
    annotate: bool   = Query(False, description="Return annotated image as base64 JPEG in 'annotated_image_b64'."),
):
    # Accept by MIME or by file extension
    mime = (file.content_type or "").lower().split(";")[0].strip()
    fname = (file.filename or "").lower()
    ext   = os.path.splitext(fname)[1] if fname else ""
    if mime not in _ALLOWED_IMAGE_MIMES and ext not in _ALLOWED_IMAGE_EXTS:
        raise HTTPException(
            400,
            f"Unsupported file type '{mime}'. "
            "Supported: JPEG, PNG, WEBP, BMP, GIF, TIFF, AVIF, HEIC"
        )
    try:
        contents  = await file.read();  await file.close()
        try:
            pil_image = Image.open(io.BytesIO(contents))
            pil_image.verify()           # catch truncated / corrupt files
            pil_image = _exif_upright(Image.open(io.BytesIO(contents)))  # re-open after verify
        except Exception as img_err:
            raise HTTPException(400, f"Cannot read image file: {img_err}")
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        elif pil_image.mode == "L":
            pil_image = pil_image.convert("RGB")
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        result    = run_full_validation(pil_image, cv2_image)
        category  = result["category"]
        output_path = filename = annotated_b64 = None
        if save or annotate:
            annotated_frame = draw_annotations(_annotation_frame(cv2_image), result)
            if save:
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename    = f"{category}_{timestamp}.jpg"
                output_path = os.path.join(CATEGORY_FOLDERS.get(category, BASE_OUTPUT), filename)
                cv2.imwrite(output_path, annotated_frame)
                logger.info(f"Saved: {filename} -> {category}")
            if annotate:
                _, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                annotated_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        result.update({"output_image": output_path, "filename": filename,
                        "saved": save,
                        "annotated_image_b64": annotated_b64,
                        "success": result.get("is_valid", category == "front")})
        status_code = (CATEGORY_STATUS_MAP.get("blurred", 422)
                       if "blurred" in result.get("categories", [])
                       else CATEGORY_STATUS_MAP.get(category, 400))
        logger.info(f"Done | {category} | HTTP {status_code} | "
                    f"faces={result['total_faces']} | {result['processing_ms']}ms")
        return JSONResponse(status_code=status_code, content=jsonable_encoder(_clean_response(result, detail=detail)))
    except HTTPException:
        raise                      # keep intended 4xx responses (bad/corrupt upload)
    except Exception as exc:
        import traceback as _tb
        logger.error(f"Processing error: {exc}\n{_tb.format_exc()}")
        raise HTTPException(500, str(exc))


@app.get("/get-image")
async def get_image(filename: str = Query(...)):
    path = os.path.join(BASE_OUTPUT, filename)
    if not os.path.exists(path):
        for f in CATEGORY_FOLDERS.values():
            cand = os.path.join(f, filename)
            if os.path.exists(cand): path = cand; break
    if not os.path.exists(path):
        raise HTTPException(404, f"'{filename}' not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/get-latest")
async def get_latest(
    category: str = Query("front", description=(
        "front|side|mixed|eyes_closed|multiple_person|sunglasses|"
        "no_human|irrelevant|no_landmarks"
    ))
):
    folder = CATEGORY_FOLDERS.get(category)
    if not folder: raise HTTPException(400, f"Unknown category: {category}")
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    if not files: raise HTTPException(404, f"No images in '{category}'")
    return FileResponse(os.path.join(folder, files[-1]), media_type="image/jpeg")


@app.get("/list-outputs")
async def list_outputs():
    output = {}; total = 0
    for cat, folder in CATEGORY_FOLDERS.items():
        files = sorted(os.listdir(folder), reverse=True)
        output[cat] = {"count": len(files), "files": files}
        total += len(files)
    return JSONResponse({"success": True, "total": total, "categories": output})


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device),
            "gpu": torch.cuda.is_available(),
            "version": "2.5.0",
            "categories": list(CATEGORY_FOLDERS.keys())}


@app.get("/debug-gender")
async def debug_gender():
    result = {
        "model_path": _GENDER_MODEL_PATH,
        "model_file_exists": os.path.isfile(_GENDER_MODEL_PATH),
        "model_loaded": _gender_model is not None,
        "model_failed": _gender_model_failed,
        "model_error": _gender_model_error,
        "architecture": _gender_model_arch,
    }
    model = _load_gender_model()
    result["model_loaded"] = model is not None
    result["model_failed"] = _gender_model_failed
    result["model_error"] = _gender_model_error
    if model is not None:
        try:
            from PIL import Image as _PIL
            dummy_pil = _PIL.new("RGB", (224, 224), color=(128, 128, 128))
            test_result = model(dummy_pil)
            result["dummy_inference_smoke_test"] = test_result
            result["dummy_inference_passed"] = True
        except Exception as exc:
            result["dummy_inference_passed"] = False
            result["dummy_inference_error"] = f"{type(exc).__name__}: {exc}"
    return result


@app.post("/validate-folder")
async def validate_folder(
    folder_path: str = Query(...),
    save: bool       = Query(False),
):
    if not os.path.exists(folder_path):
        raise HTTPException(404, "Folder does not exist")
    if not os.path.isdir(folder_path):
        raise HTTPException(400, "Path is not a folder")
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not images: raise HTTPException(404, "No supported images found")

    batch_results: list = []; total_processed = 0; t0 = time.time()
    for img_name in sorted(images):
        img_path  = os.path.join(folder_path, img_name)
        file_size = os.path.getsize(img_path)
        if file_size > BATCH_MAX_FILE_BYTES:
            logger.warning(f"Skipping {img_name}: {file_size/1e6:.1f}MB > limit")
            batch_results.append({"original_file": img_name,
                "error": f"File too large ({file_size/1e6:.1f}MB)"})
            continue
        try:
            with Image.open(img_path) as pil_img:
                pil_img = _exif_upright(pil_img)
                if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
                pil_copy = pil_img.copy()
            cv2_img  = cv2.cvtColor(np.array(pil_copy), cv2.COLOR_RGB2BGR)
            result   = run_full_validation(pil_copy, cv2_img)
            category = result["category"]
            saved_as = None
            if save:
                annotated = draw_annotations(_annotation_frame(cv2_img), result)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename  = f"{category}_{timestamp}.jpg"
                out_path  = os.path.join(CATEGORY_FOLDERS.get(category, BASE_OUTPUT), filename)
                if not cv2.imwrite(out_path, annotated):
                    raise RuntimeError("cv2.imwrite failed")
                saved_as = filename
            batch_results.append({"original_file": img_name, "category": category,
                                   "faces_detected": result["total_faces"], "saved_as": saved_as})
            total_processed += 1
        except Exception as exc:
            logger.error(f"Batch error — {img_name}: {exc}")
            batch_results.append({"original_file": img_name, "error": str(exc)})

    return JSONResponse({"success": True,
                         "folder_path": os.path.abspath(folder_path),
                         "total_images_found": len(images),
                         "total_processed": total_processed,
                         "processing_time_ms": round((time.time() - t0)*1000, 2),
                         "results": batch_results})




import asyncio
import io
import os
import tempfile
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import aiohttp
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Known column → URL prefix mapping (applied when value is NOT a full URL)
# Add or change entries here if column names or prefixes change.
# ─────────────────────────────────────────────────────────────
KNOWN_COLUMN_PREFIXES: dict[str, str] = {
    "revice_photograph": "https://ih.imagicahealth.in/teammindnext/doc_kyc/revise_doctors/",
    "photograph":        "https://ih.imagicahealth.in/teammindnext/doctors/",
    # Add more known columns here as needed
}

# ─────────────────────────────────────────────────────────────
# Human-readable comments per category
# ─────────────────────────────────────────────────────────────
CATEGORY_COMMENT: dict[str, str] = {
    "front":            "Valid frontal face",
    "side":             "Face is not frontal — side pose detected",
    "mixed":            "Mixed poses detected",
    "eyes_closed":      "Eyes are closed",
    "sunglasses":       "Dark goggles or sunglasses detected",
    "headwear":         "Cap, helmet, or face mask detected",
    "multiple_person":  "Multiple persons detected in image",
    "no_human":         "No human face detected",
    "irrelevant":       "Image does not contain a human",
    "no_landmarks":     "Face landmarks could not be extracted",
    "blurred":          "Image/face has strong multi-signal blur evidence; pose may still be classified",
    "blur_uncertain":   "Sharpness evidence is inconclusive; manual review or additional analysis recommended",
    "low_resolution":  "Face has insufficient pixels for a reliable sharpness decision",
    "focused":          "Face framing/composition needs correction",
}

# ─────────────────────────────────────────────────────────────
# Concurrency / retry settings
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 10
IMAGE_DOWNLOAD_TIMEOUT   = 15       # seconds per attempt
BATCH_SIZE               = 200
THREAD_WORKERS           = min(8, max(4, os.cpu_count() or 4))
MAX_RETRIES              = 2        # retry on transient failures (timeout/503)
RETRY_DELAY              = 1.5      # seconds between retries

# ─────────────────────────────────────────────────────────────
# In-memory job store
# Replace with Redis / DB for multi-process deployments
# ─────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
# Schema per job:
# {
#   "status":       "pending" | "running" | "done" | "failed",
#   "submitted_at": str,
#   "started_at":   str | None,
#   "finished_at":  str | None,
#   "total_rows":   int,
#   "rows_done":    int,
#   "total_valid":  int | None,
#   "total_invalid":int | None,
#   "elapsed_ms":   float | None,
#   "output_path":  str | None,
#   "output_name":  str | None,
#   "error":        str | None,
# }


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _detect_image_columns(df: pd.DataFrame) -> list[str]:
    """
    Scan all columns and return those that look like image URL/filename columns.
    Detection rules (applied to a sample of non-null values):
      - Value starts with http:// or https://             → URL column
      - Value ends with known image extension             → filename column
      - Column name contains known keywords               → included
    Returns columns in priority order: known columns first (revice > photo), then others.
    """
    import re
    IMG_EXT = re.compile(r"\.(jpe?g|png|webp|gif|bmp|tiff?|avif|heic|heif)$", re.I)
    IMG_URL_RE = re.compile(r"^https?://", re.I)

    def _looks_like_image(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sum(
            1 for v in sample
            if IMG_URL_RE.match(v.strip()) or IMG_EXT.search(v.strip())
        )
        return hits / len(sample) >= 0.10  # at least 10% look like images

    # Priority: known columns first (revice before photo), then auto-detected
    found = []
    for col in KNOWN_COLUMN_PREFIXES:
        if col in df.columns:
            found.append(col)
    for col in df.columns:
        if col not in found and _looks_like_image(df[col]):
            found.append(col)
    return found


def _build_image_url_smart(row: pd.Series, url_cols: list[str]) -> tuple[str | None, str]:
    """
    Try each detected image column in priority order.
    - If value is a full URL (http/https), use as-is.
    - If column is in KNOWN_COLUMN_PREFIXES, prepend that prefix.
    - Otherwise use value as-is (assume full URL).
    """
    for col in url_cols:
        val = str(row.get(col, "") or "").strip()
        if not val or val.lower() in ("nan", "none", ""):
            continue
        if val.startswith("http://") or val.startswith("https://"):
            return val, col
        prefix = KNOWN_COLUMN_PREFIXES.get(col, "")
        return prefix + val, col
    return None, "none"


def _build_image_url(row: pd.Series) -> tuple[str | None, str]:
    """Legacy fallback — used only when url_cols not passed."""
    return _build_image_url_smart(row, list(KNOWN_COLUMN_PREFIXES.keys()))


def _empty_result(comment: str, status_code: int = 404) -> dict:
    return {
        "status_code":       status_code,
        "validation_status": "no_human",
        "is_valid":          0,
        "comment":           comment,
    }


def _run_validation_sync(pil_image: Image.Image) -> dict:
    import cv2
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return run_full_validation(pil_image, cv2_image)   # from your main.py


async def _immediate_result(res: dict) -> dict:
    return res


async def _download_and_validate(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    url:       str,
) -> dict:
    """
    Download one image and validate.
    Retries up to MAX_RETRIES times on timeout / 503.
    Uses asyncio.get_running_loop() for thread-safe execution in async context.
    """
    async with semaphore:
        raw = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):   # attempts: 1, 2, 3
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                ) as resp:
                    if resp.status == 503:
                        last_error = "HTTP 503 Service Unavailable"
                        if attempt <= MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * attempt)
                            continue
                        return _empty_result(last_error, status_code=503)

                    if resp.status != 200:
                        return _empty_result(
                            f"HTTP {resp.status} when downloading image",
                            status_code=resp.status,
                        )
                    raw = await resp.read()
                    break   # success

            except asyncio.TimeoutError:
                last_error = "Image request timed out"
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                    continue
                return _empty_result(last_error, status_code=408)

            except aiohttp.ClientError as exc:
                return _empty_result(f"Network error: {exc}", status_code=503)

            except Exception as exc:
                return _empty_result(f"Download error: {exc}", status_code=500)

    if raw is None:
        return _empty_result(last_error or "Unknown download failure", status_code=500)

    # Decode image
    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_image = _exif_upright(img).convert("RGB")
    except Exception as exc:
        return _empty_result(f"Cannot decode image: {exc}", status_code=422)

    # Run CPU-bound validation in thread pool using the running event loop
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _run_validation_sync, pil_image)
    except Exception as exc:
        return _empty_result(f"Validation error: {exc}", status_code=500)

    category = result.get("category", "no_human")
    categories = result.get("categories", [category])
    quality_issue = any(label in ("blurred", "blur_uncertain", "low_resolution")
                        for label in categories)
    quality_label = next((label for label in ("blurred", "low_resolution", "blur_uncertain")
                          if label in categories), None)
    comments = [CATEGORY_COMMENT.get(label, label) for label in categories]
    # Build audit_category (same logic as _clean_response)
    _faces = result.get("faces") or []
    _app_label = (_faces[0].get("appearance") or {}).get("appearance_label") if _faces else None
    if category == "sunglasses":
        _audit = "goggles"
    elif category == "multiple_person":
        _audit = "multiple_person"
    elif category in ("side", "mixed"):
        _audit = "side_face"
    elif category in ("blurred", "blur_uncertain", "blur_sunglasses_uncertain"):
        _audit = "blur"
    elif category == "low_resolution":
        _audit = "low_resolution"
    elif _app_label:
        _audit = _app_label
    else:
        _audit = "front"
    return {
        "status_code":          CATEGORY_STATUS_MAP.get(quality_label, CATEGORY_STATUS_MAP.get(category, 400)) if quality_issue else CATEGORY_STATUS_MAP.get(category, 400),
        "validation_status":    category,  # backward-compatible single primary label
        "validation_categories": categories,
        "audit_category":       _audit,
        "appearance_label":     _app_label,
        "hair_color":           (_faces[0].get("hair") or {}).get("hair_color") if _faces else None,
        "specs_detected":       (_faces[0].get("specs") or {}).get("detected") if _faces else None,
        "headwear_kind":        (_faces[0].get("headwear") or {}).get("kind") if _faces else None,
        "is_valid":             1 if category == "front" and not quality_issue else 0,
        "comment":              "; ".join(dict.fromkeys(comments)),
    }


async def _process_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    batch:     list[tuple[str | None, str]],
) -> list[dict]:
    tasks = [
        _immediate_result(_empty_result("No image address provided in this row"))
        if url is None
        else _download_and_validate(session, semaphore, executor, url)
        for url, _ in batch
    ]
    return await asyncio.gather(*tasks, return_exceptions=False)


# ═══════════════════════════════════════════════════════════════
# Background worker  (runs after POST returns job_id)
# ═══════════════════════════════════════════════════════════════

async def _run_validation_job(job_id: str, df: pd.DataFrame) -> None:
    job = _jobs[job_id]
    job["status"]     = "running"
    job["started_at"] = datetime.now().isoformat()
    t0 = time.time()

    total_rows = len(df)
    job["total_rows"] = total_rows
    all_results: list[dict] = []

    try:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        executor  = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)

        async with aiohttp.ClientSession(connector=connector) as session:
            # Process rows in batches to keep memory usage low
            for batch_start in range(0, total_rows, BATCH_SIZE):
                batch_df = df.iloc[batch_start : batch_start + BATCH_SIZE]
                batch    = [_build_image_url(row) for _, row in batch_df.iterrows()]

                batch_results = await _process_batch(session, semaphore, executor, batch)
                all_results.extend(batch_results)

                job["rows_done"] = min(batch_start + BATCH_SIZE, total_rows)
                logger.info(
                    f"job={job_id} | progress {job['rows_done']}/{total_rows}"
                )

        executor.shutdown(wait=True)

        # Append result columns
        df["status_code"]       = [r["status_code"]       for r in all_results]
        df["validation_status"] = [r["validation_status"] for r in all_results]
        df["audit_category"]    = [r.get("audit_category")    for r in all_results]
        df["appearance_label"]  = [r.get("appearance_label")  for r in all_results]
        df["hair_color"]        = [r.get("hair_color")        for r in all_results]
        df["specs_detected"]    = [r.get("specs_detected")    for r in all_results]
        df["headwear_kind"]     = [r.get("headwear_kind")     for r in all_results]
        df["is_valid"]          = [r["is_valid"]           for r in all_results]
        df["comment"]           = [r["comment"]            for r in all_results]

        total_valid   = int(df["is_valid"].sum())
        total_invalid = total_rows - total_valid
        elapsed_ms    = round((time.time() - t0) * 1000, 2)

        # Microsecond-precision timestamp ensures zero filename collision risk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_name  = f"validated_doctors_{timestamp}.csv"
        out_path  = os.path.join(tempfile.gettempdir(), out_name)
        df.to_csv(out_path, index=False)

        job.update({
            "status":        "done",
            "finished_at":   datetime.now().isoformat(),
            "total_valid":   total_valid,
            "total_invalid": total_invalid,
            "elapsed_ms":    elapsed_ms,
            "output_path":   out_path,
            "output_name":   out_name,
        })
        logger.info(
            f"job={job_id} complete | rows={total_rows} | "
            f"valid={total_valid} | invalid={total_invalid} | {elapsed_ms}ms"
        )

    except Exception as exc:
        job.update({
            "status":      "failed",
            "finished_at": datetime.now().isoformat(),
            "error":       str(exc),
        })
        logger.error(f"job={job_id} failed: {exc}")



import asyncio
import io
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import numpy as np
import pandas as pd
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Known column → URL prefix mapping (applied when value is NOT a full URL)
# Add or change entries here if column names or prefixes change.
# ─────────────────────────────────────────────────────────────
KNOWN_COLUMN_PREFIXES: dict[str, str] = {
    "revice_photograph": "https://ih.imagicahealth.in/teammindnext/doc_kyc/revise_doctors/",
    "photograph":        "https://ih.imagicahealth.in/teammindnext/doctors/",
    # Add more known columns here as needed
}

# ─────────────────────────────────────────────────────────────
# Human-readable comments per category
# ─────────────────────────────────────────────────────────────
CATEGORY_COMMENT: dict[str, str] = {
    "front":            "Valid frontal face",
    "side":             "Face is not frontal — side pose detected",
    "mixed":            "Mixed poses detected",
    "eyes_closed":      "Eyes are closed",
    "sunglasses":       "Dark goggles or sunglasses detected",
    "headwear":         "Cap, helmet, or face mask detected",
    "multiple_person":  "Multiple persons detected in image",
    "no_human":         "No human face detected",
    "irrelevant":       "Image does not contain a human",
    "no_landmarks":     "Face landmarks could not be extracted",
    "blurred":          "Image/face has strong multi-signal blur evidence; pose may still be classified",
    "blur_uncertain":   "Sharpness evidence is inconclusive; manual review or additional analysis recommended",
    "low_resolution":  "Face has insufficient pixels for a reliable sharpness decision",
    "focused":          "Face framing/composition needs correction",
}

# ─────────────────────────────────────────────────────────────
# Concurrency / retry settings
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 10
IMAGE_DOWNLOAD_TIMEOUT   = 15
BATCH_SIZE               = 200
THREAD_WORKERS           = min(8, max(4, os.cpu_count() or 4))
MAX_RETRIES              = 2
RETRY_DELAY              = 1.5


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _detect_image_columns(df: pd.DataFrame) -> list[str]:
    """
    Scan all columns and return those that look like image URL/filename columns.
    Detection rules (applied to a sample of non-null values):
      - Value starts with http:// or https://             → URL column
      - Value ends with known image extension             → filename column
      - Column name contains known keywords               → included
    Returns columns in priority order: known columns first (revice > photo), then others.
    """
    import re
    IMG_EXT = re.compile(r"\.(jpe?g|png|webp|gif|bmp|tiff?|avif|heic|heif)$", re.I)
    IMG_URL_RE = re.compile(r"^https?://", re.I)

    def _looks_like_image(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sum(
            1 for v in sample
            if IMG_URL_RE.match(v.strip()) or IMG_EXT.search(v.strip())
        )
        return hits / len(sample) >= 0.10  # at least 10% look like images

    # Priority: known columns first (revice before photo), then auto-detected
    found = []
    for col in KNOWN_COLUMN_PREFIXES:
        if col in df.columns:
            found.append(col)
    for col in df.columns:
        if col not in found and _looks_like_image(df[col]):
            found.append(col)
    return found


def _build_image_url_smart(row: pd.Series, url_cols: list[str]) -> tuple[str | None, str]:
    """
    Try each detected image column in priority order.
    - If value is a full URL (http/https), use as-is.
    - If column is in KNOWN_COLUMN_PREFIXES, prepend that prefix.
    - Otherwise use value as-is (assume full URL).
    """
    for col in url_cols:
        val = str(row.get(col, "") or "").strip()
        if not val or val.lower() in ("nan", "none", ""):
            continue
        if val.startswith("http://") or val.startswith("https://"):
            return val, col
        prefix = KNOWN_COLUMN_PREFIXES.get(col, "")
        return prefix + val, col
    return None, "none"


def _build_image_url(row: pd.Series) -> tuple[str | None, str]:
    """Legacy fallback — used only when url_cols not passed."""
    return _build_image_url_smart(row, list(KNOWN_COLUMN_PREFIXES.keys()))


def _empty_result(comment: str, status_code: int = 404) -> dict:
    return {
        "status_code":       status_code,
        "validation_status": "no_human",
        "is_valid":          0,
        "comment":           comment,
    }


def _run_validation_sync(pil_image: Image.Image) -> dict:
    import cv2
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return run_full_validation(pil_image, cv2_image)   # from your main.py


async def _immediate_result(res: dict) -> dict:
    return res


async def _download_and_validate(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    url:       str,
) -> dict:
    async with semaphore:
        raw       = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                ) as resp:
                    if resp.status == 503:
                        last_error = "HTTP 503 Service Unavailable"
                        if attempt <= MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * attempt)
                            continue
                        return _empty_result(last_error, status_code=503)
                    if resp.status != 200:
                        return _empty_result(
                            f"HTTP {resp.status} when downloading image",
                            status_code=resp.status,
                        )
                    raw = await resp.read()
                    break
            except asyncio.TimeoutError:
                last_error = "Image request timed out"
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                    continue
                return _empty_result(last_error, status_code=408)
            except aiohttp.ClientError as exc:
                return _empty_result(f"Network error: {exc}", status_code=503)
            except Exception as exc:
                return _empty_result(f"Download error: {exc}", status_code=500)

    if raw is None:
        return _empty_result(last_error or "Unknown download failure", status_code=500)

    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_image = _exif_upright(img).convert("RGB")
    except Exception as exc:
        return _empty_result(f"Cannot decode image: {exc}", status_code=422)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _run_validation_sync, pil_image)
    except Exception as exc:
        return _empty_result(f"Validation error: {exc}", status_code=500)

    category = result.get("category", "no_human")
    categories = result.get("categories", [category])
    quality_issue = any(label in ("blurred", "blur_uncertain", "low_resolution")
                        for label in categories)
    quality_label = next((label for label in ("blurred", "low_resolution", "blur_uncertain")
                          if label in categories), None)
    comments = [CATEGORY_COMMENT.get(label, label) for label in categories]
    # Build audit_category (same logic as _clean_response)
    _faces = result.get("faces") or []
    _app_label = (_faces[0].get("appearance") or {}).get("appearance_label") if _faces else None
    if category == "sunglasses":
        _audit = "goggles"
    elif category == "multiple_person":
        _audit = "multiple_person"
    elif category in ("side", "mixed"):
        _audit = "side_face"
    elif category in ("blurred", "blur_uncertain", "blur_sunglasses_uncertain"):
        _audit = "blur"
    elif category == "low_resolution":
        _audit = "low_resolution"
    elif _app_label:
        _audit = _app_label
    else:
        _audit = "front"
    return {
        "status_code":          CATEGORY_STATUS_MAP.get(quality_label, CATEGORY_STATUS_MAP.get(category, 400)) if quality_issue else CATEGORY_STATUS_MAP.get(category, 400),
        "validation_status":    category,  # backward-compatible single primary label
        "validation_categories": categories,
        "audit_category":       _audit,
        "appearance_label":     _app_label,
        "hair_color":           (_faces[0].get("hair") or {}).get("hair_color") if _faces else None,
        "specs_detected":       (_faces[0].get("specs") or {}).get("detected") if _faces else None,
        "headwear_kind":        (_faces[0].get("headwear") or {}).get("kind") if _faces else None,
        "is_valid":             1 if category == "front" and not quality_issue else 0,
        "comment":              "; ".join(dict.fromkeys(comments)),
    }


async def _process_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    batch:     list[tuple[str | None, str]],
) -> list[dict]:
    tasks = [
        _immediate_result(_empty_result("No image address provided in this row"))
        if url is None
        else _download_and_validate(session, semaphore, executor, url)
        for url, _ in batch
    ]
    return await asyncio.gather(*tasks, return_exceptions=False)



import asyncio
import io
import os
import tempfile
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Known column → URL prefix mapping (applied when value is NOT a full URL)
# Add or change entries here if column names or prefixes change.
# ─────────────────────────────────────────────────────────────
KNOWN_COLUMN_PREFIXES: dict[str, str] = {
    "revice_photograph": "https://ih.imagicahealth.in/teammindnext/doc_kyc/revise_doctors/",
    "photograph":        "https://ih.imagicahealth.in/teammindnext/doctors/",
    # Add more known columns here as needed
}

# ─────────────────────────────────────────────────────────────
# Human-readable comments per category
# ─────────────────────────────────────────────────────────────
CATEGORY_COMMENT: dict[str, str] = {
    "front":            "Valid frontal face",
    "side":             "Face is not frontal — side pose detected",
    "mixed":            "Mixed poses detected",
    "eyes_closed":      "Eyes are closed",
    "sunglasses":       "Dark goggles or sunglasses detected",
    "headwear":         "Cap, helmet, or face mask detected",
    "multiple_person":  "Multiple persons detected in image",
    "no_human":         "No human face detected",
    "irrelevant":       "Image does not contain a human",
    "no_landmarks":     "Face landmarks could not be extracted",
    "blurred":          "Image/face has strong multi-signal blur evidence; pose may still be classified",
    "blur_uncertain":   "Sharpness evidence is inconclusive; manual review or additional analysis recommended",
    "low_resolution":  "Face has insufficient pixels for a reliable sharpness decision",
    "focused":          "Face framing/composition needs correction",
}

# ─────────────────────────────────────────────────────────────
# Concurrency / retry settings
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 10
IMAGE_DOWNLOAD_TIMEOUT   = 15
BATCH_SIZE               = 200
THREAD_WORKERS           = min(8, max(4, os.cpu_count() or 4))
MAX_RETRIES              = 2
RETRY_DELAY              = 1.5

# ─────────────────────────────────────────────────────────────
# In-memory job store  {job_id: job_dict}
# ─────────────────────────────────────────────────────────────
_JOBS: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _detect_image_columns(df: pd.DataFrame) -> list[str]:
    """
    Scan all columns and return those that look like image URL/filename columns.
    Detection rules (applied to a sample of non-null values):
      - Value starts with http:// or https://             → URL column
      - Value ends with known image extension             → filename column
      - Column name contains known keywords               → included
    Returns columns in priority order: known columns first (revice > photo), then others.
    """
    import re
    IMG_EXT = re.compile(r"\.(jpe?g|png|webp|gif|bmp|tiff?|avif|heic|heif)$", re.I)
    IMG_URL_RE = re.compile(r"^https?://", re.I)

    def _looks_like_image(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sum(
            1 for v in sample
            if IMG_URL_RE.match(v.strip()) or IMG_EXT.search(v.strip())
        )
        return hits / len(sample) >= 0.10  # at least 10% look like images

    # Priority: known columns first (revice before photo), then auto-detected
    found = []
    for col in KNOWN_COLUMN_PREFIXES:
        if col in df.columns:
            found.append(col)
    for col in df.columns:
        if col not in found and _looks_like_image(df[col]):
            found.append(col)
    return found


def _build_image_url_smart(row: pd.Series, url_cols: list[str]) -> tuple[str | None, str]:
    """
    Try each detected image column in priority order.
    - If value is a full URL (http/https), use as-is.
    - If column is in KNOWN_COLUMN_PREFIXES, prepend that prefix.
    - Otherwise use value as-is (assume full URL).
    """
    for col in url_cols:
        val = str(row.get(col, "") or "").strip()
        if not val or val.lower() in ("nan", "none", ""):
            continue
        if val.startswith("http://") or val.startswith("https://"):
            return val, col
        prefix = KNOWN_COLUMN_PREFIXES.get(col, "")
        return prefix + val, col
    return None, "none"


def _build_image_url(row: pd.Series) -> tuple[str | None, str]:
    """Legacy fallback — used only when url_cols not passed."""
    return _build_image_url_smart(row, list(KNOWN_COLUMN_PREFIXES.keys()))


def _empty_result(comment: str, status_code: int = 404) -> dict:
    return {
        "status_code":       status_code,
        "validation_status": "no_human",
        "is_valid":          0,
        "comment":           comment,
    }


def _run_validation_sync(pil_image: Image.Image) -> dict:
    import cv2
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return run_full_validation(pil_image, cv2_image)   # from your main.py


async def _immediate_result(res: dict) -> dict:
    return res


async def _download_and_validate(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    url:       str,
) -> dict:
    """Download one image with retry, then run face validation."""
    async with semaphore:
        raw        = None
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):   # attempts: 1, 2, 3
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT),
                ) as resp:
                    if resp.status == 503:
                        last_error = "HTTP 503 Service Unavailable"
                        if attempt <= MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * attempt)
                            continue
                        return _empty_result(last_error, status_code=503)
                    if resp.status != 200:
                        return _empty_result(
                            f"HTTP {resp.status} when downloading image",
                            status_code=resp.status,
                        )
                    raw = await resp.read()
                    break   # success

            except asyncio.TimeoutError:
                last_error = f"Image request timed out (attempt {attempt})"
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                    continue
                return _empty_result(last_error, status_code=408)
            except aiohttp.ClientError as exc:
                return _empty_result(f"Network error: {exc}", status_code=503)
            except Exception as exc:
                return _empty_result(f"Download error: {exc}", status_code=500)

    if raw is None:
        return _empty_result(last_error or "Unknown download failure", status_code=500)

    # Decode image
    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_image = _exif_upright(img).convert("RGB")
    except Exception as exc:
        return _empty_result(f"Cannot decode image: {exc}", status_code=422)

    # Run CPU-bound validation in thread pool
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _run_validation_sync, pil_image)
    except Exception as exc:
        return _empty_result(f"Validation error: {exc}", status_code=500)

    category = result.get("category", "no_human")
    categories = result.get("categories", [category])
    quality_issue = any(label in ("blurred", "blur_uncertain", "low_resolution")
                        for label in categories)
    quality_label = next((label for label in ("blurred", "low_resolution", "blur_uncertain")
                          if label in categories), None)
    comments = [CATEGORY_COMMENT.get(label, label) for label in categories]
    # Build audit_category (same logic as _clean_response)
    _faces = result.get("faces") or []
    _app_label = (_faces[0].get("appearance") or {}).get("appearance_label") if _faces else None
    if category == "sunglasses":
        _audit = "goggles"
    elif category == "multiple_person":
        _audit = "multiple_person"
    elif category in ("side", "mixed"):
        _audit = "side_face"
    elif category in ("blurred", "blur_uncertain", "blur_sunglasses_uncertain"):
        _audit = "blur"
    elif category == "low_resolution":
        _audit = "low_resolution"
    elif _app_label:
        _audit = _app_label
    else:
        _audit = "front"
    return {
        "status_code":          CATEGORY_STATUS_MAP.get(quality_label, CATEGORY_STATUS_MAP.get(category, 400)) if quality_issue else CATEGORY_STATUS_MAP.get(category, 400),
        "validation_status":    category,  # backward-compatible single primary label
        "validation_categories": categories,
        "audit_category":       _audit,
        "appearance_label":     _app_label,
        "hair_color":           (_faces[0].get("hair") or {}).get("hair_color") if _faces else None,
        "specs_detected":       (_faces[0].get("specs") or {}).get("detected") if _faces else None,
        "headwear_kind":        (_faces[0].get("headwear") or {}).get("kind") if _faces else None,
        "is_valid":             1 if category == "front" and not quality_issue else 0,
        "comment":              "; ".join(dict.fromkeys(comments)),
    }


async def _process_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    executor:  ThreadPoolExecutor,
    batch_rows: list[pd.Series],
    url_cols: list[str] | None = None,
) -> list[dict]:
    """Run a batch of rows concurrently."""
    _cols = url_cols or list(KNOWN_COLUMN_PREFIXES.keys())
    tasks = []
    for row in batch_rows:
        url, _ = _build_image_url_smart(row, _cols)
        tasks.append(
            _immediate_result(_empty_result("No image address provided in this row"))
            if url is None
            else _download_and_validate(session, semaphore, executor, url)
        )
    return await asyncio.gather(*tasks, return_exceptions=False)


# ═══════════════════════════════════════════════════════════════
# Background worker
# ═══════════════════════════════════════════════════════════════

async def _run_validation_job(job_id: str, df: pd.DataFrame) -> None:
    """
    Runs in background after POST returns.
    - Processes rows in BATCH_SIZE chunks (memory safe).
    - Appends each result to job["results"] immediately (live polling).
    - Saves final CSV to temp dir for download.
    """
    job   = _JOBS[job_id]
    t0    = time.time()
    total = len(df)
    rows     = [row for _, row in df.iterrows()]
    url_cols = job.get("url_cols") or list(KNOWN_COLUMN_PREFIXES.keys())

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    executor  = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)

    job["status"]    = "processing"
    job["processed"] = 0
    job["progress"]  = 0.0
    job["results"]   = []   # live per-row results, readable via /job-status at any time

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            for batch_start in range(0, total, BATCH_SIZE):
                chunk         = rows[batch_start : batch_start + BATCH_SIZE]
                batch_results = await _process_batch(session, semaphore, executor, chunk, url_cols)

                # Append each row result immediately so /job-status can serve live data
                for i, result in enumerate(batch_results):
                    row        = chunk[i]
                    row_number = batch_start + i + 1
                    # Include all detected image column values in live status
                    img_vals = {col: str(row.get(col, "") or "").strip() for col in url_cols}
                    job["results"].append({
                        "row":               row_number,
                        **img_vals,
                        "status_code":       result["status_code"],
                        "validation_status": result["validation_status"],
                        "audit_category":    result.get("audit_category"),
                        "appearance_label":  result.get("appearance_label"),
                        "hair_color":        result.get("hair_color"),
                        "specs_detected":    result.get("specs_detected"),
                        "headwear_kind":     result.get("headwear_kind"),
                        "is_valid":          result["is_valid"],
                        "comment":           result["comment"],
                    })

                done             = min(batch_start + BATCH_SIZE, total)
                job["processed"] = done
                job["progress"]  = round(done / total * 100, 1)
                logger.info(f"job={job_id} | {done}/{total} rows done")

        executor.shutdown(wait=True)

        # ── Build summary ─────────────────────────────────────
        total_valid   = sum(r["is_valid"] for r in job["results"])
        total_invalid = total - total_valid
        elapsed_ms    = round((time.time() - t0) * 1000, 2)

        # ── Save result CSV (microsecond timestamp = no collision) ──
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_name  = f"validated_doctors_{timestamp}.csv"
        out_path  = os.path.join(tempfile.gettempdir(), out_name)

        result_df = df.copy()
        result_df["status_code"]       = [r["status_code"]       for r in job["results"]]
        result_df["validation_status"] = [r["validation_status"] for r in job["results"]]
        result_df["audit_category"]    = [r.get("audit_category")    for r in job["results"]]
        result_df["appearance_label"]  = [r.get("appearance_label")  for r in job["results"]]
        result_df["hair_color"]        = [r.get("hair_color")        for r in job["results"]]
        result_df["specs_detected"]    = [r.get("specs_detected")    for r in job["results"]]
        result_df["headwear_kind"]     = [r.get("headwear_kind")     for r in job["results"]]
        result_df["is_valid"]          = [r["is_valid"]           for r in job["results"]]
        result_df["comment"]           = [r["comment"]            for r in job["results"]]
        result_df.to_csv(out_path, index=False)

        job.update({
            "status":        "done",
            "total_rows":    total,
            "total_valid":   total_valid,
            "total_invalid": total_invalid,
            "elapsed_ms":    elapsed_ms,
            "out_path":      out_path,
            "out_name":      out_name,
        })
        logger.info(
            f"job={job_id} complete | rows={total} | "
            f"valid={total_valid} | invalid={total_invalid} | {elapsed_ms}ms"
        )

    except Exception as exc:
        executor.shutdown(wait=False)
        job["status"] = "error"
        job["error"]  = str(exc)
        logger.error(f"job={job_id} failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# Endpoint 1 — Upload CSV → get job_id immediately
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/validate-csv",
    summary="Bulk CSV face validation — start job",
    description=(
        "Upload a CSV file containing `photograph` and `revice_photograph` columns. "
        "Returns a `job_id` immediately — no waiting. "
        "Poll `/job-status/{job_id}` for live progress and per-row results. "
        "Download the final CSV from `/job-download/{job_id}` when done."
    ),
    tags=["Batch"],
)
async def validate_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Upload your CSV file here"),
):
    # ── 1. Validate file type ─────────────────────────────────
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a .csv file, got: '{file.filename}'",
        )

    # ── 2. Read and parse ─────────────────────────────────────
    raw_bytes = await file.read()
    await file.close()

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}")

    # ── 3. Auto-detect image URL columns ─────────────────────
    url_cols = _detect_image_columns(df)
    if not url_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                "No image URL or filename column detected. "
                "Make sure at least one column contains image URLs (http://...) "
                "or image filenames (.jpg / .png / .webp etc.). "
                f"Columns found: {list(df.columns)}"
            ),
        )

    # ── 4. Register job and kick off background task ──────────
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "status":    "queued",
        "filename":  file.filename,
        "total":     len(df),
        "processed": 0,
        "progress":  0.0,
        "results":   [],
        "url_cols":  url_cols,
    }

    background_tasks.add_task(_run_validation_job, job_id, df)
    logger.info(f"validate-csv job queued | job_id={job_id} | rows={len(df)}")

    return JSONResponse(
        status_code=202,
        content={
            "job_id":       job_id,
            "total_rows":   len(df),
            "status":       "queued",
            "status_url":   f"/job-status/{job_id}",
            "download_url": f"/job-download/{job_id}",
            "message":      "Job started. Poll status_url for live progress.",
        },
    )


# ═══════════════════════════════════════════════════════════════
# Endpoint 2 — Live status + per-row results (JSON)
# ═══════════════════════════════════════════════════════════════

@app.get(
    "/job-status/{job_id}",
    summary="Poll CSV validation job — live progress + results",
    description=(
        "Returns live processing status and all results collected so far. "
        "Keep polling until `status` is `done` or `error`."
    ),
    tags=["Batch"],
)
async def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    response: dict = {
        "job_id":          job_id,
        "status":          job["status"],          # queued | processing | done | error
        "filename":        job["filename"],
        "total_rows":      job["total"],
        "processed":       job["processed"],
        "progress":        f"{job['progress']}%",
        "results_so_far":  job.get("results", []),
    }

    if job["status"] == "done":
        response.update({
            "total_valid":   job["total_valid"],
            "total_invalid": job["total_invalid"],
            "elapsed_ms":    job["elapsed_ms"],
            "download_url":  f"/job-download/{job_id}",
        })

    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")

    return JSONResponse(content=response)


# ═══════════════════════════════════════════════════════════════
# Endpoint 3 — Download final result CSV
# ═══════════════════════════════════════════════════════════════

@app.get(
    "/job-download/{job_id}",
    summary="Download validated CSV result",
    description="Download the final CSV with 4 appended result columns. Only available when job status is `done`.",
    tags=["Batch"],
)
async def job_download(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job["status"] in ("queued", "processing"):
        raise HTTPException(
            status_code=202,
            detail=f"Job still running — {job['progress']}% done. Try again shortly.",
        )
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.get('error')}")

    return FileResponse(
        path=job["out_path"],
        media_type="text/csv",
        filename=job["out_name"],
        headers={
            "X-Total-Rows":    str(job["total_rows"]),
            "X-Total-Valid":   str(job["total_valid"]),
            "X-Total-Invalid": str(job["total_invalid"]),
            "X-Processing-Ms": str(job["elapsed_ms"]),
        },
    )


import asyncio
import csv
import io
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import aiohttp
import pandas as pd
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse


# ---------------------------------------------------------------
# In-memory stream registry  {stream_id: stream_dict}
# ---------------------------------------------------------------
_STREAMS: dict[str, dict] = {}

# CSV result columns written per row
RESULT_COLUMNS = [
    "row",
    "photograph",
    "revice_photograph",
    "source_used",
    "status_code",
    "validation_status",
    "is_valid",
    "comment",
    "row_ms",
]


# ===============================================================
# Endpoint 1 — Stream row by row + write partial CSV as we go
# ===============================================================

@app.post(
    "/validate-csv-stream",
    summary="CSV face validation — live stream + partial CSV saved on disk",
    description=(
        "Upload a CSV file. Results stream back ONE ROW AT A TIME as NDJSON. "
        "Every row is also written to a CSV file on disk immediately. "
        "If you stop/disconnect at any point, all rows processed so far are saved. "
        "The first line of the stream contains `stream_id`. "
        "Use GET /stream-download/{stream_id} to download the partial or full CSV anytime."
    ),
    tags=["Batch"],
    response_class=StreamingResponse,
)
async def validate_csv_stream(
    file: UploadFile = File(..., description="Upload your CSV file here"),
):
    # 1. Validate file type
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a .csv file, got: '{file.filename}'",
        )

    # 2. Read and parse CSV
    raw_bytes = await file.read()
    await file.close()

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}")

    # 3. Validate required columns
    missing = {"photograph", "revice_photograph"} - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required column(s): {', '.join(sorted(missing))}. "
                f"CSV must contain: photograph, revice_photograph"
            ),
        )

    total_rows = len(df)
    rows       = [row for _, row in df.iterrows()]

    # 4. Create stream_id + output CSV file path
    stream_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_name  = f"stream_{timestamp}.csv"
    out_path  = os.path.join(tempfile.gettempdir(), out_name)

    # Register stream in memory
    _STREAMS[stream_id] = {
        "status":        "streaming",
        "filename":      file.filename,
        "out_path":      out_path,
        "out_name":      out_name,
        "total_rows":    total_rows,
        "processed":     0,
        "total_valid":   0,
        "total_invalid": 0,
    }

    # Write CSV header immediately so file is valid even after 0 rows
    original_columns = list(df.columns)
    all_columns      = original_columns + [
        "status_code", "validation_status", "is_valid", "comment"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()

    logger.info(
        f"validate-csv-stream started | "
        f"stream_id={stream_id} | file={file.filename} | rows={total_rows}"
    )

    # 5. Async generator — one row at a time
    async def row_generator():
        t0            = time.time()
        valid_count   = 0
        invalid_count = 0

        # Limit to one concurrent download for gentle streaming load
        executor  = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
        semaphore = asyncio.Semaphore(1)
        connector = aiohttp.TCPConnector(limit=1)

        # First line — send stream_id so client can save it for download later
        yield json.dumps({
            "event":       "start",
            "stream_id":   stream_id,
            "total_rows":  total_rows,
            "filename":    file.filename,
            "download_url": f"/stream-download/{stream_id}",
            "message":     "Save stream_id. Use download_url anytime to get partial/full CSV.",
        }) + "\n"

        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                for row_number, row in enumerate(rows, start=1):
                    row_t0 = time.time()

                    # Build URL and validate
                    url, source_col = _build_image_url(row)

                    if url is None:
                        result = _empty_result("No image address provided in this row")
                    else:
                        result = await _download_and_validate(
                            session, semaphore, executor, url
                        )

                    # Count
                    if result["is_valid"] == 1:
                        valid_count += 1
                    else:
                        invalid_count += 1

                    row_ms = round((time.time() - row_t0) * 1000, 2)

                    # Write this row to CSV immediately (partial save)
                    row_data = dict(row)   # original CSV columns
                    row_data.update({
                        "status_code":       result["status_code"],
                        "validation_status": result["validation_status"],
                        "is_valid":          result["is_valid"],
                        "comment":           result["comment"],
                    })
                    with open(out_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=all_columns, extrasaction="ignore"
                        )
                        writer.writerow(row_data)

                    # Update stream registry
                    _STREAMS[stream_id]["processed"]     = row_number
                    _STREAMS[stream_id]["total_valid"]   = valid_count
                    _STREAMS[stream_id]["total_invalid"] = invalid_count

                    # Yield row result to client immediately
                    yield json.dumps({
                        "event":             "row",
                        "stream_id":         stream_id,
                        "row":               row_number,
                        "total_rows":        total_rows,
                        "progress":          f"{round(row_number / total_rows * 100, 1)}%",
                        "photograph":        str(row.get("photograph",        "") or "").strip(),
                        "revice_photograph": str(row.get("revice_photograph", "") or "").strip(),
                        "source_used":       source_col,
                        "status_code":       result["status_code"],
                        "validation_status": result["validation_status"],
                        "is_valid":          result["is_valid"],
                        "comment":           result["comment"],
                        "row_ms":            row_ms,
                        "download_url":      f"/stream-download/{stream_id}",
                    }) + "\n"

                    logger.info(
                        f"stream={stream_id} | row {row_number}/{total_rows} | "
                        f"{result['validation_status']} | {row_ms}ms"
                    )

        except asyncio.CancelledError:
            # Client disconnected — this is normal, partial CSV is already saved
            logger.warning(
                f"stream={stream_id} | client disconnected at row "
                f"{_STREAMS[stream_id]['processed']}/{total_rows} | "
                f"partial CSV saved at {out_path}"
            )
            _STREAMS[stream_id]["status"] = "partial"
            return   # stop generator cleanly

        finally:
            executor.shutdown(wait=False)

        # All rows done
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        _STREAMS[stream_id]["status"]     = "done"
        _STREAMS[stream_id]["elapsed_ms"] = elapsed_ms

        # Final summary line
        yield json.dumps({
            "event":         "done",
            "stream_id":     stream_id,
            "total_rows":    total_rows,
            "total_valid":   valid_count,
            "total_invalid": invalid_count,
            "elapsed_ms":    elapsed_ms,
            "download_url":  f"/stream-download/{stream_id}",
            "message":       "All rows done. Download your full CSV at download_url.",
        }) + "\n"

        logger.info(
            f"stream={stream_id} complete | rows={total_rows} | "
            f"valid={valid_count} | invalid={invalid_count} | {elapsed_ms}ms"
        )

    # 6. Return streaming response
    return StreamingResponse(
        row_generator(),
        media_type="application/x-ndjson",
        headers={
            "X-Stream-Id":       stream_id,
            "X-Total-Rows":      str(total_rows),
            "X-Download-Url":    f"/stream-download/{stream_id}",
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disables NGINX buffering on Render
        },
    )


# ===============================================================
# Endpoint 2 — Download partial or full CSV anytime
# ===============================================================

@app.get(
    "/stream-download/{stream_id}",
    summary="Download partial or full stream result CSV",
    description=(
        "Download the CSV file for a stream — works anytime, even mid-stream. "
        "If the stream is still running or was stopped early, "
        "you get all rows processed so far. "
        "If the stream is done, you get the complete validated CSV."
    ),
    tags=["Batch"],
)
async def stream_download(stream_id: str):
    stream = _STREAMS.get(stream_id)

    if not stream:
        raise HTTPException(
            status_code=404,
            detail=f"Stream not found: {stream_id}. It may have expired or never started.",
        )

    out_path = stream["out_path"]

    if not os.path.exists(out_path):
        raise HTTPException(
            status_code=404,
            detail="CSV file not found on disk. It may have been cleaned up.",
        )

    processed    = stream["processed"]
    total_rows   = stream["total_rows"]
    status       = stream["status"]   # streaming | partial | done

    # Label the filename clearly so user knows if it is partial or full
    if status == "done":
        download_name = stream["out_name"].replace("stream_", "validated_full_")
    else:
        download_name = stream["out_name"].replace(
            "stream_", f"validated_partial_{processed}of{total_rows}_"
        )

    logger.info(
        f"stream-download | stream_id={stream_id} | "
        f"status={status} | rows={processed}/{total_rows}"
    )

    return FileResponse(
        path=out_path,
        media_type="text/csv",
        filename=download_name,
        headers={
            "X-Stream-Status":   status,
            "X-Rows-Processed":  str(processed),
            "X-Total-Rows":      str(total_rows),
            "X-Total-Valid":     str(stream.get("total_valid",   0)),
            "X-Total-Invalid":   str(stream.get("total_invalid", 0)),
     
        },
    )


# ===============================================================
# Endpoint 3 — Check stream status (without downloading)
# ===============================================================

@app.get(
    "/stream-status/{stream_id}",
    summary="Check stream progress and download availability",
    tags=["Batch"],
)
async def stream_status(stream_id: str):
    stream = _STREAMS.get(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail=f"Stream not found: {stream_id}")

    processed  = stream["processed"]
    total_rows = stream["total_rows"]

    return {
        "stream_id":     stream_id,
        "status":        stream["status"],    # streaming | partial | done
        "filename":      stream["filename"],
        "processed":     processed,
        "total_rows":    total_rows,
        "progress":      f"{round(processed / total_rows * 100, 1)}%" if total_rows else "0%",
        "total_valid":   stream.get("total_valid",   0),
        "total_invalid": stream.get("total_invalid", 0),
        "download_url":  f"/stream-download/{stream_id}",
    }


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--eval-scene":
        raise SystemExit(run_scene_eval(_sys.argv[2:]))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

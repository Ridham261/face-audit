# Face Validation Engine — Complete Project Documentation

> **Purpose of this document**: A second developer (or your future self) can read
> this top-to-bottom and fully understand what was built, why every decision was
> made, and how to run, test, and extend the system.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Python Environment](#2-python-environment)
3. [Project Folder Structure](#3-project-folder-structure)
4. [Core Files — Root Level](#4-core-files--root-level)
5. [Models Folder](#5-models-folder)
6. [Face Validation Engine — main.py Deep Dive](#6-face-validation-engine--mainpy-deep-dive)
7. [Custom YOLO Goggles Model](#7-custom-yolo-goggles-model)
8. [Goggles Wrapper API](#8-goggles-wrapper-api)
9. [Dataset Structure](#9-dataset-structure)
10. [Test Results](#10-test-results)
11. [How to Run — Step by Step](#11-how-to-run--step-by-step)
12. [Training Folder Contents](#12-training-folder-contents)
13. [What Was Cleaned Up and Why](#13-what-was-cleaned-up-and-why)
14. [Pending Tasks](#14-pending-tasks)

---

## 1. What This Project Does

This is the **Face Audit System** — an automated image validation pipeline that
decides whether a face photo is acceptable (e.g., for ID documents, healthcare
profiles, or professional directories).

It performs **multi-layer validation** in this exact priority order:

| Priority | Check | Reject Reason |
|---|---|---|
| 1 | Dimension guard | Image too small (< 100 × 100 px) |
| 2 | No human face | No face detected at all |
| 3 | Irrelevant image | Not a person photo (OIV7 object detection) |
| 4 | Multiple persons | More than one face in frame |
| 5 | No landmarks | Face found but landmarks missing |
| 6 | **Sunglasses / Goggles** | Glasses that hide eyes |
| 7 | Eyes closed | Eyes are shut |
| 8 | Pose | Not front-facing |
| 9 | Appearance | Classify hair, headwear, specs type |
| 10 | Blur | Image too blurry |
| 11 | Resolution | Image too low-resolution |
| 12 | Gender | Classify male / female |

Each check fires in order. The first check that fails returns a `rejected` or
`review` status and **stops** — later checks are not run.

The final output is a structured JSON response with:
- `is_valid` (1 / 0)
- `status` (accepted / review / rejected)
- `audit_category` (black_hair, specs_black_hair, doctor_cap, doctor_cap_specs,
  white_hair, specs_white_hair, turban, turban_specs)
- `gender` (male / female) + confidence
- `appearance_label` + confidence
- `pose` (front / side)
- detailed per-check scores and notes

---

## 2. Python Environment

**Only one environment is used for everything: `face-validation`**

| Property | Value |
|---|---|
| Conda env name | `face-validation` |
| Python version | 3.11.18 |
| Python executable | `D:\Users\ADMIN\anaconda3\envs\face-validation\python.exe` |

### Key packages in this environment

| Package | Purpose |
|---|---|
| `ultralytics` | Load and run YOLO models (goggles + OIV7) |
| `torch`, `torchvision` | Deep learning (ConvNeXt gender model) |
| `onnxruntime` | Run insightface ONNX models |
| `insightface` | Face detection + landmark extraction |
| `mediapipe` | Additional landmark detection fallback |
| `facenet_pytorch` | Face detection (MTCNN) |
| `deepface` | Auxiliary face analysis |
| `fastapi`, `uvicorn` | Web API server |
| `opencv-python` (cv2) | Image preprocessing |
| `Pillow` (PIL) | Image loading and EXIF handling |
| `numpy` 1.26.4 | Array operations |
| `aiohttp` | Async HTTP (imported by main.py) |
| `pandas` | Data manipulation in test scripts |

> **Important**: Do NOT use the `faceswapih` or any other conda environment.
> Everything — the validation engine, the YOLO goggles model, the test runners —
> all run inside `face-validation`.

### How to activate the environment

```bat
conda activate face-validation
```

Or use the full Python path directly (no conda needed in PATH):

```powershell
& "D:\Users\ADMIN\anaconda3\envs\face-validation\python.exe" script.py
```

---

## 3. Project Folder Structure

```
D:\audit\
│
├── main.py                     ← CORE: Face Validation Engine v2.5.0
├── goggles_wrapper_api.py      ← CORE: Goggles pre-check + API wrapper (port 8000)
├── run_test_dataset.py         ← CORE: Full test suite runner (all 244 images)
├── start_audit_servers.bat     ← CORE: One-click server startup
│
├── models\                     ← All trained model weights
│   ├── goggles_model.pt        ← Custom YOLOv8n goggles detector (6.2 MB)
│   ├── yolov8m-oiv7.pt         ← YOLOv8m OIV7 object classifier (50 MB)
│   ├── best_convnext_large.pth ← ConvNeXt-Large gender classifier (2.2 GB)
│   ├── appearance_model.pt     ← Appearance category model (1.1 GB)
│   ├── appearance_head.npz     ← Appearance classifier head weights
│   ├── clip\                   ← CLIP transformer model (577 MB pytorch_model.bin)
│   ├── face_parsing\           ← Face parsing segmentation model (323 MB)
│   └── models\                 ← InsightFace ONNX models
│       ├── buffalo_l\          ← buffalo_l: w600k_r50.onnx + det_10g.onnx etc.
│       └── antelopev2\         ← antelopev2: glintr100.onnx + scrfd etc.
│
├── dataset\                    ← Test image dataset
│   └── test\
│       ├── black_hair\         ← 45 images
│       ├── doctor_cap\         ← 49 images
│       ├── doctor_cap_specs\   ← 49 images
│       ├── specs_black_hair\   ← 42 images
│       ├── specs_white_hair\   ← 46 images
│       └── white_hair\         ← 13 images
│           TOTAL: 244 images
│
├── results\                    ← All CSV test output files
│   ├── dataset_test_results.csv    ← Full test run (60+ columns per image)
│   └── dataset_goggles_results.csv ← Goggles-only model test
│
└── training\                   ← All training scripts and data (do not edit)
    ├── prepare_and_train.py    ← YOLO training preparation
    ├── run_train.py            ← YOLO training launcher
    ├── batch_annotate_goggles.py  ← Auto-annotate goggles images
    ├── batch_annotate_specs.py    ← Auto-annotate specs images
    ├── fake_albumentations.py  ← Augmentation shim used during training
    ├── run_test_goggles.py     ← Goggles-only model accuracy test
    ├── runs\                   ← YOLO training run output
    │   └── glasses_detector\
    │       └── weights\
    │           ├── best.pt     ← Same as models\goggles_model.pt (original)
    │           └── last.pt
    ├── yolo_glasses_dataset\   ← YOLO format dataset used for training
    ├── goggles_annotations\    ← 450 annotated goggles images
    └── specs_annotations\      ← 801 annotated specs images
```

---

## 4. Core Files — Root Level

### `main.py` — Face Validation Engine v2.5.0

- **Size**: ~587 KB (very large because it has all models + logic inline)
- **Source**: Restored from `F:\Programs\audit\main.py`
- **Version**: v2.5.0 (confirmed in file header)
- **What it does**: The entire validation pipeline in one file. When imported,
  it loads all models at startup. The two public functions are:
  - `run_full_validation(pil_image, bgr_image)` — runs all checks, returns a dict
  - `_clean_response(response_dict)` — normalises the dict for CSV export
- **Model loading**: All model paths are **absolute** using
  `os.path.dirname(os.path.abspath(__file__))` as the base, so the script can
  be called from any working directory.
- **Key path constants** (inside main.py):
  ```python
  _GENDER_MODEL_PATH  = <D:\audit>\models\best_convnext_large.pth
  _APP_MODEL_PATH     = <D:\audit>\models\appearance_head.npz
  _yolo_oiv7_model    = <D:\audit>\models\yolov8m-oiv7.pt   # was relative, now absolute
  ```

> **History note**: Earlier in this project, `main.py` was accidentally replaced
> with the Face Swap API v10.0 file (a different project). It was recovered from
> `F:\Programs\audit\main.py`. The original bytecode files (`main.cpython-310.pyc`
> and `main.cpython-313.pyc` from 11/16) confirmed the correct file was 586–600 KB.
> The Face Swap API version was ~763 KB. The correct file is now in place.

---

### `goggles_wrapper_api.py` — Goggles Wrapper API

- **Port**: 8000
- **Endpoint**: `POST /validate`  (multipart form: `file=<image>`)
- **What it does**:
  1. Receives an image upload
  2. Runs the custom YOLO goggles model (`models/goggles_model.pt`)
  3. If class 1 (goggles/sunglasses) detected with confidence ≥ 0.50 → reject immediately
  4. If no goggles → forward the image to port 8001 (full face validation engine)
- **Why it exists**: The face validation engine on port 8001 uses the OIV7 YOLO
  model for general sunglasses detection. Our custom YOLO model was trained
  specifically for goggles (sports goggles, ski goggles, swimming goggles,
  dark sunglasses). Adding the wrapper catches these before the main engine.

> **Current state**: The wrapper architecture requires TWO servers — port 8000
> (this file) and port 8001 (main.py as a FastAPI server). As of this writing,
> `run_test_dataset.py` calls `main.py` directly as a Python import (no HTTP),
> so the goggles wrapper is NOT used in the test runner. The wrapper is for
> production API use.

---

### `run_test_dataset.py` — Full Test Suite Runner

- **What it does**: Imports `main.py` directly, runs every image in
  `dataset/test/` through `run_full_validation()`, and writes a comprehensive
  CSV to `results/dataset_test_results.csv`
- **Output columns**: 60+ columns covering identity, appearance scores, gender
  scores, pose scores, check scores (goggles, blur, etc.), check details
  (ok/fail + note per check), detected categories, warnings
- **Image count**: 244 images across 6 categories
- **How to run**:
  ```powershell
  & "D:\Users\ADMIN\anaconda3\envs\face-validation\python.exe" D:\audit\run_test_dataset.py
  ```
- **Prerequisite**: `aiohttp` must be installed in `face-validation` env
  (main.py imports it). Install with:
  ```powershell
  & "D:\Users\ADMIN\anaconda3\envs\face-validation\python.exe" -m pip install aiohttp pandas
  ```

---

### `start_audit_servers.bat` — Server Startup

- Activates `face-validation` env
- Changes to `D:\audit`
- Starts `goggles_wrapper_api.py` on port 8000

> For production: run `start_audit_servers.bat` to start the goggles wrapper.
> The face validation engine would need to be started separately on port 8001
> (as a FastAPI/uvicorn server wrapping `main.py`).

---

## 5. Models Folder

All models live in `D:\audit\models\`. They are split into 3 functional categories:

### Category 1 — Goggles Detection

| File | Size | Purpose |
|---|---|---|
| `goggles_model.pt` | 6.2 MB | Custom-trained YOLOv8n. Detects goggles/sunglasses. Class 0 = specs (OK), Class 1 = goggles/sunglasses (REJECT). Confidence threshold: 0.50. mAP50 = 0.929. |
| `yolov8m-oiv7.pt` | 50.3 MB | YOLOv8m trained on Open Images V7 (600 classes). Used by main.py to detect "Sunglasses" category for general sunglass detection. This is the OIV7 pre-trained model, not our custom one. |

### Category 2 — Gender Classification

| File | Size | Purpose |
|---|---|---|
| `best_convnext_large.pth` | 2.2 GB | ConvNeXt-Large architecture fine-tuned for gender classification (male/female). Loaded by main.py at startup via `_GENDER_MODEL_PATH`. |

### Category 3 — Appearance Classification

| File | Size | Purpose |
|---|---|---|
| `appearance_model.pt` | 1.1 GB | Full appearance model for 8-class classification: black_hair, specs_black_hair, white_hair, specs_white_hair, doctor_cap, doctor_cap_specs, turban, turban_specs |
| `appearance_head.npz` | <1 MB | Classifier head weights used with feature extraction. Loaded via `_APP_MODEL_PATH` in main.py. |
| `clip\` | 577 MB | CLIP (Contrastive Language-Image Pretraining) model. Used for feature extraction as part of appearance pipeline. Contains: pytorch_model.bin, tokenizer, preprocessor config. |
| `face_parsing\` | 323 MB | Face parsing / segmentation model (safetensors format). Used to understand face regions (hair, skin, etc.) for appearance classification. |
| `models\buffalo_l\` | ~325 MB | InsightFace buffalo_l pack: face recognition (w600k_r50.onnx), detection (det_10g.onnx), 2D landmarks (2d106det.onnx), 3D landmarks (1k3d68.onnx), gender/age (genderage.onnx). |
| `models\antelopev2\` | ~408 MB | InsightFace antelopev2 pack: same set of models, higher accuracy variant. |

---

## 6. Face Validation Engine — main.py Deep Dive

### Validation Priority Chain

The engine processes each image through these checks **in strict order**.
The first failure returns immediately and no further checks run.

```
Image input
    │
    ▼
[1] Dimension Guard
    Rule: width >= 100 AND height >= 100
    Fail → "image_too_small"

    ▼
[2] No Human (InsightFace + MTCNN)
    Rule: at least one face must be detected
    Fail → "no_face"

    ▼
[3] Irrelevant Image (OIV7 YOLO — yolov8m-oiv7.pt)
    Rule: image must contain a person
    Fail → "irrelevant"

    ▼
[4] Multiple Persons
    Rule: exactly one face in frame
    Fail → "multiple_person"

    ▼
[5] No Landmarks
    Rule: facial landmarks must be extractable
    Fail → "no_landmarks"

    ▼
[6] Sunglasses / Goggles  ← KEY CHECK
    Rule: no sunglasses/goggles on face
    Uses: OIV7 YOLO for "Sunglasses" class (sunglasses_check)
          + our custom goggles_model.pt via wrapper
    Fail → "sunglasses"

    ▼
[7] Eyes Closed
    Rule: both eyes must be open
    Uses: MediaPipe landmarks + EAR (Eye Aspect Ratio)
    Fail → "eyes_closed"

    ▼
[8] Pose (front-facing)
    Rule: face must be frontal (not profile)
    Uses: landmark angles
    Fail → "side_face"

    ▼
[9] Appearance Classification
    8 categories: black_hair, specs_black_hair, white_hair, specs_white_hair,
                  doctor_cap, doctor_cap_specs, turban, turban_specs
    Uses: appearance_model.pt + clip/ + face_parsing/
    Result → sets audit_category

    ▼
[10] Blur Check
    Rule: blur score must be below threshold
    Uses: Laplacian variance
    Fail → "blurred"

    ▼
[11] Resolution Check
    Rule: effective face resolution must meet minimum
    Fail → "low_resolution"

    ▼
[12] Gender Classification
    Uses: best_convnext_large.pth (ConvNeXt-Large)
    Result → sets gender field (male / female + confidence)

    ▼
PASS → status = "accepted", is_valid = 1
```

### How to call main.py from Python

```python
import sys
sys.path.insert(0, r"D:\audit")
import main as M

# Load an image
from PIL import Image, ImageOps
import cv2, numpy as np

pil = ImageOps.exif_transpose(Image.open("photo.jpg")).convert("RGB")
bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# Run all checks
result = M.run_full_validation(pil, bgr)

# Get clean flat dict for CSV
clean = M._clean_response(result)
print(clean["status"])          # accepted / review / rejected
print(clean["audit_category"])  # black_hair, doctor_cap, etc.
print(clean["gender"])          # male / female
```

---

## 7. Custom YOLO Goggles Model

### Why we trained a custom model

The OIV7 YOLO model inside `main.py` detects "Sunglasses" (a standard OIV7
class). It works for typical sunglasses but misses:
- Sports goggles (ski goggles, swimming goggles, safety goggles)
- Wrap-around dark goggles
- Oversized sunglasses with unusual shapes

We trained a **custom YOLOv8n** model specifically for these cases.

### Model details

| Property | Value |
|---|---|
| Architecture | YOLOv8n (nano — smallest and fastest) |
| File | `D:\audit\models\goggles_model.pt` |
| Original training output | `D:\audit\training\runs\glasses_detector\weights\best.pt` |
| File size | 6.2 MB |
| Training completed | 11/18/2026 |
| mAP50 | **0.929** (92.9%) |

### Classes

| Class ID | Name | Action |
|---|---|---|
| 0 | `specs` | Regular clear glasses / spectacles → **ACCEPT** (do not reject) |
| 1 | `goggles` | Goggles or dark sunglasses → **REJECT** |

> Class 0 (specs) is intentionally not rejected. The face validation engine has
> a category `specs_black_hair` and `specs_white_hair` for people wearing clear
> glasses — these are valid photos.

### Confidence threshold

Rejection threshold: **0.50** (50% confidence)
If class 1 confidence >= 0.50 → reject as "goggles"

### Training data

| Folder | Description | Count |
|---|---|---|
| `training\goggles_annotations\` | Images annotated as class 1 (goggles) | 450 |
| `training\specs_annotations\` | Images annotated as class 0 (specs) | 801 |
| `training\yolo_glasses_dataset\` | Final YOLO-format dataset (images + labels) | — |

Annotation was done using `batch_annotate_goggles.py` and
`batch_annotate_specs.py` (semi-automatic with manual review).

### Test accuracy on 244-image dataset

| Category | Correct | Total | Accuracy |
|---|---|---|---|
| black_hair | 44 | 45 | 97.8% |
| doctor_cap | 49 | 49 | 100% |
| doctor_cap_specs | 49 | 49 | 100% |
| specs_black_hair | 42 | 42 | 100% |
| specs_white_hair | 46 | 46 | 100% |
| white_hair | 13 | 13 | 100% |
| **TOTAL** | **243** | **244** | **99.6%** |

**1 false positive**: `black_hair/9229.png` — goggles confidence 0.644
(no actual goggles, model triggered incorrectly on this one image)

---

## 8. Goggles Wrapper API

### Architecture

```
Client
  │
  │  POST /validate  (image file)
  ▼
goggles_wrapper_api.py   ← port 8000
  │
  ├── [STEP 1] Run models\goggles_model.pt
  │       class 1 confidence >= 0.50?
  │       YES → return {"is_valid": 0, "reason": "goggles"} immediately
  │       NO  → continue
  │
  └── [STEP 2] Forward image to port 8001 (main.py FastAPI)
              Return full validation response
```

### File: `goggles_wrapper_api.py`

```python
MODEL_PATH        = D:\audit\models\goggles_model.pt
VALIDATION_URL    = http://localhost:8001/validate
GOGGLES_CLASS_ID  = 1
CONF_THRESHOLD    = 0.50
```

### How to start

```bat
D:\audit\start_audit_servers.bat
```

This starts the wrapper on port 8000. The face validation server on port 8001
must be started separately.

> **Note for the full test pipeline**: `run_test_dataset.py` does NOT use the
> HTTP wrapper. It imports `main.py` directly as a Python module, which is
> faster for batch testing. The goggles check in the wrapper is an additional
> layer ON TOP of main.py's built-in OIV7 sunglasses check — both run in
> production use.

---

## 9. Dataset Structure

```
D:\audit\dataset\test\
├── black_hair\          45 images   Normal photos, black hair, no headwear
├── doctor_cap\          49 images   Medical scrub cap, no glasses
├── doctor_cap_specs\    49 images   Medical scrub cap + clear glasses
├── specs_black_hair\    42 images   Clear glasses + black hair
├── specs_white_hair\    46 images   Clear glasses + white hair
└── white_hair\          13 images   Normal photos, white hair, no headwear
                        ───
TOTAL                   244 images
```

**Expected validation results** (what the engine should output per category):

| Category | Expected audit_category | Expected status |
|---|---|---|
| black_hair | black_hair | accepted |
| doctor_cap | doctor_cap | accepted |
| doctor_cap_specs | doctor_cap_specs | accepted |
| specs_black_hair | specs_black_hair | accepted |
| specs_white_hair | specs_white_hair | accepted |
| white_hair | white_hair | accepted |

Images in this dataset contain **no goggles** — all should be accepted by the
goggles check (which is why the custom model's 1 false positive is notable).

---

## 10. Test Results

### Goggles model only (`results\dataset_goggles_results.csv`)

- **Total images**: 244
- **Correctly handled**: 243 (99.6%)
- **False positives** (wrongly rejected): 1 — `black_hair/9229.png`
- **False negatives** (goggles missed): 0 (no goggles images in test set)

### Full validation pipeline (`results\dataset_test_results.csv`)

- **Total images**: 244
- **Columns**: 60+ per row
- **Covers**: all checks (blur, pose, eyes, appearance, gender, goggles, etc.)
- **Generated by**: `run_test_dataset.py`
- **Status**: This file exists from the last run on 11/18/2026. Re-run to get
  updated results with the latest model paths.

---

## 11. How to Run — Step by Step

### Step 0: Install missing packages (one-time)

```powershell
& "D:\Users\ADMIN\anaconda3\envs\face-validation\python.exe" -m pip install aiohttp pandas
```

### Step 1: Run the full test suite

```powershell
& "D:\Users\ADMIN\anaconda3\envs\face-validation\python.exe" D:\audit\run_test_dataset.py
```

Output: `D:\audit\results\dataset_test_results.csv`

### Step 2: Run the goggles model test only (optional)

```powershell
& "D:\Users\ADMIN\anaconda3\envs\face-validation\python.exe" D:\audit\training\run_test_goggles.py
```

Output: `D:\audit\results\dataset_goggles_results.csv`

### Step 3: Start the production API server

```bat
D:\audit\start_audit_servers.bat
```

This starts goggles_wrapper_api.py on port 8000.

### Step 4: Send an image to the API

```powershell
curl -X POST http://localhost:8000/validate -F "file=@photo.jpg"
```

---

## 12. Training Folder Contents

The `D:\audit\training\` folder contains everything used to build the custom
goggles YOLO model. **You do not need to touch this folder for normal use.**

| File | Purpose |
|---|---|
| `prepare_and_train.py` | Prepares YOLO dataset format and launches training |
| `run_train.py` | Simple launcher for the training process |
| `batch_annotate_goggles.py` | Auto-generates YOLO annotations for goggles images |
| `batch_annotate_specs.py` | Auto-generates YOLO annotations for specs images |
| `generate_annotation_example.py` | Shows annotation examples visually |
| `fake_albumentations.py` | Stub module that bypasses albumentations dependency during training |
| `fake_albumentations.py` | Augmentation compatibility shim |
| `run_test_goggles.py` | Tests goggles model accuracy on dataset (uses `D:\audit\models\goggles_model.pt`) |
| `train_appearance.py` | Training script for the appearance classifier |
| `extract_features.py` | Feature extraction for appearance model training |
| `extract_convnext.py` | ConvNeXt feature extractor |
| `build_bundle.py` | Bundles model weights |
| `compare_models.py` | Model comparison utility |
| `smoke_test_service.py` | Quick smoke test for the validation API |
| `verify_integration.py` | Integration verification script |

| Folder | Contents |
|---|---|
| `runs\glasses_detector\weights\` | YOLO training output: best.pt and last.pt (best.pt = same as models\goggles_model.pt) |
| `yolo_glasses_dataset\` | YOLO-format dataset: images + labels in YOLO txt format |
| `goggles_annotations\` | 450 annotated images for goggles class |
| `specs_annotations\` | 801 annotated images for specs class |

---

## 13. What Was Cleaned Up and Why

This documents everything removed from the project so nothing is a mystery.

### Files deleted from root

| File | Why removed |
|---|---|
| `appearance_classifier.py` | Early standalone prototype, superseded by main.py |
| `eval_appearance.py` | Old evaluation script, superseded by run_test_dataset.py |
| `test_categories.py` | Old debugging script, superseded by run_test_dataset.py |
| `test_focused_goggles.py` | Old debugging script, superseded by run_test_goggles.py |
| `test_goggles_model.py` | Old debugging script, superseded by run_test_goggles.py |
| `verify_fixes.py` | One-off fix verification, no longer needed |
| `verify_fixes.csv` | Output of verify_fixes.py |
| `category_test_results.csv` | Old partial results from initial testing |
| `category_test_run.err` | Old error log |
| `category_test_run.log` | Old run log |

### Folders deleted from root

| Folder | Why removed |
|---|---|
| `__pycache__\` | Python bytecode cache (auto-regenerated, not source code) |
| `annotation_output\` | 403 annotated image previews generated during training prep (interim artifact) |
| `annotation_specs_review\` | 23 annotation review images (interim artifact) |
| `category_test\` | Old test category staging folder |
| `detection_outputs\` | Interim detection preview images |

### Models removed from `models\`

These belonged to the **Face Swap API** (a completely separate project that was
accidentally mixed in):

| File | Size | Project |
|---|---|---|
| `arcface_converter_ghost.onnx` | 20 MB | Face Swap API |
| `arcface_converter_simswap.onnx` | 20 MB | Face Swap API |
| `ghost_3_256.onnx` | 816 MB | Face Swap API |
| `hyperswap_1c_256.onnx` | 384 MB | Face Swap API |
| `inswapper_128_fp16.onnx` | 265 MB | Face Swap API |
| `simswap_unofficial_512.onnx` | 228 MB | Face Swap API |
| `codeformer\` folder | — | Face Swap API |

**Total removed**: ~1.75 GB of Face Swap API weights that had no role in
face validation.

### Scripts moved from root → `training\`

| File | Now in |
|---|---|
| `prepare_and_train.py` | training\ |
| `run_train.py` | training\ |
| `batch_annotate_goggles.py` | training\ |
| `batch_annotate_specs.py` | training\ |
| `generate_annotation_example.py` | training\ |
| `fake_albumentations.py` | training\ |
| `run_test_goggles.py` | training\ |

### Folders moved from root → `training\`

| Folder | Now in |
|---|---|
| `runs\` | training\runs\ |
| `yolo_glasses_dataset\` | training\yolo_glasses_dataset\ |
| `goggles_annotations\` | training\goggles_annotations\ |
| `specs_annotations\` | training\specs_annotations\ |

### Model moved

| File | Old location | New location | Why |
|---|---|---|---|
| `yolov8m-oiv7.pt` | `D:\audit\yolov8m-oiv7.pt` (root) | `D:\audit\models\yolov8m-oiv7.pt` | Centralised into models\ with all other weights. main.py path updated. |
| `goggles_model.pt` | `training\runs\glasses_detector\weights\best.pt` | `models\goggles_model.pt` | Copy moved to models\ for clean access. Original kept in training\. |

---

## 14. Pending Tasks

| Task | Command |
|---|---|
| Install missing packages | `& "D:\Users\ADMIN\anaconda3\envs\face-validation\python.exe" -m pip install aiohttp pandas` |
| Run full test pipeline | `& "D:\Users\ADMIN\anaconda3\envs\face-validation\python.exe" D:\audit\run_test_dataset.py` |
| Delete locked leftover file | Close whatever has `D:\audit\dataset_goggles_results.csv` open (Excel?), then delete it — the copy is in `results\` |

---

*Document prepared: 2026-11-19*
*Engine version: Face Validation Engine v2.5.0*
*Environment: face-validation (Python 3.11.18)*

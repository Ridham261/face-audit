"""
Test the custom YOLO goggles model against dataset/test/.
Outputs dataset_goggles_results.csv

Columns:
  file, true_category, goggles_detected, goggles_conf,
  specs_detected, specs_conf, status, correct

  - goggles_detected : True if class-1 (goggles) conf >= 0.50
  - specs_detected   : True if class-0 (specs)   conf >= 0.50
  - status           : rejected (goggles found) | accepted (no goggles)
  - correct          : True if model correctly handled the category
      * specs categories  -> should be accepted (specs OK)
      * other categories  -> should be accepted (no goggles)
      * (no goggles folder in test set, so all should be accepted)

Run with:
    conda activate face-validation
    python D:\audit\training\run_test_goggles.py
"""
import csv
import io
import os

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(HERE)   # D:\audit

MODEL_PATH  = os.path.join(ROOT, "models", "goggles_model.pt")
TEST_ROOT   = os.path.join(ROOT, "dataset", "test")
OUT_CSV     = os.path.join(ROOT, "results", "dataset_goggles_results.csv")
EXTS        = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

GOGGLES_CLASS = 1   # class 1 = goggles / sunglasses  → reject
SPECS_CLASS   = 0   # class 0 = specs / clear glasses → accept
CONF_THRESH   = 0.50

# Categories where specs are expected (model should detect class 0, not class 1)
SPECS_CATEGORIES = {"specs_black_hair", "specs_white_hair", "doctor_cap_specs"}

print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded OK\n")


def run_one(path: str) -> dict:
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    results = model(bgr, verbose=False)

    best_goggles = 0.0
    best_specs   = 0.0
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == GOGGLES_CLASS:
                best_goggles = max(best_goggles, conf)
            elif cls == SPECS_CLASS:
                best_specs = max(best_specs, conf)

    goggles_det = best_goggles >= CONF_THRESH
    specs_det   = best_specs   >= CONF_THRESH
    status      = "rejected" if goggles_det else "accepted"
    return {
        "goggles_detected": goggles_det,
        "goggles_conf":     round(best_goggles, 4),
        "specs_detected":   specs_det,
        "specs_conf":       round(best_specs, 4),
        "status":           status,
    }


entries = []
for cat in sorted(os.listdir(TEST_ROOT)):
    cat_dir = os.path.join(TEST_ROOT, cat)
    if not os.path.isdir(cat_dir):
        continue
    for fn in sorted(os.listdir(cat_dir)):
        if fn.lower().endswith(EXTS):
            entries.append({"true_category": cat, "file": fn,
                             "path": os.path.join(cat_dir, fn)})

total = len(entries)
print(f"Found {total} images in {len(set(e['true_category'] for e in entries))} categories\n")

rows = []
for i, e in enumerate(entries, 1):
    try:
        out = run_one(e["path"])
        # All categories in this test set should be accepted (no goggles images)
        correct = (out["status"] == "accepted")
        mark    = "OK " if correct else "BAD"
        row = {
            "file":             e["file"],
            "true_category":    e["true_category"],
            "goggles_detected": out["goggles_detected"],
            "goggles_conf":     out["goggles_conf"],
            "specs_detected":   out["specs_detected"],
            "specs_conf":       out["specs_conf"],
            "status":           out["status"],
            "correct":          correct,
        }
        print(
            f"[{i:3d}/{total}] {mark}  {e['true_category']:20s}  "
            f"{e['file']:40s}  goggles={out['goggles_conf']:.3f}  "
            f"specs={out['specs_conf']:.3f}  -> {out['status']}",
            flush=True,
        )
    except Exception as exc:
        row = {
            "file": e["file"], "true_category": e["true_category"],
            "goggles_detected": None, "goggles_conf": None,
            "specs_detected": None, "specs_conf": None,
            "status": f"ERROR:{type(exc).__name__}",
            "correct": False,
        }
        print(f"[{i:3d}/{total}] ERR  {e['true_category']:20s}  {e['file']}  — {exc}", flush=True)
    rows.append(row)

# Write CSV
fieldnames = ["file", "true_category", "goggles_detected", "goggles_conf",
              "specs_detected", "specs_conf", "status", "correct"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
    w.writeheader()
    w.writerows(rows)

print(f"\nSaved {len(rows)} rows → {OUT_CSV}")

# Accuracy summary
from collections import Counter, defaultdict
by_cat = defaultdict(list)
for r in rows:
    by_cat[r["true_category"]].append(r)

print("\n========= GOGGLES MODEL ACCURACY =========")
tot_ok = tot = 0
for cat in sorted(by_cat):
    rs  = by_cat[cat]
    ok  = sum(1 for r in rs if r["correct"])
    bad = [r for r in rs if not r["correct"]]
    tot_ok += ok; tot += len(rs)
    conf_note = ""
    if bad:
        confs = [r["goggles_conf"] for r in bad if r["goggles_conf"] is not None]
        if confs:
            conf_note = f"  false-pos goggles_conf: {[f'{c:.3f}' for c in confs]}"
    print(
        f"  {cat:22s}  {ok:3d}/{len(rs):<3d} = {100*ok/len(rs):5.1f}%{conf_note}"
    )
print(f"  {'ALL':22s}  {tot_ok:3d}/{tot:<3d} = {100*tot_ok/max(tot,1):.1f}%")
print()
goggles_fp = sum(1 for r in rows if r["status"] == "rejected")
print(f"False positives (wrongly rejected): {goggles_fp}/{tot}")

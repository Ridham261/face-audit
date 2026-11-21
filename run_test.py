"""
Run all test images through full face validation and write results CSV.
Output: D:\audit\results\dataset_test_results.csv
"""
import csv, json, os, sys, warnings, statistics, platform
import mlflow
warnings.filterwarnings("ignore")

import cv2, numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(HERE, 'mlflow.db')}")
mlflow.set_experiment("face-audit-test")

TEST_ROOT  = os.path.join(HERE, "dataset", "test")
_csv_base  = os.path.join(HERE, "results", "dataset_test_results.csv")
# If locked (open in Excel), write to a timestamped copy instead
def _pick_csv(base):
    try:
        with open(base, "a"):
            pass
        return base
    except PermissionError:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return base.replace(".csv", f"_{ts}.csv")
OUT_CSV    = _pick_csv(_csv_base)
EXTS       = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PER_CAT    = 999  # all images per category

APP_LABELS   = ["black_hair","specs_black_hair","white_hair","specs_white_hair",
                "doctor_cap","doctor_cap_specs","turban","turban_specs"]
CHECK_SCORES = ["goggles","multiple_person","blur","low_resolution",
                "eyes_closed","focused","headwear","no_face"]
CHECK_DETAIL = ["one_person","front_facing","eyes_open","no_sunglasses","sharp","resolution"]

FIELDNAMES = [
    # --- identity ---
    "file", "true_category", "correct",
    # --- top level ---
    "status", "needs_manual_review", "is_valid", "processing_ms",
    # --- result ---
    "result_code", "result_title", "result_message",
    # --- detected (fired categories + scores) ---
    "detected_json",
    # --- not triggered ---
    "not_triggered_json",
    # --- appearance ---
    "appearance_label", "appearance_title", "appearance_confidence",
    "appearance_source", "appearance_uncertain",
    "hair_color", "specs_detected", "headwear_kind",
    # --- appearance scores (8 classes) ---
] + [f"score_app_{l}" for l in APP_LABELS] + [
    # --- gender ---
    "gender", "gender_confidence",
    "score_gender_male", "score_gender_female",
    # --- pose / face ---
    "pose", "face_total", "face_frontal", "face_side",
    "score_pose_front", "score_pose_side",
    # --- check scores ---
] + [f"score_check_{k}" for k in CHECK_SCORES] + [
    # --- check details (ok + note per check) ---
] + [item for k in CHECK_DETAIL for item in [f"check_{k}_ok", f"check_{k}_note"]] + [
    # --- quality ---
    "blur_score", "is_blurred", "low_res", "sunglasses",
    # --- matched ---
    "matched_appearance", "matched_gender", "matched_pose",
    "matched_issues", "matched_categories",
    # --- warnings ---
    "warnings",
]

# ── load models once ──────────────────────────────────────────────────────────
print("Loading models...", flush=True)
import main as M
print("Models ready.\n", flush=True)

_mlflow_run = mlflow.start_run()

import torch
mlflow.log_params({
    "dataset":        "D:/audit/dataset/test",
    "python_version": platform.python_version(),
    "torch_version":  torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "os":             platform.system(),
})
mlflow.set_tags({
    "run_type":    "full_test",
    "model_ver":   "v2.5.0",
    "dataset":     "244_images",
    "triggered_by": "manual",
})

# ── collect 10 images per category ───────────────────────────────────────────
entries = []
for cat in sorted(os.listdir(TEST_ROOT)):
    cat_path = os.path.join(TEST_ROOT, cat)
    if not os.path.isdir(cat_path):
        continue
    files = sorted([f for f in os.listdir(cat_path) if f.lower().endswith(EXTS)])[:PER_CAT]
    for fn in files:
        entries.append({"true_category": cat, "file": fn,
                        "path": os.path.join(cat_path, fn)})

total = len(entries)
print(f"Images: {total}  (all categories)\n", flush=True)

# ── run ───────────────────────────────────────────────────────────────────────
rows = []
for idx, e in enumerate(entries, 1):
    try:
        pil = ImageOps.exif_transpose(Image.open(e["path"])).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        res = M.run_full_validation(pil, bgr)
        c   = M._clean_response(res)

        sc_app  = (c.get("scores") or {}).get("appearance", {})
        sc_gen  = (c.get("scores") or {}).get("gender", {})
        sc_pose = (c.get("scores") or {}).get("pose", {})
        sc_chk  = (c.get("scores") or {}).get("checks", {})
        app     = c.get("appearance") or {}
        face    = c.get("face") or {}
        chks    = c.get("checks") or {}
        qual    = c.get("quality") or {}
        result  = c.get("result") or {}
        matched = c.get("matched") or {}
        detected   = c.get("detected") or {}
        not_trig   = c.get("not_triggered") or {}

        audit_cat = c.get("audit_category") or result.get("code", "")
        correct   = (audit_cat == e["true_category"])

        row = {
            # identity
            "file":                e["file"],
            "true_category":       e["true_category"],
            "correct":             correct,
            # top level
            "status":              c.get("status", ""),
            "needs_manual_review": c.get("needs_manual_review", False),
            "is_valid":            c.get("is_valid", False),
            "processing_ms":       round(c.get("processing_ms", 0), 1),
            # result
            "result_code":         result.get("code", ""),
            "result_title":        result.get("title", ""),
            "result_message":      result.get("message", ""),
            # detected / not_triggered
            "detected_json":       json.dumps(detected),
            "not_triggered_json":  json.dumps(not_trig),
            # appearance
            "appearance_label":    app.get("label", ""),
            "appearance_title":    app.get("title", ""),
            "appearance_confidence": app.get("confidence"),
            "appearance_source":   app.get("source", ""),
            "appearance_uncertain": app.get("uncertain", False),
            "hair_color":          app.get("hair_color", ""),
            "specs_detected":      app.get("specs", False),
            "headwear_kind":       app.get("headwear", ""),
            # gender
            "gender":              face.get("gender", ""),
            "gender_confidence":   face.get("gender_confidence"),
            "score_gender_male":   sc_gen.get("male"),
            "score_gender_female": sc_gen.get("female"),
            # pose / face
            "pose":                face.get("pose", ""),
            "face_total":          face.get("total"),
            "face_frontal":        face.get("frontal"),
            "face_side":           face.get("side"),
            "score_pose_front":    sc_pose.get("front"),
            "score_pose_side":     sc_pose.get("side"),
            # quality
            "blur_score":          qual.get("blur_score"),
            "is_blurred":          qual.get("is_blurred", False),
            "low_res":             qual.get("low_res", False),
            "sunglasses":          qual.get("sunglasses", False),
            # matched
            "matched_appearance":  matched.get("appearance", ""),
            "matched_gender":      matched.get("gender", ""),
            "matched_pose":        matched.get("pose", ""),
            "matched_issues":      ", ".join(matched.get("issues") or []),
            "matched_categories":  ", ".join(matched.get("categories") or []),
            # warnings
            "warnings":            " | ".join(c.get("warnings") or []),
        }
        for lbl in APP_LABELS:
            row[f"score_app_{lbl}"] = sc_app.get(lbl, "")
        for k in CHECK_SCORES:
            row[f"score_check_{k}"] = sc_chk.get(k, "")
        for k in CHECK_DETAIL:
            d = chks.get(k, {})
            row[f"check_{k}_ok"]   = d.get("ok", "")
            row[f"check_{k}_note"] = d.get("note", "")

        mark = "OK " if correct else "BAD"
        print(f"[{idx:3d}/{total}] {mark}  {e['true_category']:22s}  {e['file']:28s}  "
              f"{c.get('status',''):10s}  {audit_cat:22s}  "
              f"gender={face.get('gender','?')}", flush=True)

    except Exception as exc:
        print(f"[{idx:3d}/{total}] ERR  {e['true_category']:22s}  {e['file']}  — {exc}", flush=True)
        row = {f: "" for f in FIELDNAMES}
        row.update({"file": e["file"], "true_category": e["true_category"],
                    "correct": False, "status": f"ERROR", "warnings": str(exc)})

    rows.append(row)

# ── write CSV ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore", restval="")
    w.writeheader()
    w.writerows(rows)

print(f"\nSaved {len(rows)} rows  →  {OUT_CSV}")

# ── summary ───────────────────────────────────────────────────────────────────
correct_n = sum(1 for r in rows if r.get("correct") is True)
total_n   = max(len(rows), 1)
accuracy  = correct_n / total_n

print(f"\n{'='*55}")
print(f"  TOTAL  {correct_n}/{len(rows)} correct  ({100*correct_n//total_n}%)")
print(f"{'='*55}")
from collections import defaultdict
by_cat = defaultdict(list)
for r in rows:
    by_cat[r["true_category"]].append(r)
for cat in sorted(by_cat):
    rs = by_cat[cat]
    ok = sum(1 for r in rs if r.get("correct") is True)
    print(f"  {cat:24s}  {ok:2d}/{len(rs)}")
print()

# ── log to MLflow ─────────────────────────────────────────────────────────────

# 1. overall accuracy
mlflow.log_metrics({
    "accuracy": round(accuracy, 4),
    "correct":  correct_n,
    "total":    len(rows),
})

# 2. per-category accuracy
for cat in sorted(by_cat):
    rs = by_cat[cat]
    ok = sum(1 for r in rs if r.get("correct") is True)
    mlflow.log_metric(f"acc_{cat.replace('-','_')}", round(ok / max(len(rs), 1), 4))

# 3. per-status breakdown (accepted / rejected / review)
from collections import Counter
statuses = Counter(r.get("status", "") for r in rows)
for status in ["accepted", "rejected", "review"]:
    cnt = statuses.get(status, 0)
    mlflow.log_metric(f"status_{status}", cnt)
    mlflow.log_metric(f"rate_{status}",   round(cnt / total_n, 4))

# 4. processing speed
times = [r["processing_ms"] for r in rows if r.get("processing_ms")]
if times:
    mlflow.log_metrics({
        "avg_ms": round(statistics.mean(times), 1),
        "max_ms": round(max(times), 1),
        "min_ms": round(min(times), 1),
    })

# 5. gender accuracy
gender_rows = [r for r in rows if r.get("gender")]
if gender_rows:
    correct_gender = sum(1 for r in gender_rows if r.get("matched_gender") == "correct")
    mlflow.log_metric("gender_accuracy", round(correct_gender / len(gender_rows), 4))

# 6. rejection reason breakdown
reasons = Counter(r.get("result_code", "") for r in rows if r.get("status") == "rejected")
for code, cnt in reasons.items():
    if code:
        mlflow.log_metric(f"rejected_{code}", cnt)

# 7. attach CSV
mlflow.log_artifact(OUT_CSV, artifact_path="results")

mlflow.end_run()
print(f"MLflow run: {_mlflow_run.info.run_id}")

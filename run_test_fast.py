"""
Parallel dataset test — same CSV format as run_test.py, faster via multiprocessing.
Usage: python run_test_fast.py [--workers N]
"""
import csv, json, os, sys, warnings, argparse, multiprocessing, time
warnings.filterwarnings("ignore")

HERE     = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.path.join(HERE, "dataset", "test")
EXTS     = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PER_CAT  = 999

APP_LABELS   = ["black_hair","specs_black_hair","white_hair","specs_white_hair",
                "doctor_cap","doctor_cap_specs","turban","turban_specs"]
CHECK_SCORES = ["goggles","multiple_person","blur","low_resolution",
                "eyes_closed","focused","headwear","no_face"]
CHECK_DETAIL = ["one_person","front_facing","eyes_open","no_sunglasses","sharp","resolution"]

FIELDNAMES = [
    "file", "true_category", "correct",
    "status", "needs_manual_review", "is_valid", "processing_ms",
    "result_code", "result_title", "result_message",
    "detected_json", "not_triggered_json",
    "appearance_label", "appearance_title", "appearance_confidence",
    "appearance_source", "appearance_uncertain",
    "hair_color", "specs_detected", "headwear_kind",
] + [f"score_app_{l}" for l in APP_LABELS] + [
    "gender", "gender_confidence",
    "score_gender_male", "score_gender_female",
    "pose", "face_total", "face_frontal", "face_side",
    "score_pose_front", "score_pose_side",
] + [f"score_check_{k}" for k in CHECK_SCORES] + [
    item for k in CHECK_DETAIL for item in [f"check_{k}_ok", f"check_{k}_note"]
] + [
    "blur_score", "is_blurred", "low_res", "sunglasses",
    "matched_appearance", "matched_gender", "matched_pose",
    "matched_issues", "matched_categories",
    "warnings",
]


def _pick_csv(base):
    try:
        with open(base, "a"):
            pass
        return base
    except PermissionError:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return base.replace(".csv", f"_{ts}.csv")


def _process_chunk(entries):
    """Worker: load models once, process all assigned entries, return rows."""
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    warnings.filterwarnings("ignore")
    import cv2, numpy as np
    from PIL import Image, ImageOps

    pid = os.getpid()
    print(f"[worker {pid}] loading models ({len(entries)} images)...", flush=True)
    import main as M
    print(f"[worker {pid}] models ready", flush=True)

    rows = []
    total = len(entries)
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
                "file":                e["file"],
                "true_category":       e["true_category"],
                "correct":             correct,
                "status":              c.get("status", ""),
                "needs_manual_review": c.get("needs_manual_review", False),
                "is_valid":            c.get("is_valid", False),
                "processing_ms":       round(c.get("processing_ms", 0), 1),
                "result_code":         result.get("code", ""),
                "result_title":        result.get("title", ""),
                "result_message":      result.get("message", ""),
                "detected_json":       json.dumps(detected),
                "not_triggered_json":  json.dumps(not_trig),
                "appearance_label":    app.get("label", ""),
                "appearance_title":    app.get("title", ""),
                "appearance_confidence": app.get("confidence"),
                "appearance_source":   app.get("source", ""),
                "appearance_uncertain": app.get("uncertain", False),
                "hair_color":          app.get("hair_color", ""),
                "specs_detected":      app.get("specs", False),
                "headwear_kind":       app.get("headwear", ""),
                "gender":              face.get("gender", ""),
                "gender_confidence":   face.get("gender_confidence"),
                "score_gender_male":   sc_gen.get("male"),
                "score_gender_female": sc_gen.get("female"),
                "pose":                face.get("pose", ""),
                "face_total":          face.get("total"),
                "face_frontal":        face.get("frontal"),
                "face_side":           face.get("side"),
                "score_pose_front":    sc_pose.get("front"),
                "score_pose_side":     sc_pose.get("side"),
                "blur_score":          qual.get("blur_score"),
                "is_blurred":          qual.get("is_blurred", False),
                "low_res":             qual.get("low_res", False),
                "sunglasses":          qual.get("sunglasses", False),
                "matched_appearance":  matched.get("appearance", ""),
                "matched_gender":      matched.get("gender", ""),
                "matched_pose":        matched.get("pose", ""),
                "matched_issues":      ", ".join(matched.get("issues") or []),
                "matched_categories":  ", ".join(matched.get("categories") or []),
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
            print(f"[worker {pid}] [{idx:3d}/{total}] {mark}  {e['true_category']:22s}  {e['file']}", flush=True)

        except Exception as exc:
            print(f"[worker {pid}] [{idx:3d}/{total}] ERR  {e['true_category']:22s}  {e['file']}  — {exc}", flush=True)
            row = {f: "" for f in FIELDNAMES}
            row.update({"file": e["file"], "true_category": e["true_category"],
                        "correct": False, "status": "ERROR", "warnings": str(exc)})

        rows.append(row)
    return rows


def _run_pool(entries, n_workers):
    chunks = [entries[i::n_workers] for i in range(n_workers)]
    t0 = time.time()
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(_process_chunk, chunks)
    elapsed = time.time() - t0
    flat = []
    for chunk_rows in results:
        flat.extend(chunk_rows)
    return flat, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    # Collect all entries
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
    print(f"Images: {total}\n", flush=True)

    # Phase 1: 1 worker loads models and processes 1 image to confirm working
    print("=== PHASE 1: preflight (1 worker, 1 image) ===", flush=True)
    preflight, _ = _run_pool(entries[:1], 1)
    if preflight and preflight[0].get("status") not in ("", "ERROR"):
        print(f"Preflight OK — {preflight[0]['file']} → {preflight[0].get('result_code','?')}\n", flush=True)
    elif preflight and preflight[0].get("status") == "ERROR":
        print(f"Preflight FAILED: {preflight[0].get('warnings','')}\nAborting.", flush=True)
        return
    else:
        print("Preflight done.\n", flush=True)

    # Phase 2: full run with requested workers
    n_workers = min(args.workers, total)
    print(f"=== PHASE 2: full test ({n_workers} workers, {total} images) ===", flush=True)
    flat, elapsed = _run_pool(entries, n_workers)

    # Re-sort by original entry order
    order = {(e["true_category"], e["file"]): i for i, e in enumerate(entries)}
    flat.sort(key=lambda r: order.get((r.get("true_category",""), r.get("file","")), 999999))

    print(f"\nAll done in {elapsed:.0f}s", flush=True)

    # Write CSV
    out_csv = _pick_csv(os.path.join(HERE, "results", "dataset_test_results.csv"))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(flat)

    print(f"Saved {len(flat)} rows  →  {out_csv}")

    # Summary
    correct_n = sum(1 for r in flat if r.get("correct") is True)
    print(f"\n{'='*55}")
    print(f"  TOTAL  {correct_n}/{len(flat)} correct  ({100*correct_n//max(len(flat),1)}%)")
    print(f"{'='*55}")
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in flat:
        by_cat[r["true_category"]].append(r)
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        ok = sum(1 for r in rs if r.get("correct") is True)
        print(f"  {cat:24s}  {ok:2d}/{len(rs)}")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

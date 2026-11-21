"""
Batch bounding-box annotation for spectacles images.
Outputs per image into annotation_specs_review\:
  <name>_annotated.jpg   – zoomed preview with red bbox + green dots
  labels\<name>.txt      – YOLO label: 0 x_c y_c w h  (all 0-1)

Padding strategy:
  Horizontal  – outer eye corners + 1.0x eye-width on each side
  Top         – eye_top  - 2.5x eye_height  (covers full frame top + brow)
  Bottom      – eye_bot  + 1.2x eye_height  (covers lower lens rim)
  This scales automatically with face size regardless of resolution.

YOLO class 0 = spectacles/glasses

Usage:
    python batch_annotate_specs.py           # first 200 from specs_black_hair
    python batch_annotate_specs.py 50        # quick test with 50
"""
import os, sys, json
import cv2
import numpy as np
import mediapipe as mp

HERE    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "dataset", "train", "specs_white_hair")
OUT_DIR = os.path.join(HERE, "specs_annotations")
LBL_DIR = os.path.join(OUT_DIR, "labels")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

LIMIT    = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CLASS_ID = 0

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Eye landmark indices (MediaPipe FaceMesh)
L_OUTER, R_OUTER = 33,  263   # outer eye corners
L_INNER, R_INNER = 133, 362   # inner eye corners
L_TOP,   R_TOP   = 158, 385   # top of each eye
L_BOT,   R_BOT   = 144, 380   # bottom of each eye
# Extra brow landmarks for better top boundary
L_BROW_OUT, R_BROW_OUT = 70, 300  # outer brow upper
L_BROW_IN,  R_BROW_IN  = 107, 336 # inner brow peak
BROW_IDX = [L_BROW_OUT, R_BROW_OUT, L_BROW_IN, R_BROW_IN, 63, 293, 66, 296]
EYE_IDX  = [L_OUTER, R_OUTER, L_INNER, R_INNER, L_TOP, R_TOP, L_BOT, R_BOT]
ALL_IDX  = EYE_IDX + BROW_IDX


def get_lm(lm_obj, idx, w, h):
    lm = lm_obj.landmark[idx]
    return lm.x * w, lm.y * h   # floats for precise math


def compute_bbox(face_lm, w, h):
    # ── Eye region ────────────────────────────────────────────────────────────
    eye_pts = [get_lm(face_lm, i, w, h) for i in EYE_IDX]
    xs = [p[0] for p in eye_pts]
    ys = [p[1] for p in eye_pts]

    eye_w  = max(xs) - min(xs)   # span outer-to-outer corner
    eye_h  = max(ys) - min(ys)   # span top-to-bottom of eye

    # Safety: for very small/far faces
    eye_w  = max(eye_w, 10)
    eye_h  = max(eye_h, 4)

    # ── Brow top ──────────────────────────────────────────────────────────────
    brow_pts = [get_lm(face_lm, i, w, h) for i in BROW_IDX]
    brow_top_y = min(p[1] for p in brow_pts)   # smallest y = highest on image

    # ── Bbox ─────────────────────────────────────────────────────────────────
    # Horizontal: outer corners ± generous side padding for frame
    x1 = min(xs) - 0.80 * eye_w
    x2 = max(xs) + 0.80 * eye_w

    # Vertical top: whichever is higher — brow points or eye_top - 2.5×eye_h
    y_top_via_brow = brow_top_y - 0.25 * eye_h       # slightly above brow
    y_top_via_pad  = min(ys)    - 2.50 * eye_h
    y1 = min(y_top_via_brow, y_top_via_pad)

    # Vertical bottom: below eye with 1.2× eye height for lower rim
    y2 = max(ys) + 1.20 * eye_h

    # Clamp to image
    x1 = max(0,     int(x1))
    y1 = max(0,     int(y1))
    x2 = min(w - 1, int(x2))
    y2 = min(h - 1, int(y2))
    return x1, y1, x2, y2


def to_yolo(x1, y1, x2, y2, w, h):
    return ((x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h)


def process_image(img_path, fm):
    stem    = os.path.splitext(os.path.basename(img_path))[0]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, "cannot_load"
    h, w = img_bgr.shape[:2]
    result = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None, "no_face"
    face_lm = result.multi_face_landmarks[0]

    x1, y1, x2, y2 = compute_bbox(face_lm, w, h)
    xc, yc, bw, bh = to_yolo(x1, y1, x2, y2, w, h)

    # ── Annotated preview ────────────────────────────────────────────────────
    vis = img_bgr.copy()
    thick = max(3, (y2 - y1) // 20)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 220), thick)      # red box

    # ── Zoom: crop to 2× bbox size centred on bbox ───────────────────────────
    bw_px = x2 - x1
    bh_px = y2 - y1
    margin_x = int(bw_px * 1.0)
    margin_y = int(bh_px * 1.0)
    cx1 = max(0,     x1 - margin_x)
    cy1 = max(0,     y1 - margin_y)
    cx2 = min(w - 1, x2 + margin_x)
    cy2 = min(h - 1, y2 + margin_y)
    crop = vis[cy1:cy2, cx1:cx2]

    # Resize crop to fixed height for easy visual comparison
    TARGET_H = 400
    scale    = TARGET_H / max(crop.shape[0], 1)
    out_w    = max(1, int(crop.shape[1] * scale))
    zoomed   = cv2.resize(crop, (out_w, TARGET_H), interpolation=cv2.INTER_LINEAR)

    # Overlay file info
    cv2.putText(zoomed, f"{stem}  {w}x{h}  box {bw_px}x{bh_px}px",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(zoomed, f"YOLO w={bw:.3f} h={bh:.3f}",
                (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    out_img = os.path.join(OUT_DIR, stem + "_annotated.jpg")
    cv2.imwrite(out_img, zoomed, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ── YOLO label ────────────────────────────────────────────────────────────
    with open(os.path.join(LBL_DIR, stem + ".txt"), "w") as f:
        f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    return {"stem": stem, "w": w, "h": h,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "xc": round(xc, 4), "yc": round(yc, 4),
            "bw": round(bw, 4), "bh": round(bh, 4)}, "ok"


def main():
    files = sorted(
        f for f in os.listdir(SRC_DIR)
        if os.path.splitext(f)[1].lower() in EXTS
    )[:LIMIT]
    total = len(files)
    print(f"Processing {total} images → {OUT_DIR}\n")

    mp_fm = mp.solutions.face_mesh
    ok_rows, failed = [], []

    with mp_fm.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.4) as fm:
        for i, fn in enumerate(files, 1):
            row, status = process_image(os.path.join(SRC_DIR, fn), fm)
            if status == "ok":
                ok_rows.append(row)
                print(f"  [{i:3d}/{total}] OK    {fn:35s}  "
                      f"box {row['x2']-row['x1']}x{row['y2']-row['y1']}px  "
                      f"YOLO w={row['bw']} h={row['bh']}")
            else:
                failed.append({"file": fn, "reason": status})
                print(f"  [{i:3d}/{total}] SKIP  {fn:35s}  ({status})")

    print(f"\n{'='*65}")
    print(f"  Annotated : {len(ok_rows)}/{total}")
    print(f"  Skipped   : {len(failed)}/{total}")
    print(f"  Review    : {OUT_DIR}")
    print(f"  Labels    : {LBL_DIR}")
    if failed:
        print("\n  Skipped files:")
        for r in failed:
            print(f"    {r['file']:40s} -> {r['reason']}")
    print(f"{'='*65}")

    summary = {"annotated": len(ok_rows), "skipped": len(failed),
               "rows": ok_rows, "failed": failed}
    with open(os.path.join(OUT_DIR, "annotation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

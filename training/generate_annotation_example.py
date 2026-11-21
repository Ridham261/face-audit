"""
Generate a YOLO bounding-box annotation example for sunglasses/goggles.

Outputs (in D:\audit\annotation_output\):
  annotated.png   – original image with red bounding box drawn on the eye/lens region
  annotated.txt   – YOLO-format label:  0 x_center y_center width height  (all 0-1)

Usage:
    python generate_annotation_example.py [image_path]

Defaults to synthetic_sunglasses_1.png if no path is given.

MediaPipe landmarks used:
    33  – left eye outer corner
    263 – right eye outer corner
    158 – left eye top
    385 – right eye top
    144 – left eye bottom
    380 – right eye bottom
"""
import os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── MediaPipe ──────────────────────────────────────────────────────────────────
import mediapipe as mp

# ── Config ─────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DEFAULT  = os.path.join(HERE, "category_test", "goggles", "synthetic_sunglasses_1.png")
OUT_DIR  = os.path.join(HERE, "annotation_output")
os.makedirs(OUT_DIR, exist_ok=True)

# YOLO class 0 = sunglasses / goggles (class 1 would be regular glasses)
CLASS_ID = 0

# Extra padding around the eye-landmark bbox (fraction of image dimensions)
PAD_X    = 0.045   # extra left/right padding for frame
PAD_Y_T  = 0.030   # extra top padding for frame top
PAD_Y_B  = 0.025   # extra bottom padding for lens bottom

# ── Landmark indices ───────────────────────────────────────────────────────────
# Combined eye region that covers both lenses + nose bridge
L_OUTER  = 33
R_OUTER  = 263
L_INNER  = 133
R_INNER  = 362
L_TOP    = 158   # left eye top
R_TOP    = 385   # right eye top
L_BOT    = 144   # left eye bottom
R_BOT    = 380   # right eye bottom


def get_landmark_px(face_landmarks, idx, w, h):
    lm = face_landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


def compute_lens_bbox(face_landmarks, w, h):
    """Return (x1, y1, x2, y2) pixel bbox for the combined lens region."""
    pts = [
        get_landmark_px(face_landmarks, L_OUTER, w, h),
        get_landmark_px(face_landmarks, R_OUTER, w, h),
        get_landmark_px(face_landmarks, L_INNER, w, h),
        get_landmark_px(face_landmarks, R_INNER, w, h),
        get_landmark_px(face_landmarks, L_TOP,   w, h),
        get_landmark_px(face_landmarks, R_TOP,   w, h),
        get_landmark_px(face_landmarks, L_BOT,   w, h),
        get_landmark_px(face_landmarks, R_BOT,   w, h),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    pad_px_x = int(PAD_X  * w)
    pad_px_yt= int(PAD_Y_T * h)
    pad_px_yb= int(PAD_Y_B * h)

    x1 = max(0,     min(xs) - pad_px_x)
    x2 = min(w - 1, max(xs) + pad_px_x)
    y1 = max(0,     min(ys) - pad_px_yt)
    y2 = min(h - 1, max(ys) + pad_px_yb)
    return x1, y1, x2, y2


def to_yolo(x1, y1, x2, y2, w, h):
    """Convert pixel bbox to YOLO normalised format."""
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def annotate(img_path: str):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    out_img  = os.path.join(OUT_DIR, "annotated.png")
    out_txt  = os.path.join(OUT_DIR, "annotated.txt")

    # Load image
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {img_path}")
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run MediaPipe FaceMesh
    mp_fm = mp.solutions.face_mesh
    with mp_fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
    ) as fm:
        result = fm.process(img_rgb)

    if not result.multi_face_landmarks:
        print("  No face detected — trying with lower confidence...")
        with mp_fm.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.2,
        ) as fm:
            result = fm.process(img_rgb)

    if not result.multi_face_landmarks:
        raise RuntimeError("MediaPipe found no face in this image.")

    face_lm = result.multi_face_landmarks[0]
    x1, y1, x2, y2 = compute_lens_bbox(face_lm, w, h)
    xc, yc, bw, bh = to_yolo(x1, y1, x2, y2, w, h)

    print(f"  Image         : {img_path}")
    print(f"  Image size    : {w} x {h} px")
    print(f"  Bbox (pixels) : x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"  Bbox (YOLO)   : class={CLASS_ID}  x_c={xc:.4f}  y_c={yc:.4f}  w={bw:.4f}  h={bh:.4f}")

    # ── Draw on image ──────────────────────────────────────────────────────────
    vis = img_bgr.copy()
    BOX_COLOR   = (0, 0, 220)      # red (BGR)
    BOX_THICK   = max(2, h // 150)
    cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)

    # Label
    label = f"sunglasses (class {CLASS_ID})"
    font_scale = max(0.5, h / 800)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + bl), (0, 0, 0), -1)
    cv2.putText(vis, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # Also draw key landmark dots (small circles) for reference
    DOT_COLOR = (0, 200, 0)
    DOT_R = max(3, h // 200)
    for idx in [L_OUTER, R_OUTER, L_INNER, R_INNER, L_TOP, R_TOP, L_BOT, R_BOT]:
        px, py = get_landmark_px(face_lm, idx, w, h)
        cv2.circle(vis, (px, py), DOT_R, DOT_COLOR, -1)

    # Dimension overlay (top-left corner)
    dim_txt = f"{w}x{h}px | bbox: {x2-x1}x{y2-y1}px"
    cv2.putText(vis, dim_txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_img, vis)
    print(f"  Annotated PNG : {out_img}")

    # ── Write YOLO .txt ────────────────────────────────────────────────────────
    yolo_line = f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
    with open(out_txt, "w") as f:
        f.write(yolo_line)
    print(f"  YOLO label    : {out_txt}")
    print(f"  Label content : {yolo_line.strip()}")

    return out_img, out_txt, (xc, yc, bw, bh)


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    out_img, out_txt, bbox = annotate(img_path)
    print("\nDone.")
    print(f"  View annotated image : {out_img}")
    print(f"  YOLO label file      : {out_txt}")

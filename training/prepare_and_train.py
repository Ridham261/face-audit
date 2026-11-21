"""
Prepare YOLO dataset from specs + goggles annotations, then train YOLOv8.

Classes:
  0 = specs  (regular eyeglasses)
  1 = goggles (sunglasses / goggles)

Dataset layout built at D:\audit\yolo_glasses_dataset\
  images/train/   images/val/
  labels/train/   labels/val/
  data.yaml
"""
import os, shutil, random, json
from pathlib import Path

HERE = Path(r"D:\audit")

# ── Source folders ────────────────────────────────────────────────────────────
SPECS_IMG_DIRS = [
    HERE / "dataset" / "train" / "specs_black_hair",
    HERE / "dataset" / "train" / "specs_white_hair",
]
SPECS_LBL_DIR  = HERE / "specs_annotations"  / "labels"

GOG_IMG_DIR    = HERE / "dataset" / "train" / "goggles"
GOG_LBL_DIR    = HERE / "goggles_annotations" / "labels"

# ── Output ────────────────────────────────────────────────────────────────────
DS_DIR = HERE / "yolo_glasses_dataset"
for split in ("train", "val"):
    (DS_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DS_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VAL_RATIO = 0.2
random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_image(stem, img_dirs):
    """Return first matching image path across img_dirs."""
    for d in img_dirs:
        for ext in EXTS:
            p = d / (stem + ext)
            if p.exists():
                return p
    return None

def rewrite_label(src_txt, dst_txt, new_class):
    """Copy label file, optionally replacing class id."""
    lines = Path(src_txt).read_text().splitlines()
    out = []
    for line in lines:
        parts = line.split()
        if parts:
            parts[0] = str(new_class)
            out.append(" ".join(parts))
    Path(dst_txt).write_text("\n".join(out) + "\n")

# ── Collect pairs ─────────────────────────────────────────────────────────────
specs_pairs, gog_pairs = [], []

for lbl in sorted(SPECS_LBL_DIR.glob("*.txt")):
    img = find_image(lbl.stem, SPECS_IMG_DIRS)
    if img:
        specs_pairs.append((img, lbl, 0))   # class 0 = specs

random.shuffle(specs_pairs)
specs_pairs = specs_pairs[:len(specs_pairs) // 2]   # use half of specs

for lbl in sorted(GOG_LBL_DIR.glob("*.txt")):
    img = find_image(lbl.stem, [GOG_IMG_DIR])
    if img:
        gog_pairs.append((img, lbl, 1))     # class 1 = goggles

print(f"Specs pairs  : {len(specs_pairs)}")
print(f"Goggles pairs: {len(gog_pairs)}")

# ── Train / val split ─────────────────────────────────────────────────────────
def split_pairs(pairs):
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_RATIO))
    return pairs[n_val:], pairs[:n_val]

specs_train, specs_val = split_pairs(specs_pairs)
gog_train,   gog_val   = split_pairs(gog_pairs)

all_train = specs_train + gog_train
all_val   = specs_val   + gog_val
random.shuffle(all_train)
random.shuffle(all_val)

print(f"\nTrain : {len(all_train)}  (specs={len(specs_train)}, goggles={len(gog_train)})")
print(f"Val   : {len(all_val)}  (specs={len(specs_val)}, goggles={len(gog_val)})")

# ── Copy images + labels ──────────────────────────────────────────────────────
def copy_pairs(pairs, split):
    img_dir = DS_DIR / "images" / split
    lbl_dir = DS_DIR / "labels" / split
    for img_path, lbl_path, cls in pairs:
        dst_img = img_dir / img_path.name
        dst_lbl = lbl_dir / (img_path.stem + ".txt")
        shutil.copy2(img_path, dst_img)
        rewrite_label(lbl_path, dst_lbl, cls)

copy_pairs(all_train, "train")
copy_pairs(all_val,   "val")
print("\nImages + labels copied.")

# ── data.yaml ─────────────────────────────────────────────────────────────────
yaml_path = DS_DIR / "data.yaml"
yaml_path.write_text(
    f"path: {DS_DIR.as_posix()}\n"
    f"train: images/train\n"
    f"val:   images/val\n"
    f"\nnc: 2\n"
    f"names: ['specs', 'goggles']\n"
)
print(f"data.yaml → {yaml_path}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Starting YOLOv8 training ...")
print("="*60 + "\n")

from ultralytics import YOLO

model = YOLO("yolov8s.pt")   # small model — good balance of speed vs accuracy
results = model.train(
    data    = str(yaml_path),
    epochs  = 60,
    imgsz   = 640,
    batch   = 8,
    name    = "glasses_detector",
    project = str(HERE / "runs"),
    patience= 20,
    workers = 0,
    device  = "cpu",
    verbose = True,
)

print("\nTraining complete.")
print(f"Best weights: {HERE / 'runs' / 'glasses_detector' / 'weights' / 'best.pt'}")

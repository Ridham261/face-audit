"""
Build D:\\audit\\category_test\\<category>\\ with 5 images per category (13 categories).

  8 appearance categories : real labelled photos (held-out test photos; turban ones are only in train)
  low_resolution          : the real 'Reject - Low Resolution Photo' rows from the CSV (+ synthetic to reach 5)
  multiple_person         : SYNTHETIC - two real faces placed side by side
  blur                    : SYNTHETIC - a real photo with heavy Gaussian blur
  goggles                 : SYNTHETIC - dark sunglasses drawn over the eyes of a real photo
  side_face               : no real images available -> left empty (add your own)
Synthetic files are named synthetic_*.png so they are reported separately.
"""
import csv
import glob
import os
import random
import shutil
import sys

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps
from facenet_pytorch import MTCNN

OUT = r"D:\audit\category_test"
CSV_PATH = sys.argv[1]
random.seed(11)
shutil.rmtree(OUT, ignore_errors=True)
mt = MTCNN(keep_all=True, device="cpu", post_process=False)


def photos(cat, split):
    return sorted(glob.glob(rf"D:\audit\dataset\{split}\{cat}\*.png"))


def save(cat, name, im):
    os.makedirs(os.path.join(OUT, cat), exist_ok=True)
    im.convert("RGB").save(os.path.join(OUT, cat, name))


def load(p, long_side=800):
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    s = long_side / max(im.size)
    return im.resize((int(im.width * s), int(im.height * s))) if s < 1 else im


# 1) the 8 appearance categories: real photos
for cat in ["black_hair", "specs_black_hair", "white_hair", "specs_white_hair", "doctor_cap", "doctor_cap_specs"]:
    for p in random.sample(photos(cat, "test"), 5):
        shutil.copy(p, os.path.join(OUT, cat, os.path.basename(p))) if os.makedirs(os.path.join(OUT, cat), exist_ok=True) is None else None
for cat in ["turban", "turban_specs"]:                       # no held-out turban photos exist
    for p in random.sample(photos(cat, "train"), 5):
        os.makedirs(os.path.join(OUT, cat), exist_ok=True)
        shutil.copy(p, os.path.join(OUT, cat, "seen_in_training_" + os.path.basename(p)))

# 2) base photos for synthetic cases: held-out black_hair photos not used above
used = {os.path.basename(p) for p in glob.glob(os.path.join(OUT, "black_hair", "*.png"))}
bases = [p for p in photos("black_hair", "test") if os.path.basename(p) not in used]
random.shuffle(bases)

# multiple_person: two different faces side by side
for i in range(5):
    a, b = load(bases[2 * i], 600), load(bases[2 * i + 1], 600)
    h = 480
    a = a.resize((int(a.width * h / a.height), h)); b = b.resize((int(b.width * h / b.height), h))
    canvas = Image.new("RGB", (a.width + b.width, h)); canvas.paste(a, (0, 0)); canvas.paste(b, (a.width, 0))
    save("multiple_person", f"synthetic_two_people_{i+1}.png", canvas)

# blur: heavy Gaussian blur
for i in range(5):
    im = load(bases[10 + i], 700)
    sig = 0.03 * max(im.size) + 8
    arr = cv2.GaussianBlur(np.array(im), (0, 0), sig)
    save("blur", f"synthetic_blur_{i+1}.png", Image.fromarray(arr))

# goggles: dark sunglasses drawn on the eyes
n = 0
for p in bases[15:]:
    if n == 5:
        break
    im = load(p, 800)
    boxes, probs, lms = mt.detect(im, landmarks=True)
    if boxes is None:
        continue
    j = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
    (lx, ly), (rx, ry) = lms[j][0], lms[j][1]
    fw = boxes[j][2] - boxes[j][0]
    d = ImageDraw.Draw(im)
    rw, rh = 0.30 * fw, 0.17 * fw
    for cx, cy in ((lx, ly), (rx, ry)):
        d.rounded_rectangle([cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2], radius=rh / 2.5, fill=(18, 18, 20))
    d.line([lx + rw / 2, ly, rx - rw / 2, ry], fill=(18, 18, 20), width=max(2, int(fw * 0.03)))
    d.line([lx - rw / 2, ly, lx - rw / 2 - fw * 0.15, ly - fw * 0.02], fill=(18, 18, 20), width=max(2, int(fw * 0.03)))
    d.line([rx + rw / 2, ry, rx + rw / 2 + fw * 0.15, ry - fw * 0.02], fill=(18, 18, 20), width=max(2, int(fw * 0.03)))
    save("goggles", f"synthetic_sunglasses_{n+1}.png", im)
    n += 1

# low_resolution: real rejected rows from the CSV, topped up with synthetic tiny photos
rows = [r for r in csv.DictReader(open(CSV_PATH, encoding="utf-8-sig", newline="")) if r["Photo Review Status"].startswith("Reject - Low Resolution")]
k = 0
for r in rows[:5]:
    try:
        resp = requests.get(r["Photo"].strip(), timeout=30)
        os.makedirs(os.path.join(OUT, "low_resolution"), exist_ok=True)
        open(os.path.join(OUT, "low_resolution", f"real_rejected_{r['ID']}.png"), "wb").write(resp.content); k += 1
    except Exception as exc:
        print("download failed", r["ID"], exc)
for i in range(5 - k):
    im = load(bases[40 + i], 700)
    small = im.resize((60, int(60 * im.height / im.width)))
    save("low_resolution", f"synthetic_tiny_{i+1}.png", small)

os.makedirs(os.path.join(OUT, "side_face"), exist_ok=True)
for c in sorted(os.listdir(OUT)):
    print(f"{c:18s} {len(os.listdir(os.path.join(OUT, c)))} images")

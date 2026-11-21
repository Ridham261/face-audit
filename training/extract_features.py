"""
Extract CLIP image features for every downloaded photo (train + test).

For each photo:  head crop (MTCNN face box expanded) + horizontally-flipped head crop + whole photo.
Feature = concat( L2norm(mean(head, head_flip)), L2norm(whole) )  -> 1024-d

Usage:  python extract_features.py D:\\audit\\dataset D:\\audit\\dataset\\features.npz
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from transformers import CLIPModel, CLIPProcessor

ROOT = sys.argv[1]
OUT = sys.argv[2]
CLIP_DIR = r"D:\audit\models\clip"
torch.set_num_threads(os.cpu_count() or 4)

mtcnn = MTCNN(keep_all=True, device="cpu", post_process=False)
model = CLIPModel.from_pretrained(CLIP_DIR).eval()
proc = CLIPProcessor.from_pretrained(CLIP_DIR)


def load(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im


def head_crop(im):
    """Largest MTCNN face, expanded to include hair / headwear / glasses. Fallback: upper-centre."""
    w, h = im.size
    s = 800 / max(w, h)
    small = im.resize((int(w * s), int(h * s))) if s < 1 else im
    k = 1 / s if s < 1 else 1.0
    boxes, probs = mtcnn.detect(small)
    if boxes is not None and len(boxes):
        i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
        x1, y1, x2, y2 = [float(v) * k for v in boxes[i]]
        fw, fh = x2 - x1, y2 - y1
        box = (max(0, x1 - 0.6 * fw), max(0, y1 - 0.9 * fh), min(w, x2 + 0.6 * fw), min(h, y2 + 0.35 * fh))
        return im.crop(tuple(int(v) for v in box)), True
    return im.crop((int(w * 0.15), 0, int(w * 0.85), int(h * 0.65))), False


def square(im):
    return ImageOps.pad(im, (224, 224), color=(128, 128, 128))


@torch.no_grad()
def embed(images):
    px = proc(images=images, return_tensors="pt", do_center_crop=False, size={"height": 224, "width": 224})["pixel_values"]
    f = model.get_image_features(pixel_values=px)
    if not torch.is_tensor(f):          # newer transformers may return an output object
        f = f.pooler_output
    return torch.nn.functional.normalize(f, dim=-1).numpy()


X, labels, splits, ids, found = [], [], [], [], []
files = []
for split in ("train", "test"):
    sd = os.path.join(ROOT, split)
    if not os.path.isdir(sd):
        continue
    for lab in sorted(os.listdir(sd)):
        for fn in sorted(os.listdir(os.path.join(sd, lab))):
            files.append((split, lab, fn))
print("images:", len(files), flush=True)

for n, (split, lab, fn) in enumerate(files, 1):
    try:
        im = load(os.path.join(ROOT, split, lab, fn))
        head, ok = head_crop(im)
        v = embed([square(head), square(ImageOps.mirror(head)), square(im)])
        feat = np.concatenate([
            torch.nn.functional.normalize(torch.from_numpy(v[:2].mean(0)), dim=0).numpy(),
            v[2],
        ])
        X.append(feat); labels.append(lab); splits.append(split); ids.append(os.path.splitext(fn)[0]); found.append(ok)
    except Exception as exc:
        print("skip", split, lab, fn, repr(exc), flush=True)
    if n % 50 == 0:
        print(f"{n}/{len(files)}", flush=True)

np.savez(OUT, X=np.array(X, dtype=np.float32), label=np.array(labels), split=np.array(splits),
         id=np.array(ids), face_found=np.array(found))
print("saved", OUT, "face detected in", int(np.sum(found)), "of", len(found))

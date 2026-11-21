"""
Check that the service-side feature code in main.py reproduces the training features.
Loads only the appearance block of main.py (not the whole service).
"""
import logging
import os
import re
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from transformers import CLIPModel, CLIPProcessor
from typing import Optional

src = open(r"D:\audit\main.py", encoding="utf-8").read()
a = src.index("APP_ML_ENABLED  =")
b = src.index("def _combine_appearance_labels")
block = src[a:b]

logger = logging.getLogger("t")
_m = CLIPModel.from_pretrained(r"D:\audit\models\clip").eval()
ns = {"np": np, "cv2": cv2, "Image": Image, "Optional": Optional, "os": os, "logger": logger,
      "__file__": r"D:\audit\main.py", "_load_clip": lambda: (_m, None)}
exec(block, ns)

d = np.load(r"D:\audit\dataset\features.npz", allow_pickle=True)
c = np.load(r"D:\audit\dataset\cnx.npz", allow_pickle=True)
key = {k: i for i, k in enumerate(c["key"])}
mt = MTCNN(keep_all=True, device="cpu", post_process=False)

idx = [i for i in range(len(d["id"])) if d["split"][i] == "test"][::25][:10]
sims, same = [], 0
for i in idx:
    lab, sid, split = d["label"][i], d["id"][i], d["split"][i]
    path = rf"D:\audit\dataset\{split}\{lab}\{sid}.png"
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    boxes, _ = mt.detect(im)                       # full-resolution box, as the service does
    bi = int(np.argmax([(x[2] - x[0]) * (x[3] - x[1]) for x in boxes]))
    p = c["prob"][key[f"{split}/{lab}/{sid}.png"]]
    g = {"all_probs": dict(zip(["cap", "female", "hijab", "male", "turban"], p.tolist()))}
    x = ns["_app_features"](frame, boxes[bi], g)
    ref = np.concatenate([d["X"][i], np.repeat(p[[0, 2, 4]], 8)])
    cos = float(np.dot(x, ref) / (np.linalg.norm(x) * np.linalg.norm(ref)))
    r = ns["predict_appearance_ml"](frame, boxes[bi], g)
    same += int(r["appearance_label"] == lab)
    sims.append(cos)
    print(f"{lab:18s} -> {r['appearance_label']:18s} conf={r['confidence']:.2f} cos={cos:.4f}")
print("mean cosine to training features:", round(float(np.mean(sims)), 4), "| label matches:", same, "/", len(idx))

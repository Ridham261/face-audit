"""
Extract features from the project's own ConvNeXt-Large (trained on cap/female/hijab/male/turban)
on the same face crop the service uses (face box + 10% pad):  5 sigmoid probs + 1536-d pooled features.

Usage: python extract_convnext.py D:\\audit\\dataset D:\\audit\\dataset\\cnx.npz
"""
import os
import sys

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from torchvision import transforms

ROOT, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(os.cpu_count() or 4)
mtcnn = MTCNN(keep_all=True, device="cpu", post_process=False)
model = timm.create_model("convnext_large", pretrained=False, num_classes=5)
ck = torch.load(r"D:\audit\models\best_convnext_large.pth", map_location="cpu")
model.load_state_dict(ck.get("model_state_dict", ck))
model.eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

files = []
for split in ("train", "test"):
    sd = os.path.join(ROOT, split)
    for lab in sorted(os.listdir(sd)):
        for fn in sorted(os.listdir(os.path.join(sd, lab))):
            files.append((split, lab, fn))
print("images:", len(files), flush=True)

F, P, ids = [], [], []
for n, (split, lab, fn) in enumerate(files, 1):
    im = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, split, lab, fn))).convert("RGB")
    w, h = im.size
    s = 800 / max(w, h)
    small = im.resize((int(w * s), int(h * s))) if s < 1 else im
    k = 1 / s if s < 1 else 1.0
    boxes, _ = mtcnn.detect(small)
    if boxes is not None and len(boxes):
        i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
        x1, y1, x2, y2 = [float(v) * k for v in boxes[i]]
        pad = max(x2 - x1, y2 - y1) * 0.10
        crop = im.crop((int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(w, x2 + pad)), int(min(h, y2 + pad))))
    else:
        crop = im
    with torch.no_grad():
        x = tf(crop).unsqueeze(0)
        fm = model.forward_features(x)
        feat = model.forward_head(fm, pre_logits=True)
        logit = model.forward_head(fm)
    F.append(feat[0].numpy()); P.append(torch.sigmoid(logit[0]).numpy()); ids.append(f"{split}/{lab}/{fn}")
    if n % 50 == 0:
        print(f"{n}/{len(files)}", flush=True)

np.savez(OUT, feat=np.array(F, np.float32), prob=np.array(P, np.float32), key=np.array(ids))
print("saved", OUT)

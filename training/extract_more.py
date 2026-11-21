"""
Extract CLIP + ConvNeXt features ONLY for photos not already in features.npz, then write merged
features_all.npz / cnx_all.npz (same layout the training script already reads).

Usage: python extract_more.py D:\\audit\\dataset
"""
import os
import sys

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from torchvision import transforms
from transformers import CLIPModel

ROOT = sys.argv[1]
torch.set_num_threads(os.cpu_count() or 4)
mtcnn = MTCNN(keep_all=True, device="cpu", post_process=False)
clip = CLIPModel.from_pretrained(r"D:\audit\models\clip").eval()
cnx = timm.create_model("convnext_large", pretrained=False, num_classes=5)
ck = torch.load(r"D:\audit\models\best_convnext_large.pth", map_location="cpu")
cnx.load_state_dict(ck.get("model_state_dict", ck)); cnx.eval()
cnx_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

old = np.load(os.path.join(ROOT, "features.npz"), allow_pickle=True)
oldc = np.load(os.path.join(ROOT, "cnx.npz"), allow_pickle=True)
have = {f"{s}/{l}/{i}.png" for s, l, i in zip(old["split"], old["label"], old["id"])}

# resume support: finished chunks are saved in parts/ and skipped on the next run
PARTS = os.path.join(ROOT, "parts")
os.makedirs(PARTS, exist_ok=True)
for pf_ in sorted(os.listdir(PARTS)):
    have |= set(np.load(os.path.join(PARTS, pf_), allow_pickle=True)["keys"].tolist())

todo = []
for split in ("train", "test"):
    for lab in sorted(os.listdir(os.path.join(ROOT, split))):
        for fn in sorted(os.listdir(os.path.join(ROOT, split, lab))):
            if f"{split}/{lab}/{fn}" not in have:
                todo.append((split, lab, fn))
print("new images:", len(todo), flush=True)


def clip_embed(pils):
    arr = np.stack([(np.asarray(p, np.float32) / 255.0 - MEAN) / STD for p in pils]).transpose(0, 3, 1, 2).copy()
    with torch.no_grad():
        f = clip.get_image_features(pixel_values=torch.from_numpy(arr))
        if not torch.is_tensor(f):
            f = f.pooler_output
    return torch.nn.functional.normalize(f, dim=-1).numpy()


sq = lambda im: ImageOps.pad(im, (224, 224), color=(128, 128, 128))
X, labels, splits, ids, found, CF, CP, keys = [], [], [], [], [], [], [], []
for n, (split, lab, fn) in enumerate(todo, 1):
    try:
        im = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, split, lab, fn))).convert("RGB")
        w, h = im.size
        s = 800 / max(w, h)
        small = im.resize((int(w * s), int(h * s))) if s < 1 else im
        k = 1 / s if s < 1 else 1.0
        boxes, _ = mtcnn.detect(small)
        ok = boxes is not None and len(boxes) > 0
        if ok:
            i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
            x1, y1, x2, y2 = [float(v) * k for v in boxes[i]]
            fw, fh = x2 - x1, y2 - y1
            head = im.crop(tuple(int(v) for v in (max(0, x1 - .6 * fw), max(0, y1 - .9 * fh), min(w, x2 + .6 * fw), min(h, y2 + .35 * fh))))
            pad = max(fw, fh) * 0.10
            fcrop = im.crop((int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(w, x2 + pad)), int(min(h, y2 + pad))))
        else:
            head = im.crop((int(w * .15), 0, int(w * .85), int(h * .65))); fcrop = im
        v = clip_embed([sq(head), sq(ImageOps.mirror(head)), sq(im)])
        hv = torch.nn.functional.normalize(torch.from_numpy(v[:2].mean(0)), dim=0).numpy()
        with torch.no_grad():
            fm = cnx.forward_features(cnx_tf(fcrop).unsqueeze(0))
            pf = cnx.forward_head(fm, pre_logits=True)[0].numpy()
            pr = torch.sigmoid(cnx.forward_head(fm)[0]).numpy()
        X.append(np.concatenate([hv, v[2]])); labels.append(lab); splits.append(split)
        ids.append(os.path.splitext(fn)[0]); found.append(ok); CF.append(pf); CP.append(pr)
        keys.append(f"{split}/{lab}/{fn}")
    except Exception as exc:
        print("skip", split, lab, fn, repr(exc), flush=True)
    if n % 50 == 0:
        print(f"{n}/{len(todo)}", flush=True)
    if len(keys) >= 100 or (n == len(todo) and keys):          # checkpoint a chunk
        np.savez(os.path.join(PARTS, f"part_{len(os.listdir(PARTS)):04d}.npz"),
                 X=np.array(X, np.float32), label=np.array(labels), split=np.array(splits),
                 id=np.array(ids), face_found=np.array(found), CF=np.array(CF, np.float32),
                 CP=np.array(CP, np.float32), keys=np.array(keys))
        X, labels, splits, ids, found, CF, CP, keys = [], [], [], [], [], [], [], []

# merge old + every saved chunk
parts = [np.load(os.path.join(PARTS, f), allow_pickle=True) for f in sorted(os.listdir(PARTS))]
cat = lambda name, base: np.concatenate([base] + [p[name] for p in parts])
np.savez(os.path.join(ROOT, "features_all.npz"),
         X=cat("X", old["X"]), label=cat("label", old["label"]), split=cat("split", old["split"]),
         id=cat("id", old["id"]), face_found=cat("face_found", old["face_found"]))
np.savez(os.path.join(ROOT, "cnx_all.npz"),
         feat=cat("CF", oldc["feat"]), prob=cat("CP", oldc["prob"]), key=cat("keys", oldc["key"]))
print("saved merged features; total", len(old["id"]) + sum(len(p["id"]) for p in parts), flush=True)

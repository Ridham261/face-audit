"""Run the real service pipeline (main.py) in the service environment on a few photos."""
import glob
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, r"D:\audit")
import main as M   # noqa: E402

labels = ["black_hair", "specs_black_hair", "white_hair", "specs_white_hair",
          "doctor_cap", "doctor_cap_specs", "turban", "turban_specs"]
ok = 0
for lab in labels:
    f = sorted(glob.glob(rf"D:\audit\dataset\test\{lab}\*.png") or glob.glob(rf"D:\audit\dataset\train\{lab}\*.png"))[1]
    pil = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    t = time.time()
    res = M.run_full_validation(pil, bgr)
    out = M._clean_response(res)
    app = out["appearance"]
    print(f"{lab:18s} category={out['category']:12s} audit={out['audit_category']:18s} "
          f"source={app.get('source')} conf={app.get('confidence')} {time.time()-t:.1f}s")
    ok += out["audit_category"] == lab
print("matches:", ok, "/", len(labels))

"""Show the new /validate-face JSON for: a normal photo, a turban+specs photo, and a blank (no-person) image."""
import glob
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, r"D:\audit")
import main as M   # noqa: E402


def run(pil, detail=False):
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return M._clean_response(M.run_full_validation(pil, bgr), detail=detail)


f = sorted(glob.glob(r"D:\audit\dataset\test\specs_white_hair\*.png"))[1]
print("=== 1) normal photo (specs_white_hair) ===")
print(json.dumps(run(ImageOps.exif_transpose(Image.open(f)).convert("RGB")), indent=2, default=str))

print("\n=== 2) blank grey image (no person) ===")
r = run(Image.new("RGB", (600, 600), (128, 128, 128)))
print(json.dumps({k: r[k] for k in ("status", "result", "needs_manual_review", "warnings", "checks", "audit_category")}, indent=2, default=str))

print("\n=== 3) keys with ?detail=true ===")
print(sorted(run(ImageOps.exif_transpose(Image.open(f)).convert("RGB"), detail=True).keys()))

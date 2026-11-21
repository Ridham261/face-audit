"""Show the first part of the new /validate-face JSON and the 2-section terminal report for one photo."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)
import cv2                        # noqa: E402
import numpy as np                # noqa: E402
from PIL import Image, ImageOps   # noqa: E402
import main as M                  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(r"D:\audit\dataset\test\specs_white_hair\*.png"))[1]
pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
res = M.run_full_validation(pil, bgr)
res["image_size"] = [pil.width, pil.height]
out = M._clean_response(res)
top = {k: out[k] for k in ("detected", "not_triggered", "status", "result", "scores", "matched", "errors")}
print("=== JSON (top part) ===")
print(json.dumps(top, indent=2))
M._print_category_report(out)

"""Contact sheets of every wrong prediction (full held-out test set) + the 2 wrong cross-validated turban photos."""
import csv
import glob
import os
import shutil

from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT = r"D:\audit\training\wrong_images"
os.makedirs(OUT, exist_ok=True)
rows = [r for r in csv.DictReader(open(r"D:\audit\training\test_all_results.csv")) if r["correct"] == "0"]
extra = [("3069", "turban", "turban_specs"), ("3770", "turban_specs", "turban")]   # cross-validated turban misses
items = [(r["id"], r["true"], r["predicted"], r["confidence"]) for r in rows] + [(i, t, p, "") for i, t, p in extra]
items.sort(key=lambda x: (x[1], x[2]))

W, H, LAB, COLS, ROWS = 300, 300, 46, 4, 3
try:
    font = ImageFont.truetype("arial.ttf", 14)
    big = ImageFont.truetype("arialbd.ttf", 14)
except Exception:
    font = big = ImageFont.load_default()

with open(os.path.join(OUT, "wrong_list.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["no", "id", "true_label", "model_said", "confidence", "file"])
    paths = []
    for n, (i, t, p, c) in enumerate(items, 1):
        src = glob.glob(rf"D:\audit\dataset\*\{t}\{i}.png")[0]
        dst = os.path.join(OUT, f"{n:02d}_{t}__said_{p}__{i}.png")
        shutil.copyfile(src, dst); paths.append((dst, i, t, p, c))
        w.writerow([n, i, t, p, c, dst])

per = COLS * ROWS
for s in range(0, len(paths), per):
    sheet = Image.new("RGB", (COLS * W, ROWS * (H + LAB)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (path, i, t, p, c) in enumerate(paths[s:s + per]):
        x, y = (k % COLS) * W, (k // COLS) * (H + LAB)
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        sheet.paste(ImageOps.fit(im, (W - 4, H - 4)), (x + 2, y + 2))
        d.text((x + 6, y + H), f"#{s + k + 1}  id {i}", fill="black", font=big)
        d.text((x + 6, y + H + 16), f"TRUE: {t}", fill=(0, 110, 0), font=font)
        d.text((x + 6, y + H + 30), f"SAID: {p}" + (f"  ({float(c):.2f})" if c else ""), fill=(200, 0, 0), font=font)
    sheet.save(os.path.join(OUT, f"sheet_{s // per + 1}.png"))
print(len(items), "wrong photos ->", OUT, "| sheets:", (len(items) + per - 1) // per)

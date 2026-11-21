"""
Add MORE training photos without touching the existing split.

Re-creates the exact shuffle used by download_sample.py (seed 42), so rows [0:100] stay train and
[100:150] stay test; the NEW training photos are rows [150 : 150+EXTRA].
"""
import csv
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

CSV_PATH, OUT = sys.argv[1], sys.argv[2]
BIG_EXTRA = int(sys.argv[3]) if len(sys.argv) > 3 else 400      # cap for the two huge classes
N_TRAIN, N_TEST, SEED = 100, 50, 42
LABELS = {"Black Hair": "black_hair", "Specs & Black Hair": "specs_black_hair", "White Hair": "white_hair",
          "Specs & White Hair": "specs_white_hair", "Doctor Cap": "doctor_cap",
          "Doctor Cap & Spec": "doctor_cap_specs", "Turban": "turban", "Turban & Spec": "turban_specs"}

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig", newline="")))
random.seed(SEED)
by = {v: [] for v in LABELS.values()}
for r in rows:
    lab = LABELS.get(r["Face Type"].strip())
    if lab and r["Photo Review Status"].strip() == "Approved" and r["Photo"].strip():
        by[lab].append(r)

jobs = []
for lab, items in by.items():
    random.shuffle(items)                       # identical call sequence to download_sample.py
    rest = items[N_TRAIN + N_TEST:]             # never used before (train=[:100], test=[100:150])
    extra = rest[:BIG_EXTRA] if lab in ("black_hair", "specs_black_hair") else rest
    for r in extra:
        jobs.append(("train", lab, r["ID"], r["Photo"].strip()))
    print(f"{lab:18s} new training photos: {len(extra)}")


def fetch(job):
    split, lab, rid, url = job
    d = os.path.join(OUT, split, lab)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{rid}.png")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return job, True
    for _ in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and resp.content:
                open(path, "wb").write(resp.content)
                return job, True
        except Exception:
            pass
    return job, False


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(fetch, jobs))
print(f"downloaded {sum(1 for _, s in res if s)}/{len(jobs)}")

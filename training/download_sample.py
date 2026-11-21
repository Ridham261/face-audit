"""
Download a balanced sample from the labelled CSV.

  train: up to N_TRAIN per category (all rows if the category has fewer)
  test : up to N_TEST extra, never-used rows per category (held out, for honest accuracy)

Output: D:\\audit\\dataset\\{train,test}\\<label>\\<ID>.png  and  manifest.csv
"""
import csv
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

CSV_PATH = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else r"D:\audit\dataset"
N_TRAIN, N_TEST, SEED = 100, 50, 42

LABELS = {
    "Black Hair": "black_hair",
    "Specs & Black Hair": "specs_black_hair",
    "White Hair": "white_hair",
    "Specs & White Hair": "specs_white_hair",
    "Doctor Cap": "doctor_cap",
    "Doctor Cap & Spec": "doctor_cap_specs",
    "Turban": "turban",
    "Turban & Spec": "turban_specs",
}

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig", newline="")))
random.seed(SEED)
by = {v: [] for v in LABELS.values()}
for r in rows:
    lab = LABELS.get(r["Face Type"].strip())
    # only reviewed/approved rows with a photo; use the original 'Photo' column
    if lab and r["Photo Review Status"].strip() == "Approved" and r["Photo"].strip():
        by[lab].append(r)

jobs = []
for lab, items in by.items():
    random.shuffle(items)
    train, rest = items[:N_TRAIN], items[N_TRAIN:]
    test = rest[:N_TEST]
    for split, part in (("train", train), ("test", test)):
        for r in part:
            jobs.append((split, lab, r["ID"], r["Photo"].strip()))
    print(f"{lab:18s} available={len(items):5d} train={len(train):4d} test={len(test):3d}")


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
                with open(path, "wb") as f:
                    f.write(resp.content)
                return job, True
        except Exception:
            pass
    return job, False


with ThreadPoolExecutor(8) as ex:
    results = list(ex.map(fetch, jobs))

ok = sum(1 for _, s in results if s)
print(f"\ndownloaded {ok}/{len(jobs)}")
with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "label", "id", "url", "ok"])
    for (split, lab, rid, url), s in results:
        w.writerow([split, lab, rid, url, int(s)])

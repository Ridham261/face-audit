"""10 photos per category, judged by a model that never saw them."""
import random
import warnings
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
d = np.load(r"D:\audit\dataset\features_all.npz", allow_pickle=True)   # larger training set
c = np.load(r"D:\audit\dataset\cnx_all.npz", allow_pickle=True)
Xc, lab, split, ids = d["X"], d["label"], d["split"], d["id"]
key = {k: i for i, k in enumerate(c["key"])}
prob = c["prob"][[key[f"{s}/{l}/{i}.png"] for s, l, i in zip(split, lab, ids)]]
X = np.hstack([Xc, np.repeat(prob[:, [0, 2, 4]], 8, axis=1)])

CLASSES = ["black_hair", "specs_black_hair", "white_hair", "specs_white_hair",
           "doctor_cap", "doctor_cap_specs", "turban", "turban_specs"]


def parts(l):
    return ("turban" if l.startswith("turban") else "cap" if l.startswith("doctor_cap") else "none",
            int(l.endswith("specs") or l.startswith("specs")),
            "white" if "white" in l else "black" if "black" in l else "")


def compose(hw, sp, hair):
    if hw == "turban": return "turban_specs" if sp else "turban"
    if hw == "cap": return "doctor_cap_specs" if sp else "doctor_cap"
    b = "white_hair" if hair == "white" else "black_hair"
    return ("specs_" + b) if sp else b


P = [parts(l) for l in lab]
Y = {"hw": np.array([p[0] for p in P]), "sp": np.array([p[1] for p in P]), "hr": np.array([p[2] for p in P])}
HEADS = {"hw": ("hw", 1), "sp": ("sp", 3), "hr": ("hr", 10)}       # C values picked earlier by cross-validation
lr = lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=4000)
tr, te = split == "train", split == "test"

# model 1: trained on train split only -> predicts the held-out test split
models = {}
for h, (y, C) in HEADS.items():
    m = tr & (Y["hw"] == "none") if h == "hr" else tr
    models[h] = lr(C).fit(X[m], Y[y][m])


def predict(model_set, Xs):
    hw = model_set["hw"].predict(Xs); sp = model_set["sp"].predict(Xs).astype(int); hr = model_set["hr"].predict(Xs)
    conf = np.minimum(model_set["hw"].predict_proba(Xs).max(1), model_set["sp"].predict_proba(Xs).max(1))
    return np.array([compose(a, b, c_) for a, b, c_ in zip(hw, sp, hr)]), conf


pred_te, conf_te = predict(models, X)
# turban classes: cross-validated predictions inside the train split (each photo unseen by its own model)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
cvh = {}
for h, (y, C) in HEADS.items():
    m = tr & (Y["hw"] == "none") if h == "hr" else tr
    cvh[h] = (np.where(m)[0], cross_val_predict(lr(C), X[m], Y[y][m], cv=cv))
cv_hw = dict(zip(*[cvh["hw"][0], cvh["hw"][1]]))
cv_sp = dict(zip(*[cvh["sp"][0], cvh["sp"][1]]))
cv_hr = dict(zip(*[cvh["hr"][0], cvh["hr"][1]]))

import sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
random.seed(7)
tot = ok_tot = 0
summary = []
for cl in CLASSES:
    pool = [i for i in range(len(lab)) if lab[i] == cl and te[i]]
    how = "held-out test photos"
    if not pool:
        pool = [i for i in range(len(lab)) if lab[i] == cl and tr[i]]
        how = "cross-validated (unseen by its own model)"
    pick = random.sample(pool, min(N, len(pool)))
    print(f"\n--- {cl}  [{how}]")
    ok = 0
    for i in pick:
        if how.startswith("held"):
            p = pred_te[i]
        else:
            p = compose(cv_hw[i], int(cv_sp[i]), cv_hr.get(i, "black"))
        good = p == cl
        ok += good
        if N <= 10 or not good:                  # for large runs only list the wrong ones
            print(f"  {'OK ' if good else 'BAD'} id={ids[i]:>6s}  predicted={p}")
    print(f"  => {ok}/{len(pick)}")
    summary.append((cl, ok, len(pick)))
    tot += len(pick); ok_tot += ok

print("\n=== SUMMARY (10 photos per category) ===")
for cl, ok, n in summary:
    print(f"  {cl:18s} {ok:2d}/{n}")
print(f"  TOTAL              {ok_tot}/{tot} = {100*ok_tot/tot:.1f}%")

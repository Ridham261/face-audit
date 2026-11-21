"""
Train the 8-category appearance classifier.

Three independent heads (more reliable than one 8-way guess on ~100 images/class):
    headwear : none | cap | turban
    specs    : no | yes
    hair     : black | white      (trained only on photos without headwear)
Final label = combination, e.g. headwear=turban + specs=yes -> turban_specs.

Feature sets compared (all computed once by extract_features.py / extract_convnext.py):
    clip            CLIP head-crop + whole photo
    clip+cnxprob    + the project's ConvNeXt cap/hijab/turban probabilities
    clip+cnx        + ConvNeXt pooled features too

Usage: python train_appearance.py FEATURES.npz OUT.joblib [CNX.npz]
"""
import sys
import warnings
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
FEATS, OUT = sys.argv[1], sys.argv[2]
CNX = sys.argv[3] if len(sys.argv) > 3 else None
d = np.load(FEATS, allow_pickle=True)
Xc, lab, split = d["X"], d["label"], d["split"]

CLASSES = ["black_hair", "specs_black_hair", "white_hair", "specs_white_hair",
           "doctor_cap", "doctor_cap_specs", "turban", "turban_specs"]


def parts(l):
    hw = "turban" if l.startswith("turban") else "cap" if l.startswith("doctor_cap") else "none"
    sp = int(l.endswith("specs") or l.startswith("specs"))
    hair = "white" if "white" in l else "black" if "black" in l else ""
    return hw, sp, hair


def compose(hw, sp, hair):
    if hw == "turban":
        return "turban_specs" if sp else "turban"
    if hw == "cap":
        return "doctor_cap_specs" if sp else "doctor_cap"
    base = "white_hair" if hair == "white" else "black_hair"
    return ("specs_" + base) if sp else base


P = [parts(l) for l in lab]
y_hw = np.array([p[0] for p in P]); y_sp = np.array([p[1] for p in P]); y_hair = np.array([p[2] for p in P])
tr, te = split == "train", split == "test"
print("train", int(tr.sum()), "test", int(te.sum()), dict(Counter(map(str, lab[tr]))))

sets = {"clip": Xc}
if CNX:
    c = np.load(CNX, allow_pickle=True)
    key = {k: i for i, k in enumerate(c["key"])}
    idx = [key[f"{s}/{l}/{i}.png"] for s, l, i in zip(split, lab, d["id"])]
    prob, feat = c["prob"][idx], c["feat"][idx]
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    pw = np.repeat(prob[:, [0, 2, 4]], 8, axis=1) * 1.0        # cap, hijab, turban (repeated = more weight)
    sets["clip+cnxprob"] = np.hstack([Xc, pw])
    sets["clip+cnx"] = np.hstack([Xc, pw, feat])


if len(sys.argv) > 4:                      # force a feature set (must match what main.py builds)
    sets = {k: v for k, v in sets.items() if k == sys.argv[4]}


def lr(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=4000)


def pick_C(Xs, ys):
    k = max(2, min(5, min(Counter(ys).values())))
    best, bc = -1, 1.0
    for C in (0.3, 1, 3, 10, 30, 100):
        pred = cross_val_predict(lr(C), Xs, ys, cv=StratifiedKFold(k, shuffle=True, random_state=0))
        a = float(np.mean(pred == ys))
        if a > best:
            best, bc = a, C
    return bc, best


def fit_heads(X, mask_fn):
    heads, cvs = {}, {}
    for name, y, mask in (("headwear", y_hw, mask_fn()), ("specs", y_sp, mask_fn()),
                          ("hair", y_hair, mask_fn() & (y_hw == "none"))):
        C, cv = pick_C(X[mask], y[mask])
        heads[name] = lr(C).fit(X[mask], y[mask]); cvs[name] = cv
    return heads, cvs


def predict_heads(heads, Xs):
    hw = heads["headwear"].predict_proba(Xs); sp = heads["specs"].predict_proba(Xs); ha = heads["hair"].predict_proba(Xs)
    out, conf = [], []
    for i in range(len(Xs)):
        h = heads["headwear"].classes_[hw[i].argmax()]
        s = int(heads["specs"].classes_[sp[i].argmax()])
        r = heads["hair"].classes_[ha[i].argmax()]
        c = [hw[i].max(), sp[i].max()] + ([ha[i].max()] if h == "none" else [])
        out.append(compose(h, s, r)); conf.append(min(c))
    return np.array(out), np.array(conf)


def report(title, pred, true, conf=None):
    print(f"\n=== {title}: {np.mean(pred == true)*100:.1f}% ({int(np.sum(pred == true))}/{len(true)})")
    for cl in CLASSES:
        m = true == cl
        if m.any():
            wrong = Counter(map(str, pred[m & (pred != cl)]))
            print(f"  {cl:18s} {int(np.sum(pred[m] == cl)):3d}/{int(m.sum()):<3d} {np.mean(pred[m] == cl)*100:5.1f}%  wrong->{dict(wrong)}")
    if conf is not None:
        for t in (0.6, 0.7, 0.8, 0.9):
            keep = conf >= t
            if keep.any():
                print(f"  confidence>={t}: keeps {keep.mean()*100:4.0f}% of photos, accuracy {np.mean(pred[keep] == true[keep])*100:5.1f}%")


best = None
for name, X in sets.items():
    heads, cvs = fit_heads(X, lambda: tr.copy())
    p, conf = predict_heads(heads, X[te])
    acc = float(np.mean(p == lab[te]))
    print(f"\n[{name}] CV headwear={cvs['headwear']*100:.1f}% specs={cvs['specs']*100:.1f}% hair={cvs['hair']*100:.1f}%  | held-out 8-label acc={acc*100:.1f}%")
    if best is None or acc > best[0]:
        best = (acc, name, heads, p, conf, X)

acc, name, heads, p, conf, X = best
print(f"\n>>> best feature set: {name}")
report(f"3-head model [{name}] on held-out test", p, lab[te], conf)

# deployment model: refit on train + test (all labelled photos) with the same C per head
final = {}
allm = np.ones(len(X), bool)
for hname, y, mask in (("headwear", y_hw, allm), ("specs", y_sp, allm), ("hair", y_hair, y_hw == "none")):
    final[hname] = lr(heads[hname].C).fit(X[mask], y[mask])
joblib.dump({"heads": final, "feature_set": name, "classes": CLASSES}, OUT)
print("\nsaved", OUT)

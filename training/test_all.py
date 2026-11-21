"""Full held-out test set, judged by a model trained on the train split only."""
import csv
import warnings
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
import sys
FEAT = sys.argv[1] if len(sys.argv) > 1 else r"D:\audit\dataset\features.npz"
CNXF = sys.argv[2] if len(sys.argv) > 2 else r"D:\audit\dataset\cnx.npz"
d = np.load(FEAT, allow_pickle=True)
c = np.load(CNXF, allow_pickle=True)
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
tr, te = split == "train", split == "test"
lr = lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=4000)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def best_C(Xs, ys):
    sc = {C: np.mean(cross_val_predict(lr(C), Xs, ys, cv=StratifiedKFold(3, shuffle=True, random_state=0)) == ys)
          for C in (0.3, 1, 3, 10, 30)}
    return max(sc, key=sc.get)


mh = tr & (Y["hw"] == "none")
M = {"hw": lr(best_C(X[tr], Y["hw"][tr])).fit(X[tr], Y["hw"][tr]),
     "sp": lr(best_C(X[tr], Y["sp"][tr])).fit(X[tr], Y["sp"][tr]),
     "hr": lr(best_C(X[mh], Y["hr"][mh])).fit(X[mh], Y["hr"][mh])}
print("train photos:", int(tr.sum()), dict(Counter(map(str, lab[tr]))))
hw, sp, hr = M["hw"].predict(X[te]), M["sp"].predict(X[te]).astype(int), M["hr"].predict(X[te])
conf = np.minimum(M["hw"].predict_proba(X[te]).max(1), M["sp"].predict_proba(X[te]).max(1))
pred = np.array([compose(a, b, c_) for a, b, c_ in zip(hw, sp, hr)])
true, tid = lab[te], ids[te]

print(f"HELD-OUT TEST SET: {int(te.sum())} photos never used for training")
print(f"OVERALL: {int(np.sum(pred == true))}/{len(true)} = {100*np.mean(pred == true):.1f}%\n")
for cl in CLASSES:
    m = true == cl
    if m.any():
        wrong = Counter(map(str, pred[m & (pred != cl)]))
        print(f"{cl:18s} {int(np.sum(pred[m] == cl)):3d}/{int(m.sum()):<3d} {100*np.mean(pred[m] == cl):5.1f}%  wrong->{dict(wrong)}")

# component-level accuracy: shows which question causes the errors
tw = np.array([parts(l)[0] for l in true]); ts = np.array([parts(l)[1] for l in true]); th = np.array([parts(l)[2] for l in true])
nh = tw == "none"
print(f"\nheadwear (none/cap/turban): {100*np.mean(hw == tw):.1f}%   specs: {100*np.mean(sp == ts):.1f}%   "
      f"hair colour (no-headwear photos): {100*np.mean(hr[nh] == th[nh]):.1f}%")
for t in (0.6, 0.7, 0.8):
    k = conf >= t
    print(f"confidence>={t}: answers {100*k.mean():.0f}% of photos at {100*np.mean(pred[k] == true[k]):.1f}% accuracy")

with open(r"D:\audit\training\test_all_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "true", "predicted", "correct", "confidence"])
    for i, t, p, cf in zip(tid, true, pred, conf):
        w.writerow([i, t, p, int(t == p), round(float(cf), 3)])
print("\nwrong ids:", ", ".join(f"{i}({t}->{p})" for i, t, p in zip(tid, true, pred) if t != p))
print("details: D:\\audit\\training\\test_all_results.csv")

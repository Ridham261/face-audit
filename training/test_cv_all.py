"""5-fold cross-validation over ALL labelled photos: every photo is predicted by a model that never saw it."""
import csv
import warnings
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
d = np.load(r"D:\audit\dataset\features_all.npz", allow_pickle=True)
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
lr = lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=4000)
pred = np.empty(len(lab), dtype=object); conf = np.zeros(len(lab))
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, lab), 1):
    mh = Y["hw"][a] == "none"
    mhw, msp, mhr = lr(1).fit(X[a], Y["hw"][a]), lr(3).fit(X[a], Y["sp"][a]), lr(10).fit(X[a][mh], Y["hr"][a][mh])
    hw, sp, hr = mhw.predict(X[b]), msp.predict(X[b]).astype(int), mhr.predict(X[b])
    pred[b] = [compose(p, q, r) for p, q, r in zip(hw, sp, hr)]
    conf[b] = np.minimum(mhw.predict_proba(X[b]).max(1), msp.predict_proba(X[b]).max(1))
    print(f"fold {f} done", flush=True)

print(f"\nALL {len(lab)} labelled photos, each judged by a model that never saw it")
print(f"OVERALL: {int(np.sum(pred == lab))}/{len(lab)} = {100*np.mean(pred == lab):.1f}%\n")
for cl in CLASSES:
    m = lab == cl
    wrong = Counter(map(str, pred[m & (pred != cl)]))
    print(f"{cl:18s} {int(np.sum(pred[m] == cl)):4d}/{int(m.sum()):<4d} {100*np.mean(pred[m] == cl):5.1f}%  wrong->{dict(wrong.most_common(3))}")
tw = np.array([p[0] for p in P]); ts = np.array([p[1] for p in P])
print("\nconfidence cut-offs (answer only when sure; rest -> manual review):")
for t in (0.6, 0.7, 0.8, 0.9):
    k = conf >= t
    print(f"  >={t}: answers {100*k.mean():3.0f}% of photos at {100*np.mean(pred[k] == lab[k]):.1f}% accuracy")
with open(r"D:\audit\training\cv_all_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["split", "id", "true", "predicted", "correct", "confidence"])
    for s_, i, t, p, cf in zip(split, ids, lab, pred, conf):
        w.writerow([s_, i, t, p, int(t == p), round(float(cf), 3)])
print("details: D:\\audit\\training\\cv_all_results.csv")

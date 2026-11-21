"""Try stronger classifiers / feature sets on the same train/test split (3-head structure)."""
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
d = np.load(r"D:\audit\dataset\features_all.npz", allow_pickle=True)
c = np.load(r"D:\audit\dataset\cnx_all.npz", allow_pickle=True)
Xc, lab, split, ids = d["X"], d["label"], d["split"], d["id"]
key = {k: i for i, k in enumerate(c["key"])}
ix = [key[f"{s}/{l}/{i}.png"] for s, l, i in zip(split, lab, ids)]
prob, feat = c["prob"][ix], c["feat"][ix]
feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
pw = np.repeat(prob[:, [0, 2, 4]], 8, axis=1)
SETS = {"clip+cnxprob": np.hstack([Xc, pw]), "clip+cnx": np.hstack([Xc, pw, feat]), "clip": Xc}


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
mh = tr & (Y["hw"] == "none")
CANDS = {
    "LR C=3": lambda: LogisticRegression(C=3, class_weight="balanced", max_iter=4000),
    "LR C=10": lambda: LogisticRegression(C=10, class_weight="balanced", max_iter=4000),
    "SVM rbf C=10": lambda: SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced"),
    "SVM rbf C=30": lambda: SVC(C=30, kernel="rbf", gamma="scale", class_weight="balanced"),
    "MLP 512": lambda: MLPClassifier((512,), alpha=1e-3, max_iter=400, early_stopping=True, random_state=0),
}
print(f"{'features':14s}{'classifier':14s}{'headwear':>9s}{'specs':>7s}{'hair':>7s}{'8-label':>9s}")
best = (0, None)
for sn, X in SETS.items():
    for cn, mk in CANDS.items():
        hw = mk().fit(X[tr], Y["hw"][tr]).predict(X[te])
        sp = mk().fit(X[tr], Y["sp"][tr]).predict(X[te]).astype(int)
        hr = mk().fit(X[mh], Y["hr"][mh]).predict(X[te])
        pred = np.array([compose(a, b, c_) for a, b, c_ in zip(hw, sp, hr)])
        true = lab[te]
        acc = float(np.mean(pred == true))
        tw = np.array([parts(l)[0] for l in true]); ts = np.array([parts(l)[1] for l in true]); th = np.array([parts(l)[2] for l in true])
        nh = tw == "none"
        print(f"{sn:14s}{cn:14s}{100*np.mean(hw==tw):8.1f}%{100*np.mean(sp==ts):6.1f}%{100*np.mean(hr[nh]==th[nh]):6.1f}%{100*acc:8.1f}%", flush=True)
        if acc > best[0]:
            best = (acc, (sn, cn))
print("\nBEST:", best[1], f"{100*best[0]:.1f}%")

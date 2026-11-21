"""Convert appearance_head.joblib (scikit-learn) to a plain .npz so the service needs only numpy."""
import sys

import joblib
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
m = joblib.load(src)
out = {"feature_set": np.array(m["feature_set"])}
for name, clf in m["heads"].items():
    out[f"{name}__classes"] = np.array([str(c) for c in clf.classes_])
    out[f"{name}__coef"] = clf.coef_.astype(np.float32)
    out[f"{name}__intercept"] = clf.intercept_.astype(np.float32)
    print(name, list(clf.classes_), clf.coef_.shape)
np.savez(dst, **out)
print("saved", dst, "feature_set =", m["feature_set"])

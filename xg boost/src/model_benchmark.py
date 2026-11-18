"""Quick group-CV benchmark of candidate-scoring models (LightGBM and MLP excluded: taken by teammates).

Train partition only, 2 trips with unreliable labels excluded, GroupKFold by trip.
Metrics: observation-level top-1 accuracy (argmax candidate per GPS point), log loss, fit time.
Writes results/model_benchmark.csv
"""
import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BAD = {"00000054", "00000033"}      # GPS ~130 m off the reference route: labels unreliable
FEATS = ["dist", "d_gap", "rank", "n_cand", "t_frac", "along_m", "arc_len", "lateral", "heading_cos",
         "heading_ok", "speed", "dt_prev", "dt_next", "out_deg_to", "in_deg_from"]
SEED = 42

df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "candidates.csv.gz"), dtype={"trip": str})
n_before = df.trip.nunique()
df = df[(df.part == "train") & (~df.trip.isin(BAD))].reset_index(drop=True)
assert not (set(df.trip) & BAD), "bad trips were not excluded"
X, y, g = df[FEATS].values.astype(np.float32), df["y"].values, df["trip"].values
key = df["trip"] + "_" + df["pt"].astype(str)
print("rows", len(df), "trips", df.trip.nunique(), "positive rate %.3f" % y.mean(), flush=True)

pos_w = (1 - y.mean()) / y.mean()
MODELS = {
    "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED),
    "CatBoost": lambda: CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1, random_seed=SEED, verbose=0),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED),
    "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
}

rows = []
for name, make in MODELS.items():
    accs, lls, t_fit = [], [], 0.0
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=4).split(X, y, g)):
        m = make()
        t0 = time.time(); m.fit(X[tr], y[tr]); t_fit += time.time() - t0
        p = m.predict_proba(X[te])[:, 1]
        d = pd.DataFrame({"k": key.values[te], "p": p, "y": y[te]})
        top1 = d.loc[d.groupby("k")["p"].idxmax(), "y"]     # candidate with highest score per point
        n_pts = key.values[te]
        accs.append(top1.sum() / len(np.unique(n_pts)))     # over points that have candidates
        lls.append(log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6)))
    rows.append(dict(model=name, top1_acc=np.mean(accs), top1_std=np.std(accs), log_loss=np.mean(lls),
                     fit_s_per_fold=t_fit / 4))
    print(rows[-1], flush=True)

res = pd.DataFrame(rows).sort_values("top1_acc", ascending=False)
os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
res.to_csv(os.path.join(HERE, "..", "results", "model_benchmark.csv"), index=False)
print(res.to_string(index=False))

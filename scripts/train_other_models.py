"""Baseline model families evaluated with the same protocol as LightGBM, for the technical report.

Models: logistic regression, Gaussian NB, Extra Trees, distance-weighted KNN (CPU, 100k stored rows) and the
nearest-arc heuristic. Also records train-set Top-1 for LightGBM and the Random Forest so that over/underfitting can
be judged from the train / CV / test gap.

Protocol: GroupKFold(5) by trip on the training trips (two off-route trips excluded), then fit on all training trips and
evaluate on student_test. Seeds are fixed. Outputs go to results/other_models/.

Usage: uv run python scripts/train_other_models.py [model ...]
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pathsnap.features import FEATURES  # noqa: E402
from train_lightgbm import BAD_TRIP_INVALID_FRAC, LGB_PARAMS, SEED, evaluate, load  # noqa: E402

OUT = ROOT / "results" / "other_models"
KNN_ROWS, N_JOBS = 100_000, 4


CONFIGS = {
    "logreg": ("Linear", "StandardScaler + LogisticRegression(C=1.0, max_iter=300, lbfgs)"),
    "gaussian_nb": ("Probabilistic", "GaussianNB(var_smoothing=1e-9), median imputation"),
    "extra_trees": ("Tree ensemble (randomized)", "150 trees, min_samples_leaf=20, max_features=sqrt, max_samples=0.5"),
    "knn": ("Instance-based", "k=25, distance weighted, standardized features, 100k stored training rows, CPU brute force"),
    "nearest_arc": ("Heuristic (no learning)", "score = 1 / (1 + distance to arc)"),
    "lightgbm": ("Boosting", "reference: train-set accuracy only"),
    "random_forest": ("Ensemble (bagging)", "reference: train-set accuracy only"),
}


def make(name):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=300, C=1.0, random_state=SEED))
    if name == "gaussian_nb":
        return make_pipeline(SimpleImputer(strategy="median"), GaussianNB(var_smoothing=1e-9))
    if name == "extra_trees":
        return make_pipeline(SimpleImputer(strategy="constant", fill_value=-999.0),
                             ExtraTreesClassifier(n_estimators=150, min_samples_leaf=20, max_features="sqrt",
                                                  max_samples=0.5, bootstrap=True, n_jobs=N_JOBS, random_state=SEED))
    if name == "knn":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", algorithm="brute", n_jobs=N_JOBS))
    if name == "lightgbm":
        return lgb.LGBMClassifier(**{**LGB_PARAMS, "n_jobs": N_JOBS})
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="constant", fill_value=-999.0),
                             RandomForestClassifier(n_estimators=150, min_samples_leaf=20, max_features="sqrt",
                                                    max_samples=0.5, n_jobs=N_JOBS, random_state=SEED))
    raise KeyError(name)


def scores(name, model, d):
    if name == "nearest_arc":
        return 1.0 / (1.0 + d["d"].to_numpy())
    return model.predict_proba(d[FEATURES])[:, 1]


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    tr, rep = load("train")
    te, te_rep = load("student_test")
    bad = set(rep.loc[rep["frac_label_invalid"] > BAD_TRIP_INVALID_FRAC, "trip"])
    rep = rep[~rep["trip"].isin(bad)]
    tr = tr[~tr["trip"].isin(bad) & (tr["label_valid"] == 1)].reset_index(drop=True)
    te = te[te["label_valid"] == 1].reset_index(drop=True)
    pts, pts_te = dict(zip(rep["trip"], rep["n_valid_pts"])), int(te_rep["n_valid_pts"].sum())
    X, y, groups = tr[FEATURES], tr["label"].to_numpy(), tr["trip"].to_numpy()
    rng = np.random.default_rng(SEED)
    res, fold_rows = {}, []
    for name in names:
        fam, cfg = CONFIGS[name]
        print(f"=== {name} ===", flush=True)
        cv_folds = []
        if name not in ("lightgbm", "random_forest"):
            oof = np.zeros(len(tr))
            for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
                assert not (set(groups[a]) & set(groups[b])), "trip leakage"
                if name == "nearest_arc":
                    oof[b] = scores(name, None, tr.iloc[b])
                else:
                    if name == "knn" and len(a) > KNN_ROWS:
                        a = np.sort(rng.choice(a, KNN_ROWS, replace=False))
                    m = make(name).fit(X.iloc[a], y[a])
                    oof[b] = scores(name, m, tr.iloc[b])
                sub = tr.iloc[b]
                r = evaluate(sub, oof[b], sum(pts[t] for t in sub["trip"].unique()))
                cv_folds.append(r)
                fold_rows.append({"model": name, "fold": k + 1, "top1_acc": r["top1_acc"],
                                  "cand_macro_f1": r["cand_macro_f1"], "log_loss": r["log_loss"]})
                print(f"  fold {k + 1}: top1={r['top1_acc']:.4f} macroF1={r['cand_macro_f1']:.4f}", flush=True)
        # final fit on all training trips
        if name == "nearest_arc":
            m, fit_s = None, 0.0
        else:
            idx = np.arange(len(tr))
            if name == "knn":
                idx = np.sort(rng.choice(idx, KNN_ROWS, replace=False))
            t0 = time.time()
            m = make(name).fit(X.iloc[idx], y[idx])
            fit_s = time.time() - t0
        t0 = time.time()
        p = scores(name, m, te)
        inf_s = time.time() - t0
        r = evaluate(te, p, pts_te)
        entry = {"family": fam, "config": cfg, "student_test": r, "fit_s": fit_s, "inference_s": inf_s,
                 "inference_us_per_row": inf_s / len(te) * 1e6}
        if name != "knn" and name != "nearest_arc":      # KNN memorises its fit rows, so train accuracy is not meaningful
            ptr = scores(name, m, tr)
            entry["train_top1"] = evaluate(tr, ptr, sum(pts.values()))["top1_acc"]
        if cv_folds:
            for key in ("top1_acc", "cand_macro_f1", "log_loss", "route_continuity"):
                v = [f[key] for f in cv_folds]
                entry[f"cv_{key}_mean"], entry[f"cv_{key}_std"] = float(np.mean(v)), float(np.std(v, ddof=1))
        res[name] = entry
        print(f"[{name}] test top1={r['top1_acc']:.4f} macroF1={r['cand_macro_f1']:.4f} logloss={r['log_loss']:.4f} "
              f"train_top1={entry.get('train_top1')}", flush=True)
        # persist after every model
        mp = OUT / "metrics.json"
        old = json.loads(mp.read_text()) if mp.exists() else {}
        old.update({name: entry})
        mp.write_text(json.dumps(old, indent=2))
    fp = OUT / "cv_folds.csv"
    prev = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    if len(prev):
        prev = prev[~prev["model"].isin(names)]
    pd.concat([prev, pd.DataFrame(fold_rows)], ignore_index=True).to_csv(fp, index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["logreg", "gaussian_nb", "extra_trees", "nearest_arc", "lightgbm", "random_forest", "knn"])

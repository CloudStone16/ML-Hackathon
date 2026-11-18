"""Train the Random Forest (second-best of my models) with the same protocol as LightGBM.

Reuses load/evaluate from train_lightgbm.py. GroupKFold(5) by trip on the training trips, then fit
on all training trips and evaluate on student_test. Results are merged into results/model_comparison.csv.

Usage: uv run python scripts/train_more_models.py [model ...]   (default: random_forest)
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pathsnap.features import FEATURES  # noqa: E402
from train_lightgbm import BAD_TRIP_INVALID_FRAC, SEED, evaluate, load  # noqa: E402

KNN_TRAIN_ROWS = 200_000
N_JOBS = 8   # leave CPU headroom (14 cores on this machine)


CONFIGS = {
    "random_forest": ("Ensemble (bagging)", dict(n_estimators=150, min_samples_leaf=20, max_features="sqrt",
                                                  max_samples=0.5, n_jobs=N_JOBS, random_state=SEED)),
}


def make(name):
    fam, cfg = CONFIGS[name]
    # constant fill keeps "undefined heading" distinguishable from real values
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=-999.0), RandomForestClassifier(**cfg))


def subsample_idx(name, idx, rng):
    """KNN stores its training set, so cap it; other models use all rows."""
    if name == "knn" and len(idx) > KNN_TRAIN_ROWS:
        return np.sort(rng.choice(idx, KNN_TRAIN_ROWS, replace=False))
    return idx


def persist(res_dir, names, results, fold_rows):
    """Merge this run's metrics into results/ (called after every model)."""
    mp = res_dir / "more_models_metrics.json"
    old = json.loads(mp.read_text()) if mp.exists() else {}
    old.update(results)
    mp.write_text(json.dumps(old, indent=2))

    fp = res_dir / "cv_folds.csv"
    folds = pd.read_csv(fp)
    folds = pd.concat([folds[~folds["model"].isin(names)], pd.DataFrame(fold_rows)], ignore_index=True)
    folds.to_csv(fp, index=False)

    cp = res_dir / "model_comparison.csv"
    comp = pd.read_csv(cp, index_col=0)
    new = pd.DataFrame({n: {"cv_top1": r["cv"]["top1_acc"], "cv_macro_f1": r["cv"]["cand_macro_f1"],
                            "cv_logloss": r["cv"]["log_loss"], "test_top1": r["student_test"]["top1_acc"],
                            "test_macro_f1": r["student_test"]["cand_macro_f1"],
                            "test_logloss": r["student_test"]["log_loss"],
                            "test_continuity": r["student_test"]["route_continuity"]} for n, r in results.items()}).T
    comp = pd.concat([comp.drop(index=[n for n in names if n in comp.index]), new])
    comp.to_csv(cp)
    print(comp.round(4).to_string())




def main(names):
    tr, tr_rep = load("train")
    te, te_rep = load("student_test")
    bad = set(tr_rep.loc[tr_rep["frac_label_invalid"] > BAD_TRIP_INVALID_FRAC, "trip"])
    tr_rep = tr_rep[~tr_rep["trip"].isin(bad)]
    tr = tr[~tr["trip"].isin(bad) & (tr["label_valid"] == 1)].reset_index(drop=True)
    te = te[te["label_valid"] == 1].reset_index(drop=True)
    valid_pts = dict(zip(tr_rep["trip"], tr_rep["n_valid_pts"]))
    valid_pts_te = dict(zip(te_rep["trip"], te_rep["n_valid_pts"]))
    X, y, groups = tr[FEATURES], tr["label"].to_numpy(), tr["trip"].to_numpy()
    rng = np.random.default_rng(SEED)
    res_dir = ROOT / "results"
    results, fold_rows = {}, []

    for name in names:
        fam, cfg = CONFIGS[name]
        print(f"=== {name} ({fam}) ===", flush=True)
        oof = np.zeros(len(tr))
        t_fit = 0.0
        for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
            assert not (set(groups[a]) & set(groups[b])), "trip leakage across folds"
            a = subsample_idx(name, a, rng)
            m = make(name)
            t0 = time.time(); m.fit(X.iloc[a], y[a]); t_fit += time.time() - t0
            oof[b] = m.predict_proba(X.iloc[b])[:, 1]
            sub = tr.iloc[b]
            r = evaluate(sub, oof[b], sum(valid_pts[t] for t in sub["trip"].unique()))
            fold_rows.append({"model": name, "fold": k, **{c: r[c] for c in ("top1_acc", "cand_macro_f1", "log_loss", "route_continuity")}})
            print(f"  fold {k}: top1={r['top1_acc']:.4f} macroF1={r['cand_macro_f1']:.4f} logloss={r['log_loss']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        cv = evaluate(tr, oof, sum(valid_pts.values()))
        cv["fit_time_s_total_5folds"] = t_fit

        m = make(name)
        idx = subsample_idx(name, np.arange(len(tr)), rng)
        t0 = time.time(); m.fit(X.iloc[idx], y[idx]); fit_s = time.time() - t0
        t0 = time.time(); p = m.predict_proba(te[FEATURES])[:, 1]; inf_s = time.time() - t0
        r = evaluate(te, p, sum(valid_pts_te.values()))
        r.update(fit_time_s=fit_s, inference_time_s=inf_s, inference_us_per_row=inf_s / len(te) * 1e6)
        results[name] = {"family": fam, "config": {k: str(v) for k, v in cfg.items()}, "cv": cv, "student_test": r}
        print(f"[{name}] student_test top1={r['top1_acc']:.4f} macroF1={r['cand_macro_f1']:.4f} "
              f"logloss={r['log_loss']:.4f} continuity={r['route_continuity']:.4f} fit={fit_s:.0f}s", flush=True)
        path = res_dir / f"{name}_model.joblib"
        joblib.dump(m, path)
        if path.stat().st_size > MAX_SAVE_MB * 1e6:
            print(f"  model file {path.stat().st_size / 1e6:.0f} MB > {MAX_SAVE_MB} MB, not kept")
            path.unlink()
        persist(res_dir, [name], {name: results[name]}, [f for f in fold_rows if f["model"] == name])



if __name__ == "__main__":
    main(sys.argv[1:] or list(CONFIGS))

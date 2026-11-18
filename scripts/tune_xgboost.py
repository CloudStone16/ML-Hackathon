"""Bayesian (Optuna TPE) tuning for the teammate XGBoost candidate classifier.

Uses the teammate's preprocessing (xg boost/src/preprocess.py: labels, 50 m / 20-candidate generation, 15 features)
unchanged; only hyperparameters are searched. Objective: mean observation-level Top-1 over 5-fold GroupKFold by
trip on the training trips (two off-route trips excluded, as they do). student_test / evaluator_test are not used
for selection. Trial 0 is their current tuned config, and their untuned baseline is evaluated for reference.

Usage: PATHSNAP_DATA=<dir with train/ student_test/> uv run python scripts/tune_xgboost.py [n_trials]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
XDIR = ROOT / "xg boost"
sys.path.insert(0, str(XDIR / "src"))
import preprocess as xp  # noqa: E402
from xgb_common import BAD_TRIPS, FEATS, evaluate  # noqa: E402

DATA = os.environ["PATHSNAP_DATA"]
xp.ROOT = DATA
CACHE = ROOT / "artifacts" / "xgb_data"
OUT = ROOT / "results" / "tuning"
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED, N_JOBS = 42, 8


def build(part):
    """Candidate table (+ y from their DP labeller) and points per trip; cached."""
    f, pf = CACHE / f"{part}.parquet", CACHE / f"{part}_points.csv"
    if f.exists():
        return pd.read_parquet(f), pd.read_csv(pf, dtype={"trip": str})
    CACHE.mkdir(parents=True, exist_ok=True)
    frames, pts = [], []
    for tid in sorted(t for t in os.listdir(os.path.join(DATA, part)) if not t.startswith(".")):
        track, nodes, arcs, route = xp.load_trip(part, tid)
        lab, _ = xp.label_trip(track, nodes, arcs, route)
        d = xp.build_features(track, nodes, arcs, R=50.0, K=20)
        d["y"] = (d["arc"].to_numpy() == lab[d["pt"].to_numpy()]).astype(np.int8)
        d.insert(0, "trip", tid)
        frames.append(d)
        pts.append({"trip": tid, "n_points": len(track)})
        print(part, tid, len(track), flush=True)
    df, P = pd.concat(frames, ignore_index=True), pd.DataFrame(pts)
    df.to_parquet(f)
    P.to_csv(pf, index=False)
    return df, P


def fast_top1(pt_key, y, p, n_points):
    order = np.lexsort((-p, pt_key))
    k = pt_key[order]
    first = np.r_[True, k[1:] != k[:-1]]
    return float(y[order][first].sum() / n_points)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, Ptr = build("train")
    te, Pte = build("student_test")
    tr = tr[~tr["trip"].isin(BAD_TRIPS)].reset_index(drop=True)
    npts = dict(zip(Ptr["trip"], Ptr["n_points"]))
    X, y, groups = tr[FEATS].to_numpy(np.float32), tr["y"].to_numpy(), tr["trip"].to_numpy()
    pt_key = (pd.Categorical(tr["trip"]).codes.astype(np.int64) * 10_000_000 + tr["pt"].to_numpy()).astype(np.int64)
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    fold_pts = [sum(npts[t] for t in np.unique(groups[b])) for _, b in folds]

    tuned_theirs = json.loads((XDIR / "results" / "xgb_tuned.json").read_text())["best_params"]
    base_theirs = json.loads((XDIR / "results" / "xgb_baseline.json").read_text())["params"]
    fixed = dict(tree_method="hist", n_jobs=N_JOBS, random_state=SEED, eval_metric="logloss")

    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 800),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 3.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True), **fixed)
        scores = []
        for i, (a, b) in enumerate(folds):
            m = xgb.XGBClassifier(**p).fit(X[a], y[a])
            scores.append(fast_top1(pt_key[b], y[b], m.predict_proba(X[b])[:, 1], fold_pts[i]))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    keys = ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight",
            "gamma", "reg_lambda", "reg_alpha")
    study.enqueue_trial({k: int(tuned_theirs[k]) if k in ("n_estimators", "max_depth") else float(tuned_theirs[k])
                         for k in keys})                                   # trial 0: their tuned config
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number}: {t.state.name} value={t.value} "
                                                 f"best={s.best_value:.4f} ({time.time() - t0:.0f}s)", flush=True)])
    search_s = time.time() - t0
    study.trials_dataframe().to_csv(OUT / "xgboost_trials.csv", index=False)

    out = {"objective": "mean 5-fold GroupKFold observation-level Top-1 (train trips only)", "n_trials": N_TRIALS,
           "search_seconds": search_s, "best_cv_top1_mean": study.best_value,
           "their_tuned_cv_top1_mean": study.trials[0].value, "best_params": study.best_params}
    npts_te = int(Pte["n_points"].sum())
    configs = {"their_untuned": {**base_theirs, "n_jobs": N_JOBS},
               "their_tuned": {**tuned_theirs, **fixed}, "retuned": {**study.best_params, **fixed}}
    for name, params in configs.items():
        t1 = time.time()
        m = xgb.XGBClassifier(**params).fit(X, y)
        fit_s = time.time() - t1
        t1 = time.time()
        p = m.predict_proba(te[FEATS])[:, 1]
        inf_s = time.time() - t1
        r = evaluate(te, p, npts_te)
        out[f"student_test_{name}"] = {"top1_acc": r["top1_acc"], "f1_macro": r["f1_macro"], "log_loss": r["log_loss"],
                                       "fit_s": fit_s, "inference_s": inf_s}
        if name == "retuned":
            m.save_model(str(OUT / "xgboost_retuned.json"))
        print(name, out[f"student_test_{name}"], flush=True)
    (OUT / "xgboost_tuning_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

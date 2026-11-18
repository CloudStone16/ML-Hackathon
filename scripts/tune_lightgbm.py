"""Bayesian (Optuna TPE) hyperparameter tuning for the LightGBM candidate classifier.

Objective: mean observation-level Top-1 over 5-fold GroupKFold (by trip) on the training trips only.
student_test / evaluator_test are never used to choose parameters. Trial 0 is the current (untuned) config so
the comparison is like for like.

Usage: uv run python scripts/tune_lightgbm.py [n_trials]
"""
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pathsnap.features import FEATURES  # noqa: E402
from train_lightgbm import BAD_TRIP_INVALID_FRAC, LGB_PARAMS, SEED, evaluate, load  # noqa: E402

OUT = ROOT / "results" / "tuning"
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_JOBS = 8


def prepare():
    tr, rep = load("train")
    bad = set(rep.loc[rep["frac_label_invalid"] > BAD_TRIP_INVALID_FRAC, "trip"])
    rep = rep[~rep["trip"].isin(bad)]
    tr = tr[~tr["trip"].isin(bad) & (tr["label_valid"] == 1)].reset_index(drop=True)
    valid_pts = dict(zip(rep["trip"], rep["n_valid_pts"]))
    return tr, valid_pts


def fast_top1(pt_key, y, p, n_points):
    """Observation-level Top-1: highest-scoring candidate per point is the true arc."""
    order = np.lexsort((-p, pt_key))
    k = pt_key[order]
    first = np.r_[True, k[1:] != k[:-1]]
    return float(y[order][first].sum() / n_points)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, valid_pts = prepare()
    X, y, groups = tr[FEATURES].to_numpy(np.float32), tr["label"].to_numpy(), tr["trip"].to_numpy()
    pt_key = (pd.Categorical(tr["trip"]).codes.astype(np.int64) * 10_000_000 + tr["pt"].to_numpy()).astype(np.int64)
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    fold_pts = [sum(valid_pts[t] for t in np.unique(groups[b])) for _, b in folds]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            subsample_freq=1, random_state=SEED, n_jobs=N_JOBS, verbose=-1)
        scores = []
        for i, (a, b) in enumerate(folds):
            m = lgb.LGBMClassifier(**params).fit(X[a], y[a])
            scores.append(fast_top1(pt_key[b], y[b], m.predict_proba(X[b])[:, 1], fold_pts[i]))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    base = {k: LGB_PARAMS[k] for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample",
                                       "colsample_bytree", "reg_lambda")}
    base.update(reg_alpha=1e-3, min_split_gain=0.0)
    study.enqueue_trial(base)                       # trial 0 = untuned reference
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number}: {t.state.name} value={t.value} "
                                                 f"best={s.best_value:.4f} ({time.time() - t0:.0f}s)", flush=True)])
    search_s = time.time() - t0
    study.trials_dataframe().to_csv(OUT / "lightgbm_trials.csv", index=False)

    best = dict(study.best_params, subsample_freq=1, random_state=SEED, n_jobs=N_JOBS, verbose=-1)
    untuned = {**LGB_PARAMS, "n_jobs": N_JOBS}
    # final: refit on all training trips, evaluate on student_test (dev set)
    te, te_rep = load("student_test")
    te = te[te["label_valid"] == 1].reset_index(drop=True)
    te_pts = int(te_rep["n_valid_pts"].sum())
    out = {"objective": "mean 5-fold GroupKFold observation-level Top-1 (train trips only)", "n_trials": N_TRIALS,
           "search_seconds": search_s, "best_cv_top1_mean": study.best_value,
           "untuned_cv_top1_mean": study.trials[0].value, "best_params": study.best_params}
    for name, params in (("untuned", untuned), ("tuned", best)):
        t1 = time.time()
        m = lgb.LGBMClassifier(**params).fit(X, y)
        fit_s = time.time() - t1
        t1 = time.time()
        p = m.predict_proba(te[FEATURES])[:, 1]
        inf_s = time.time() - t1
        r = evaluate(te, p, te_pts)
        out[f"student_test_{name}"] = {**{k: r[k] for k in ("top1_acc", "cand_macro_f1", "log_loss", "route_continuity")},
                                       "fit_s": fit_s, "inference_s": inf_s}
        if name == "tuned":
            joblib.dump(m, OUT / "lightgbm_tuned_model.joblib")
        print(name, out[f"student_test_{name}"], flush=True)
    (OUT / "lightgbm_tuning_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

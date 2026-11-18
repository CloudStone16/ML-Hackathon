"""Bayesian hyperparameter optimisation for XGBoost (Optuna TPE), then refit and compare with the baseline.

Search: 5-fold GroupKFold by trip on the 78 usable train trips. Objective: minimise mean CV log loss
(a smooth, calibrated-probability objective; the HMM decoder consumes these probabilities as emissions).
Observation-level top-1 accuracy is recorded for every trial and reported alongside.
student_test is NOT used for the search; it is used once afterwards to compare tuned vs untuned.

Run:  python src/xgb_tune.py [--trials 40] [--device cuda]
Writes results/xgb_trials.csv, results/xgb_optuna.db, results/xgb_tuned.json,
       results/xgb_oof_tuned.csv.gz, results/xgb_test_tuned.csv.gz, models/xgb_tuned.json
"""
import argparse
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from xgb_common import FEATS, MODELS, RES, SEED, evaluate, load_data

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    train, test, n_pts = load_data()
    X, y, g = train[FEATS].values.astype(np.float32), train["y"].values, train["trip"].values
    splits = list(GroupKFold(n_splits=5).split(X, y, g))        # same folds as the baseline
    fold_pts = [sum(n_pts[t] for t in set(g[va])) for _, va in splits]
    base = dict(tree_method="hist", device=args.device, n_jobs=-1, random_state=SEED, eval_metric="logloss")

    def cv(params, keep_oof=False):
        oof, rows = np.zeros(len(train)), []
        t0 = time.time()
        for (tr, va), npts in zip(splits, fold_pts):
            m = XGBClassifier(**base, **params)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
            rows.append(evaluate(train.iloc[va], oof[va], npts))
        return oof, rows, time.time() - t0

    def objective(trial):
        params = suggest(trial)
        _, rows, secs = cv(params)
        trial.set_user_attr("cv_top1_acc", float(np.mean([r["top1_acc"] for r in rows])))
        trial.set_user_attr("cv_f1_macro", float(np.mean([r["f1_macro"] for r in rows])))
        trial.set_user_attr("seconds", secs)
        print("trial %2d: logloss %.5f top1 %.4f (%.0fs)" % (trial.number, np.mean([r["log_loss"] for r in rows]),
                                                            trial.user_attrs["cv_top1_acc"], secs), flush=True)
        return float(np.mean([r["log_loss"] for r in rows]))

    db = os.path.join(RES, "xgb_optuna.db")
    if os.path.exists(db):
        os.remove(db)       # fresh study every run so the saved record matches the final result
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                storage="sqlite:///" + db.replace("\\", "/"), study_name="xgb_group_cv")
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials)
    search_s = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(RES, "xgb_trials.csv"), index=False)
    best = study.best_params
    print("\nbest CV log loss %.5f after %d trials (%.0f s)\nbest params: %s" % (study.best_value, len(trials), search_s, best))

    # Full-metric CV of the best configuration, same folds as the baseline
    oof, rows, _ = cv(best)
    agg = lambda k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows])))
    cv_summary = {k: agg(k) for k in ("top1_acc", "f1_macro", "precision_macro", "recall_macro", "log_loss")}

    # Refit on all train, evaluate once on student_test
    Xte, yte = test[FEATS].values.astype(np.float32), test["y"].values
    final = XGBClassifier(**base, **best)
    t0 = time.time()
    final.fit(X, y, eval_set=[(X, y), (Xte, yte)], verbose=False)
    fit_s = time.time() - t0
    t0 = time.time()
    p_te = final.predict_proba(Xte)[:, 1]
    inf_ms = 1000 * (time.time() - t0) / len(Xte) * 1000
    te = evaluate(test, p_te, sum(n_pts[t] for t in test.trip.unique()))

    with open(os.path.join(RES, "xgb_baseline.json")) as f:
        b = json.load(f)
    cmp_rows = [
        ("CV top-1 accuracy", b["cv_summary"]["top1_acc"]["mean"], cv_summary["top1_acc"]["mean"]),
        ("CV macro F1", b["cv_summary"]["f1_macro"]["mean"], cv_summary["f1_macro"]["mean"]),
        ("CV log loss", b["cv_summary"]["log_loss"]["mean"], cv_summary["log_loss"]["mean"]),
        ("student-test top-1 accuracy", b["student_test"]["top1_acc"], te["top1_acc"]),
        ("student-test macro F1", b["student_test"]["f1_macro"], te["f1_macro"]),
        ("student-test log loss", b["student_test"]["log_loss"], te["log_loss"]),
        ("inference ms / 1000 rows", b["inference_ms_per_1000_rows"], inf_ms),
    ]
    print("\n%-30s %10s %10s" % ("metric", "untuned", "tuned"))
    for name, u, t in cmp_rows:
        print("%-30s %10.4f %10.4f" % (name, u, t))
    print("search %.0f s, final fit %.1f s (device %s)" % (search_s, fit_s, args.device))

    hist = final.evals_result()
    out = dict(best_params=best, objective="minimise mean 5-fold group-CV log loss", n_trials=len(trials),
               search_seconds=search_s, device=args.device, cv_summary=cv_summary, student_test=te,
               final_fit_seconds=fit_s, inference_ms_per_1000_rows=inf_ms,
               comparison={n: dict(untuned=u, tuned=t) for n, u, t in cmp_rows},
               loss_history=dict(train=hist["validation_0"]["logloss"], student_test=hist["validation_1"]["logloss"]),
               features=FEATS, seed=SEED)
    with open(os.path.join(RES, "xgb_tuned.json"), "w") as f:
        json.dump(out, f, indent=2)
    final.save_model(os.path.join(MODELS, "xgb_tuned.json"))
    train[["trip", "pt", "arc", "y"]].assign(p=oof).to_csv(os.path.join(RES, "xgb_oof_tuned.csv.gz"), index=False)
    test[["trip", "pt", "arc", "y"]].assign(p=p_te).to_csv(os.path.join(RES, "xgb_test_tuned.csv.gz"), index=False)
    print("saved")


if __name__ == "__main__":
    main()

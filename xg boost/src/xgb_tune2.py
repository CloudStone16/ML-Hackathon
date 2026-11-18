"""Round-2 XGBoost tuning with overfitting controls, then repeated-CV verification.

Controls against overfitting
  * n_estimators is chosen by early stopping on a held-out 10% of the TRAINING trips inside every fold
    (the validation fold itself never influences training or stopping).
  * stronger regularisation ranges (min_child_weight, gamma, lambda, alpha), smaller learning rates.
  * train-vs-validation log-loss gap recorded per trial.
  * final configuration picked with the one-standard-error rule: among trials within 1 SE of the best CV log loss,
    take the one with the smallest train-validation gap.
Verification
  * repeated 5-fold GroupKFold (3 different trip partitions = 15 folds): baseline vs round-1 tuned vs round-2 final,
    with paired per-fold differences.
  * student_test evaluated once, after selection.

Run:  python src/xgb_tune2.py [--trials 30] [--device cuda]
Writes results/xgb_trials_round2.csv, results/xgb_optuna_round2.db, results/xgb_final.json,
       results/xgb_oof_final.csv.gz, results/xgb_test_final.csv.gz, models/xgb_final.json
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


def shuffled_folds(groups, seed, k=5):
    """Group-wise k-fold with a different random trip partition per seed."""
    trips = np.array(sorted(set(groups)))
    np.random.default_rng(seed).shuffle(trips)
    fold_of = {t: i % k for i, t in enumerate(trips)}
    f = np.array([fold_of[t] for t in groups])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    train, test, n_pts = load_data()
    X, y, g = train[FEATS].values.astype(np.float32), train["y"].values, train["trip"].values
    Xte, yte = test[FEATS].values.astype(np.float32), test["y"].values
    base = dict(tree_method="hist", device=args.device, n_jobs=-1, random_state=SEED, eval_metric="logloss")
    pts_of = lambda idx: sum(n_pts[t] for t in set(g[idx]))
    main_folds = list(GroupKFold(n_splits=5).split(X, y, g))        # same folds as baseline / round 1

    with open(os.path.join(RES, "xgb_baseline.json")) as f:
        b_base = json.load(f)
    with open(os.path.join(RES, "xgb_tuned.json")) as f:
        b_tuned = json.load(f)

    def fit_es(params, tr, seed):
        """Fit with early stopping on 10% of the training trips; returns model, best_iter, train/val loss at best."""
        rng = np.random.default_rng(seed)
        trips = np.unique(g[tr])
        es_trips = rng.choice(trips, max(2, int(0.1 * len(trips))), replace=False)
        es = np.isin(g[tr], es_trips)
        fit_idx, es_idx = tr[~es], tr[es]
        mon = rng.choice(fit_idx, min(200_000, len(fit_idx)), replace=False)
        m = XGBClassifier(**base, **params, n_estimators=1000, early_stopping_rounds=30)
        m.fit(X[fit_idx], y[fit_idx], eval_set=[(X[mon], y[mon]), (X[es_idx], y[es_idx])], verbose=False)
        h, bi = m.evals_result(), m.best_iteration
        return m, bi, h["validation_0"]["logloss"][bi], h["validation_1"]["logloss"][bi]

    def suggest(trial):
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 9),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 0.9),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )

    def objective(trial):
        params = suggest(trial)
        t0 = time.time()
        losses, accs, gaps, iters = [], [], [], []
        for k, (tr, va) in enumerate(main_folds):
            m, bi, l_tr, l_es = fit_es(params, tr, seed=SEED + k)
            p = m.predict_proba(X[va], iteration_range=(0, bi + 1))[:, 1]
            r = evaluate(train.iloc[va], p, pts_of(va))
            losses.append(r["log_loss"]); accs.append(r["top1_acc"]); gaps.append(l_es - l_tr); iters.append(bi + 1)
        trial.set_user_attr("loss_std", float(np.std(losses)))
        trial.set_user_attr("top1", float(np.mean(accs)))
        trial.set_user_attr("gap", float(np.mean(gaps)))
        trial.set_user_attr("mean_iters", float(np.mean(iters)))
        print("trial %2d: logloss %.5f top1 %.4f gap %.4f iters %4.0f (%.0fs)"
              % (trial.number, np.mean(losses), np.mean(accs), np.mean(gaps), np.mean(iters), time.time() - t0),
              flush=True)
        return float(np.mean(losses))

    db = os.path.join(RES, "xgb_optuna_round2.db")
    if os.path.exists(db):
        os.remove(db)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                storage="sqlite:///" + db.replace("\\", "/"), study_name="xgb_round2")
    r1 = b_tuned["best_params"]       # warm start: the round-1 optimum is trial 0, so round 2 cannot do worse by luck
    study.enqueue_trial(dict(learning_rate=min(max(r1["learning_rate"], 0.02), 0.1), max_depth=min(r1["max_depth"], 9),
                             min_child_weight=min(max(r1["min_child_weight"], 1.0), 50.0),
                             subsample=min(r1["subsample"], 0.9), colsample_bytree=r1["colsample_bytree"],
                             gamma=r1["gamma"], reg_lambda=max(r1["reg_lambda"], 0.1), reg_alpha=r1["reg_alpha"]))
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials)
    search_s = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(RES, "xgb_trials_round2.csv"), index=False)

    # one-standard-error selection: statistically tied with the best, smallest train-validation gap
    best = study.best_trial
    se = best.user_attrs["loss_std"] / np.sqrt(len(main_folds))
    tied = [t for t in study.trials if t.value is not None and t.value <= best.value + se]
    chosen = min(tied, key=lambda t: t.user_attrs["gap"])
    params = dict(chosen.params)
    n_final = int(round(1.1 * chosen.user_attrs["mean_iters"]))      # a little more data in the refit
    print("\nsearch %.0f s. best trial %d logloss %.5f (SE %.5f); %d trials within 1 SE"
          % (search_s, best.number, best.value, se, len(tied)))
    print("chosen trial %d: logloss %.5f gap %.4f iters %d\nparams %s" % (chosen.number, chosen.value,
          chosen.user_attrs["gap"], n_final, params), flush=True)
    final_params = dict(params, n_estimators=n_final)

    # ---- repeated-CV verification (3 different trip partitions x 5 folds) ----
    r1p = dict(b_tuned["best_params"])
    configs = {"baseline": dict(b_base["params"]), "round1_tuned": r1p, "round2_final": final_params}
    for c in configs.values():
        for k in ("tree_method", "n_jobs", "random_state", "eval_metric", "device"):
            c.pop(k, None)
    rows = []
    t0 = time.time()
    for seed in (1, 2, 3):
        for fi, (tr, va) in enumerate(shuffled_folds(g, seed)):
            for name, cfg in configs.items():
                m = XGBClassifier(**base, **cfg)
                m.fit(X[tr], y[tr])
                r = evaluate(train.iloc[va], m.predict_proba(X[va])[:, 1], pts_of(va))
                rows.append(dict(seed=seed, fold=fi, model=name, top1=r["top1_acc"], f1=r["f1_macro"],
                                 log_loss=r["log_loss"]))
    V = pd.DataFrame(rows)
    V.to_csv(os.path.join(RES, "xgb_repeated_cv.csv"), index=False)
    summ = V.groupby("model")[["top1", "f1", "log_loss"]].agg(["mean", "std"]).round(5)
    print("\nrepeated 5-fold group CV (3 partitions, 15 folds, %.0f s)\n%s" % (time.time() - t0, summ.to_string()))
    piv = V.pivot_table(index=["seed", "fold"], columns="model", values=["log_loss", "top1"])
    paired = {}
    for a, bname in (("round2_final", "baseline"), ("round2_final", "round1_tuned"), ("round1_tuned", "baseline")):
        dl = (piv["log_loss"][a] - piv["log_loss"][bname])
        dt = (piv["top1"][a] - piv["top1"][bname])
        paired["%s vs %s" % (a, bname)] = dict(
            log_loss_diff_mean=float(dl.mean()), log_loss_better_folds="%d/15" % int((dl < 0).sum()),
            top1_diff_mean=float(dt.mean()), top1_better_folds="%d/15" % int((dt > 0).sum()))
        print("%-30s logloss diff %+.5f (better in %d/15) | top1 diff %+.4f (better in %d/15)"
              % ("%s vs %s" % (a, bname), dl.mean(), (dl < 0).sum(), dt.mean(), (dt > 0).sum()))

    # ---- main-fold OOF predictions, final refit, student_test (once) ----
    oof, fold_rows = np.zeros(len(train)), []
    for tr, va in main_folds:
        m = XGBClassifier(**base, **final_params)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        fold_rows.append(evaluate(train.iloc[va], oof[va], pts_of(va)))
    agg = lambda k: dict(mean=float(np.mean([r[k] for r in fold_rows])), std=float(np.std([r[k] for r in fold_rows])))
    cv_summary = {k: agg(k) for k in ("top1_acc", "f1_macro", "precision_macro", "recall_macro", "log_loss")}
    final = XGBClassifier(**base, **final_params)
    t0 = time.time()
    final.fit(X, y, eval_set=[(X, y), (Xte, yte)], verbose=False)
    fit_s = time.time() - t0
    t0 = time.time()
    p_te = final.predict_proba(Xte)[:, 1]
    inf_ms = 1000 * (time.time() - t0) / len(Xte) * 1000
    te = evaluate(test, p_te, sum(n_pts[t] for t in test.trip.unique()))

    table = [("CV top-1 accuracy", b_base["cv_summary"]["top1_acc"]["mean"], b_tuned["cv_summary"]["top1_acc"]["mean"], cv_summary["top1_acc"]["mean"]),
             ("CV macro F1", b_base["cv_summary"]["f1_macro"]["mean"], b_tuned["cv_summary"]["f1_macro"]["mean"], cv_summary["f1_macro"]["mean"]),
             ("CV log loss", b_base["cv_summary"]["log_loss"]["mean"], b_tuned["cv_summary"]["log_loss"]["mean"], cv_summary["log_loss"]["mean"]),
             ("student-test top-1", b_base["student_test"]["top1_acc"], b_tuned["student_test"]["top1_acc"], te["top1_acc"]),
             ("student-test macro F1", b_base["student_test"]["f1_macro"], b_tuned["student_test"]["f1_macro"], te["f1_macro"]),
             ("student-test log loss", b_base["student_test"]["log_loss"], b_tuned["student_test"]["log_loss"], te["log_loss"])]
    print("\n%-24s %10s %10s %10s" % ("metric", "baseline", "round 1", "round 2"))
    for n, a, bb, c in table:
        print("%-24s %10.4f %10.4f %10.4f" % (n, a, bb, c))
    hist = final.evals_result()
    print("final fit %.1fs, inference %.2f ms/1000 rows; train-vs-test loss at end: %.4f vs %.4f"
          % (fit_s, inf_ms, hist["validation_0"]["logloss"][-1], hist["validation_1"]["logloss"][-1]))

    out = dict(chosen_params=final_params, chosen_trial=chosen.number, best_trial=best.number, n_trials=len(trials),
               selection_rule="one standard error: tied with best CV log loss, smallest train-validation gap",
               search_seconds=search_s, cv_summary=cv_summary, student_test=te, final_fit_seconds=fit_s,
               inference_ms_per_1000_rows=inf_ms, repeated_cv=json.loads(summ.to_json()), paired_differences=paired,
               comparison=[dict(metric=n, baseline=a, round1=bb, round2=c) for n, a, bb, c in table],
               loss_history=dict(train=hist["validation_0"]["logloss"], student_test=hist["validation_1"]["logloss"]),
               features=FEATS, seed=SEED)
    with open(os.path.join(RES, "xgb_final.json"), "w") as f:
        json.dump(out, f, indent=2)
    final.save_model(os.path.join(MODELS, "xgb_final.json"))
    train[["trip", "pt", "arc", "y"]].assign(p=oof).to_csv(os.path.join(RES, "xgb_oof_final.csv.gz"), index=False)
    test[["trip", "pt", "arc", "y"]].assign(p=p_te).to_csv(os.path.join(RES, "xgb_test_final.csv.gz"), index=False)
    print("saved")


if __name__ == "__main__":
    main()

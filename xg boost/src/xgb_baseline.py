"""XGBoost baseline: 5-fold group CV on train (by trip), then fit on all train and evaluate on student_test.

Run:  python src/xgb_baseline.py
Writes results/xgb_baseline.json, results/xgb_oof_baseline.csv.gz, results/xgb_test_baseline.csv.gz,
       models/xgb_baseline.json
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from xgb_common import FEATS, MODELS, RES, SEED, evaluate, load_data

PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss")


def main():
    train, test, n_pts = load_data()
    X, y, g = train[FEATS].values.astype(np.float32), train["y"].values, train["trip"].values
    print("train rows %d (%d trips), test rows %d (%d trips), positive rate %.3f"
          % (len(train), train.trip.nunique(), len(test), test.trip.nunique(), y.mean()), flush=True)

    folds, oof = [], np.zeros(len(train))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        m = XGBClassifier(**PARAMS)
        t0 = time.time()
        m.fit(X[tr], y[tr])
        fit_s = time.time() - t0
        oof[va] = m.predict_proba(X[va])[:, 1]
        trips_va, trips_tr = set(train.trip.values[va]), set(train.trip.values[tr])
        val = evaluate(train.iloc[va], oof[va], sum(n_pts[t] for t in trips_va))
        trn = evaluate(train.iloc[tr], m.predict_proba(X[tr])[:, 1], sum(n_pts[t] for t in trips_tr))
        val.update(fit_seconds=fit_s, train_top1_acc=trn["top1_acc"], train_log_loss=trn["log_loss"], fold=k)
        folds.append(val)
        print("fold %d: val top1 %.4f f1 %.4f logloss %.4f | train top1 %.4f logloss %.4f | %.0fs"
              % (k, val["top1_acc"], val["f1_macro"], val["log_loss"], trn["top1_acc"], trn["log_loss"], fit_s),
              flush=True)

    def agg(key):
        v = [f[key] for f in folds]
        return dict(mean=float(np.mean(v)), std=float(np.std(v)))
    cv = {k: agg(k) for k in ("top1_acc", "f1_macro", "precision_macro", "recall_macro", "log_loss",
                              "train_top1_acc", "train_log_loss", "fit_seconds")}
    print("\nCV (5 folds): top1 %.4f +- %.4f | macro F1 %.4f | log loss %.4f | train top1 %.4f"
          % (cv["top1_acc"]["mean"], cv["top1_acc"]["std"], cv["f1_macro"]["mean"], cv["log_loss"]["mean"],
             cv["train_top1_acc"]["mean"]), flush=True)

    # Final fit on all train, with loss history on train and student_test (for the loss curve; no early stopping)
    Xte, yte = test[FEATS].values.astype(np.float32), test["y"].values
    final = XGBClassifier(**PARAMS)
    t0 = time.time()
    final.fit(X, y, eval_set=[(X, y), (Xte, yte)], verbose=False)
    fit_s = time.time() - t0
    t0 = time.time()
    p_te = final.predict_proba(Xte)[:, 1]
    inf_ms_per_1k = 1000 * (time.time() - t0) / len(Xte) * 1000
    te = evaluate(test, p_te, sum(n_pts[t] for t in test.trip.unique()))
    print("student_test: top1 %.4f | macro F1 %.4f | precision %.4f | recall %.4f | log loss %.4f | miss rate %.4f"
          % (te["top1_acc"], te["f1_macro"], te["precision_macro"], te["recall_macro"], te["log_loss"],
             te["candidate_miss_rate"]), flush=True)
    print("fit %.0fs, inference %.1f ms per 1000 candidate rows" % (fit_s, inf_ms_per_1k))

    hist = final.evals_result()
    out = dict(params={k: v for k, v in PARAMS.items()}, features=FEATS, seed=SEED,
               cv_folds=folds, cv_summary=cv, student_test=te, final_fit_seconds=fit_s,
               inference_ms_per_1000_rows=inf_ms_per_1k,
               loss_history=dict(train=hist["validation_0"]["logloss"], student_test=hist["validation_1"]["logloss"]))
    with open(os.path.join(RES, "xgb_baseline.json"), "w") as f:
        json.dump(out, f, indent=2)
    final.save_model(os.path.join(MODELS, "xgb_baseline.json"))
    train[["trip", "pt", "arc", "y"]].assign(p=oof).to_csv(os.path.join(RES, "xgb_oof_baseline.csv.gz"), index=False)
    test[["trip", "pt", "arc", "y"]].assign(p=p_te).to_csv(os.path.join(RES, "xgb_test_baseline.csv.gz"), index=False)
    print("saved results and model")


if __name__ == "__main__":
    main()

"""Train LightGBM candidate classifier (+ LogReg and nearest-arc baselines).

CV: GroupKFold by trip on the 80 training trips. Final: fit on all train, evaluate on student_test.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import FEATURES, MAX_CANDS, RADIUS_M  # noqa: E402

SEED = 42
BAD_TRIP_INVALID_FRAC = 0.5   # exclude trips whose reference route does not cover the GPS track
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, n_jobs=-1, verbose=-1)


def load(split):
    df = pd.read_parquet(ROOT / "artifacts" / f"features_{split}.parquet")
    rep = pd.read_csv(ROOT / "results" / f"preprocess_report_{split}.csv", dtype={"trip": str})
    rep["trip"] = rep["trip"].str.zfill(8)
    rep["n_valid_pts"] = np.round(rep["n_pts"] * (1 - rep["frac_label_invalid"])).astype(int)
    return df, rep


def evaluate(df, p, n_valid_pts, thr=0.5):
    """df: candidate rows (label_valid==1 only); p: predicted prob per row."""
    df = df.assign(p=p)
    y = df["label"].to_numpy()
    yhat = (p >= thr).astype(int)
    out = {
        "n_rows": len(df), "n_valid_pts": int(n_valid_pts),
        "cand_accuracy": float((y == yhat).mean()),
        "cand_precision": float(precision_score(y, yhat, zero_division=0)),
        "cand_recall": float(recall_score(y, yhat, zero_division=0)),
        "cand_f1_pos": float(f1_score(y, yhat, zero_division=0)),
        "cand_macro_f1": float(f1_score(y, yhat, average="macro")),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }
    top = df.loc[df.groupby(["trip", "pt"])["p"].idxmax()]
    hits = int(top["label"].sum())
    pts_with_label = int(df.groupby(["trip", "pt"])["label"].max().sum())
    out["top1_acc"] = hits / n_valid_pts                       # misses (label not in candidates) count as errors
    out["top1_acc_given_candidate"] = hits / max(pts_with_label, 1)
    out["candidate_miss_rate"] = 1 - pts_with_label / n_valid_pts
    out["route_continuity"] = continuity(top.rename(columns={}), "arc")
    truth = df[df["label"] == 1].drop_duplicates(["trip", "pt"])
    out["route_continuity_reference"] = continuity(truth, "arc")
    return out


def continuity(picks, _):
    """Share of consecutive-point pairs whose chosen arcs are identical or head-to-tail connected."""
    p = picks.sort_values(["trip", "pt"])
    nxt = p.groupby("trip").shift(-1)
    pair = nxt["pt"] == p["pt"] + 1
    ok = (p["arc"] == nxt["arc"]) | (p["to_node"] == nxt["from_node"])
    return float(ok[pair].mean())


def make_models():
    return {
        "lightgbm": lambda: lgb.LGBMClassifier(**LGB_PARAMS),
        "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(max_iter=300, C=1.0, random_state=SEED)),
    }


def nearest_baseline_scores(df):
    """Heuristic reference: closest arc wins (score = -d mapped to (0,1))."""
    return 1.0 / (1.0 + df["d"].to_numpy())


def main():
    tr, tr_rep = load("train")
    te, te_rep = load("student_test")
    bad = set(tr_rep.loc[tr_rep["frac_label_invalid"] > BAD_TRIP_INVALID_FRAC, "trip"])
    print("excluded trips (reference route does not match track):", sorted(bad))
    tr_rep = tr_rep[~tr_rep["trip"].isin(bad)]
    tr = tr[~tr["trip"].isin(bad) & (tr["label_valid"] == 1)].reset_index(drop=True)
    te = te[te["label_valid"] == 1].reset_index(drop=True)
    valid_pts = dict(zip(tr_rep["trip"], tr_rep["n_valid_pts"]))
    valid_pts_te = dict(zip(te_rep["trip"], te_rep["n_valid_pts"]))
    print(f"train rows={len(tr)} trips={tr['trip'].nunique()} | student_test rows={len(te)} trips={te['trip'].nunique()}")

    X, y, groups = tr[FEATURES], tr["label"].to_numpy(), tr["trip"].to_numpy()
    results, fold_rows = {}, []
    gkf = GroupKFold(n_splits=5)
    for name, mk in make_models().items():
        oof = np.zeros(len(tr))
        t_fit = 0.0
        for k, (a, b) in enumerate(gkf.split(X, y, groups)):
            assert not (set(groups[a]) & set(groups[b])), "trip leakage across folds"
            m = mk()
            t0 = time.time(); m.fit(X.iloc[a], y[a]); t_fit += time.time() - t0
            oof[b] = m.predict_proba(X.iloc[b])[:, 1]
            sub = tr.iloc[b]
            r = evaluate(sub, oof[b], sum(valid_pts[t] for t in sub["trip"].unique()))
            fold_rows.append({"model": name, "fold": k, **{c: r[c] for c in ("top1_acc", "cand_macro_f1", "log_loss", "route_continuity")}})
            print(f"  [{name}] fold {k}: top1={r['top1_acc']:.4f} macroF1={r['cand_macro_f1']:.4f} logloss={r['log_loss']:.4f}", flush=True)
        cv = evaluate(tr, oof, sum(valid_pts.values()))
        cv["fit_time_s_total_5folds"] = t_fit
        results[name] = {"cv": cv}
    # nearest-arc heuristic (no training)
    results["nearest_arc_heuristic"] = {"cv": evaluate(tr, nearest_baseline_scores(tr), sum(valid_pts.values()))}

    # final fit on all training trips, evaluate on student_test
    final = {}
    for name, mk in make_models().items():
        m = mk()
        t0 = time.time(); m.fit(X, y); fit_s = time.time() - t0
        t0 = time.time(); p = m.predict_proba(te[FEATURES])[:, 1]; inf_s = time.time() - t0
        r = evaluate(te, p, sum(valid_pts_te.values()))
        r.update(fit_time_s=fit_s, inference_time_s=inf_s, inference_us_per_row=inf_s / len(te) * 1e6)
        results[name]["student_test"] = r
        final[name] = m
        print(f"[{name}] student_test top1={r['top1_acc']:.4f} macroF1={r['cand_macro_f1']:.4f} "
              f"logloss={r['log_loss']:.4f} continuity={r['route_continuity']:.4f} fit={fit_s:.0f}s", flush=True)
    results["nearest_arc_heuristic"]["student_test"] = evaluate(te, nearest_baseline_scores(te), sum(valid_pts_te.values()))

    # save
    res_dir = ROOT / "results"
    (res_dir / "lightgbm_metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(fold_rows).to_csv(res_dir / "cv_folds.csv", index=False)
    imp = pd.DataFrame({"feature": FEATURES, "gain": final["lightgbm"].booster_.feature_importance("gain")})
    imp.sort_values("gain", ascending=False).to_csv(res_dir / "lightgbm_feature_importance.csv", index=False)
    joblib.dump(final["lightgbm"], ROOT / "results" / "lightgbm_model.joblib")
    (ROOT / "results" / "config.json").write_text(json.dumps({
        "features": FEATURES, "radius_m": RADIUS_M, "max_cands": MAX_CANDS, "seed": SEED,
        "lightgbm_params": LGB_PARAMS, "excluded_train_trips": sorted(bad),
        "primary_metric": "observation-level top-1 accuracy"}, indent=2))
    summ = pd.DataFrame({n: {"cv_top1": r["cv"]["top1_acc"], "cv_macro_f1": r["cv"]["cand_macro_f1"],
                             "cv_logloss": r["cv"]["log_loss"], "test_top1": r["student_test"]["top1_acc"],
                             "test_macro_f1": r["student_test"]["cand_macro_f1"], "test_logloss": r["student_test"]["log_loss"],
                             "test_continuity": r["student_test"]["route_continuity"]} for n, r in results.items()}).T
    summ.to_csv(res_dir / "model_comparison.csv")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()

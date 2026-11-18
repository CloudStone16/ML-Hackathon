"""Tune the HMM decoder on LightGBM out-of-fold probabilities, then evaluate once on student_test.

Rules followed: decoder parameters are chosen using TRAIN trips only (out-of-fold probabilities, group CV by trip).
student_test is evaluated once, after the parameters are fixed. Evaluator trips are never used.

Run from the repo root:
    python backend/tune_decoder.py --data-root "<folder containing train/ and student_test/>" [--trials 25]
Writes backend/decoder_config.json, backend/decoder_trials.csv, backend/decoder_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend import decoder  # noqa: E402
from pathsnap.features import build_trip  # noqa: E402
from pathsnap.io import clean_track  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
TOP_K = 10
MAX_UNIQUE_ARCS = 12000          # skip trips whose reachability table would be too large to tune on
BAD_INVALID_FRAC = 0.5           # same exclusion rule as scripts/train_lightgbm.py


def load_trip(root, split, tid):
    d = Path(root) / split / tid
    return {"id": tid, "track": np.loadtxt(d / f"{tid}.track", ndmin=2), "nodes": np.loadtxt(d / f"{tid}.nodes", ndmin=2),
            "arcs": np.loadtxt(d / f"{tid}.arcs", dtype=np.int64, ndmin=2),
            "route": np.loadtxt(d / f"{tid}.route", dtype=np.int64, ndmin=1)}


def build_split(root, split):
    out = []
    for tid in sorted(p.name for p in (Path(root) / split).iterdir() if p.is_dir()):
        trip = load_trip(root, split, tid)
        df, rep = build_trip(trip, with_labels=True)
        track, _ = clean_track(trip["track"])
        out.append(dict(tid=tid, df=df, rep=rep, track=track, nodes=trip["nodes"], arcs=trip["arcs"]))
    return out


def prep(trips, probs):
    """Decoder inputs + validity-aware truth for each trip. probs: list of arrays aligned with each trip's rows."""
    res = []
    for t, p in zip(trips, probs):
        df, n = t["df"], len(t["track"])
        u = df["arc"].nunique()
        if u > MAX_UNIQUE_ARCS:
            print(f"  skipping trip {t['tid']} for decoding ({u} unique candidate arcs)")
            continue
        truth = np.full(n, -1)
        m = ((df["label"] == 1) & (df["label_valid"] == 1)).to_numpy()
        truth[df["pt"].to_numpy()[m]] = df["arc"].to_numpy()[m]
        n_valid = int(round(t["rep"]["n_pts"] * (1 - t["rep"]["frac_label_invalid"])))
        d = decoder.prepare(n, t["track"][:, 2], t["arcs"], len(t["nodes"]), df["pt"].to_numpy(), df["arc"].to_numpy(),
                            p, top_k=TOP_K)
        res.append(dict(tid=t["tid"], d=d, truth=truth, n_valid=max(n_valid, 1)))
    return res


def accuracy(items, fn):
    hit = 0
    for i in items:
        pred = fn(i["d"])
        hit += int(((pred == i["truth"]) & (i["truth"] >= 0)).sum())
    return hit / sum(i["n_valid"] for i in items)


def continuity(items, fn):
    ok = tot = 0
    for i in items:
        pred = fn(i["d"])
        u = np.flatnonzero(pred >= 0)
        pa = pred[u]
        ch = np.flatnonzero(pa[1:] != pa[:-1])
        if len(ch) and i["d"]["hop"] is not None:
            ia, ib = np.searchsorted(i["d"]["u_arcs"], pa[ch]), np.searchsorted(i["d"]["u_arcs"], pa[ch + 1])
            ok += int((i["d"]["hop"][ia, ib] < 255).sum())
        tot += len(ch)
    return ok / tot if tot else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--every", type=int, default=3, help="tune on every n-th train trip (speed)")
    args = ap.parse_args()

    with open(REPO_ROOT / "results" / "config.json") as f:
        cfg = json.load(f)
    feats, lgb_params = cfg["features"], cfg["lightgbm_params"]

    t0 = time.time()
    train_all, test = build_split(args.data_root, "train"), build_split(args.data_root, "student_test")
    bad = {t["tid"] for t in train_all if t["rep"]["frac_label_invalid"] > BAD_INVALID_FRAC}
    train = [t for t in train_all if t["tid"] not in bad]
    print(f"features built in {time.time() - t0:.0f}s; train trips {len(train)} (excluded {sorted(bad)}), test trips {len(test)}",
          flush=True)

    # out-of-fold LightGBM probabilities for every candidate row of every train trip
    import pandas as pd
    big = pd.concat([t["df"].assign(_i=i) for i, t in enumerate(train)], ignore_index=True)
    fit_mask = (big["label_valid"] == 1).to_numpy()
    groups, y = big["trip"].to_numpy(), big["label"].to_numpy()
    oof = np.zeros(len(big))
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(big, y, groups)):
        a = a[fit_mask[a]]
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(big.iloc[a][feats], y[a])
        oof[b] = m.predict_proba(big.iloc[b][feats])[:, 1]
        print(f"  OOF fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    oof_by_trip = [oof[(big["_i"] == i).to_numpy()] for i in range(len(train))]

    items_all = prep(train, oof_by_trip)
    items_tune = items_all[:: args.every]
    print(f"decoder inputs ready: {len(items_all)} train trips, tuning on {len(items_tune)}", flush=True)
    clf_acc = accuracy(items_tune, decoder.classifier_only)
    base_acc = accuracy(items_tune, lambda d: decoder.viterbi(d, **decoder.DEFAULT_PARAMS))
    print(f"tuning subset: classifier-only {clf_acc:.4f}, HMM default params {base_acc:.4f}", flush=True)

    def objective(trial):
        p = dict(alpha=trial.suggest_float("alpha", 0.2, 3.0), lam=trial.suggest_float("lam", 0.0, 6.0),
                 disc=trial.suggest_float("disc", 1.0, 60.0), uturn=trial.suggest_float("uturn", 0.0, 30.0),
                 gap_s=trial.suggest_float("gap_s", 3.0, 30.0))
        acc = accuracy(items_tune, lambda d: decoder.viterbi(d, **p))
        print(f"trial {trial.number:2d} top1 {acc:.4f} {({k: round(v, 2) for k, v in p.items()})}", flush=True)
        return acc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial(dict(alpha=1.0, lam=1.0, disc=10.0, uturn=6.0, gap_s=10.0))      # default params as trial 0
    t1 = time.time()
    study.optimize(objective, n_trials=args.trials)
    best = study.best_params
    study.trials_dataframe().to_csv(REPO_ROOT / "backend" / "decoder_trials.csv", index=False)
    print(f"\nbest tuning-subset top1 {study.best_value:.4f} in {time.time() - t1:.0f}s: {best}", flush=True)

    fn_best = lambda d: decoder.viterbi(d, **best)
    results = {
        "tuned_on": "train trips, LightGBM out-of-fold probabilities (every %d-th trip)" % args.every,
        "best_params": best, "top_k": TOP_K, "n_trials": args.trials,
        "train_oof_all": {"classifier_only": accuracy(items_all, decoder.classifier_only),
                          "hmm_default": accuracy(items_all, lambda d: decoder.viterbi(d, **decoder.DEFAULT_PARAMS)),
                          "hmm_tuned": accuracy(items_all, fn_best),
                          "continuity_classifier": continuity(items_all, decoder.classifier_only),
                          "continuity_hmm_tuned": continuity(items_all, fn_best)},
    }
    print("train OOF (all trips):", {k: round(v, 4) for k, v in results["train_oof_all"].items()}, flush=True)

    # student_test, once, with the model that the API serves
    model = joblib.load(REPO_ROOT / "results" / "lightgbm_model.joblib")
    test_items = prep(test, [model.predict_proba(t["df"][feats])[:, 1] for t in test])
    results["student_test"] = {"classifier_only": accuracy(test_items, decoder.classifier_only),
                               "hmm_default": accuracy(test_items, lambda d: decoder.viterbi(d, **decoder.DEFAULT_PARAMS)),
                               "hmm_tuned": accuracy(test_items, fn_best),
                               "continuity_classifier": continuity(test_items, decoder.classifier_only),
                               "continuity_hmm_tuned": continuity(test_items, fn_best),
                               "n_trips": len(test_items)}
    print("student_test:", {k: round(v, 4) for k, v in results["student_test"].items()}, flush=True)

    with open(REPO_ROOT / "backend" / "decoder_config.json", "w") as f:
        json.dump({"params": best, "top_k": TOP_K, "model": "lightgbm", "tuned_on": results["tuned_on"]}, f, indent=2)
    with open(REPO_ROOT / "backend" / "decoder_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved decoder_config.json, decoder_trials.csv, decoder_results.json")


if __name__ == "__main__":
    main()

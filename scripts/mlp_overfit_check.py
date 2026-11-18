"""Check the teammate Deep MLP for overfitting / leakage.

Uses their tuned config unchanged. Measures observation-level Top-1 (argmax candidate == aligned arc) on:
  fit trips (72), early-stopping trips (8), student_test (10), evaluator_test (10)   x 3 random seeds
plus a clean 5-fold GroupKFold CV where early stopping + threshold use a carve-out of the TRAIN side only
(their own CV used the held-out fold for both, which is slightly optimistic).

Usage: PATHSNAP_DATA=<dir with train/ student_test/ evaluator_test/> uv run python scripts/mlp_overfit_check.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = os.environ["PATHSNAP_DATA"]
sys.path.insert(0, str(ROOT))
from feature_engineering.candidate_features import build_or_load_partition_dataset  # noqa: E402
from models.mlp_model import PathSnapMLPClassifier  # noqa: E402

torch.set_num_threads(8)
CACHE = str(ROOT / "artifacts" / "mlp_cache")
OUT = ROOT / "results" / "mlp_overfit_check"
BEST = json.loads((ROOT / "mlp" / "results" / "mlp_complete_report_metrics.json").read_text())["best_hyperparameters"]
PER_OBS = ("X", "y", "cand_arcs", "gt_arcs", "miss_mask", "dt")


def clean(bundles):
    out = []
    for b in bundles:   # same filter as train_mlp.filter_clean_train_bundles
        valid = (~b["miss_mask"]) | (b["X"][:, 0, 0] < 25.0)
        if np.sum(valid) < 20:
            valid = np.ones(len(b["X"]), dtype=bool)
        out.append({k: (b[k][valid] if k in PER_OBS else b[k]) for k in b})
    return out


def make(seed):
    return PathSnapMLPClassifier(hidden_dim=BEST["hidden_dim"], head_dim=BEST["head_dim"], dropout=BEST["dropout"],
                                 lr=BEST["lr"], epochs=26, batch_obs=4096, max_obs_per_trip=1200,
                                 listwise_weight=BEST["listwise_weight"], pos_weight=BEST["pos_weight"],
                                 random_state=seed)


def top1(clf, bundles):
    hit = n = 0
    ll = []
    for b in bundles:
        p = clf.predict_proba_trip(b)
        k = np.argmax(p, axis=1)
        hit += int((b["cand_arcs"][np.arange(len(k)), k] == b["gt_arcs"]).sum())
        n += len(k)
        y = b["y"].ravel()
        pp = np.clip(p.ravel(), 1e-7, 1 - 1e-7)
        ll.append(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)).mean())
    return hit / n, float(np.mean(ll))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = clean(build_or_load_partition_dataset(DATA, "train", K=5, cache_dir=CACHE))
    student = build_or_load_partition_dataset(DATA, "student_test", K=5, cache_dir=CACHE)
    evalr = build_or_load_partition_dataset(DATA, "evaluator_test", K=5, cache_dir=CACHE)
    rows = []
    for seed in (42, 7, 123):
        t0 = time.time()
        clf = make(seed)
        clf.fit(train[:72], train[72:])
        parts = {"fit trips (72)": train[:72], "early-stop trips (8)": train[72:],
                 "student_test (10)": student, "evaluator_test (10)": evalr}
        for name, bs in parts.items():
            t1, ll = top1(clf, bs)
            rows.append({"run": f"seed {seed}", "partition": name, "top1": t1, "log_loss": ll})
        h = clf.history
        rows.append({"run": f"seed {seed}", "partition": "final train loss / best val loss",
                     "top1": float(h["train_loss"][-1]), "log_loss": float(min(h["val_loss"]))})
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)

    groups = np.arange(len(train))
    for f, (tr_i, ho_i) in enumerate(GroupKFold(n_splits=5).split(groups, groups, groups)):
        tr = [train[i] for i in tr_i]
        ho = [train[i] for i in ho_i]
        clf = make(42)
        clf.fit(tr[:-8], tr[-8:])               # early stopping + threshold from the TRAIN side only
        t1, ll = top1(clf, ho)
        rows.append({"run": "clean 5-fold CV", "partition": f"fold {f + 1} held-out", "top1": t1, "log_loss": ll})
        print(f"cv fold {f + 1}: top1={t1:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mlp_overfit_check.csv", index=False)
    seeds = df[df["run"].str.startswith("seed") & ~df["partition"].str.contains("loss")]
    print(seeds.groupby("partition")["top1"].agg(["mean", "std", "min", "max"]).round(4).to_string())
    cv = df[df["run"] == "clean 5-fold CV"]["top1"]
    print("clean CV top1 mean %.4f std %.4f" % (cv.mean(), cv.std()))


if __name__ == "__main__":
    main()

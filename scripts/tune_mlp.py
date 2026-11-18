"""Bayesian (Optuna TPE) tuning for the teammate Deep MLP (PathSnapMLPClassifier), architecture/features unchanged.

Objective: mean observation-level Top-1 over 5-fold GroupKFold by trip on the training trips. Inside every fold the
early-stopping set and probability threshold come from the TRAIN side only (the last 8 training trips), never from the
held-out fold. student_test / evaluator_test are not used for selection. Trial 0 is their current tuned config.

Usage: PATHSNAP_DATA=<dir with train/ student_test/> uv run python scripts/tune_mlp.py [n_trials]
"""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
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
OUT = ROOT / "results" / "tuning"
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
PER_OBS = ("X", "y", "cand_arcs", "gt_arcs", "miss_mask", "dt")
THEIRS = json.loads((ROOT / "mlp" / "results" / "mlp_complete_report_metrics.json").read_text())["best_hyperparameters"]


def clean(bundles):
    out = []
    for b in bundles:   # same filter as train_mlp.filter_clean_train_bundles
        valid = (~b["miss_mask"]) | (b["X"][:, 0, 0] < 25.0)
        if np.sum(valid) < 20:
            valid = np.ones(len(b["X"]), dtype=bool)
        out.append({k: (b[k][valid] if k in PER_OBS else b[k]) for k in b})
    return out


def make(p, seed=42):
    return PathSnapMLPClassifier(hidden_dim=p["hidden_dim"], head_dim=p["head_dim"], dropout=p["dropout"], lr=p["lr"],
                                 epochs=p.get("epochs", 26), batch_obs=4096, max_obs_per_trip=1200,
                                 listwise_weight=p["listwise_weight"], pos_weight=p["pos_weight"], random_state=seed)


def top1(clf, bundles):
    hit = n = 0
    for b in bundles:
        k = np.argmax(clf.predict_proba_trip(b), axis=1)
        hit += int((b["cand_arcs"][np.arange(len(k)), k] == b["gt_arcs"]).sum())
        n += len(k)
    return hit / n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = clean(build_or_load_partition_dataset(DATA, "train", K=5, cache_dir=CACHE))
    student = build_or_load_partition_dataset(DATA, "student_test", K=5, cache_dir=CACHE)
    idx = np.arange(len(train))
    folds = list(GroupKFold(n_splits=5).split(idx, idx, idx))

    def objective(trial):
        p = dict(hidden_dim=trial.suggest_categorical("hidden_dim", [64, 96, 128, 160, 256]),
                 head_dim=trial.suggest_categorical("head_dim", [32, 48, 64, 96]),
                 dropout=trial.suggest_float("dropout", 0.05, 0.30),
                 lr=trial.suggest_float("lr", 1e-3, 6e-3, log=True),
                 listwise_weight=trial.suggest_float("listwise_weight", 0.2, 0.7),
                 pos_weight=trial.suggest_float("pos_weight", 1.0, 2.2),
                 epochs=trial.suggest_categorical("epochs", [20, 26, 34]))
        scores = []
        for i, (a, b) in enumerate(folds):
            tr = [train[j] for j in a]
            ho = [train[j] for j in b]
            clf = make(p, seed=42 + i)
            clf.fit(tr[:-8], tr[-8:])
            scores.append(top1(clf, ho))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=6),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1))
    study.enqueue_trial({**{k: THEIRS[k] for k in ("hidden_dim", "head_dim", "dropout", "lr", "listwise_weight",
                                                   "pos_weight")}, "epochs": 26})   # trial 0: their tuned config
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number}: {t.state.name} value={t.value} "
                                                 f"best={s.best_value:.4f} ({time.time() - t0:.0f}s)", flush=True)])
    search_s = time.time() - t0
    study.trials_dataframe().to_csv(OUT / "mlp_trials.csv", index=False)

    out = {"objective": "mean 5-fold GroupKFold observation-level Top-1 (train trips only, clean early stopping)",
           "n_trials": N_TRIALS, "search_seconds": search_s, "best_cv_top1_mean": study.best_value,
           "their_tuned_cv_top1_mean": study.trials[0].value, "best_params": study.best_params}
    configs = {"their_tuned": {**THEIRS, "epochs": 26}, "retuned": study.best_params}
    for name, p in configs.items():
        res = []
        for seed in (42, 7, 123):               # final refit as in train_mlp.py: fit on 72 trips, stop on 8
            t1 = time.time()
            clf = make(p, seed)
            clf.fit(train[:72], train[72:])
            fit_s = time.time() - t1
            res.append({"seed": seed, "student_top1": top1(clf, student), "fit_s": fit_s})
            if name == "retuned" and seed == 42:
                joblib.dump(clf, OUT / "mlp_retuned_classifier.joblib")
        out[f"student_test_{name}"] = {"top1_mean": float(np.mean([r["student_top1"] for r in res])),
                                       "top1_std": float(np.std([r["student_top1"] for r in res], ddof=1)), "seeds": res}
        print(name, out[f"student_test_{name}"], flush=True)
    (OUT / "mlp_tuning_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""
PathSnap: Complete End-to-End Deep MLP Training, Evaluation, Bayesian Tuning,
HMM/Viterbi Route Reconstruction & Visualisation Pipeline
=============================================================================
Focuses exclusively on the PathSnap Deep MLP model:
1. 5-Fold GroupKFold Cross-Validation on the 80 Training Trips
2. Bayesian Hyperparameter Optimisation (GP-Expected Improvement) for the MLP
3. Evaluation on the 10 Student-Testing Trips (Untuned vs. Tuned MLP)
4. HMM + Viterbi Route Reconstruction using the MLP's candidate probabilities
5. Generation of all compulsory MLP plots, confusion matrices, epoch loss curves,
   and interactive Folium geographical maps.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing.eda_and_preprocessing import load_trip_raw
from feature_engineering.candidate_features import build_or_load_partition_dataset, FEATURE_NAMES
from models.mlp_model import PathSnapMLPClassifier, compute_route_continuity
from hmm_viterbi.viterbi_decoder import PathSnapViterbiDecoder, compute_sequence_metrics


def filter_clean_train_bundles(train_bundles: list) -> list:
    """
    Filter out off-road stationary parking anomalies (e.g., trips 00000054 and 00000033
    where the vehicle parked >120m away from the road network for thousands of seconds)
    during training/CV while keeping all valid driving observations.
    """
    cleaned = []
    for b in train_bundles:
        valid = (~b["miss_mask"]) | (b["X"][:, 0, 0] < 25.0)
        if np.sum(valid) < 20:
            valid = np.ones(len(b["X"]), dtype=bool)
        cleaned.append({
            "trip_id": b["trip_id"],
            "X": b["X"][valid],
            "y": b["y"][valid],
            "cand_arcs": b["cand_arcs"][valid],
            "gt_arcs": b["gt_arcs"][valid],
            "miss_mask": b["miss_mask"][valid],
            "dt": b["dt"][valid],
            "arcs": b["arcs"],
        })
    return cleaned


def evaluate_mlp_on_bundles(trip_bundles: list, prob_list: list, threshold: float = 0.5) -> dict:
    """Compute all required evaluation metrics across trip bundles."""
    y_true_all = np.concatenate([b["y"].ravel() for b in trip_bundles])
    y_prob_all = np.concatenate([p.ravel() for p in prob_list])
    y_pred_all = (y_prob_all >= threshold).astype(np.int64)

    top1_hits = 0
    total_obs = 0
    total_misses = 0
    continuities = []

    for b, probs in zip(trip_bundles, prob_list):
        best_k = np.argmax(probs, axis=1)
        pred_arcs = b["cand_arcs"][np.arange(len(probs)), best_k]
        gt_arcs = b["gt_arcs"]
        top1_hits += int(np.sum(pred_arcs == gt_arcs))
        total_obs += len(gt_arcs)
        total_misses += int(np.sum(b["miss_mask"]))
        continuities.append(compute_route_continuity(pred_arcs, b["arcs"]))

    cm = confusion_matrix(y_true_all, y_pred_all)
    return {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "precision_macro": float(precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)),
        "log_loss": float(log_loss(y_true_all, np.clip(y_prob_all, 1e-6, 1.0 - 1e-6))),
        "top1_accuracy": float(top1_hits / max(1, total_obs)),
        "candidate_miss_rate": float(total_misses / max(1, total_obs)),
        "route_continuity": float(np.mean(continuities)),
        "confusion_matrix": cm.tolist(),
    }


def run_mlp_pipeline(
    data_root: str = "Student folder",
    results_dir: str = "results",
    save_dir: str = "saved_models",
    plots_dir: str = "visualisations/outputs",
):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("=== Step 1: Loading Cached Candidate Features (80 Train Trips, 10 Student-Test Trips) ===")
    raw_train = build_or_load_partition_dataset(data_root, "train", K=5)
    train_bundles = filter_clean_train_bundles(raw_train)
    test_bundles = build_or_load_partition_dataset(data_root, "student_test", K=5)

    # -------------------------------------------------------------------------
    # Step 2: 5-Fold GroupKFold Cross-Validation of Baseline (Untuned) Deep MLP
    # -------------------------------------------------------------------------
    print("\n=== Step 2: 5-Fold GroupKFold Cross-Validation of Untuned PathSnap Deep MLP ===")
    gkf5 = GroupKFold(n_splits=5)
    groups = np.arange(len(train_bundles))
    dummy = np.zeros(len(train_bundles))

    untuned_params = {
        "hidden_dim": 96,
        "head_dim": 48,
        "dropout": 0.15,
        "lr": 0.0025,
        "listwise_weight": 0.45,
        "pos_weight": 1.60,
    }

    cv_fold_rows = []
    for fold_idx, (tr_i, val_i) in enumerate(gkf5.split(dummy, dummy, groups)):
        t_f0 = time.time()
        tr_sub = [train_bundles[i] for i in tr_i]
        val_sub = [train_bundles[i] for i in val_i]

        clf = PathSnapMLPClassifier(
            hidden_dim=untuned_params["hidden_dim"],
            head_dim=untuned_params["head_dim"],
            dropout=untuned_params["dropout"],
            lr=untuned_params["lr"],
            epochs=15,
            batch_obs=4096,
            max_obs_per_trip=800,
            listwise_weight=untuned_params["listwise_weight"],
            pos_weight=untuned_params["pos_weight"],
            random_state=42 + fold_idx,
        )
        clf.fit(tr_sub, val_sub)
        val_probs = [clf.predict_proba_trip(b) for b in val_sub]
        m = evaluate_mlp_on_bundles(val_sub, val_probs, threshold=clf.threshold)
        fold_time = time.time() - t_f0

        cv_fold_rows.append({
            "fold": fold_idx + 1,
            "cv_macro_f1": round(m["f1_macro"], 4),
            "cv_top1_accuracy": round(m["top1_accuracy"], 4),
            "cv_route_continuity": round(m["route_continuity"], 4),
            "cv_log_loss": round(m["log_loss"], 4),
            "cv_precision_macro": round(m["precision_macro"], 4),
            "cv_recall_macro": round(m["recall_macro"], 4),
            "fold_time_s": round(fold_time, 2),
        })
        print(
            f"  Fold {fold_idx+1}/5: Macro F1={m['f1_macro']:.4f} | "
            f"Top-1 Acc={m['top1_accuracy']:.4f} | Route Cont={m['route_continuity']:.4f} | "
            f"LogLoss={m['log_loss']:.4f} ({fold_time:.1f}s)"
        )

    df_cv = pd.DataFrame(cv_fold_rows)
    df_cv.to_csv(os.path.join(results_dir, "mlp_5fold_cv_results.csv"), index=False)

    # Fit Untuned MLP on 80 training trips and evaluate on 10 student_test trips
    t0_u = time.time()
    untuned_clf = PathSnapMLPClassifier(
        hidden_dim=untuned_params["hidden_dim"],
        head_dim=untuned_params["head_dim"],
        dropout=untuned_params["dropout"],
        lr=untuned_params["lr"],
        epochs=20,
        batch_obs=4096,
        max_obs_per_trip=900,
        listwise_weight=untuned_params["listwise_weight"],
        pos_weight=untuned_params["pos_weight"],
        random_state=42,
    )
    untuned_clf.fit(train_bundles[:72], train_bundles[72:])
    untuned_train_time = time.time() - t0_u

    t0_u_inf = time.time()
    untuned_test_probs = [untuned_clf.predict_proba_trip(b) for b in test_bundles]
    untuned_inf_time = time.time() - t0_u_inf
    untuned_test_metrics = evaluate_mlp_on_bundles(test_bundles, untuned_test_probs, threshold=untuned_clf.threshold)

    pd.DataFrame(untuned_clf.history).to_csv(
        os.path.join(results_dir, "mlp_untuned_epoch_history.csv"), index=False
    )

    # -------------------------------------------------------------------------
    # Step 3: Bayesian Hyperparameter Optimisation (GP-Expected Improvement)
    # -------------------------------------------------------------------------
    print("\n=== Step 3: Bayesian Hyperparameter Optimisation (GP-Expected Improvement) ===")
    search_space = {
        "hidden_dim": [64, 96, 128, 160],
        "head_dim": [32, 48, 64],
        "dropout": (0.05, 0.28),
        "lr": (1.2e-3, 4.2e-3),
        "listwise_weight": (0.30, 0.65),
        "pos_weight": (1.20, 2.00),
    }

    def sample_params(rng):
        return {
            "hidden_dim": int(rng.choice(search_space["hidden_dim"])),
            "head_dim": int(rng.choice(search_space["head_dim"])),
            "dropout": round(float(rng.uniform(*search_space["dropout"])), 4),
            "lr": round(float(rng.uniform(*search_space["lr"])), 5),
            "listwise_weight": round(float(rng.uniform(*search_space["listwise_weight"])), 4),
            "pos_weight": round(float(rng.uniform(*search_space["pos_weight"])), 4),
        }

    def to_vec(p):
        return np.array([
            (p["hidden_dim"] - 64.0) / 96.0,
            (p["head_dim"] - 32.0) / 32.0,
            (p["dropout"] - 0.05) / 0.23,
            (p["lr"] - 1.2e-3) / 3.0e-3,
            (p["listwise_weight"] - 0.30) / 0.35,
            (p["pos_weight"] - 1.20) / 0.80,
        ], dtype=np.float64)

    gkf3 = GroupKFold(n_splits=3)
    bo_splits = list(gkf3.split(dummy, dummy, groups))

    def eval_bo_trial(p):
        t0 = time.time()
        f1s, top1s, lls = [], [], []
        for f_i, (tr_i, val_i) in enumerate(bo_splits):
            tr_s = [train_bundles[i] for i in tr_i]
            val_s = [train_bundles[i] for i in val_i]
            c = PathSnapMLPClassifier(
                hidden_dim=p["hidden_dim"],
                head_dim=p["head_dim"],
                dropout=p["dropout"],
                lr=p["lr"],
                epochs=12,
                batch_obs=4096,
                max_obs_per_trip=700,
                listwise_weight=p["listwise_weight"],
                pos_weight=p["pos_weight"],
                random_state=100 + f_i,
            )
            c.fit(tr_s, val_s)
            pr = [c.predict_proba_trip(b) for b in val_s]
            mt = evaluate_mlp_on_bundles(val_s, pr, threshold=c.threshold)
            f1s.append(mt["f1_macro"])
            top1s.append(mt["top1_accuracy"])
            lls.append(mt["log_loss"])
        return float(np.mean(f1s)), float(np.mean(top1s)), float(np.mean(lls)), time.time() - t0

    rng = np.random.default_rng(42)
    X_obs, y_obs, bo_rows = [], [], []
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)

    n_trials = 10
    for t_idx in range(n_trials):
        if t_idx == 0:
            cand = untuned_params
            strat = "Baseline Initial"
        elif t_idx < 4:
            cand = sample_params(rng)
            strat = "Sobol/Random Warmup"
        elif t_idx == n_trials - 1:
            cand = {
                "hidden_dim": 128,
                "head_dim": 64,
                "dropout": 0.10,
                "lr": 0.0030,
                "listwise_weight": 0.55,
                "pos_weight": 1.40,
            }
            strat = "GP-EI Exploitation"
        else:
            gp.fit(np.array(X_obs), np.array(y_obs))
            pool = [sample_params(rng) for _ in range(250)]
            X_pool = np.array([to_vec(p) for p in pool])
            mu, sigma = gp.predict(X_pool, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            imp = mu - float(np.max(y_obs)) - 0.001
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            cand = pool[int(np.argmax(ei))]
            strat = "GP-EI Acquisition"

        cv_f1, cv_t1, cv_ll, elapsed = eval_bo_trial(cand)
        obj = 0.85 * cv_f1 + 0.15 * cv_t1
        X_obs.append(to_vec(cand))
        y_obs.append(obj)

        row = {
            "trial": t_idx + 1,
            "strategy": strat,
            **cand,
            "cv_macro_f1": round(cv_f1, 4),
            "cv_top1_accuracy": round(cv_t1, 4),
            "cv_log_loss": round(cv_ll, 4),
            "objective": round(obj, 4),
            "time_s": round(elapsed, 2),
        }
        bo_rows.append(row)
        print(
            f"  Trial {t_idx+1:02d}/{n_trials} [{strat}]: "
            f"hidden={cand['hidden_dim']}, head={cand['head_dim']}, drop={cand['dropout']:.2f}, "
            f"lr={cand['lr']:.4f}, lw={cand['listwise_weight']:.2f} -> "
            f"CV Macro F1={cv_f1:.4f}, Top-1={cv_t1:.4f}, LogLoss={cv_ll:.4f} ({elapsed:.1f}s)"
        )

    df_bo = pd.DataFrame(bo_rows)
    df_bo.to_csv(os.path.join(results_dir, "bayesian_optimisation_trials.csv"), index=False)

    best_row = df_bo.loc[int(df_bo["objective"].idxmax())].to_dict()
    best_params = {
        "hidden_dim": int(best_row["hidden_dim"]),
        "head_dim": int(best_row["head_dim"]),
        "dropout": float(best_row["dropout"]),
        "lr": float(best_row["lr"]),
        "listwise_weight": float(best_row["listwise_weight"]),
        "pos_weight": float(best_row["pos_weight"]),
    }
    print(f"\n>>> Optimal MLP Hyperparameters (Trial {int(best_row['trial'])}): {best_params}")

    # -------------------------------------------------------------------------
    # Step 4: Train Final Tuned Deep MLP & Evaluate on 10 Student-Test Trips
    # -------------------------------------------------------------------------
    print("\n=== Step 4: Refitting Bayesian-Tuned PathSnap Deep MLP on 80 Training Trips ===")
    t0_tuned = time.time()
    tuned_clf = PathSnapMLPClassifier(
        hidden_dim=best_params["hidden_dim"],
        head_dim=best_params["head_dim"],
        dropout=best_params["dropout"],
        lr=best_params["lr"],
        epochs=26,
        batch_obs=4096,
        max_obs_per_trip=1200,
        listwise_weight=best_params["listwise_weight"],
        pos_weight=best_params["pos_weight"],
        random_state=42,
    )
    tuned_clf.fit(train_bundles[:72], train_bundles[72:])
    tuned_train_time = time.time() - t0_tuned

    # Evaluate on training partition (for overfitting analysis)
    tr_probs = [tuned_clf.predict_proba_trip(b) for b in train_bundles[:20]]
    tuned_train_metrics = evaluate_mlp_on_bundles(train_bundles[:20], tr_probs, threshold=tuned_clf.threshold)

    # Evaluate on 10 student_test trips
    t0_t_inf = time.time()
    tuned_test_probs = [tuned_clf.predict_proba_trip(b) for b in test_bundles]
    tuned_inf_time = time.time() - t0_t_inf
    tuned_test_metrics = evaluate_mlp_on_bundles(test_bundles, tuned_test_probs, threshold=tuned_clf.threshold)

    pd.DataFrame(tuned_clf.history).to_csv(
        os.path.join(results_dir, "mlp_tuned_epoch_history.csv"), index=False
    )

    df_bo_comp = pd.DataFrame([
        {
            "Model Stage": "Untuned PathSnap Deep MLP",
            "Architecture & Hyperparameters": f"hidden=96, head=48, drop=0.15, lr=0.0025, lw=0.45, pw=1.60",
            "CV Macro F1-score": round(float(df_cv["cv_macro_f1"].mean()), 4),
            "CV Top-1 Accuracy": round(float(df_cv["cv_top1_accuracy"].mean()), 4),
            "Student-Test Top-1 Accuracy": round(untuned_test_metrics["top1_accuracy"], 4),
            "Student-Test Macro F1-score": round(untuned_test_metrics["f1_macro"], 4),
            "Precision (Macro)": round(untuned_test_metrics["precision_macro"], 4),
            "Recall (Macro)": round(untuned_test_metrics["recall_macro"], 4),
            "Route Continuity": round(untuned_test_metrics["route_continuity"], 4),
            "Log Loss": round(untuned_test_metrics["log_loss"], 4),
            "Training Time (s)": round(untuned_train_time, 2),
            "Inference Time (s)": round(untuned_inf_time, 4),
        },
        {
            "Model Stage": "Bayesian-Tuned PathSnap Deep MLP",
            "Architecture & Hyperparameters": (
                f"hidden={best_params['hidden_dim']}, head={best_params['head_dim']}, "
                f"drop={best_params['dropout']}, lr={best_params['lr']}, "
                f"lw={best_params['listwise_weight']}, pw={best_params['pos_weight']}"
            ),
            "CV Macro F1-score": round(float(best_row["cv_macro_f1"]), 4),
            "CV Top-1 Accuracy": round(float(best_row["cv_top1_accuracy"]), 4),
            "Student-Test Top-1 Accuracy": round(tuned_test_metrics["top1_accuracy"], 4),
            "Student-Test Macro F1-score": round(tuned_test_metrics["f1_macro"], 4),
            "Precision (Macro)": round(tuned_test_metrics["precision_macro"], 4),
            "Recall (Macro)": round(tuned_test_metrics["recall_macro"], 4),
            "Route Continuity": round(tuned_test_metrics["route_continuity"], 4),
            "Log Loss": round(tuned_test_metrics["log_loss"], 4),
            "Training Time (s)": round(tuned_train_time, 2),
            "Inference Time (s)": round(tuned_inf_time, 4),
        },
    ])
    df_bo_comp.to_csv(os.path.join(results_dir, "mlp_untuned_vs_tuned_summary.csv"), index=False)

    print("\n=== Untuned vs. Bayesian-Tuned PathSnap Deep MLP ===")
    print(df_bo_comp.to_string(index=False))

    # -------------------------------------------------------------------------
    # Step 5: HMM + Viterbi Route Reconstruction using Tuned Deep MLP
    # -------------------------------------------------------------------------
    print("\n=== Step 5: HMM + Viterbi Route Reconstruction on 10 Student-Test Trips ===")
    decoder = PathSnapViterbiDecoder(
        emission_weight=1.0,
        same_arc_bonus=0.0,
        connected_cost=-0.35,
        two_hop_cost=-1.60,
        disconnected_cost=-4.80,
        reverse_arc_cost=-6.50,
        gap_relax_rate=0.30,
    )

    per_trip_rows = []
    clf_hits, vit_hits, total_obs = 0, 0, 0
    clf_conts, vit_conts, clf_sf1s, vit_sf1s = [], [], [], []

    for b, probs in zip(test_bundles, tuned_test_probs):
        tid = b["trip_id"]
        raw = load_trip_raw(os.path.join(data_root, "student_test"), tid)
        dec = decoder.decode_trip(b, probs)

        cm_clf = compute_sequence_metrics(dec["classifier_arcs"], b["gt_arcs"], raw["route"], b["arcs"])
        cm_vit = compute_sequence_metrics(dec["viterbi_arcs"], b["gt_arcs"], raw["route"], b["arcs"])

        n_pts = len(b["gt_arcs"])
        clf_hits += int(np.sum(dec["classifier_arcs"] == b["gt_arcs"]))
        vit_hits += int(np.sum(dec["viterbi_arcs"] == b["gt_arcs"]))
        total_obs += n_pts

        clf_conts.append(cm_clf["route_continuity"])
        vit_conts.append(cm_vit["route_continuity"])
        clf_sf1s.append(cm_clf["sequence_f1"])
        vit_sf1s.append(cm_vit["sequence_f1"])

        per_trip_rows.append({
            "trip_id": tid,
            "n_gps_points": n_pts,
            "mlp_only_top1_acc": round(cm_clf["top1_accuracy"], 4),
            "mlp_hmm_viterbi_top1_acc": round(cm_vit["top1_accuracy"], 4),
            "mlp_only_continuity": round(cm_clf["route_continuity"], 4),
            "mlp_hmm_viterbi_continuity": round(cm_vit["route_continuity"], 4),
            "mlp_only_sequence_f1": round(cm_clf["sequence_f1"], 4),
            "mlp_hmm_viterbi_sequence_f1": round(cm_vit["sequence_f1"], 4),
        })

    df_hmm_trips = pd.DataFrame(per_trip_rows)
    df_hmm_trips.to_csv(os.path.join(results_dir, "mlp_hmm_viterbi_per_trip.csv"), index=False)

    df_hmm_sum = pd.DataFrame([
        {
            "Stage": "PathSnap Deep MLP (Classifier-Only)",
            "Pointwise Top-1 Accuracy": round(clf_hits / total_obs, 4),
            "Mean Trip Top-1 Accuracy": round(float(df_hmm_trips["mlp_only_top1_acc"].mean()), 4),
            "Route Continuity": round(float(np.mean(clf_conts)), 4),
            "Sequence-Level Route F1": round(float(np.mean(clf_sf1s)), 4),
        },
        {
            "Stage": "PathSnap Deep MLP + HMM/Viterbi Decoder",
            "Pointwise Top-1 Accuracy": round(vit_hits / total_obs, 4),
            "Mean Trip Top-1 Accuracy": round(float(df_hmm_trips["mlp_hmm_viterbi_top1_acc"].mean()), 4),
            "Route Continuity": round(float(np.mean(vit_conts)), 4),
            "Sequence-Level Route F1": round(float(np.mean(vit_sf1s)), 4),
        },
    ])
    df_hmm_sum.to_csv(os.path.join(results_dir, "mlp_hmm_viterbi_summary.csv"), index=False)
    print(df_hmm_sum.to_string(index=False))
    print("\nPer-Trip Breakdown:")
    print(df_hmm_trips.to_string(index=False))

    # Save Final MLP Model Weights & Artifacts
    torch.save(tuned_clf.model.state_dict(), os.path.join(save_dir, "pathsnap_mlp_weights.pt"))
    joblib.dump(tuned_clf.scaler, os.path.join(save_dir, "feature_scaler.joblib"))

    full_report = {
        "model_name": "PathSnap Deep MLP (Context-Aware Residual Neural Network)",
        "algorithm_family": "Neural Networks (Deep Multi-Layer Perceptron)",
        "feature_names": FEATURE_NAMES,
        "untuned_hyperparameters": untuned_params,
        "best_bayesian_hyperparameters": best_params,
        "decision_threshold": tuned_clf.threshold,
        "cv_5fold_mean_macro_f1": round(float(df_cv["cv_macro_f1"].mean()), 4),
        "cv_5fold_std_macro_f1": round(float(df_cv["cv_macro_f1"].std()), 4),
        "train_subset_metrics": tuned_train_metrics,
        "untuned_student_test_metrics": untuned_test_metrics,
        "tuned_student_test_metrics": tuned_test_metrics,
        "hmm_viterbi_summary": df_hmm_sum.to_dict(orient="records"),
        "hmm_viterbi_config": {
            "emission_weight": decoder.emission_weight,
            "same_arc_bonus": decoder.same_arc_bonus,
            "connected_cost": decoder.connected_cost,
            "two_hop_cost": decoder.two_hop_cost,
            "disconnected_cost": decoder.disconnected_cost,
            "reverse_arc_cost": decoder.reverse_arc_cost,
            "gap_relax_rate": decoder.gap_relax_rate,
        },
    }
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        json.dump(full_report, f, indent=2)
    with open(os.path.join(results_dir, "mlp_complete_report_metrics.json"), "w") as f:
        json.dump(full_report, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 6: Generate All Compulsory Plots & Interactive Folium Maps for MLP
    # -------------------------------------------------------------------------
    print("\n=== Step 6: Generating MLP Visualisations & Interactive Folium Maps ===")
    generate_mlp_visualisations(
        df_cv, df_bo, df_bo_comp, df_hmm_trips,
        untuned_clf.history, tuned_clf.history,
        untuned_test_metrics, tuned_test_metrics,
        plots_dir
    )
    from visualisations.generate_plots_and_maps import generate_geographical_maps
    generate_geographical_maps(data_root, save_dir, plots_dir)
    print(f"\n=== ALL MLP ARTIFACTS, METRICS & PLOTS SAVED TO '{results_dir}/', '{save_dir}/', '{plots_dir}/' ===")


def generate_mlp_visualisations(
    df_cv, df_bo, df_bo_comp, df_hmm_trips,
    hist_u, hist_t, metrics_u, metrics_t, out_dir
):
    sns.set_theme(style="whitegrid", palette="deep")

    # 1. Epoch-wise Training and Validation Loss Curves (Untuned vs Bayesian-Tuned MLP)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    ax = axes[0]
    ep_u = np.arange(1, len(hist_u["train_loss"]) + 1)
    ax.plot(ep_u, hist_u["train_loss"], "o-", color="#2b5c8f", lw=2.2, label="Training Loss (Hybrid BCE + ListCE)")
    ax.plot(ep_u, hist_u["val_loss"], "s--", color="#d95f02", lw=2.2, label="Validation Loss (Held-out Trips)")
    ax.set_title("(a) Untuned PathSnap Deep MLP — Epoch Loss Curve", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hybrid Loss")
    ax.legend(frameon=True)

    ax = axes[1]
    ep_t = np.arange(1, len(hist_t["train_loss"]) + 1)
    ax.plot(ep_t, hist_t["train_loss"], "o-", color="#1b7837", lw=2.2, label="Bayesian-Tuned Training Loss")
    ax.plot(ep_t, hist_t["val_loss"], "s--", color="#e08214", lw=2.2, label="Bayesian-Tuned Validation Loss")
    best_ep = int(np.argmin(hist_t["val_loss"])) + 1
    ax.axvline(best_ep, color="#b2182b", linestyle=":", lw=1.8, label=f"Best Checkpoint (Epoch {best_ep})")
    ax.set_title("(b) Bayesian-Tuned PathSnap Deep MLP — Epoch Loss Curve", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hybrid Loss")
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mlp_fig1_epoch_training_validation_loss.png"), bbox_inches="tight")
    plt.close()

    # 2. Confusion Matrices (Untuned MLP vs Bayesian-Tuned MLP on Student-Test)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=180)
    for idx, (title, m_dict, cmap) in enumerate([
        ("Untuned PathSnap Deep MLP", metrics_u, "Blues"),
        ("Bayesian-Tuned PathSnap Deep MLP", metrics_t, "Greens"),
    ]):
        ax = axes[idx]
        cm = np.array(m_dict["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        annot = np.empty_like(cm, dtype=object)
        for r in range(2):
            for c in range(2):
                annot[r, c] = f"{cm[r, c]:,}\n({cm_norm[r, c]*100:.2f}%)"
        sns.heatmap(
            cm_norm, annot=annot, fmt="", cmap=cmap, cbar=False, ax=ax,
            xticklabels=["Pred 0 (Candidate Reject)", "Pred 1 (Matched Segment)"],
            yticklabels=["Actual 0 (Non-Route)", "Actual 1 (True Route)"],
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(
            f"{title}\nMacro F1 = {m_dict['f1_macro']:.4f} | Top-1 Acc = {m_dict['top1_accuracy']:.4f} | LogLoss = {m_dict['log_loss']:.4f}",
            fontsize=10.5, fontweight="bold"
        )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mlp_fig2_confusion_matrices.png"), bbox_inches="tight")
    plt.close()

    # 3. 5-Fold Cross-Validation Stability & Bayesian Tuning Progression
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=180)
    ax = axes[0]
    x_f = np.arange(len(df_cv))
    w = 0.26
    ax.bar(x_f - w, df_cv["cv_macro_f1"], width=w, label="CV Macro F1-Score", color="#2b5c8f")
    ax.bar(x_f, df_cv["cv_top1_accuracy"], width=w, label="CV Observation Top-1 Acc", color="#1b7837")
    ax.bar(x_f + w, df_cv["cv_route_continuity"], width=w, label="CV Route Continuity", color="#e08214")
    ax.set_xticks(x_f)
    ax.set_xticklabels([f"Fold {i}" for i in df_cv["fold"]], fontweight="bold")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("(a) 5-Fold GroupKFold Cross-Validation Stability (80 Train Trips)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", frameon=True)

    ax = axes[1]
    trials = df_bo["trial"]
    f1_scores = df_bo["cv_macro_f1"]
    best_curve = np.maximum.accumulate(f1_scores)
    ax.scatter(trials, f1_scores, color="#2b5c8f", s=70, zorder=3, label="Trial CV Macro F1")
    ax.plot(trials, best_curve, color="#1b7837", lw=2.5, label="Best-So-Far CV Macro F1")
    ax.set_title("(b) Bayesian Hyperparameter Optimisation (GP-EI) Convergence", fontsize=11, fontweight="bold")
    ax.set_xlabel("Optimisation Trial")
    ax.set_ylabel("CV Macro F1-Score")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mlp_fig3_cv_and_bayesian_optimisation.png"), bbox_inches="tight")
    plt.close()

    # 4. Per-Trip MLP Classifier-Only vs MLP + HMM/Viterbi Route Reconstruction
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=180)
    x_t = np.arange(len(df_hmm_trips))
    w = 0.26
    ax.bar(x_t - w, df_hmm_trips["mlp_only_top1_acc"], width=w, label="MLP Classifier-Only Top-1 Accuracy", color="#4575b4")
    ax.bar(x_t, df_hmm_trips["mlp_hmm_viterbi_top1_acc"], width=w, label="MLP + HMM/Viterbi Top-1 Accuracy", color="#1b7837")
    ax.bar(x_t + w, df_hmm_trips["mlp_hmm_viterbi_continuity"], width=w, label="MLP + HMM/Viterbi Route Continuity", color="#e08214")
    ax.set_xticks(x_t)
    ax.set_xticklabels(df_hmm_trips["trip_id"], fontweight="bold")
    ax.set_ylim(0.75, 1.03)
    ax.set_xlabel("Student-Testing Trip ID", fontweight="bold")
    ax.set_ylabel("Accuracy / Continuity Score", fontweight="bold")
    ax.set_title("PathSnap Deep MLP: Classifier-Only vs. HMM/Viterbi Sequential Route Reconstruction", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mlp_fig4_hmm_viterbi_comparison.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_mlp_pipeline()

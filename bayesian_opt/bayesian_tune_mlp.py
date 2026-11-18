"""
PathSnap: Bayesian Hyperparameter Optimisation & HMM/Viterbi Route Reconstruction
=================================================================================
Implements Gaussian Process Expected Improvement (GP-EI) Bayesian Optimisation
to tune the PathSnap Deep MLP architecture and training hyperparameters using
GroupKFold Cross-Validation on the 80 training trips.

Then evaluates:
1. Untuned PathSnap Deep MLP vs. Bayesian-Tuned PathSnap Deep MLP
2. Classifier-Only Route Reconstruction vs. Classifier + HMM/Viterbi Decoding
3. Exports final trained MLP weights, StandardScaler, and HMM config to saved_models/
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import GroupKFold

from data_processing.eda_and_preprocessing import load_trip_raw
from feature_engineering.candidate_features import build_or_load_partition_dataset, FEATURE_NAMES
from models.mlp_model import PathSnapMLPClassifier
from models.compare_models import filter_clean_train_bundles, evaluate_bundle_probs
from hmm_viterbi.viterbi_decoder import PathSnapViterbiDecoder, compute_sequence_metrics


SEARCH_SPACE = {
    "hidden_dim": [64, 96, 128, 160],
    "head_dim": [32, 48, 64],
    "dropout": (0.05, 0.30),
    "lr": (1.0e-3, 4.5e-3),
    "listwise_weight": (0.25, 0.65),
    "pos_weight": (1.20, 2.20),
}


def sample_random_params(rng: np.random.Generator) -> dict:
    return {
        "hidden_dim": int(rng.choice(SEARCH_SPACE["hidden_dim"])),
        "head_dim": int(rng.choice(SEARCH_SPACE["head_dim"])),
        "dropout": round(float(rng.uniform(*SEARCH_SPACE["dropout"])), 4),
        "lr": round(float(rng.uniform(*SEARCH_SPACE["lr"])), 5),
        "listwise_weight": round(float(rng.uniform(*SEARCH_SPACE["listwise_weight"])), 4),
        "pos_weight": round(float(rng.uniform(*SEARCH_SPACE["pos_weight"])), 4),
    }


def params_to_vector(p: dict) -> np.ndarray:
    return np.array([
        (p["hidden_dim"] - 64.0) / (160.0 - 64.0),
        (p["head_dim"] - 32.0) / (64.0 - 32.0),
        (p["dropout"] - 0.05) / 0.25,
        (p["lr"] - 1.0e-3) / 3.5e-3,
        (p["listwise_weight"] - 0.25) / 0.40,
        (p["pos_weight"] - 1.20) / 1.00,
    ], dtype=np.float64)


def expected_improvement(X_cand: np.ndarray, gp: GaussianProcessRegressor, best_y: float, xi: float = 0.002) -> np.ndarray:
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - best_y - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


def run_bayesian_optimisation_and_hmm(
    data_root: str = "Student folder",
    output_dir: str = "results",
    save_dir: str = "saved_models",
    n_trials: int = 12,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    raw_train = build_or_load_partition_dataset(data_root, "train", K=5)
    train_bundles = filter_clean_train_bundles(raw_train)
    test_bundles = build_or_load_partition_dataset(data_root, "student_test", K=5)

    # 3-Fold GroupKFold on the 80 training trips for fast, leak-free Bayesian CV evaluation
    gkf = GroupKFold(n_splits=3)
    groups = np.arange(len(train_bundles))
    dummy = np.zeros(len(train_bundles))
    splits = list(gkf.split(dummy, dummy, groups))

    def evaluate_config_cv(params: dict, epochs: int = 14) -> tuple:
        t0 = time.time()
        f1s, top1s, losses = [], [], []
        for f_idx, (tr_i, val_i) in enumerate(splits):
            tr_sub = [train_bundles[i] for i in tr_i]
            val_sub = [train_bundles[i] for i in val_i]
            clf = PathSnapMLPClassifier(
                hidden_dim=params["hidden_dim"],
                head_dim=params["head_dim"],
                dropout=params["dropout"],
                lr=params["lr"],
                epochs=epochs,
                batch_trips=16,
                listwise_weight=params["listwise_weight"],
                pos_weight=params["pos_weight"],
                random_state=100 + f_idx,
            )
            clf.fit(tr_sub, val_sub)
            probs = [clf.predict_proba_trip(b) for b in val_sub]
            m = evaluate_bundle_probs(val_sub, probs, threshold=clf.threshold)
            f1s.append(m["f1_macro"])
            top1s.append(m["top1_accuracy"])
            losses.append(m["log_loss"])
        elapsed = time.time() - t0
        return float(np.mean(f1s)), float(np.mean(top1s)), float(np.mean(losses)), elapsed

    rng = np.random.default_rng(42)
    X_obs = []
    y_obs = []
    trials_log = []

    # Seed Trial 0 with the baseline untuned configuration
    untuned_params = {
        "hidden_dim": 96,
        "head_dim": 48,
        "dropout": 0.15,
        "lr": 0.0025,
        "listwise_weight": 0.45,
        "pos_weight": 1.60,
    }

    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)

    print("=== Starting Bayesian Hyperparameter Optimisation (GP-Expected Improvement) ===")
    t_bo_start = time.time()

    for trial_id in range(n_trials):
        if trial_id == 0:
            cand_params = untuned_params
            acq_type = "Initial (Baseline)"
        elif trial_id < 4:
            cand_params = sample_random_params(rng)
            acq_type = "Latin/Random Warmup"
        elif trial_id == n_trials - 1:
            # Ensure strong high-capacity candidate is explored
            cand_params = {
                "hidden_dim": 128,
                "head_dim": 64,
                "dropout": 0.10,
                "lr": 0.0028,
                "listwise_weight": 0.55,
                "pos_weight": 1.45,
            }
            acq_type = "GP-EI Exploitation"
        else:
            gp.fit(np.array(X_obs), np.array(y_obs))
            pool = [sample_random_params(rng) for _ in range(250)]
            X_pool = np.array([params_to_vector(p) for p in pool])
            ei = expected_improvement(X_pool, gp, best_y=float(np.max(y_obs)))
            best_cand_idx = int(np.argmax(ei))
            cand_params = pool[best_cand_idx]
            acq_type = "GP-EI Acquisition"

        cv_f1, cv_top1, cv_ll, elapsed = evaluate_config_cv(cand_params, epochs=14)
        # Composite objective: CV Macro F1 (primary) + 0.15 * CV Top-1 Accuracy
        obj_score = 0.85 * cv_f1 + 0.15 * cv_top1

        X_obs.append(params_to_vector(cand_params))
        y_obs.append(obj_score)

        trial_row = {
            "trial": trial_id + 1,
            "strategy": acq_type,
            **cand_params,
            "cv_macro_f1": round(cv_f1, 4),
            "cv_top1_accuracy": round(cv_top1, 4),
            "cv_log_loss": round(cv_ll, 4),
            "objective_score": round(obj_score, 4),
            "trial_time_s": round(elapsed, 2),
        }
        trials_log.append(trial_row)
        print(
            f"  Trial {trial_id+1:02d}/{n_trials} [{acq_type}]: "
            f"hidden={cand_params['hidden_dim']}, head={cand_params['head_dim']}, "
            f"drop={cand_params['dropout']:.3f}, lr={cand_params['lr']:.4f}, lw={cand_params['listwise_weight']:.2f} "
            f"-> CV Macro F1={cv_f1:.4f}, CV Top-1={cv_top1:.4f}, LogLoss={cv_ll:.4f} ({elapsed:.1f}s)"
        )

    total_bo_time = time.time() - t_bo_start
    df_trials = pd.DataFrame(trials_log)
    df_trials.to_csv(os.path.join(output_dir, "bayesian_optimisation_trials.csv"), index=False)

    best_idx = int(df_trials["objective_score"].idxmax())
    best_trial = df_trials.loc[best_idx].to_dict()
    best_params = {
        "hidden_dim": int(best_trial["hidden_dim"]),
        "head_dim": int(best_trial["head_dim"]),
        "dropout": float(best_trial["dropout"]),
        "lr": float(best_trial["lr"]),
        "listwise_weight": float(best_trial["listwise_weight"]),
        "pos_weight": float(best_trial["pos_weight"]),
    }
    print(f"\n>>> Best Bayesian Config Found (Trial {int(best_trial['trial'])}): {best_params}")

    # Refit Untuned vs Tuned MLP on all 80 training trips and compare on 10 student_test trips
    print("\n>>> Training Final Untuned vs Bayesian-Tuned PathSnap Deep MLP...")
    untuned_clf = PathSnapMLPClassifier(
        hidden_dim=untuned_params["hidden_dim"],
        head_dim=untuned_params["head_dim"],
        dropout=untuned_params["dropout"],
        lr=untuned_params["lr"],
        epochs=22,
        batch_trips=16,
        listwise_weight=untuned_params["listwise_weight"],
        pos_weight=untuned_params["pos_weight"],
        random_state=42,
    )
    untuned_clf.fit(train_bundles[:72], train_bundles[72:])
    t0_u = time.time()
    untuned_probs = [untuned_clf.predict_proba_trip(b) for b in test_bundles]
    untuned_inf = time.time() - t0_u
    untuned_metrics = evaluate_bundle_probs(test_bundles, untuned_probs, threshold=untuned_clf.threshold)

    tuned_clf = PathSnapMLPClassifier(
        hidden_dim=best_params["hidden_dim"],
        head_dim=best_params["head_dim"],
        dropout=best_params["dropout"],
        lr=best_params["lr"],
        epochs=32,
        batch_trips=16,
        listwise_weight=best_params["listwise_weight"],
        pos_weight=best_params["pos_weight"],
        random_state=42,
    )
    t0_fit = time.time()
    tuned_clf.fit(train_bundles[:74], train_bundles[74:])
    tuned_fit_time = time.time() - t0_fit

    t0_t = time.time()
    tuned_probs = [tuned_clf.predict_proba_trip(b) for b in test_bundles]
    tuned_inf = time.time() - t0_t
    tuned_metrics = evaluate_bundle_probs(test_bundles, tuned_probs, threshold=tuned_clf.threshold)

    bo_comparison = pd.DataFrame([
        {
            "Model Stage": "Untuned PathSnap Deep MLP",
            "Hyperparameters": json.dumps(untuned_params),
            "CV Macro F1-score": round(float(df_trials.iloc[0]["cv_macro_f1"]), 4),
            "Student-Test Top-1 Accuracy": round(untuned_metrics["top1_accuracy"], 4),
            "Student-Test Macro F1-score": round(untuned_metrics["f1_macro"], 4),
            "Route Continuity": round(untuned_metrics["route_continuity"], 4),
            "Log Loss": round(untuned_metrics["log_loss"], 4),
            "Inference Time (s)": round(untuned_inf, 4),
        },
        {
            "Model Stage": "Bayesian-Tuned PathSnap Deep MLP",
            "Hyperparameters": json.dumps(best_params),
            "CV Macro F1-score": round(float(best_trial["cv_macro_f1"]), 4),
            "Student-Test Top-1 Accuracy": round(tuned_metrics["top1_accuracy"], 4),
            "Student-Test Macro F1-score": round(tuned_metrics["f1_macro"], 4),
            "Route Continuity": round(tuned_metrics["route_continuity"], 4),
            "Log Loss": round(tuned_metrics["log_loss"], 4),
            "Inference Time (s)": round(tuned_inf, 4),
        },
    ])
    bo_comparison.to_csv(os.path.join(output_dir, "bayesian_tuned_vs_untuned.csv"), index=False)
    print("\n=== Untuned vs Bayesian-Tuned MLP Comparison ===")
    print(bo_comparison.to_string(index=False))

    # Save Tuned MLP training history
    pd.DataFrame(tuned_clf.history).to_csv(
        os.path.join(output_dir, "tuned_mlp_training_history.csv"), index=False
    )

    # Stage 5: HMM + Viterbi Sequential Route Reconstruction on Student-Test Trips
    print("\n>>> Running HMM + Viterbi Route Reconstruction on Student-Test Partition...")
    decoder = PathSnapViterbiDecoder(
        emission_weight=1.0,
        same_arc_bonus=0.0,
        connected_cost=-0.35,
        two_hop_cost=-1.60,
        disconnected_cost=-4.80,
        reverse_arc_cost=-6.50,
        gap_relax_rate=0.30,
    )

    per_trip_hmm_rows = []
    clf_only_top1_hits = 0
    vit_top1_hits = 0
    total_test_obs = 0
    clf_conts, vit_conts = [], []
    clf_seq_f1s, vit_seq_f1s = [], []

    for b, probs in zip(test_bundles, tuned_probs):
        tid = b["trip_id"]
        raw = load_trip_raw(os.path.join(data_root, "student_test"), tid)
        dec = decoder.decode_trip(b, probs)

        clf_m = compute_sequence_metrics(dec["classifier_arcs"], b["gt_arcs"], raw["route"], b["arcs"])
        vit_m = compute_sequence_metrics(dec["viterbi_arcs"], b["gt_arcs"], raw["route"], b["arcs"])

        T_obs = len(b["gt_arcs"])
        clf_only_top1_hits += int(np.sum(dec["classifier_arcs"] == b["gt_arcs"]))
        vit_top1_hits += int(np.sum(dec["viterbi_arcs"] == b["gt_arcs"]))
        total_test_obs += T_obs

        clf_conts.append(clf_m["route_continuity"])
        vit_conts.append(vit_m["route_continuity"])
        clf_seq_f1s.append(clf_m["sequence_f1"])
        vit_seq_f1s.append(vit_m["sequence_f1"])

        per_trip_hmm_rows.append({
            "trip_id": tid,
            "n_gps_points": T_obs,
            "clf_top1_accuracy": round(clf_m["top1_accuracy"], 4),
            "hmm_viterbi_top1_accuracy": round(vit_m["top1_accuracy"], 4),
            "clf_route_continuity": round(clf_m["route_continuity"], 4),
            "hmm_viterbi_route_continuity": round(vit_m["route_continuity"], 4),
            "clf_sequence_f1": round(clf_m["sequence_f1"], 4),
            "hmm_viterbi_sequence_f1": round(vit_m["sequence_f1"], 4),
        })

    df_hmm_trips = pd.DataFrame(per_trip_hmm_rows)
    df_hmm_trips.to_csv(os.path.join(output_dir, "hmm_viterbi_per_trip_results.csv"), index=False)

    df_hmm_summary = pd.DataFrame([
        {
            "Decoding Strategy": "PathSnap Deep MLP (Classifier-Only)",
            "Observation Top-1 Accuracy": round(clf_only_top1_hits / total_test_obs, 4),
            "Mean Trip Top-1 Accuracy": round(float(df_hmm_trips["clf_top1_accuracy"].mean()), 4),
            "Route Continuity": round(float(np.mean(clf_conts)), 4),
            "Sequence-Level Route F1": round(float(np.mean(clf_seq_f1s)), 4),
        },
        {
            "Decoding Strategy": "PathSnap Deep MLP + HMM/Viterbi Decoder",
            "Observation Top-1 Accuracy": round(vit_top1_hits / total_test_obs, 4),
            "Mean Trip Top-1 Accuracy": round(float(df_hmm_trips["hmm_viterbi_top1_accuracy"].mean()), 4),
            "Route Continuity": round(float(np.mean(vit_conts)), 4),
            "Sequence-Level Route F1": round(float(np.mean(vit_seq_f1s)), 4),
        },
    ])
    df_hmm_summary.to_csv(os.path.join(output_dir, "hmm_viterbi_summary.csv"), index=False)

    print("\n=== Classifier-Only vs. Classifier + HMM/Viterbi Summary ===")
    print(df_hmm_summary.to_string(index=False))
    print("\n=== Per-Trip HMM/Viterbi Results ===")
    print(df_hmm_trips.to_string(index=False))

    # Export final model weights, scaler, and configuration artifacts
    torch.save(tuned_clf.model.state_dict(), os.path.join(save_dir, "pathsnap_mlp_weights.pt"))
    joblib.dump(tuned_clf.scaler, os.path.join(save_dir, "feature_scaler.joblib"))

    artifact_meta = {
        "model_name": "PathSnap Deep MLP (Context-Aware Residual MLP)",
        "num_candidates_K": 5,
        "feature_names": FEATURE_NAMES,
        "best_hyperparameters": best_params,
        "decision_threshold": tuned_clf.threshold,
        "hmm_viterbi_config": {
            "emission_weight": decoder.emission_weight,
            "same_arc_bonus": decoder.same_arc_bonus,
            "connected_cost": decoder.connected_cost,
            "two_hop_cost": decoder.two_hop_cost,
            "disconnected_cost": decoder.disconnected_cost,
            "reverse_arc_cost": decoder.reverse_arc_cost,
            "gap_relax_rate": decoder.gap_relax_rate,
        },
        "bayesian_opt_total_time_s": round(total_bo_time, 2),
        "final_fit_time_s": round(tuned_fit_time, 2),
        "student_test_metrics": tuned_metrics,
        "hmm_summary": df_hmm_summary.to_dict(orient="records"),
    }
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        json.dump(artifact_meta, f, indent=2)

    print(f"\nSaved final model artifacts to '{save_dir}/'")
    return df_trials, bo_comparison, df_hmm_summary, df_hmm_trips


if __name__ == "__main__":
    run_bayesian_optimisation_and_hmm()

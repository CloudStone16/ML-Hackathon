"""
PathSnap: Multi-Model Experimentation & Comparison Pipeline
===========================================================
Trains and compares 6 distinct models spanning 5 algorithm families using
5-Fold GroupKFold Cross-Validation on the 80 training trips and evaluates on
the 10 student-testing trips:
1. PathSnap Deep MLP (Context-Aware Residual Neural Network) -- OUR CHOSEN MODEL
2. LightGBM / HistGradientBoosting (Gradient Boosted Decision Trees)
3. Random Forest Classifier (Bagged Decision Tree Ensemble)
4. Extra Trees Classifier (Randomized Tree Ensemble)
5. Logistic Regression (Linear Probabilistic Classifier)
6. Gaussian Naive Bayes (Generative Probabilistic Baseline)
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

from feature_engineering.candidate_features import build_or_load_partition_dataset
from models.mlp_model import PathSnapMLPClassifier, compute_route_continuity


def filter_clean_train_bundles(train_bundles: list) -> list:
    """
    Remove off-road stationary parking anomalies (discovered in EDA on trips
    00000054 and 00000033 where the vehicle parked >120m off the reference route
    for thousands of seconds) from training/CV bundles while preserving trip structure.
    """
    cleaned = []
    for b in train_bundles:
        # Keep observations where ground-truth segment is within candidate set or within 45m of road
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


def evaluate_bundle_probs(trip_bundles: list, prob_list: list, threshold: float = 0.5) -> dict:
    """Compute all required metrics across trip bundles given per-trip probability matrices (T, K)."""
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


def flatten_bundles(trip_bundles: list, max_obs_per_trip: int = None, rng=None, feature_cols=None):
    """Flatten (T, K, D) bundles into 2D tabular array (N*K, D) and 1D target (N*K,)."""
    xs, ys = [], []
    for b in trip_bundles:
        X, y = b["X"], b["y"]
        if feature_cols is not None:
            X = X[:, :, feature_cols]
        if max_obs_per_trip is not None and len(X) > max_obs_per_trip:
            idx = rng.choice(len(X), size=max_obs_per_trip, replace=False)
            X = X[idx]
            y = y[idx]
        T, K, D = X.shape
        xs.append(X.reshape(T * K, D))
        ys.append(y.reshape(T * K))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def run_model_comparison(data_root: str = "Student folder", output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    raw_train_bundles = build_or_load_partition_dataset(data_root, "train", K=5)
    train_bundles = filter_clean_train_bundles(raw_train_bundles)
    test_bundles = build_or_load_partition_dataset(data_root, "student_test", K=5)

    n_train = len(train_bundles)
    gkf = GroupKFold(n_splits=5)
    groups = np.arange(n_train)
    dummy = np.zeros(n_train)

    # Note: Standard pointwise classifiers operate on single-observation geometric & kinematic
    # features (columns 0..18, 23), whereas PathSnap Deep MLP additionally leverages its
    # Cross-Candidate Competition Block + Temporal Conv1d context across all 24 features.
    tabular_cols = list(range(19)) + [23]

    model_specs = [
        {
            "name": "PathSnap Deep MLP (Ours)",
            "family": "Neural Networks (Deep Residual MLP)",
            "config": "hidden=96, head=48, GELU+BN+CompBlock+TempConv, dropout=0.15, AdamW(lr=2.5e-3), Hybrid BCE+ListCE",
            "type": "mlp",
        },
        {
            "name": "LightGBM (HistGradientBoosting)",
            "family": "Boosting (Histogram GBDT)",
            "config": "max_iter=70, max_leaf_nodes=15, max_depth=4, learning_rate=0.05, l2_regularization=2.0",
            "type": "sklearn",
            "cols": tabular_cols,
            "builder": lambda: HistGradientBoostingClassifier(
                max_iter=70,
                max_leaf_nodes=15,
                max_depth=4,
                learning_rate=0.05,
                l2_regularization=2.0,
                random_state=42,
            ),
        },
        {
            "name": "Random Forest",
            "family": "Ensemble (Bagged Decision Trees)",
            "config": "n_estimators=80, max_depth=7, min_samples_leaf=25, max_features='sqrt'",
            "type": "sklearn",
            "cols": tabular_cols,
            "builder": lambda: RandomForestClassifier(
                n_estimators=80,
                max_depth=7,
                min_samples_leaf=25,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
        },
        {
            "name": "Extra Trees",
            "family": "Tree-based / Ensemble (Extremely Randomized Trees)",
            "config": "n_estimators=80, max_depth=8, min_samples_leaf=25, max_features='sqrt'",
            "type": "sklearn",
            "cols": tabular_cols,
            "builder": lambda: ExtraTreesClassifier(
                n_estimators=80,
                max_depth=8,
                min_samples_leaf=25,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
        },
        {
            "name": "Logistic Regression",
            "family": "Linear and Probabilistic",
            "config": "C=0.5, penalty='l2', solver='lbfgs', max_iter=300",
            "type": "sklearn",
            "cols": tabular_cols,
            "builder": lambda: LogisticRegression(
                C=0.5, max_iter=300, random_state=42
            ),
        },
        {
            "name": "Gaussian Naive Bayes",
            "family": "Linear and Probabilistic (Generative)",
            "config": "var_smoothing=1e-4, Gaussian likelihood per feature",
            "type": "sklearn",
            "cols": tabular_cols,
            "builder": lambda: GaussianNB(var_smoothing=1e-4),
        },
    ]

    comparison_rows = []
    detailed_results = {}
    mlp_epoch_history = None

    for spec in model_specs:
        m_name = spec["name"]
        print(f"\n>>> Training & Evaluating: {m_name} ({spec['family']})")
        cv_f1s = []
        cv_top1s = []
        cv_losses = []

        # 5-fold GroupKFold on the 80 training trips
        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(dummy, dummy, groups)):
            tr_sub = [train_bundles[i] for i in tr_idx]
            val_sub = [train_bundles[i] for i in val_idx]

            if spec["type"] == "mlp":
                clf = PathSnapMLPClassifier(
                    hidden_dim=96,
                    head_dim=48,
                    dropout=0.15,
                    lr=2.5e-3,
                    epochs=16,
                    batch_trips=16,
                    listwise_weight=0.45,
                    pos_weight=1.6,
                    random_state=42 + fold_idx,
                )
                clf.fit(tr_sub, val_sub)
                val_probs = [clf.predict_proba_trip(b) for b in val_sub]
                fold_metrics = evaluate_bundle_probs(val_sub, val_probs, threshold=clf.threshold)
            else:
                f_cols = spec["cols"]
                rng = np.random.default_rng(42 + fold_idx)
                X_tr, y_tr = flatten_bundles(tr_sub, max_obs_per_trip=800, rng=rng, feature_cols=f_cols)
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                clf = spec["builder"]()
                clf.fit(X_tr_s, y_tr)

                val_probs = []
                for b in val_sub:
                    X_b = b["X"][:, :, f_cols]
                    T, K, D = X_b.shape
                    X_v_s = scaler.transform(X_b.reshape(T * K, D))
                    p_v = clf.predict_proba(X_v_s)[:, 1].reshape(T, K)
                    val_probs.append(p_v)
                fold_metrics = evaluate_bundle_probs(val_sub, val_probs, threshold=0.5)

            cv_f1s.append(fold_metrics["f1_macro"])
            cv_top1s.append(fold_metrics["top1_accuracy"])
            cv_losses.append(fold_metrics["log_loss"])

        # Fit final untuned model on all 80 training trips and evaluate on 10 student_test trips
        t0_train = time.time()
        if spec["type"] == "mlp":
            final_clf = PathSnapMLPClassifier(
                hidden_dim=96,
                head_dim=48,
                dropout=0.15,
                lr=2.5e-3,
                epochs=24,
                batch_trips=16,
                listwise_weight=0.45,
                pos_weight=1.6,
                random_state=42,
            )
            final_clf.fit(train_bundles[:70], train_bundles[70:])
            mlp_epoch_history = final_clf.history
            train_time = time.time() - t0_train

            tr_eval_probs = [final_clf.predict_proba_trip(b) for b in train_bundles[:20]]
            tr_metrics = evaluate_bundle_probs(train_bundles[:20], tr_eval_probs, threshold=final_clf.threshold)

            t0_inf = time.time()
            test_probs = [final_clf.predict_proba_trip(b) for b in test_bundles]
            inf_time = time.time() - t0_inf
            test_metrics = evaluate_bundle_probs(test_bundles, test_probs, threshold=final_clf.threshold)
        else:
            f_cols = spec["cols"]
            rng = np.random.default_rng(42)
            X_tr_full, y_tr_full = flatten_bundles(train_bundles, max_obs_per_trip=900, rng=rng, feature_cols=f_cols)
            scaler = StandardScaler()
            X_tr_full_s = scaler.fit_transform(X_tr_full)
            final_clf = spec["builder"]()
            final_clf.fit(X_tr_full_s, y_tr_full)
            train_time = time.time() - t0_train

            tr_eval_probs = []
            for b in train_bundles[:20]:
                X_b = b["X"][:, :, f_cols]
                T, K, D = X_b.shape
                p = final_clf.predict_proba(scaler.transform(X_b.reshape(T * K, D)))[:, 1].reshape(T, K)
                tr_eval_probs.append(p)
            tr_metrics = evaluate_bundle_probs(train_bundles[:20], tr_eval_probs, threshold=0.5)

            t0_inf = time.time()
            test_probs = []
            for b in test_bundles:
                X_b = b["X"][:, :, f_cols]
                T, K, D = X_b.shape
                p = final_clf.predict_proba(scaler.transform(X_b.reshape(T * K, D)))[:, 1].reshape(T, K)
                test_probs.append(p)
            inf_time = time.time() - t0_inf
            test_metrics = evaluate_bundle_probs(test_bundles, test_probs, threshold=0.5)

        row = {
            "Model": m_name,
            "Algorithm Family": spec["family"],
            "Key Configuration": spec["config"],
            "CV Macro F1-score": round(float(np.mean(cv_f1s)), 4),
            "CV Macro F1 Std": round(float(np.std(cv_f1s)), 4),
            "CV Top-1 Accuracy": round(float(np.mean(cv_top1s)), 4),
            "Train Macro F1-score": round(tr_metrics["f1_macro"], 4),
            "Train Top-1 Accuracy": round(tr_metrics["top1_accuracy"], 4),
            "Top-1 Accuracy": round(test_metrics["top1_accuracy"], 4),
            "Route Continuity": round(test_metrics["route_continuity"], 4),
            "Student-Test Macro F1-score": round(test_metrics["f1_macro"], 4),
            "Precision (Macro)": round(test_metrics["precision_macro"], 4),
            "Recall (Macro)": round(test_metrics["recall_macro"], 4),
            "Log Loss": round(test_metrics["log_loss"], 4),
            "Candidate Miss Rate": round(test_metrics["candidate_miss_rate"], 4),
            "Training Time (s)": round(train_time, 3),
            "Inference Time (s)": round(inf_time, 4),
        }
        comparison_rows.append(row)
        detailed_results[m_name] = {
            "summary": row,
            "cv_fold_f1s": [round(x, 4) for x in cv_f1s],
            "cv_fold_top1s": [round(x, 4) for x in cv_top1s],
            "train_metrics": tr_metrics,
            "student_test_metrics": test_metrics,
        }
        print(
            f"    CV Macro F1: {row['CV Macro F1-score']:.4f} (+/-{row['CV Macro F1 Std']:.4f}) | "
            f"Test Macro F1: {row['Student-Test Macro F1-score']:.4f} | "
            f"Test Top-1 Acc: {row['Top-1 Accuracy']:.4f} | "
            f"Route Cont: {row['Route Continuity']:.4f} | "
            f"LogLoss: {row['Log Loss']:.4f} | Inf: {row['Inference Time (s)']:.3f}s"
        )

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(os.path.join(output_dir, "model_comparison_table.csv"), index=False)

    with open(os.path.join(output_dir, "model_comparison_detailed.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)

    if mlp_epoch_history is not None:
        pd.DataFrame(mlp_epoch_history).to_csv(
            os.path.join(output_dir, "mlp_training_history.csv"), index=False
        )

    return df_comp, detailed_results


if __name__ == "__main__":
    df_comp, _ = run_model_comparison()
    print("\n=== CONSOLIDATED MODEL COMPARISON TABLE ===")
    cols = [
        "Model",
        "CV Macro F1-score",
        "Top-1 Accuracy",
        "Route Continuity",
        "Student-Test Macro F1-score",
        "Precision (Macro)",
        "Recall (Macro)",
        "Log Loss",
        "Inference Time (s)",
    ]
    print(df_comp[cols].to_string(index=False))

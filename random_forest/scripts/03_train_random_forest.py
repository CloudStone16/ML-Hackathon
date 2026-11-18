import os
import sys

# Ensure random_forest root is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if RF_ROOT not in sys.path:
    sys.path.insert(0, RF_ROOT)

import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from src.config import DATA_OUTPUT_DIR, MODEL_OUTPUT_DIR, OUTPUTS_DIR, FEATURE_COLUMNS, RANDOM_STATE
from src.metrics import compute_all_metrics

def train_and_evaluate():
    print("=" * 60)
    print("TRAINING RANDOM FOREST BASELINE MODEL")
    print("=" * 60)

    train_path = os.path.join(DATA_OUTPUT_DIR, "train_features.parquet")
    test_path = os.path.join(DATA_OUTPUT_DIR, "test_features.parquet")

    print(f"Loading data from {train_path}...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Train set: {len(train_df)} rows, {train_df['trip_id'].nunique()} trips")
    print(f"Test set: {len(test_df)} rows, {test_df['trip_id'].nunique()} trips")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].values
    groups = train_df["trip_id"].values

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].values

    # 1. Group-aware 5-Fold Cross Validation
    print("\nRunning 5-Fold GroupKFold Cross-Validation on 80 training trips...")
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(train_df))
    cv_metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
        fold_X_tr, fold_y_tr = X_train.iloc[train_idx], y_train[train_idx]
        fold_X_val, fold_y_val = X_train.iloc[val_idx], y_train[val_idx]

        rf_fold = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE + fold,
            n_jobs=-1
        )
        rf_fold.fit(fold_X_tr, fold_y_tr)
        val_probs = rf_fold.predict_proba(fold_X_val)[:, 1]
        oof_preds[val_idx] = val_probs

        val_df_fold = train_df.iloc[val_idx].copy()
        val_df_fold["pred_prob"] = val_probs
        fold_m = compute_all_metrics(val_df_fold, FEATURE_COLUMNS)
        cv_metrics_list.append(fold_m)
        print(f"Fold {fold + 1}: Top-1 Acc = {fold_m['top1_accuracy'] * 100:.2f}%, Macro F1 = {fold_m['macro_f1']:.4f}, Log Loss = {fold_m['log_loss']:.4f}")

    cv_top1 = np.mean([m["top1_accuracy"] for m in cv_metrics_list])
    cv_f1 = np.mean([m["macro_f1"] for m in cv_metrics_list])
    cv_logloss = np.mean([m["log_loss"] for m in cv_metrics_list])

    print("\n--- Cross-Validation Averages ---")
    print(f"Mean CV Top-1 Accuracy: {cv_top1 * 100:.2f}%")
    print(f"Mean CV Macro F1:       {cv_f1:.4f}")
    print(f"Mean CV Log Loss:       {cv_logloss:.4f}")

    # 2. Fit Full Model on all 80 Training Trips
    print("\nFitting final baseline model on all 80 training trips...")
    final_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    t_tr0 = time.time()
    final_rf.fit(X_train, y_train)
    t_tr1 = time.time()
    training_time = t_tr1 - t_tr0
    print(f"Trained in {training_time:.2f} seconds")

    # 3. Evaluate on 10 Student-Testing Trips
    t_inf0 = time.time()
    test_probs = final_rf.predict_proba(X_test)[:, 1]
    t_inf1 = time.time()
    inference_time = t_inf1 - t_inf0
    inference_per_sample_ms = (inference_time / len(X_test)) * 1000.0

    test_df_eval = test_df.copy()
    test_df_eval["pred_prob"] = test_probs
    test_metrics = compute_all_metrics(test_df_eval, FEATURE_COLUMNS)

    print("\n--- Student-Testing Results ---")
    print(f"Student-Test Top-1 Accuracy: {test_metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Student-Test Macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"Student-Test Precision:      {test_metrics['precision']:.4f}")
    print(f"Student-Test Recall:         {test_metrics['recall']:.4f}")
    print(f"Student-Test Log Loss:       {test_metrics['log_loss']:.4f}")
    print(f"Candidate Miss Rate:         {test_metrics['candidate_miss_rate']:.2f}%")
    print(f"Total Inference Time:        {inference_time:.4f}s ({inference_per_sample_ms:.3f} ms/candidate)")

    # 4. Feature Importances
    importances = dict(zip(FEATURE_COLUMNS, [round(float(x), 4) for x in final_rf.feature_importances_]))
    print("\nFeature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:15s}: {imp * 100:.2f}%")

    # Save artifacts
    model_path = os.path.join(MODEL_OUTPUT_DIR, "random_forest_baseline.joblib")
    joblib.dump(final_rf, model_path, compress=3)
    print(f"\nSaved baseline model to {model_path}")

    results = {
        "model_name": "Random Forest (Baseline)",
        "algorithm_family": "Ensemble (Bagging)",
        "key_configuration": {
            "n_estimators": 150,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced"
        },
        "cv_top1_accuracy": round(float(cv_top1), 4),
        "cv_macro_f1": round(float(cv_f1), 4),
        "cv_log_loss": round(float(cv_logloss), 4),
        "test_top1_accuracy": round(float(test_metrics["top1_accuracy"]), 4),
        "test_macro_f1": round(float(test_metrics["macro_f1"]), 4),
        "test_precision": round(float(test_metrics["precision"]), 4),
        "test_recall": round(float(test_metrics["recall"]), 4),
        "test_log_loss": round(float(test_metrics["log_loss"]), 4),
        "candidate_miss_rate": round(float(test_metrics["candidate_miss_rate"]), 4),
        "training_time_sec": round(float(training_time), 2),
        "inference_time_sec": round(float(inference_time), 4),
        "feature_importances": importances
    }

    results_path = os.path.join(OUTPUTS_DIR, "baseline_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved baseline metrics to {results_path}")

if __name__ == "__main__":
    train_and_evaluate()

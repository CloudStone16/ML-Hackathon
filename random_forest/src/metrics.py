import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score

def compute_observation_top1_accuracy(df_preds):
    """
    Computes observation-level Top-1 accuracy.
    df_preds must contain ['obs_id', 'pred_prob', 'label'].
    """
    correct = 0
    total = 0
    for obs_id, group in df_preds.groupby("obs_id"):
        top_cand = group.loc[group["pred_prob"].idxmax()]
        if top_cand["label"] == 1:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def compute_candidate_miss_rate(df_candidates):
    """
    Computes candidate miss rate: percentage of observations where NO candidate has label=1.
    """
    missed = 0
    total = 0
    for obs_id, group in df_candidates.groupby("obs_id"):
        if (group["label"] == 1).sum() == 0:
            missed += 1
        total += 1
    return (missed / total) * 100.0 if total > 0 else 0.0

def compute_route_continuity(predicted_arcs, arcs_array):
    """
    Computes route continuity: fraction of consecutive steps where either:
    1) arc[t] == arc[t+1] (vehicle stays on same segment)
    2) arc[t].to_node == arc[t+1].from_node (consecutive segment in road network)
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    continuous_steps = 0
    total_steps = len(predicted_arcs) - 1

    for t in range(total_steps):
        a1 = predicted_arcs[t]
        a2 = predicted_arcs[t + 1]

        if a1 == a2:
            continuous_steps += 1
        else:
            v1 = arcs_array[a1, 1]
            u2 = arcs_array[a2, 0]
            if v1 == u2:
                continuous_steps += 1

    return continuous_steps / total_steps if total_steps > 0 else 1.0

def compute_all_metrics(df_preds, feature_cols, arcs_array=None):
    """
    Comprehensive evaluation metrics reporting.
    """
    y_true = df_preds["label"].values
    y_prob = df_preds["pred_prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    top1_acc = compute_observation_top1_accuracy(df_preds)
    miss_rate = compute_candidate_miss_rate(df_preds)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        loss = log_loss(y_true, y_prob)
    except Exception:
        loss = float("nan")

    # Sequence level continuity for top-1 predictions
    continuity = float("nan")
    if arcs_array is not None and "obs_id" in df_preds.columns:
        seq_top1 = []
        for obs_id, group in df_preds.groupby("obs_id"):
            top_cand = group.loc[group["pred_prob"].idxmax()]
            seq_top1.append(int(top_cand["arc_id"]))
        continuity = compute_route_continuity(seq_top1, arcs_array)

    return {
        "top1_accuracy": top1_acc,
        "macro_f1": f1,
        "precision": prec,
        "recall": rec,
        "log_loss": loss,
        "candidate_miss_rate": miss_rate,
        "route_continuity": continuity
    }

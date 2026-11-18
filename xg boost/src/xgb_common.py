"""Shared helpers for the XGBoost model: data loading and metrics."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "processed")
RES = os.path.join(HERE, "..", "results")
MODELS = os.path.join(HERE, "..", "models")
for _d in (RES, MODELS):
    os.makedirs(_d, exist_ok=True)

SEED = 42
# Trips whose GPS is ~130 m off the reference route (median distance to labelled arc 139 m and 122 m):
# the labels are unreliable, so they are excluded from fitting and cross-validation.
BAD_TRIPS = {"00000054", "00000033"}
FEATS = ["dist", "d_gap", "rank", "n_cand", "t_frac", "along_m", "arc_len", "lateral", "heading_cos",
         "heading_ok", "speed", "dt_prev", "dt_next", "out_deg_to", "in_deg_from"]


def load_data():
    """Return (train_df, test_df, points_per_trip) with trip IDs kept as strings."""
    df = pd.read_csv(os.path.join(PROC, "candidates.csv.gz"), dtype={"trip": str})
    P = pd.read_csv(os.path.join(PROC, "trip_points.csv"), dtype={"trip": str})
    train = df[(df.part == "train") & (~df.trip.isin(BAD_TRIPS))].reset_index(drop=True)
    test = df[df.part == "student_test"].reset_index(drop=True)
    assert not (set(train.trip) & BAD_TRIPS)
    assert not (set(train.trip) & set(test.trip))
    return train, test, dict(zip(P.trip, P.n_points))


def point_key(d):
    return d["trip"].astype("category").cat.codes.astype("int64") * 1_000_000 + d["pt"].astype("int64")


def evaluate(d, p, n_points_total, threshold=0.5):
    """d: candidate rows (trip, pt, y); p: predicted probability per row.

    Top-1 is observation-level: for each GPS point take the highest-scoring candidate. Points whose true arc is
    not among the candidates count as wrong (so the candidate miss rate caps the score)."""
    d = d[["trip", "pt", "y"]].copy().reset_index(drop=True)
    d["p"] = np.asarray(p)
    d["k"] = point_key(d).values
    best = d.loc[d.groupby("k")["p"].idxmax()]
    n_cand_points = d["k"].nunique()
    hit_points = d.groupby("k")["y"].max().sum()
    pred = (d["p"] >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(d["y"], pred, average="macro", zero_division=0)
    return dict(
        top1_acc=float(best["y"].sum() / n_points_total),
        top1_acc_on_hit_points=float(best["y"].sum() / max(hit_points, 1)),
        candidate_miss_rate=float(1 - hit_points / n_points_total),
        log_loss=float(log_loss(d["y"], np.clip(d["p"], 1e-6, 1 - 1e-6))),
        row_accuracy=float((pred == d["y"]).mean()),
        precision_macro=float(prec), recall_macro=float(rec), f1_macro=float(f1),
        confusion_matrix=confusion_matrix(d["y"], pred).tolist(),
        n_points=int(n_points_total), n_rows=int(len(d)),
        points_without_candidates=int(n_points_total - n_cand_points),
    )

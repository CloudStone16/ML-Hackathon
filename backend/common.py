"""Helpers shared by the model pipelines."""
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TRIP_ID = re.compile(r"^\d{1,12}$")


class TripStore:
    """Lists trips and reads PREDICTION inputs only (.track, .nodes, .arcs). The .route file is never read here."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def trip_ids(self):
        if not self.data_dir.is_dir():
            return []
        return sorted(p.name for p in self.data_dir.iterdir()
                      if p.is_dir() and TRIP_ID.match(p.name) and (p / f"{p.name}.track").exists())

    def n_points(self, tid):
        with open(self.data_dir / tid / f"{tid}.track") as f:
            return sum(1 for _ in f)

    def read_inputs(self, tid):
        d = self.data_dir / tid
        return {
            "id": tid,
            "track": np.loadtxt(d / f"{tid}.track", ndmin=2),
            "nodes": np.loadtxt(d / f"{tid}.nodes", ndmin=2),
            "arcs": np.loadtxt(d / f"{tid}.arcs", dtype=np.int64, ndmin=2),
            "route": None,
        }

    def read_route(self, tid):
        """Hidden reference route. Only the server-side scoring step calls this, after predicting."""
        p = self.data_dir / tid / f"{tid}.route"
        return np.loadtxt(p, dtype=np.int64, ndmin=1) if p.exists() else None


def fill_missing(pred):
    """Points with no candidate are -1: carry the previous arc forward (back-fill at the start)."""
    pred = pred.copy()
    valid = np.flatnonzero(pred >= 0)
    if len(valid) == 0:
        return pred
    idx = np.searchsorted(valid, np.arange(len(pred)), side="right") - 1
    idx = np.clip(idx, 0, len(valid) - 1)
    return pred[valid[idx]]


def geometry(track, nodes, arcs, pred_arcs):
    """Everything the UI needs to draw the result (no ground truth). track columns: lon, lat, t."""
    va = pred_arcs
    seq = va[np.r_[True, va[1:] != va[:-1]]]
    a, b = arcs[seq, 0], arcs[seq, 1]
    segments = np.stack([np.stack([nodes[a, 1], nodes[a, 0]], -1),
                         np.stack([nodes[b, 1], nodes[b, 0]], -1)], 1)
    return {
        "gps_points": np.round(track[:, [1, 0]], 6).tolist(),
        "predicted_arc_ids": va.astype(int).tolist(),
        "predicted_route": np.round(segments, 6).tolist(),
        "n_route_segments": int(len(seq)),
    }

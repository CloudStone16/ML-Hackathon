"""LightGBM inference pipeline: raw trip files -> candidates/features -> LightGBM -> HMM/Viterbi -> route.

predict() uses only .track / .nodes / .arcs. score() reads the hidden .route afterwards, server-side only.
Uses the team's LightGBM feature code (src/pathsnap) and trained model (results/lightgbm_model.joblib).
"""
import json
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np

from backend import decoder
from backend.common import REPO_ROOT, TripStore, fill_missing, geometry

for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hmm_viterbi.viterbi_decoder import compute_sequence_metrics  # noqa: E402
from pathsnap.features import build_trip  # noqa: E402
from pathsnap.geo import LocalProjection  # noqa: E402
from pathsnap.io import clean_track  # noqa: E402
from pathsnap.labeling import align_to_route  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.track import kinematics  # noqa: E402

LABEL_MAX_DIST_M = 60.0     # same validity rule the LightGBM training script uses
DECODER_CONFIG = REPO_ROOT / "backend" / "decoder_config.json"


class LightGBMPipeline:
    def __init__(self, model_dir, data_dir):
        self.model_dir = Path(model_dir)
        self.store = TripStore(data_dir)
        self.data_dir = self.store.data_dir
        self._lock = threading.Lock()
        for need in (self.model_dir / "lightgbm_model.joblib", self.model_dir / "config.json"):
            if not need.exists():
                raise FileNotFoundError(f"Missing model artifact: {need}")
        self.model = joblib.load(self.model_dir / "lightgbm_model.joblib")
        with open(self.model_dir / "config.json") as f:
            self.cfg = json.load(f)
        self.features = self.cfg["features"]
        if DECODER_CONFIG.exists():
            with open(DECODER_CONFIG) as f:
                dc = json.load(f)
            self.decoder_params, self.top_k = dc["params"], int(dc.get("top_k", 10))
        else:
            self.decoder_params, self.top_k = dict(decoder.DEFAULT_PARAMS), 10

    def model_info(self):
        return {
            "name": "LightGBM + HMM/Viterbi",
            "model": "LightGBM gradient-boosted trees",
            "hyperparameters": self.cfg.get("lightgbm_params", {}),
            "n_features": len(self.features),
            "candidates": {"radius_m": self.cfg.get("radius_m"), "max_per_point": self.cfg.get("max_cands")},
            "decoder": dict(self.decoder_params, top_k=self.top_k),
        }

    def trip_ids(self):
        return self.store.trip_ids()

    def n_points(self, tid):
        return self.store.n_points(tid)

    def predict(self, tid):
        t0 = time.perf_counter()
        trip = self.store.read_inputs(tid)
        df, _ = build_trip(trip, with_labels=False)
        track, _ = clean_track(trip["track"])
        t1 = time.perf_counter()
        with self._lock:
            p = self.model.predict_proba(df[self.features])[:, 1]
        t2 = time.perf_counter()
        t = decoder.prepare(len(track), track[:, 2], trip["arcs"], len(trip["nodes"]),
                            df["pt"].to_numpy(), df["arc"].to_numpy(), p, top_k=self.top_k)
        hmm = decoder.viterbi(t, **self.decoder_params)
        clf = decoder.classifier_only(t)
        t3 = time.perf_counter()
        return {"trip": trip, "track": track, "hmm": hmm, "clf": clf,
                "timings_s": {"features": t1 - t0, "model": t2 - t1, "viterbi": t3 - t2, "total": t3 - t0}}

    def score(self, tid, res):
        """Server-side evaluation against the hidden reference route. Returns metrics only."""
        route = self.store.read_route(tid)
        if route is None:
            return None
        trip, track = res["trip"], res["track"]
        lonlat = trip["nodes"]
        proj = LocalProjection(lonlat[:, 0].mean(), lonlat[:, 1].mean())
        net = RoadNetwork(lonlat, trip["arcs"], proj)
        P = proj.to_xy(track[:, :2])
        label_arc, label_d = align_to_route(P, kinematics(P, track[:, 2]), net, route)
        valid = label_d <= LABEL_MAX_DIST_M
        out = {}
        for key, pred in (("hmm_viterbi", res["hmm"]), ("classifier_only", res["clf"])):
            pf = fill_missing(pred)
            m = compute_sequence_metrics(pf, label_arc, route, trip["arcs"])
            m["top1_accuracy_valid_labels"] = float(np.mean(pf[valid] == label_arc[valid])) if valid.any() else float("nan")
            out[key] = {k: float(v) for k, v in m.items()}
        out["reference_route_arcs"] = int(len(route))
        return out

    @staticmethod
    def geometry(res):
        return geometry(res["track"], res["trip"]["nodes"], res["trip"]["arcs"], fill_missing(res["hmm"]))

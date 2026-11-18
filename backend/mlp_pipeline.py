"""PathSnap inference pipeline: raw trip files -> candidates/features -> MLP -> HMM/Viterbi -> route.

Separation of concerns (evaluation rules):
  * predict()  reads ONLY .track / .nodes / .arcs. It never opens the .route (ground-truth) file.
  * score()    is server-side only: it reads .route after the prediction is made and returns metrics.
                Ground-truth labels are never passed to the model and never returned to the UI.
"""
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.eda_and_preprocessing import clean_and_project_trip  # noqa: E402
from feature_engineering.candidate_features import (  # noqa: E402
    align_ground_truth_route,
    extract_trip_candidates_and_features,
)
from hmm_viterbi.viterbi_decoder import PathSnapViterbiDecoder, compute_sequence_metrics  # noqa: E402
from models.mlp_model import PathSnapMLPClassifier, PathSnapMLPNet  # noqa: E402

TRIP_ID = re.compile(r"^\d{1,12}$")


class MLPPipeline:
    def __init__(self, model_dir, data_dir):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._load_model()

    # ------------------------------------------------------------------ model
    def _load_model(self):
        cfg_path = self.model_dir / "model_config.json"
        for need in (cfg_path, self.model_dir / "pathsnap_mlp_weights.pt", self.model_dir / "feature_scaler.joblib"):
            if not need.exists():
                raise FileNotFoundError(
                    f"Missing model artifact: {need}\n"
                    "Copy pathsnap_mlp_weights.pt, feature_scaler.joblib and model_config.json (written by "
                    "models/train_mlp.py into saved_models/) into PATHSNAP_MODEL_DIR."
                )
        with open(cfg_path) as f:
            self.cfg = json.load(f)
        hp = self.cfg["best_bayesian_hyperparameters"]
        self.n_features = len(self.cfg["feature_names"])
        self.K = int(self.cfg.get("K", 5))

        self.clf = PathSnapMLPClassifier(hidden_dim=hp["hidden_dim"], head_dim=hp["head_dim"], dropout=hp["dropout"])
        self.clf.model = PathSnapMLPNet(in_features=self.n_features, hidden_dim=hp["hidden_dim"],
                                        head_dim=hp["head_dim"], dropout=hp["dropout"])
        self.clf.model.load_state_dict(
            torch.load(self.model_dir / "pathsnap_mlp_weights.pt", map_location="cpu", weights_only=True))
        self.clf.model.eval()
        self.clf.scaler = joblib.load(self.model_dir / "feature_scaler.joblib")
        self.clf.threshold = float(self.cfg.get("decision_threshold", 0.5))
        self.decoder = PathSnapViterbiDecoder(**self.cfg["hmm_viterbi_config"])

    def model_info(self):
        hp = self.cfg["best_bayesian_hyperparameters"]
        return {
            "name": self.cfg.get("model_name", "PathSnap Deep MLP"),
            "hyperparameters": hp,
            "n_features": self.n_features,
            "candidates_per_point": self.K,
            "decoder": self.cfg["hmm_viterbi_config"],
        }

    # ------------------------------------------------------------------ data
    def trip_ids(self):
        if not self.data_dir.is_dir():
            return []
        return sorted(p.name for p in self.data_dir.iterdir()
                      if p.is_dir() and TRIP_ID.match(p.name) and (p / f"{p.name}.track").exists())

    def _read_inputs(self, tid):
        """Prediction inputs only. The .route file is deliberately NOT read here."""
        d = self.data_dir / tid
        return {
            "trip_id": tid,
            "track": np.loadtxt(d / f"{tid}.track", dtype=np.float64, ndmin=2),
            "nodes": np.loadtxt(d / f"{tid}.nodes", dtype=np.float64, ndmin=2),
            "arcs": np.loadtxt(d / f"{tid}.arcs", dtype=np.int32, ndmin=2),
            "route": None,
        }

    def n_points(self, tid):
        with open(self.data_dir / tid / f"{tid}.track") as f:
            return sum(1 for _ in f)

    # --------------------------------------------------------------- predict
    def predict(self, tid):
        t0 = time.perf_counter()
        proc = clean_and_project_trip(self._read_inputs(tid))
        t1 = time.perf_counter()
        bundle = extract_trip_candidates_and_features(proc, K=self.K)
        if bundle["X"].shape[-1] != self.n_features:
            raise RuntimeError(f"Feature count {bundle['X'].shape[-1]} != model expects {self.n_features}")
        t2 = time.perf_counter()
        with self._lock:
            probs = self.clf.predict_proba_trip(bundle)
        t3 = time.perf_counter()
        dec = self.decoder.decode_trip(bundle, probs)
        t4 = time.perf_counter()
        return {
            "proc": proc, "bundle": bundle, "dec": dec,
            "timings_s": {"preprocess": t1 - t0, "features": t2 - t1, "model": t3 - t2, "viterbi": t4 - t3,
                          "total": t4 - t0},
        }

    # ----------------------------------------------------------------- score
    def score(self, tid, res):
        """Server-side evaluation against the hidden reference route. Returns metrics only."""
        route_path = self.data_dir / tid / f"{tid}.route"
        if not route_path.exists():
            return None
        route = np.loadtxt(route_path, dtype=np.int32, ndmin=1)
        proc = res["proc"]
        gt = align_ground_truth_route(proc["tx"], proc["ty"], proc["nx"], proc["ny"], proc["arcs"], route)
        arcs = proc["arcs"]
        hmm = compute_sequence_metrics(res["dec"]["viterbi_arcs"], gt, route, arcs)
        clf = compute_sequence_metrics(res["dec"]["classifier_arcs"], gt, route, arcs)
        return {
            "hmm_viterbi": {k: float(v) for k, v in hmm.items()},
            "classifier_only": {k: float(v) for k, v in clf.items()},
            "reference_route_arcs": int(len(route)),
        }

    # -------------------------------------------------------------- geometry
    @staticmethod
    def geometry(res):
        """Everything the UI needs to draw the result (no ground truth)."""
        proc = res["proc"]
        nodes, arcs = proc["nodes"], proc["arcs"]
        va = res["dec"]["viterbi_arcs"]
        seq = va[np.r_[True, va[1:] != va[:-1]]]                      # collapse repeated arcs
        a, b = arcs[seq, 0], arcs[seq, 1]
        segments = np.stack([np.stack([nodes[a, 1], nodes[a, 0]], -1),
                             np.stack([nodes[b, 1], nodes[b, 0]], -1)], 1)   # (S, 2, [lat, lon])
        return {
            "gps_points": np.round(proc["track"][:, [1, 0]], 6).tolist(),     # [lat, lon]
            "predicted_arc_ids": va.astype(int).tolist(),                     # one arc per GPS point
            "predicted_route": np.round(segments, 6).tolist(),                # polyline segments [lat, lon]
            "n_route_segments": int(len(seq)),
        }

"""Score the tuned LightGBM / XGBoost / MLP on evaluator_test and student_test with the shared truth.

Reporting only: the tuned parameters were chosen on train-trip CV (see scripts/tune_*.py), never on these sets.
Reuses the runners and metrics in eval_evaluator_test.py.

Usage: PATHSNAP_DATA=<dir with train/ student_test/ evaluator_test/> uv run python scripts/eval_tuned_models.py
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "xg boost" / "src"))
TUNE = ROOT / "results" / "tuning"


def score_split(split, which):
    os.environ["PATHSNAP_SPLIT"] = split
    for m in [m for m in list(sys.modules) if m == "eval_evaluator_test"]:
        del sys.modules[m]
    E = __import__("eval_evaluator_test")
    truth = E.common_truth()
    rows = []

    def add(name, stage, cand=None, point=None):
        pick = E.top_pick(cand) if point is None else point
        rows.append({"split": split, "model": name, "stage": stage, **E.metrics(truth, pick, cand if point is None else None)})

    if which == "lightgbm":
        _lightgbm(E, split, truth, add)
    elif which == "xgboost":
        _xgboost(E, split, truth, add)
    else:
        _mlp(E, split, truth, add)
    return rows


def _lightgbm(E, split, truth, add):
    import pathsnap.io as pio
    from pathsnap.features import FEATURES, build_trip
    pio.DATA_ROOT = E.DATA
    fp = E.PRED_DIR / "features.parquet"
    if not fp.exists():
        pd.concat([build_trip(pio.load_trip(split, t), with_labels=False)[0] for t in E.trips()]).to_parquet(fp)
    df = pd.read_parquet(fp)
    m = joblib.load(TUNE / "lightgbm_tuned_model.joblib")
    add("LightGBM (tuned)", "classifier only",
        cand=pd.DataFrame({"trip": df["trip"], "pt": df["pt"], "arc": df["arc"], "p": m.predict_proba(df[FEATURES])[:, 1]}))



def _xgboost(E, split, truth, add):
    import hmm_decoder as hd
    import preprocess as xp
    import xgboost as xgb
    from xgb_common import FEATS
    xp.ROOT = str(E.DATA)
    model = xgb.XGBClassifier()
    model.load_model(str(TUNE / "xgboost_retuned.json"))
    frames = []
    for tid in E.trips():
        track, nodes, arcs, _ = xp.load_trip(split, tid)
        f = xp.build_features(track, nodes, arcs, R=50.0, K=20)
        f.insert(0, "trip", tid)
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    cand = pd.DataFrame({"trip": d["trip"], "pt": d["pt"], "arc": d["arc"], "p": model.predict_proba(d[FEATS])[:, 1]})
    add("XGBoost (re-tuned)", "classifier only", cand=cand)
    cfg = json.loads((ROOT / "xg boost" / "models" / "hmm_config.json").read_text())
    hd.MAXHOPS = cfg.get("maxhops", hd.MAXHOPS)
    c = cand.merge(truth[["trip", "pt", "truth_arc"]], on=["trip", "pt"])
    c["y"] = (c["arc"] == c["truth_arc"]).astype(int)
    c = c.sort_values(["trip", "pt"], kind="stable")
    pts = []
    for tid, g in c.groupby("trip", sort=True):
        pred = hd.viterbi(hd.prep_trip(split, tid, g.reset_index(drop=True)), **cfg["params"])
        pts.append(pd.DataFrame({"trip": tid, "pt": np.arange(len(pred)), "pred_arc": pred}))
    add("XGBoost (re-tuned)", "classifier + HMM/Viterbi", point=pd.concat(pts, ignore_index=True))



def _mlp(E, split, truth, add):
    import torch
    torch.set_num_threads(8)
    from feature_engineering.candidate_features import build_or_load_partition_dataset
    from hmm_viterbi.viterbi_decoder import PathSnapViterbiDecoder
    clf = joblib.load(TUNE / "mlp_retuned_classifier.joblib")
    bundles = build_or_load_partition_dataset(str(E.DATA), split, K=5, cache_dir=str(ROOT / "artifacts" / "mlp_cache"))
    dec = PathSnapViterbiDecoder(emission_weight=1.0, same_arc_bonus=0.0, connected_cost=-0.35, two_hop_cost=-1.60,
                                 disconnected_cost=-4.80, reverse_arc_cost=-6.50, gap_relax_rate=0.30)
    crow, prow = [], []
    for b in bundles:
        pr = clf.predict_proba_trip(b)
        ca = np.asarray(b["cand_arcs"])
        n, k = ca.shape
        keep = ca.ravel() >= 0
        crow.append(pd.DataFrame({"trip": b["trip_id"], "pt": np.repeat(np.arange(n), k)[keep], "arc": ca.ravel()[keep],
                                  "p": np.asarray(pr).ravel()[keep]}))
        prow.append(pd.DataFrame({"trip": b["trip_id"], "pt": np.arange(n), "pred_arc": dec.decode_trip(b, pr)["viterbi_arcs"]}))
    add("Deep MLP (re-tuned)", "classifier only", cand=pd.concat(crow, ignore_index=True))
    add("Deep MLP (re-tuned)", "classifier + HMM/Viterbi", point=pd.concat(prow, ignore_index=True))


if __name__ == "__main__":
    import subprocess
    if len(sys.argv) == 4 and sys.argv[1] == "one":      # one model x one split, in its own process (OpenMP clash)
        _, _, split, which = sys.argv
        pd.DataFrame(score_split(split, which)).to_csv(TUNE / f"_tmp_{split}_{which}.csv", index=False)
    else:
        parts = []
        for split in ("student_test", "evaluator_test"):
            for which in ("lightgbm", "xgboost", "mlp"):
                subprocess.run([sys.executable, __file__, "one", split, which], check=True, cwd=ROOT)
                parts.append(pd.read_csv(TUNE / f"_tmp_{split}_{which}.csv"))
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(TUNE / "tuned_models_scores.csv", index=False)
        for f in TUNE.glob("_tmp_*.csv"):
            f.unlink()
        print(out[["split", "model", "stage", "top1_acc", "route_continuity", "macro_f1", "log_loss"]].round(4).to_string(index=False))

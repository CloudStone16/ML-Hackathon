"""LightGBM (untuned and tuned) with and without HMM/Viterbi decoding, on student_test and evaluator_test.

The decoder is the teammate's (xg boost/src/hmm_decoder.py) with its saved parameters (xg boost/models/hmm_config.json).
Nothing is re-tuned on either test set. Scoring uses the shared per-point truth from eval_evaluator_test.py.

Usage: PATHSNAP_DATA=<dir with student_test/ evaluator_test/> uv run python scripts/eval_lightgbm_hmm.py
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "xg boost" / "src"))
OUT = ROOT / "results" / "lightgbm_hmm"


def run(split):
    os.environ["PATHSNAP_SPLIT"] = split
    sys.modules.pop("eval_evaluator_test", None)
    E = __import__("eval_evaluator_test")
    import hmm_decoder as hd
    import preprocess as xp
    from pathsnap.features import FEATURES
    xp.ROOT = str(E.DATA)
    cfg = json.loads((ROOT / "xg boost" / "models" / "hmm_config.json").read_text())
    hd.MAXHOPS = cfg.get("maxhops", hd.MAXHOPS)
    truth = E.common_truth()
    df = pd.read_parquet(E.PRED_DIR / "features.parquet")
    models = {"LightGBM (untuned)": joblib.load(ROOT / "results" / "lightgbm_model.joblib"),
              "LightGBM (tuned)": joblib.load(ROOT / "results" / "tuning" / "lightgbm_tuned_model.joblib")}
    rows = []
    for name, m in models.items():
        cand = pd.DataFrame({"trip": df["trip"], "pt": df["pt"], "arc": df["arc"],
                             "p": m.predict_proba(df[FEATURES])[:, 1]})
        rows.append({"split": split, "model": name, "stage": "classifier only", **E.metrics(truth, E.top_pick(cand), cand)})
        c = cand.merge(truth[["trip", "pt", "truth_arc"]], on=["trip", "pt"])
        c["y"] = (c["arc"] == c["truth_arc"]).astype(int)       # y only feeds the decoder's scoring helper
        c = c.sort_values(["trip", "pt"], kind="stable")
        pts = []
        for tid, g in c.groupby("trip", sort=True):
            pred = hd.viterbi(hd.prep_trip(split, tid, g.reset_index(drop=True)), **cfg["params"])
            pts.append(pd.DataFrame({"trip": tid, "pt": np.arange(len(pred)), "pred_arc": pred}))
        rows.append({"split": split, "model": name, "stage": "classifier + HMM/Viterbi",
                     **E.metrics(truth, pd.concat(pts, ignore_index=True))})
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([r for s in ("student_test", "evaluator_test") for r in run(s)])
    out.to_csv(OUT / "scores.csv", index=False)
    print(out[["split", "model", "stage", "top1_acc", "route_continuity"]].round(4).to_string(index=False))

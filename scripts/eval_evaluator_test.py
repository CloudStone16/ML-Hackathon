"""Run every kept model on the evaluator_test trips and score them with one common metric.

Each model keeps its OWN preprocessing, candidate generation and trained weights (nothing is re-tuned).
All models are then scored against the SAME per-point truth, so the numbers are comparable:

  truth(trip, point) = route arc aligned to the point with pathsnap.labeling.align_to_route
  top1  = share of GPS points whose highest-scoring candidate arc == truth (no candidate = wrong)

Usage (from repo root, each model runs in its own process to avoid module-name clashes):
  uv run python scripts/eval_evaluator_test.py run <model>     # lightgbm | logreg | rf | xgboost | mlp
  uv run python scripts/eval_evaluator_test.py score           # merge predictions -> results/evaluator_test/
  uv run python scripts/eval_evaluator_test.py all
Data layout expected under $PATHSNAP_DATA: train/, student_test/, evaluator_test/ (symlinks are fine).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PATHSNAP_DATA", ROOT / "student_dataset"))
SPLIT = os.environ.get("PATHSNAP_SPLIT", "evaluator_test")   # evaluator_test | student_test
EVAL_DIR = DATA / SPLIT
PRED_DIR = ROOT / "artifacts" / ("evaluator_preds" if SPLIT == "evaluator_test" else f"{SPLIT}_preds")
OUT_DIR = ROOT / "results" / SPLIT
MODELS = ["lightgbm", "logreg", "rf", "xgboost", "mlp"]
TEAM_THREADS = 8


# ---------------------------------------------------------------- common truth + scoring
def trips():
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir())


def common_truth():
    """DataFrame (trip, pt, truth_arc) from the shared labeller; cached."""
    path = PRED_DIR / "truth.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sys.path.insert(0, str(ROOT / "src"))
    import pathsnap.io as pio
    from pathsnap.geo import LocalProjection
    from pathsnap.labeling import align_to_route
    from pathsnap.network import RoadNetwork
    from pathsnap.track import kinematics
    pio.DATA_ROOT = DATA
    rows = []
    for tid in trips():
        tr = pio.load_trip(SPLIT, tid)
        track, _ = pio.clean_track(tr["track"])
        ll = tr["nodes"]
        proj = LocalProjection(ll[:, 0].mean(), ll[:, 1].mean())
        net = RoadNetwork(ll, tr["arcs"], proj)
        P = proj.to_xy(track[:, :2])
        arc, dist = align_to_route(P, kinematics(P, track[:, 2]), net, tr["route"])
        rows.append(pd.DataFrame({"trip": tid, "pt": np.arange(len(P)), "truth_arc": arc, "truth_dist": dist}))
    df = pd.concat(rows, ignore_index=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def save_preds(name, cand=None, point=None, seconds=None):
    """cand: DataFrame(trip, pt, arc, p) candidate scores. point: DataFrame(trip, pt, pred_arc) decoded arcs."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    if cand is not None:
        cand.to_parquet(PRED_DIR / f"{name}_cand.parquet")
    if point is not None:
        point.to_parquet(PRED_DIR / f"{name}_point.parquet")
    if seconds is not None:
        (PRED_DIR / f"{name}_time.json").write_text(json.dumps(seconds))


def top_pick(cand):
    c = cand.sort_values(["trip", "pt", "p"], ascending=[True, True, False])
    return c.drop_duplicates(["trip", "pt"])[["trip", "pt", "arc"]].rename(columns={"arc": "pred_arc"})


def arcs_of(tid):
    return np.loadtxt(EVAL_DIR / tid / f"{tid}.arcs", dtype=np.int64, ndmin=2)


def metrics(truth, pick, cand=None):
    """truth: common_truth(); pick: (trip, pt, pred_arc); cand: optional candidate scores for F1/log loss."""
    from sklearn.metrics import f1_score, log_loss
    m = truth.merge(pick, on=["trip", "pt"], how="left")
    m["pred_arc"] = m["pred_arc"].fillna(-1).astype(int)
    m["ok"] = m["pred_arc"] == m["truth_arc"]
    out = {"n_points": len(m), "top1_acc": float(m["ok"].mean()),
           "mean_trip_top1": float(m.groupby("trip")["ok"].mean().mean())}
    cont_num = cont_den = 0
    for tid, g in m.groupby("trip"):
        a = arcs_of(tid)
        p = g.sort_values("pt")["pred_arc"].to_numpy()
        p1, p2 = p[:-1], p[1:]
        valid = (p1 >= 0) & (p2 >= 0)
        same = p1 == p2
        conn = np.zeros(len(p1), bool)
        v = valid & ~same
        conn[v] = a[p1[v], 1] == a[p2[v], 0]
        cont_num += int((same | conn)[valid].sum())
        cont_den += int(valid.sum())
    out["route_continuity"] = cont_num / max(cont_den, 1)
    if cand is not None:
        c = cand.merge(truth[["trip", "pt", "truth_arc"]], on=["trip", "pt"], how="inner")
        y = (c["arc"] == c["truth_arc"]).astype(int)
        p = c["p"].to_numpy(float)
        hit = c.assign(y=y).groupby(["trip", "pt"])["y"].max().sum()
        out["candidate_miss_rate"] = float(1 - hit / len(truth))
        out["macro_f1"] = float(f1_score(y, (p >= 0.5).astype(int), average="macro"))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    return out


# ---------------------------------------------------------------- per-model runners
def run_lightgbm(retrain_rf=False):
    """My LightGBM (and, for rf_mine, my Random Forest retrained with the same config) on my features."""
    import joblib
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import pathsnap.io as pio
    from pathsnap.features import FEATURES, build_trip
    pio.DATA_ROOT = DATA
    fpath = PRED_DIR / "features.parquet"
    if not fpath.exists():
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        frames = [build_trip(pio.load_trip(SPLIT, t), with_labels=False)[0] for t in trips()]
        pd.concat(frames, ignore_index=True).to_parquet(fpath)
    df = pd.read_parquet(fpath)
    if not retrain_rf:
        model = joblib.load(ROOT / "results" / "lightgbm_model.joblib")
        t0 = time.time()
        p = model.predict_proba(df[FEATURES])[:, 1]
        name = "lightgbm"
    else:
        from train_more_models import make
        tr = pd.read_parquet(ROOT / "artifacts" / "features_train.parquet")
        rep = pd.read_csv(ROOT / "results" / "preprocess_report_train.csv", dtype={"trip": str})
        rep["trip"] = rep["trip"].str.zfill(8)
        bad = set(rep.loc[rep["frac_label_invalid"] > 0.5, "trip"])
        tr = tr[~tr["trip"].isin(bad) & (tr["label_valid"] == 1)]
        model = make("random_forest")
        model.fit(tr[FEATURES], tr["label"].to_numpy())
        t0 = time.time()
        p = model.predict_proba(df[FEATURES])[:, 1]
        name = "rf_mine"
    cand = pd.DataFrame({"trip": df["trip"], "pt": df["pt"], "arc": df["arc"], "p": p})
    inf = time.time() - t0
    point = None
    if name == "lightgbm":
        # HMM/Viterbi with the decoder and saved parameters from xg boost/ (tuned on XGBoost out-of-fold train
        # probabilities only; not re-tuned for LightGBM). y only feeds the decoder's scoring helper.
        sys.path.insert(0, str(ROOT / "xg boost" / "src"))
        import hmm_decoder as hd
        import preprocess as xp
        xp.ROOT = str(DATA)
        cfg = json.loads((ROOT / "xg boost" / "models" / "hmm_config.json").read_text())
        hd.MAXHOPS = cfg.get("maxhops", hd.MAXHOPS)
        truth = common_truth()
        c = cand.merge(truth[["trip", "pt", "truth_arc"]], on=["trip", "pt"])
        c["y"] = (c["arc"] == c["truth_arc"]).astype(int)
        c = c.sort_values(["trip", "pt"], kind="stable")
        rows = []
        t1 = time.time()
        for tid, g in c.groupby("trip", sort=True):
            pred = hd.viterbi(hd.prep_trip(SPLIT, tid, g.reset_index(drop=True)), **cfg["params"])
            rows.append(pd.DataFrame({"trip": tid, "pt": np.arange(len(pred)), "pred_arc": pred}))
        point = pd.concat(rows, ignore_index=True)
        inf_hmm = time.time() - t1
    save_preds(name, cand=cand, point=point,
               seconds={"inference_s": inf, **({"hmm_s": inf_hmm} if point is not None else {})})


def run_logreg():
    """Logistic regression (standardised features) trained on the training trips, scored on the same features."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from pathsnap.features import FEATURES
    from train_other_models import make
    from train_lightgbm import load, BAD_TRIP_INVALID_FRAC
    run_lightgbm()                              # makes sure the feature table for this split exists
    df = pd.read_parquet(PRED_DIR / "features.parquet")
    tr, rep = load("train")
    bad = set(rep.loc[rep["frac_label_invalid"] > BAD_TRIP_INVALID_FRAC, "trip"])
    tr = tr[~tr["trip"].isin(bad) & (tr["label_valid"] == 1)]
    model = make("logreg").fit(tr[FEATURES], tr["label"].to_numpy())
    t0 = time.time()
    p = model.predict_proba(df[FEATURES])[:, 1]
    save_preds("logreg", cand=pd.DataFrame({"trip": df["trip"], "pt": df["pt"], "arc": df["arc"], "p": p}),
               seconds={"inference_s": time.time() - t0})


def run_rf():
    """Teammate Random Forest: their loader, candidate generator, features and saved model."""
    import joblib
    rf_dir = ROOT / "random_forest"
    sys.path.insert(0, str(rf_dir))
    os.chdir(rf_dir)
    from src.config import FEATURE_COLUMNS
    from src.data_loader import load_trip_data
    from src.feature_extraction import extract_features_for_trip
    model = joblib.load(rf_dir / "models" / "random_forest_baseline.joblib")
    frames = []
    for tid in trips():
        d = load_trip_data(str(EVAL_DIR / tid), tid)
        d["route"] = None                      # route is never needed to build features
        frames.append(extract_features_for_trip(d))
    df = pd.concat(frames, ignore_index=True)
    t0 = time.time()
    p = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    save_preds("rf", cand=pd.DataFrame({"trip": df["trip_id"], "pt": df["obs_id"], "arc": df["arc_id"], "p": p}),
               seconds={"inference_s": time.time() - t0})


def run_xgboost():
    """Teammate XGBoost (tuned): their candidate/feature builder, saved model and tuned HMM decoder."""
    import xgboost as xgb
    xdir = ROOT / "xg boost"
    sys.path.insert(0, str(xdir / "src"))
    import preprocess as xp
    import hmm_decoder as hd
    from xgb_common import FEATS
    xp.ROOT = str(DATA)                         # redirect their loader to our data folder
    model = xgb.XGBClassifier()
    model.load_model(str(xdir / "models" / "xgb_tuned.json"))
    truth = common_truth()
    frames = []
    for tid in trips():
        track, nodes, arcs, _ = xp.load_trip(SPLIT, tid)
        f = xp.build_features(track, nodes, arcs, R=50.0, K=20)
        f.insert(0, "trip", tid)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    t0 = time.time()
    p = model.predict_proba(df[FEATS])[:, 1]
    inf = time.time() - t0
    cand = pd.DataFrame({"trip": df["trip"], "pt": df["pt"], "arc": df["arc"], "p": p})
    # HMM/Viterbi with their tuned parameters (y only feeds their scoring helper, not the decoding)
    cfg = json.loads((xdir / "models" / "hmm_config.json").read_text())
    hd.MAXHOPS = cfg.get("maxhops", hd.MAXHOPS)
    c = cand.merge(truth[["trip", "pt", "truth_arc"]], on=["trip", "pt"])
    c["y"] = (c["arc"] == c["truth_arc"]).astype(int)
    c = c.sort_values(["trip", "pt"], kind="stable")
    rows = []
    t0 = time.time()
    for tid, g in c.groupby("trip", sort=True):
        t = hd.prep_trip(SPLIT, tid, g.reset_index(drop=True))
        pred = hd.viterbi(t, **cfg["params"])
        rows.append(pd.DataFrame({"trip": tid, "pt": np.arange(len(pred)), "pred_arc": pred}))
    save_preds("xgboost", cand=cand, point=pd.concat(rows, ignore_index=True),
               seconds={"inference_s": inf, "hmm_s": time.time() - t0})


def run_mlp():
    """Teammate Deep MLP: weights were not committed, so refit the tuned config exactly as train_mlp.py does."""
    import torch
    torch.set_num_threads(TEAM_THREADS)
    sys.path.insert(0, str(ROOT))
    from feature_engineering.candidate_features import build_or_load_partition_dataset
    from hmm_viterbi.viterbi_decoder import PathSnapViterbiDecoder
    from models.mlp_model import PathSnapMLPClassifier
    cache = str(ROOT / "artifacts" / "mlp_cache")
    best = json.loads((ROOT / "mlp" / "results" / "mlp_complete_report_metrics.json").read_text())["best_hyperparameters"]
    raw = build_or_load_partition_dataset(str(DATA), "train", K=5, cache_dir=cache)
    per_obs = ("X", "y", "cand_arcs", "gt_arcs", "miss_mask", "dt")
    train = []
    for b in raw:                               # same filter as train_mlp.filter_clean_train_bundles
        valid = (~b["miss_mask"]) | (b["X"][:, 0, 0] < 25.0)
        if np.sum(valid) < 20:
            valid = np.ones(len(b["X"]), dtype=bool)
        train.append({k: (b[k][valid] if k in per_obs else b[k]) for k in b})
    t0 = time.time()
    clf = PathSnapMLPClassifier(hidden_dim=best["hidden_dim"], head_dim=best["head_dim"], dropout=best["dropout"],
                                lr=best["lr"], epochs=26, batch_obs=4096, max_obs_per_trip=1200,
                                listwise_weight=best["listwise_weight"], pos_weight=best["pos_weight"], random_state=42)
    clf.fit(train[:72], train[72:])
    fit_s = time.time() - t0
    bundles = build_or_load_partition_dataset(str(DATA), SPLIT, K=5, cache_dir=cache)
    t0 = time.time()
    probs = [clf.predict_proba_trip(b) for b in bundles]
    inf = time.time() - t0
    decoder = PathSnapViterbiDecoder(emission_weight=1.0, same_arc_bonus=0.0, connected_cost=-0.35, two_hop_cost=-1.60,
                                     disconnected_cost=-4.80, reverse_arc_cost=-6.50, gap_relax_rate=0.30)
    cand_rows, pt_rows = [], []
    t0 = time.time()
    for b, pr in zip(bundles, probs):
        ca = np.asarray(b["cand_arcs"])
        n, k = ca.shape
        keep = ca.ravel() >= 0
        cand_rows.append(pd.DataFrame({"trip": b["trip_id"], "pt": np.repeat(np.arange(n), k)[keep],
                                       "arc": ca.ravel()[keep], "p": np.asarray(pr).ravel()[keep]}))
        dec = decoder.decode_trip(b, pr)
        pt_rows.append(pd.DataFrame({"trip": b["trip_id"], "pt": np.arange(n), "pred_arc": dec["viterbi_arcs"]}))
    save_preds("mlp", cand=pd.concat(cand_rows, ignore_index=True), point=pd.concat(pt_rows, ignore_index=True),
               seconds={"fit_s": fit_s, "inference_s": inf, "hmm_s": time.time() - t0, "threshold": float(clf.threshold)})


# ---------------------------------------------------------------- scoring
LABELS = {"lightgbm": "LightGBM", "logreg": "Logistic Regression", "rf": "Random Forest", "xgboost": "XGBoost", "mlp": "Deep MLP"}


def _norm(df):
    df = df.copy()
    df["trip"] = df["trip"].astype(str).str.zfill(8)
    df["pt"] = df["pt"].astype(int)
    return df


def score():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    truth = common_truth()
    rows, per_trip = [], []
    for name in MODELS:
        f = PRED_DIR / f"{name}_cand.parquet"
        if not f.exists():
            print("missing predictions for", name)
            continue
        cand = _norm(pd.read_parquet(f))
        pick = top_pick(cand)
        tf = PRED_DIR / f"{name}_time.json"
        t = json.loads(tf.read_text()) if tf.exists() else {}
        rows.append({"model": LABELS[name], "stage": "classifier only", **metrics(truth, pick, cand),
                     **{k: v for k, v in t.items() if k.endswith("_s")}})
        m = truth.merge(pick, on=["trip", "pt"], how="left")
        m["ok"] = m["pred_arc"].fillna(-1).astype(int) == m["truth_arc"]
        per_trip.append(m.groupby("trip")["ok"].mean().rename(LABELS[name]))
        pf = PRED_DIR / f"{name}_point.parquet"
        if pf.exists():
            pp = _norm(pd.read_parquet(pf))
            rows.append({"model": LABELS[name], "stage": "classifier + HMM/Viterbi", **metrics(truth, pp)})
            m = truth.merge(pp, on=["trip", "pt"], how="left")
            m["ok"] = m["pred_arc"].fillna(-1).astype(int) == m["truth_arc"]
            per_trip.append(m.groupby("trip")["ok"].mean().rename(LABELS[name] + " + HMM"))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / f"{SPLIT}_summary.csv", index=False)
    pd.concat(per_trip, axis=1).round(4).to_csv(OUT_DIR / f"{SPLIT}_per_trip_top1.csv")
    cols = ["model", "stage", "top1_acc", "mean_trip_top1", "route_continuity", "candidate_miss_rate", "macro_f1", "log_loss"]
    print(res[[c for c in cols if c in res]].round(4).to_string(index=False))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "run":
        name = sys.argv[2]
        common_truth()
        {"lightgbm": run_lightgbm, "logreg": run_logreg, "rf": run_rf, "xgboost": run_xgboost, "mlp": run_mlp}[name]()
        print("done", name)
    elif cmd == "score":
        score()
    else:
        for m in MODELS:
            print(f"=== {m} ===", flush=True)
            subprocess.run([sys.executable, __file__, "run", m], check=True, cwd=ROOT, env=os.environ)
        score()

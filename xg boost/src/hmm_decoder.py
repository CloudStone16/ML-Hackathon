"""HMM / Viterbi route reconstruction on top of the XGBoost candidate probabilities.

Hidden state at GPS point t = one of the candidate arcs of that point (candidate <-> arc correspondence preserved).
Emission    : alpha * log(p), p = classifier probability that this candidate is the true arc.
Transition  : from the number of road steps (hops) between consecutive arcs, via a bounded search on the road graph:
                same arc / next arc  -> 0,   skipping h-1 arcs -> -lam*(h-1),   U-turn onto the reverse arc -> -uturn,
                not reachable within maxhops (disconnected / teleport) -> -disc  (finite, so the chain can recover).
Gaps        : if two consecutive usable GPS points are more than gap_s seconds apart the transition is unconstrained
              (the chain restarts); GPS points with no candidate are skipped and reported as -1.
Decoding    : Viterbi, delta_t(j) = emit_t(j) + max_i [ delta_{t-1}(i) + trans(i, j) ], then backtracking.

Run:  python src/hmm_decoder.py            # default parameters, train OOF + student_test
      python src/hmm_decoder.py --tune     # Optuna search on train OOF only, then one evaluation on student_test
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from preprocess import load_trip
from xgb_common import MODELS, RES

MAXHOPS = 4
DEFAULT = dict(alpha=1.0, lam=1.0, disc=10.0, uturn=6.0, gap_s=10.0)


def reach_matrix(arcs, n_nodes, u_arcs, maxhops=MAXHOPS):
    """hop[i, j] = fewest arc-to-arc steps from u_arcs[i] to u_arcs[j] (0 = same arc); 255 = not within maxhops."""
    order = np.argsort(arcs[:, 0], kind="stable")
    starts = np.searchsorted(arcs[order, 0], np.arange(n_nodes + 1))
    idx = {int(a): i for i, a in enumerate(u_arcs)}
    hop = np.full((len(u_arcs), len(u_arcs)), 255, dtype=np.uint8)
    for i, a0 in enumerate(u_arcs):
        hop[i, i] = 0
        seen, frontier = {int(a0)}, [int(a0)]
        for depth in range(1, maxhops + 1):
            nxt = []
            for a in frontier:
                to = arcs[a, 1]
                for b in order[starts[to]:starts[to + 1]]:
                    b = int(b)
                    if b not in seen:
                        seen.add(b)
                        nxt.append(b)
                        j = idx.get(b)
                        if j is not None:
                            hop[i, j] = depth
            frontier = nxt
            if not frontier:
                break
    return hop


def prep_trip(part, tid, g):
    """Everything needed to decode one trip. g: candidate rows (pt, arc, y, p) sorted by pt."""
    track, nodes, arcs, route = load_trip(part, tid)
    pt, arc, p, y = g["pt"].values, g["arc"].values, g["p"].values, g["y"].values
    starts = np.r_[0, np.flatnonzero(np.diff(pt)) + 1]
    ends = np.r_[starts[1:], len(pt)]
    u_arcs = np.unique(arc)
    truth = np.full(len(track), -1)
    truth[pt[y == 1]] = arc[y == 1]
    return dict(tid=tid, n_points=len(track), times=track[:, 2], arcs=arcs, route=route, pt=pt, arc=arc, p=p,
                starts=starts, ends=ends, u_arcs=u_arcs, uidx=np.searchsorted(u_arcs, arc), truth=truth,
                hop=reach_matrix(arcs, len(nodes), u_arcs))


def classifier_only(t):
    pred = np.full(t["n_points"], -1)
    for s, e in zip(t["starts"], t["ends"]):
        pred[t["pt"][s]] = t["arc"][s + np.argmax(t["p"][s:e])]
    return pred


def viterbi(t, alpha, lam, disc, uturn, gap_s):
    arcs, hop = t["arcs"], t["hop"]
    frm, to = arcs[:, 0], arcs[:, 1]
    lut = np.full(256, -disc)
    lut[: MAXHOPS + 1] = -lam * np.maximum(np.arange(MAXHOPS + 1) - 1, 0)
    logp = alpha * np.log(np.clip(t["p"], 1e-6, 1.0))
    S, E, T = t["starts"], t["ends"], len(t["starts"])
    ptime = t["times"][t["pt"][S]]
    score = logp[S[0]:E[0]].copy()
    back = [None]
    for s in range(1, T):
        pa, ca = t["arc"][S[s - 1]:E[s - 1]], t["arc"][S[s]:E[s]]
        emis = logp[S[s]:E[s]]
        if ptime[s] - ptime[s - 1] > gap_s:
            bi = np.full(len(ca), int(np.argmax(score)))
            score = score.max() + emis
        else:
            H = hop[np.ix_(t["uidx"][S[s - 1]:E[s - 1]], t["uidx"][S[s]:E[s]])]
            tr = lut[H]
            rev = (to[pa][:, None] == frm[ca][None, :]) & (frm[pa][:, None] == to[ca][None, :])
            tr = tr - uturn * rev
            cand = score[:, None] + tr
            bi = cand.argmax(0)
            score = cand[bi, np.arange(len(ca))] + emis
        back.append(bi)
    j = int(np.argmax(score))
    pred = np.full(t["n_points"], -1)
    for s in range(T - 1, -1, -1):
        pred[t["pt"][S[s]]] = t["arc"][S[s] + j]
        if s > 0:
            j = int(back[s][j])
    return pred


def score_trip(t, pred):
    ok = (pred >= 0) & (pred == t["truth"])
    used = np.flatnonzero(pred >= 0)
    pa = pred[used]
    ch = np.flatnonzero(pa[1:] != pa[:-1])
    if len(ch):
        ia, ib = np.searchsorted(t["u_arcs"], pa[ch]), np.searchsorted(t["u_arcs"], pa[ch + 1])
        conn = int((t["hop"][ia, ib] < 255).sum())
    else:
        conn = 0
    pset, rset = set(pa.tolist()), set(t["route"].tolist())
    tp = len(pset & rset)
    prec, rec = tp / max(len(pset), 1), tp / max(len(rset), 1)
    return dict(correct=int(ok.sum()), n=t["n_points"], changes=int(len(ch)), connected=conn,
                route_f1=2 * prec * rec / max(prec + rec, 1e-9), route_prec=prec, route_rec=rec)


def aggregate(rows):
    n = sum(r["n"] for r in rows)
    ch = sum(r["changes"] for r in rows)
    return dict(top1_acc=sum(r["correct"] for r in rows) / n,
                route_continuity=(sum(r["connected"] for r in rows) / ch) if ch else 1.0,
                route_f1=float(np.mean([r["route_f1"] for r in rows])),
                route_precision=float(np.mean([r["route_prec"] for r in rows])),
                route_recall=float(np.mean([r["route_rec"] for r in rows])),
                arc_changes=ch, n_points=n, n_trips=len(rows))


def evaluate(trips, params):
    clf = aggregate([score_trip(t, classifier_only(t)) for t in trips])
    hmm = aggregate([score_trip(t, viterbi(t, **params)) for t in trips])
    return clf, hmm


def load_trips(part, csv, max_u=None, every=1):
    df = pd.read_csv(os.path.join(RES, csv), dtype={"trip": str})
    out = []
    for i, (tid, g) in enumerate(df.groupby("trip", sort=True)):
        if i % every:
            continue
        if max_u and g["arc"].nunique() > max_u:
            continue
        out.append(prep_trip(part, tid, g.reset_index(drop=True)))
    return out


def show(title, clf, hmm):
    print("\n%s" % title)
    print("%-22s %12s %12s" % ("metric", "classifier", "clf + HMM"))
    for k in ("top1_acc", "route_continuity", "route_f1", "route_precision", "route_recall"):
        print("%-22s %12.4f %12.4f" % (k, clf[k], hmm[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--probs", default="tuned", help="which XGBoost predictions to decode (baseline|tuned)")
    args = ap.parse_args()

    t0 = time.time()
    train = load_trips("train", "xgb_oof_%s.csv.gz" % args.probs, max_u=12000, every=1 if not args.tune else 2)
    test = load_trips("student_test", "xgb_test_%s.csv.gz" % args.probs)
    print("prepared %d train trips (out-of-fold probabilities) and %d student_test trips in %.0f s"
          % (len(train), len(test), time.time() - t0), flush=True)

    out = dict(probs=args.probs, maxhops=MAXHOPS)
    t0 = time.time()
    clf, hmm = evaluate(train, DEFAULT)
    show("train OOF, DEFAULT params %s (%.0f s)" % (DEFAULT, time.time() - t0), clf, hmm)
    out["train_default"] = dict(classifier=clf, hmm=hmm, params=DEFAULT)
    params = dict(DEFAULT)

    if args.tune:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            p = dict(alpha=trial.suggest_float("alpha", 0.3, 3.0), lam=trial.suggest_float("lam", 0.0, 6.0),
                     disc=trial.suggest_float("disc", 1.0, 40.0), uturn=trial.suggest_float("uturn", 0.0, 20.0),
                     gap_s=trial.suggest_float("gap_s", 3.0, 30.0))
            acc = aggregate([score_trip(t, viterbi(t, **p)) for t in train])["top1_acc"]
            print("trial %2d top1 %.4f %s" % (trial.number, acc, {k: round(v, 2) for k, v in p.items()}), flush=True)
            return acc

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        t0 = time.time()
        study.optimize(objective, n_trials=args.trials)
        params = study.best_params
        study.trials_dataframe().to_csv(os.path.join(RES, "hmm_trials.csv"), index=False)
        print("\nbest train-OOF top1 %.4f (%d trials, %.0f s)\nbest params %s"
              % (study.best_value, args.trials, time.time() - t0, params), flush=True)
        out["tuning"] = dict(objective="maximise observation-level top-1 on train out-of-fold probabilities "
                                       "(every 2nd train trip)", n_trials=args.trials, best_params=params,
                             best_value=study.best_value)
        trn_all = load_trips("train", "xgb_oof_%s.csv.gz" % args.probs, max_u=12000)
        c, h = evaluate(trn_all, params)
        show("train OOF (all usable trips), TUNED params", c, h)
        out["train_tuned"] = dict(classifier=c, hmm=h)

    clf, hmm = evaluate(test, params)
    show("student_test, params %s" % {k: round(v, 2) for k, v in params.items()}, clf, hmm)
    out["student_test"] = dict(classifier=clf, hmm=hmm, params=params)
    out["params"] = params
    suffix = "tuned" if args.tune else "default"
    with open(os.path.join(RES, "hmm_%s.json" % suffix), "w") as f:
        json.dump(out, f, indent=2)
    if args.tune:
        with open(os.path.join(MODELS, "hmm_config.json"), "w") as f:
            json.dump(dict(params=params, maxhops=MAXHOPS), f, indent=2)
    print("saved")


if __name__ == "__main__":
    main()

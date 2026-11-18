"""HMM / Viterbi route reconstruction over candidate arcs (works with any model that outputs a probability per candidate).

Hidden state at GPS point t = one of its candidate arcs (top_k by probability).
Emission    : alpha * log(p)
Transition  : by road-graph hops between consecutive arcs
              same arc / next arc -> 0, skipping h-1 arcs -> -lam*(h-1), U-turn onto the reverse twin -> -uturn,
              not reachable within MAXHOPS -> -disc (finite, so the chain can recover).
Gaps        : if two consecutive usable points are more than gap_s seconds apart the transition is unconstrained.
Decoding    : Viterbi recurrence delta_t(j) = emit_t(j) + max_i [delta_{t-1}(i) + trans(i, j)], then backtracking.
Points with no candidate are returned as -1.
"""
import numpy as np

MAXHOPS = 4
DEFAULT_PARAMS = dict(alpha=1.0, lam=1.0, disc=10.0, uturn=6.0, gap_s=10.0)


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


def prepare(n_points, times, arcs, n_nodes, pt, arc, p, top_k=10):
    """Group candidate rows by GPS point, keep the top_k by probability, build the arc reachability table."""
    pt, arc, p = np.asarray(pt), np.asarray(arc), np.asarray(p, dtype=float)
    order = np.lexsort((-p, pt))
    pt, arc, p = pt[order], arc[order], p[order]
    first = np.r_[True, pt[1:] != pt[:-1]] if len(pt) else np.zeros(0, dtype=bool)
    grp_start = np.flatnonzero(first)
    rank = np.arange(len(pt)) - grp_start[np.cumsum(first) - 1]
    keep = rank < top_k
    pt, arc, p = pt[keep], arc[keep], p[keep]
    starts = np.flatnonzero(np.r_[True, pt[1:] != pt[:-1]]) if len(pt) else np.zeros(0, dtype=int)
    ends = np.r_[starts[1:], len(pt)] if len(pt) else np.zeros(0, dtype=int)
    u_arcs = np.unique(arc)
    return dict(n_points=int(n_points), times=np.asarray(times), arcs=np.asarray(arcs), pt=pt, arc=arc, p=p,
                starts=starts, ends=ends, u_arcs=u_arcs, uidx=np.searchsorted(u_arcs, arc),
                hop=reach_matrix(np.asarray(arcs), n_nodes, u_arcs) if len(u_arcs) else None)


def classifier_only(t):
    pred = np.full(t["n_points"], -1)
    if len(t["starts"]):
        pred[t["pt"][t["starts"]]] = t["arc"][t["starts"]]      # rows are sorted by descending p within a point
    return pred


def viterbi(t, alpha, lam, disc, uturn, gap_s):
    pred = np.full(t["n_points"], -1)
    S, E = t["starts"], t["ends"]
    T = len(S)
    if T == 0:
        return pred
    arcs, hop = t["arcs"], t["hop"]
    frm, to = arcs[:, 0], arcs[:, 1]
    lut = np.full(256, -disc)
    lut[: MAXHOPS + 1] = -lam * np.maximum(np.arange(MAXHOPS + 1) - 1, 0)
    logp = alpha * np.log(np.clip(t["p"], 1e-6, 1.0))
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
            cand = score[:, None] + tr - uturn * rev
            bi = cand.argmax(0)
            score = cand[bi, np.arange(len(ca))] + emis
        back.append(bi)
    j = int(np.argmax(score))
    for s in range(T - 1, -1, -1):
        pred[t["pt"][S[s]]] = t["arc"][S[s] + j]
        if s > 0:
            j = int(back[s][j])
    return pred

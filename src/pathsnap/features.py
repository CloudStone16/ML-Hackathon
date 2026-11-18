"""Candidate generation + feature engineering for (GPS point, candidate arc) rows.

Features use the track and road network only. The reference route is never read here.
"""
import numpy as np
import pandas as pd

from .geo import LocalProjection
from .io import clean_track
from .labeling import align_to_route
from .network import RoadNetwork
from .track import kinematics

RADIUS_M = 50.0
LABEL_MAX_DIST_M = 60.0   # aligned route arc farther than this from the GPS point = reference route not trustworthy there
MAX_CANDS = 40
WINDOW = (-3, -2, -1, 1, 2, 3)

FEATURES = [
    "d", "frac", "arc_len", "along", "to_end", "rank_d", "d_minus_min", "d_ratio_min", "n_cands",
    "min_d", "second_d_gap", "cos_head", "head_valid", "cos_head5", "cos_head10", "cos_head_fill", "speed", "dt_prev", "dt_next", "gap",
    "has_twin", "twin_cand", "twin_d_diff", "deg_out_to", "deg_in_from",
    "d_m3", "d_m1", "d_p1", "d_p3", "win_mean_d", "win_count",
    "prog_signed", "conn_next1", "conn_prev1", "conn_next3", "conn_prev3",
]


def _shift_lookup(df, offset, col, name):
    """Value of `col` for the same arc at pt+offset (NaN if not a candidate there)."""
    other = df[["trip", "pt", "arc", col]].copy()
    other["pt"] = other["pt"] - offset
    other = other.rename(columns={col: name})
    return df[["trip", "pt", "arc"]].merge(other, on=["trip", "pt", "arc"], how="left")[name].to_numpy()


def _conn(df, offset, from_node, to_node):
    """1 if arc at pt is the same as / head-to-tail adjacent to some candidate at pt+offset."""
    if offset > 0:
        # candidate arc c at pt+offset is next if from_node[c] == to_node[a]  or c == a
        tgt = df[["trip", "pt", "from_node"]].copy()
        tgt["pt"] -= offset
        tgt = tgt.drop_duplicates().rename(columns={"from_node": "to_node"})
        tgt["hit"] = 1
        m = df[["trip", "pt", "to_node"]].merge(tgt, on=["trip", "pt", "to_node"], how="left")["hit"].to_numpy()
    else:
        tgt = df[["trip", "pt", "to_node"]].copy()
        tgt["pt"] -= offset  # offset negative -> pt + |offset|
        tgt = tgt.drop_duplicates().rename(columns={"to_node": "from_node"})
        tgt["hit"] = 1
        m = df[["trip", "pt", "from_node"]].merge(tgt, on=["trip", "pt", "from_node"], how="left")["hit"].to_numpy()
    same = _shift_lookup(df, offset, "d", "same")
    return (np.nan_to_num(m) > 0) | ~np.isnan(same)


def build_trip(trip: dict, with_labels: bool = True):
    """Returns (rows DataFrame, point-level report dict)."""
    track, clean_report = clean_track(trip["track"])
    lonlat = trip["nodes"]
    proj = LocalProjection(lonlat[:, 0].mean(), lonlat[:, 1].mean())
    net = RoadNetwork(lonlat, trip["arcs"], proj)
    P = proj.to_xy(track[:, :2])
    t = track[:, 2]
    kin = kinematics(P, t)
    n = len(P)

    pt, arc, d, frac = net.candidates(P, RADIUS_M, MAX_CANDS)
    df = pd.DataFrame({"trip": trip["id"], "pt": pt, "arc": arc, "d": d, "frac": frac})

    # --- labels (dev/train only) ---
    report = dict(clean_report, trip=trip["id"], n_pts=n, n_arcs=len(net.arcs), n_nodes=len(net.nodes))
    if with_labels and trip["route"] is not None:
        label_arc, label_d = align_to_route(P, kin, net, trip["route"])
        df["label"] = (df["arc"].to_numpy() == label_arc[df["pt"].to_numpy()]).astype(np.int8)
        valid = label_d <= LABEL_MAX_DIST_M
        df["label_valid"] = valid[df["pt"].to_numpy()].astype(np.int8)
        in_cands = df.groupby("pt")["label"].max().reindex(range(n)).fillna(0).to_numpy() > 0
        report.update(label_dist_med=float(np.median(label_d)), label_dist_p95=float(np.percentile(label_d, 95)),
                      frac_label_invalid=float(1 - valid.mean()),
                      miss_rate=float(1 - in_cands[valid].mean()) if valid.any() else float("nan"))

    # --- geometry / candidate-set features ---
    arc_a = df["arc"].to_numpy()
    df["from_node"] = net.arcs[arc_a, 0]
    df["to_node"] = net.arcs[arc_a, 1]
    df["arc_len"] = net.length[arc_a]
    df["along"] = df["frac"] * df["arc_len"]
    df["to_end"] = (1 - df["frac"]) * df["arc_len"]
    g = df.groupby("pt")["d"]
    df["rank_d"] = g.rank(method="first").astype(int) - 1
    df["min_d"] = g.transform("min")
    df["d_minus_min"] = df["d"] - df["min_d"]
    df["d_ratio_min"] = df["d"] / (df["min_d"] + 1.0)
    df["n_cands"] = g.transform("size")
    second = df[df["rank_d"] == 1].set_index("pt")["d"].reindex(range(n))
    df["second_d_gap"] = (second.to_numpy()[df["pt"]] - df["min_d"]).astype(float)

    # --- heading / kinematics ---
    arc_dir = np.stack([np.cos(net.bearing[arc_a]), np.sin(net.bearing[arc_a])], 1)
    hv = kin["hvec"][df["pt"].to_numpy()]
    df["cos_head"] = (hv * arc_dir).sum(1)           # NaN when heading unreliable
    for key, col in (("hvec5", "cos_head5"), ("hvec10", "cos_head10"), ("hvec_fill", "cos_head_fill")):
        df[col] = (kin[key][df["pt"].to_numpy()] * arc_dir).sum(1)
    df["head_valid"] = (~np.isnan(df["cos_head"])).astype(np.int8)
    for k in ("speed", "dt_prev", "dt_next"):
        df[k] = kin[k][df["pt"].to_numpy()]
    df["gap"] = kin["gap"][df["pt"].to_numpy()].astype(np.int8)

    # --- reverse twin ---
    twin = net.twin[arc_a]
    df["has_twin"] = (twin >= 0).astype(np.int8)
    twin_d = pd.DataFrame({"trip": df["trip"], "pt": df["pt"], "arc": twin, "twin_d": df["d"]})
    # d of twin arc *as candidate at same point*: look up (pt, arc=twin)
    lk = df[["trip", "pt", "arc", "d"]].rename(columns={"d": "twin_dist"})
    j = df[["trip", "pt"]].assign(arc=twin).merge(lk, on=["trip", "pt", "arc"], how="left")
    df["twin_cand"] = (~j["twin_dist"].isna()).to_numpy().astype(np.int8)
    df["twin_d_diff"] = (j["twin_dist"].to_numpy() - df["d"].to_numpy())
    df["deg_out_to"] = net.deg_out[df["to_node"].to_numpy()]
    df["deg_in_from"] = net.deg_in[df["from_node"].to_numpy()]

    # --- temporal window features ---
    shifted = {o: _shift_lookup(df, o, "d", f"dw{o}") for o in WINDOW}
    df["d_m3"], df["d_m1"], df["d_p1"], df["d_p3"] = shifted[-3], shifted[-1], shifted[1], shifted[3]
    stack = np.stack([shifted[o] for o in WINDOW] + [df["d"].to_numpy()], 1)
    df["win_count"] = (~np.isnan(stack)).sum(1)
    df["win_mean_d"] = np.where(np.isnan(stack), RADIUS_M + 10, stack).mean(1)
    # signed progress along the arc direction between pt-2 and pt+2 (needs arc seen at both)
    a_m2 = _shift_lookup(df, -2, "along", "a_m2")
    a_p2 = _shift_lookup(df, 2, "along", "a_p2")
    df["prog_signed"] = a_p2 - a_m2
    for off in (1, 3):
        df[f"conn_next{off}"] = _conn(df, off, "from_node", "to_node").astype(np.int8)
        df[f"conn_prev{off}"] = _conn(df, -off, "from_node", "to_node").astype(np.int8)

    df = df.drop(columns=[]).reset_index(drop=True)
    return df, report

"""PathSnap preprocessing: label GPS points from the route, generate candidate arcs, build features.

Run from the project root:  python src/preprocess.py [--radius 50] [--k 20]
Writes data/processed/candidates.csv.gz and data/processed/trip_points.csv
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from shapely import STRtree, linestrings, points as shp_points

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "data", "Student folder")
OUT = os.path.join(HERE, "..", "data", "processed")


def load_trip(part, tid):
    d = os.path.join(ROOT, part, tid)
    track = np.loadtxt(os.path.join(d, tid + ".track"), ndmin=2)
    nodes = np.loadtxt(os.path.join(d, tid + ".nodes"), ndmin=2)
    arcs = np.loadtxt(os.path.join(d, tid + ".arcs"), dtype=int, ndmin=2)
    route = np.loadtxt(os.path.join(d, tid + ".route"), dtype=int, ndmin=1)
    return track, nodes, arcs, route


def to_xy(lon, lat, lon0, lat0):
    """Local flat projection to metres (origin = first GPS point of the trip)."""
    return (lon - lon0) * 111320.0 * np.cos(np.radians(lat0)), (lat - lat0) * 110540.0


def point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.where(L == 0, 1, L), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def monotone_labels(D):
    """One route arc per GPS point, never moving backwards along the route (dynamic programming)."""
    T, R = D.shape
    cost = D[0].copy()
    bp = np.zeros((T, R), dtype=np.int32)
    ar = np.arange(R)
    for t in range(1, T):
        pm = np.minimum.accumulate(cost)
        bp[t] = np.maximum.accumulate(np.where(cost == pm, ar, 0))
        cost = D[t] + pm
    j = int(cost.argmin())
    idx = np.empty(T, dtype=int)
    for t in range(T - 1, -1, -1):
        idx[t] = j
        j = bp[t, j]
    return idx


def label_trip(track, nodes, arcs, route):
    """Needs the route, so only for train / student_test. Returns (true arc id, distance to it) per point."""
    lon0, lat0 = track[0, 0], track[0, 1]
    gx, gy = to_xy(track[:, 0], track[:, 1], lon0, lat0)
    x, y = to_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    A = arcs[route]
    D = point_seg_dist(gx[:, None], gy[:, None], x[A[:, 0]][None, :], y[A[:, 0]][None, :],
                       x[A[:, 1]][None, :], y[A[:, 1]][None, :])
    pos = monotone_labels(D)
    return route[pos], D[np.arange(len(D)), pos]


def build_features(track, nodes, arcs, R=50.0, K=20):
    """Candidate arcs + features for ONE trip. Uses only GPS and the road network, never the route."""
    lon0, lat0 = track[0, 0], track[0, 1]
    gx, gy = to_xy(track[:, 0], track[:, 1], lon0, lat0)
    ts = track[:, 2]
    x, y = to_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    ax, ay, bx, by = x[arcs[:, 0]], y[arcs[:, 0]], x[arcs[:, 1]], y[arcs[:, 1]]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    L = np.sqrt(L2)
    out_deg = np.bincount(arcs[:, 0], minlength=len(nodes))
    in_deg = np.bincount(arcs[:, 1], minlength=len(nodes))

    vx, vy = np.gradient(gx, ts), np.gradient(gy, ts)
    speed = np.hypot(vx, vy)
    d_t = np.diff(ts)
    dt_prev, dt_next = np.r_[0.0, d_t], np.r_[d_t, 0.0]
    heading_ok = (speed > 1.0) & (np.maximum(dt_prev, dt_next) <= 5)

    lines = linestrings(np.stack([np.stack([ax, ay], 1), np.stack([bx, by], 1)], 1))
    pt, arc = STRtree(lines).query(shp_points(gx, gy), predicate="dwithin", distance=R)

    t = np.clip(((gx[pt] - ax[arc]) * dx[arc] + (gy[pt] - ay[arc]) * dy[arc])
                / np.where(L2[arc] == 0, 1, L2[arc]), 0, 1)
    cx, cy = ax[arc] + t * dx[arc], ay[arc] + t * dy[arc]
    dist = np.hypot(gx[pt] - cx, gy[pt] - cy)
    side = np.sign(dx[arc] * (gy[pt] - ay[arc]) - dy[arc] * (gx[pt] - ax[arc]))
    cos_h = (vx[pt] * dx[arc] + vy[pt] * dy[arc]) / (speed[pt] * L[arc] + 1e-9)
    cos_h = np.where(heading_ok[pt] & (L[arc] > 0), cos_h, 0.0)

    df = pd.DataFrame({
        "pt": pt, "arc": arc, "dist": dist, "t_frac": t, "along_m": t * L[arc], "arc_len": L[arc],
        "lateral": side * dist, "heading_cos": cos_h, "heading_ok": heading_ok[pt].astype(np.int8),
        "speed": speed[pt], "dt_prev": dt_prev[pt], "dt_next": dt_next[pt],
        "out_deg_to": out_deg[arcs[arc, 1]], "in_deg_from": in_deg[arcs[arc, 0]]})
    df["n_cand"] = df.groupby("pt")["pt"].transform("size")
    df = df.sort_values(["pt", "dist"], kind="stable")
    df["rank"] = df.groupby("pt").cumcount()
    df = df[df["rank"] < K].copy()
    df["d_gap"] = df["dist"] - df.groupby("pt")["dist"].transform("min")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    trips = [(p, t) for p in ("train", "student_test") for t in sorted(os.listdir(os.path.join(ROOT, p)))]
    print(len(trips), "trips")

    # Step 3: labels + radius table
    t0 = time.time()
    labels = {}
    for part, tid in trips:
        trk, nds, arc, rte = load_trip(part, tid)
        labels[(part, tid)] = label_trip(trk, nds, arc, rte)
    print("labelled in %.0f s" % (time.time() - t0))
    train_d = np.concatenate([v[1] for k, v in labels.items() if k[0] == "train"])
    print("distance from GPS point to its true arc (train): median %.1f, p95 %.1f, p99 %.1f, max %.1f"
          % (np.median(train_d), *np.percentile(train_d, [95, 99]), train_d.max()))
    for r in (20, 30, 40, 50, 75, 100):
        print("  R = %3d m -> %.2f%% of train points would miss their true arc" % (r, 100 * (train_d > r).mean()))

    # Steps 4-5: features for all trips
    frames, pts = [], []
    for i, ((part, tid), (lab, _)) in enumerate(labels.items()):
        trk, nds, arc, _ = load_trip(part, tid)
        f = build_features(trk, nds, arc, R=args.radius, K=args.k)
        f["y"] = (f["arc"].values == lab[f["pt"].values]).astype(np.int8)
        f.insert(0, "trip", tid)
        f.insert(0, "part", part)
        frames.append(f)
        pts.append(dict(part=part, trip=tid, n_points=len(trk)))
        if i % 10 == 0:
            print(i, part, tid, "rows so far:", sum(len(x) for x in frames), "%.0f s" % (time.time() - t0))
    df = pd.concat(frames, ignore_index=True)
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "candidates.csv.gz"), index=False)
    P = pd.DataFrame(pts)
    P.to_csv(os.path.join(OUT, "trip_points.csv"), index=False)
    print("saved", df.shape, "radius", args.radius, "K", args.k)

    # Step 6: sanity checks
    n_points = P.groupby("part")["n_points"].sum()
    hit = df.groupby(["part", "trip", "pt"])["y"].max()
    nearest_ok = df[df["rank"] == 0].groupby("part")["y"].sum()
    print("\npoints per partition:", n_points.to_dict())
    print("candidate miss rate (%):", (100 * (1 - hit.groupby("part").sum() / n_points)).round(2).to_dict())
    print("nearest-candidate top-1 accuracy (%):", (100 * nearest_ok / n_points).round(2).to_dict())
    print("positive rate among candidate rows (%):", (100 * df.groupby("part")["y"].mean()).round(2).to_dict())
    print("mean candidates per point:", round(df.groupby(["part", "trip", "pt"]).size().mean(), 1))
    print("NaNs in features:", int(df.drop(columns=["part", "trip"]).isna().sum().sum()))
    print("trips per partition:", df.groupby("part")["trip"].nunique().to_dict())
    print("trips shared between train and student_test:",
          set(df[df.part == "train"].trip) & set(df[df.part == "student_test"].trip))


if __name__ == "__main__":
    main()

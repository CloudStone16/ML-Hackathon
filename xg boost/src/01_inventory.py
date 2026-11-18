"""Dataset inventory: per-trip counts, time gaps, extents, network connectivity, route checks."""
import os, sys
import numpy as np
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "Student folder")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def load_trip(part, tid):
    d = os.path.join(ROOT, part, tid)
    track = np.loadtxt(os.path.join(d, tid + ".track"), ndmin=2)
    nodes = np.loadtxt(os.path.join(d, tid + ".nodes"), ndmin=2)
    arcs = np.loadtxt(os.path.join(d, tid + ".arcs"), dtype=int, ndmin=2)
    route = np.loadtxt(os.path.join(d, tid + ".route"), dtype=int, ndmin=1)
    return track, nodes, arcs, route


rows = []
for part in ["train", "student_test"]:
    for tid in sorted(os.listdir(os.path.join(ROOT, part))):
        t, n, a, r = load_trip(part, tid)
        dt = np.diff(t[:, 2])
        g = nx.Graph(); g.add_nodes_from(range(len(n))); g.add_edges_from(a.tolist())
        comps = nx.number_connected_components(g)
        deg0 = sum(1 for _, d in g.degree() if d == 0)
        # route continuity: consecutive arcs share a node?
        ok = sum(1 for x, y in zip(r[:-1], r[1:]) if {*a[x]} & {*a[y]})
        rows.append(dict(part=part, trip=tid, n_gps=len(t), n_nodes=len(n), n_arcs=len(a),
                         n_route=len(r), uniq_route=len(np.unique(r)),
                         dur_s=t[-1, 2] - t[0, 2], dt_ne1=int((dt != 1).sum()), dt_max=dt.max(),
                         dt_nonmono=int((dt <= 0).sum()),
                         nan=int(np.isnan(t).sum()), lon_min=t[:, 0].min(), lon_max=t[:, 0].max(),
                         lat_min=t[:, 1].min(), lat_max=t[:, 1].max(),
                         comps=comps, isolated=deg0,
                         route_adj=f"{ok}/{len(r)-1}", max_arc=int(r.max()), arcs_ok=bool(r.max() < len(a))))

cols = list(rows[0].keys())
with open(os.path.join(OUT, "inventory.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r[c]) for c in cols) + "\n")

for part in ["train", "student_test"]:
    rs = [r for r in rows if r["part"] == part]
    print(f"\n== {part}: {len(rs)} trips")
    for c in ["n_gps", "n_nodes", "n_arcs", "n_route", "dur_s", "dt_ne1", "dt_max", "comps", "isolated"]:
        v = np.array([r[c] for r in rs], float)
        print(f"  {c:9s} min={v.min():9.0f} med={np.median(v):9.0f} max={v.max():9.0f}")
    print("  NaNs:", sum(r["nan"] for r in rs), " nonmono dt:", sum(r["dt_nonmono"] for r in rs),
          " route idx out of range:", sum(not r["arcs_ok"] for r in rs))
    print("  route fully adjacent:", sum(r["route_adj"].split("/")[0] == str(r["n_route"] - 1) for r in rs), "/", len(rs))

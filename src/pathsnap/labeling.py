"""Derive per-point arc labels by aligning the track to the ordered reference route.

The .route file is a sequence of arc ids, not a per-point label. We align points to
route arcs with a monotone dynamic program (arc index never decreases), minimising
distance-to-arc plus a heading-mismatch penalty. Used ONLY to build training/dev
labels, never as a feature.
"""
import numpy as np

from .geo import point_segment

DIST_CLIP_M = 60.0       # cap per-point distance cost so outliers don't dominate
HEADING_WEIGHT_M = 20.0  # metres-equivalent penalty for a fully opposite heading
SKIP_PENALTY_M = 0.5     # cost per route arc skipped between consecutive points


def align_to_route(P, kin, net, route):
    """Returns (label_arc[N], dist_to_label[N])."""
    n, m = len(P), len(route)
    A, B = net.A[route], net.B[route]
    d, _ = point_segment(P[:, None, :], A[None], B[None])             # (N, M)
    cost = np.minimum(d, DIST_CLIP_M)
    arc_dir = np.stack([np.cos(net.bearing[route]), np.sin(net.bearing[route])], 1)  # (M,2)
    h = kin["hvec"]
    cos = np.nan_to_num(h @ arc_dir.T, nan=1.0)                       # no heading -> no penalty
    cost = cost + HEADING_WEIGHT_M * 0.5 * (1.0 - cos)

    j = np.arange(m)
    dp = cost[0].copy()
    bp = np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        v = dp - SKIP_PENALTY_M * j
        cm = np.minimum.accumulate(v)
        new_min = v <= cm
        ki = np.maximum.accumulate(np.where(new_min, j, 0))
        dp = cost[i] + cm + SKIP_PENALTY_M * j
        bp[i] = ki
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    return route[path], d[np.arange(n), path]

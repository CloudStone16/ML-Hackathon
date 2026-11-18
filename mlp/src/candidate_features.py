"""
PathSnap: Candidate Generation, Ground-Truth Alignment & Feature Engineering
============================================================================
Extracts top-K candidate road segments per GPS observation and computes 24
geometric, directional, kinematic, displacement-tangent, candidate-relative,
and forward-backward topological features.
All feature scaling is fit strictly on the 80 training trips to prevent data leakage.
"""

import os
import time
import joblib
import numpy as np
from scipy.spatial import cKDTree
from data_processing.eda_and_preprocessing import load_trip_raw, clean_and_project_trip


FEATURE_NAMES = [
    "perp_dist_m",              # 0: Exact perpendicular distance to segment (m)
    "gaussian_spatial_ll",      # 1: Smooth Gaussian spatial log-likelihood -0.5*(d/8)^2
    "proj_fraction",            # 2: Projection parameter u in [0, 1] along directed arc
    "interior_margin",          # 3: Distance from segment endpoints min(u, 1-u) in [0, 0.5]
    "dist_to_start_m",          # 4: Euclidean distance from GPS point to arc start node (m)
    "dist_to_end_m",            # 5: Euclidean distance from GPS point to arc end node (m)
    "segment_length_m",         # 6: Length of candidate road segment (m)
    "cos_heading_inst",         # 7: Cosine of angle between instantaneous GPS velocity and arc
    "sin_heading_inst",         # 8: Absolute sine of angle between instantaneous velocity and arc
    "cos_heading_smooth5",      # 9: Cosine of angle using 5-step smoothed GPS velocity
    "cos_heading_disp20m",      # 10: Cosine of angle using 20m displacement-triggered tangent vector
    "dist_prev_step_m",         # 11: Perpendicular distance from previous GPS point (t-1) to arc
    "dist_next_step_m",         # 12: Perpendicular distance from next GPS point (t+1) to arc
    "dist_window5_mean_m",      # 13: Mean perpendicular distance over 5-step window [t-2..t+2]
    "proj_progress_3step",      # 14: Forward progress along segment u(t+1) - u(t-1)
    "rel_dist_diff_best_m",     # 15: Distance minus best candidate distance at step t
    "rel_dist_ratio_best",      # 16: Ratio of distance to best candidate distance at step t
    "cand_geom_softmax_prob",   # 17: Geometric+heading softmax score across K candidates at step t
    "cand_rank_norm",           # 18: Normalized candidate rank k / (K - 1)
    "conn_from_prev_top1",      # 19: Graph connectivity from t-1 top-1 candidate (1=same, 0.85=connected)
    "conn_to_next_top1",        # 20: Graph connectivity to t+1 top-1 candidate (1=same, 0.85=connected)
    "fb_chain_rel_cost",        # 21: Unsupervised forward-backward topological chain relative cost
    "fb_chain_softmax_prob",    # 22: Unsupervised forward-backward topological chain softmax probability
    "gps_speed_mps",            # 23: Smoothed vehicle speed at timestamp t (m/s)
]


def point_to_segments_dist(px, py, ax, ay, bx, by):
    """Vectorized perpendicular distance and projection parameter t in [0, 1]."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    len_sq_safe = np.where(len_sq < 1e-12, 1.0, len_sq)
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq_safe
    t = np.clip(t, 0.0, 1.0)
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t


def compute_displacement_tangent(tx: np.ndarray, ty: np.ndarray, min_move_m: float = 20.0):
    """
    Compute unit tangent direction vector at each step t by looking forward and backward
    until the vehicle has displaced by at least `min_move_m` meters.
    Robust to red-light stops and low-speed traffic jitter on two-way streets.
    """
    T = len(tx)
    fdx = np.zeros(T, dtype=np.float64)
    fdy = np.zeros(T, dtype=np.float64)
    bdx = np.zeros(T, dtype=np.float64)
    bdy = np.zeros(T, dtype=np.float64)

    j = 0
    for i in range(T):
        if j < i:
            j = i
        while j < T - 1 and np.hypot(tx[j] - tx[i], ty[j] - ty[i]) < min_move_m:
            j += 1
        d = np.hypot(tx[j] - tx[i], ty[j] - ty[i]) + 1e-9
        fdx[i] = (tx[j] - tx[i]) / d
        fdy[i] = (ty[j] - ty[i]) / d

    j = T - 1
    for i in range(T - 1, -1, -1):
        if j > i:
            j = i
        while j > 0 and np.hypot(tx[i] - tx[j], ty[i] - ty[j]) < min_move_m:
            j -= 1
        d = np.hypot(tx[i] - tx[j], ty[i] - ty[j]) + 1e-9
        bdx[i] = (tx[i] - tx[j]) / d
        bdy[i] = (ty[i] - ty[j]) / d

    vx = fdx + bdx
    vy = fdy + bdy
    norm = np.hypot(vx, vy) + 1e-9
    return vx / norm, vy / norm


def align_ground_truth_route(tx, ty, nx, ny, arcs, route):
    """
    Monotonic Dynamic Programming alignment of each GPS observation t in [0..T-1]
    to its true traversed road segment along the chronological .route sequence.
    """
    r_from = arcs[route, 0]
    r_to = arcs[route, 1]
    dist, t_proj = point_to_segments_dist(
        tx[:, None], ty[:, None],
        nx[r_from][None, :], ny[r_from][None, :],
        nx[r_to][None, :], ny[r_to][None, :]
    )
    ux_d, uy_d = compute_displacement_tangent(tx, ty, 20.0)
    dx_arc = nx[r_to] - nx[r_from]
    dy_arc = ny[r_to] - ny[r_from]
    len_arc = np.hypot(dx_arc, dy_arc) + 1e-9
    cos_disp = (ux_d[:, None] * dx_arc[None, :] + uy_d[:, None] * dy_arc[None, :]) / len_arc[None, :]

    endpoint_pen = np.where((t_proj <= 0.001) | (t_proj >= 0.999), 0.05, 0.0)
    dir_pen = np.where(cos_disp < 0, 5.0 * (-cos_disp), -0.40 * cos_disp)
    eff_dist = dist + dir_pen + endpoint_pen

    T, M = dist.shape
    dp = np.full((T, M), np.inf, dtype=np.float64)
    back = np.zeros((T, M), dtype=np.int32)
    dp[0] = eff_dist[0]

    max_shift = min(15, M)
    for t in range(1, T):
        best_prev = dp[t - 1].copy()
        best_idx = np.arange(M, dtype=np.int32)
        for s in range(1, max_shift):
            pen = 0.02 * s if s <= 3 else 0.5 * (s - 3)
            cand = dp[t - 1, :-s] + pen
            mask = cand < best_prev[s:]
            best_prev[s:][mask] = cand[mask]
            best_idx[s:][mask] = np.arange(M - s, dtype=np.int32)[mask]
        dp[t] = eff_dist[t] + best_prev
        back[t] = best_idx

    m_idx = np.zeros(T, dtype=np.int32)
    m_idx[-1] = int(np.argmin(dp[-1]))
    for t in range(T - 1, 0, -1):
        m_idx[t - 1] = back[t, m_idx[t]]
    return route[m_idx]


def Smooth1D(arr, window: int):
    if len(arr) < window or window <= 1:
        return arr.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def extract_trip_candidates_and_features(proc_trip: dict, K: int = 5, pre_k: int = 45) -> dict:
    """
    Generate top-K candidate road segments for each GPS point in a preprocessed trip
    and extract the 24-feature matrix X of shape (T, K, 24).
    """
    tx = proc_trip["tx"]
    ty = proc_trip["ty"]
    dt = proc_trip["dt"]
    nx = proc_trip["nx"]
    ny = proc_trip["ny"]
    arcs = proc_trip["arcs"]
    route = proc_trip["route"]
    T = len(tx)
    num_arcs = len(arcs)

    a_from = arcs[:, 0]
    a_to = arcs[:, 1]
    ax, ay = nx[a_from], ny[a_from]
    bx, by = nx[a_to], ny[a_to]
    mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
    q1x, q1y = 0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by
    q3x, q3y = 0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by

    pts = np.vstack([
        np.column_stack([mx, my]),
        np.column_stack([q1x, q1y]),
        np.column_stack([q3x, q3y]),
    ])
    tree = cKDTree(pts)
    k_query = min(pre_k, len(pts))
    _, idx = tree.query(np.column_stack([tx, ty]), k=k_query)
    if idx.ndim == 1:
        idx = idx[:, None]
    cand_pool = idx % num_arcs  # (T, k_query)

    # Deduplicate candidate arc_ids per GPS point
    order = np.argsort(cand_pool, axis=1)
    sorted_pool = np.take_along_axis(cand_pool, order, axis=1)
    dup_sorted = np.zeros_like(sorted_pool, dtype=bool)
    dup_sorted[:, 1:] = (sorted_pool[:, 1:] == sorted_pool[:, :-1])
    inv_order = np.argsort(order, axis=1)
    dup_mask = np.take_along_axis(dup_sorted, inv_order, axis=1)

    c_from = arcs[cand_pool, 0]
    c_to = arcs[cand_pool, 1]
    c_ax, c_ay = nx[c_from], ny[c_from]
    c_bx, c_by = nx[c_to], ny[c_to]

    dist_pool, _ = point_to_segments_dist(
        tx[:, None], ty[:, None], c_ax, c_ay, c_bx, c_by
    )

    # Velocity vectors (instantaneous, 5-step smooth, and 20m displacement-triggered tangent)
    vx_inst = np.gradient(tx) / np.maximum(dt, 1e-3)
    vy_inst = np.gradient(ty) / np.maximum(dt, 1e-3)
    vx_s5 = Smooth1D(vx_inst, 5)
    vy_s5 = Smooth1D(vy_inst, 5)
    ux_disp, uy_disp = compute_displacement_tangent(tx, ty, 20.0)

    speed_inst = np.hypot(vx_inst, vy_inst)
    speed_s5 = np.hypot(vx_s5, vy_s5)

    arc_dx = c_bx - c_ax
    arc_dy = c_by - c_ay
    arc_len = np.hypot(arc_dx, arc_dy) + 1e-9

    cos_s5_pool = (vx_s5[:, None] * arc_dx + vy_s5[:, None] * arc_dy) / (
        (speed_s5[:, None] + 1e-9) * arc_len
    )
    cos_disp_pool = (ux_disp[:, None] * arc_dx + uy_disp[:, None] * arc_dy) / arc_len

    eff_cos_pool = np.where(
        (speed_s5 > 1.5)[:, None],
        0.6 * cos_s5_pool + 0.4 * cos_disp_pool,
        cos_disp_pool,
    )

    rank_score = dist_pool + np.where(
        eff_cos_pool < 0, 5.0 * (-eff_cos_pool), -0.40 * eff_cos_pool
    )
    rank_score = np.where(dup_mask, np.inf, rank_score)

    top_k_idx = np.argsort(rank_score, axis=1)[:, :K]
    cand_arcs = np.take_along_axis(cand_pool, top_k_idx, axis=1)   # (T, K)
    cand_rscore = np.take_along_axis(rank_score, top_k_idx, axis=1)

    # Re-fetch exact geometry for top-K candidates
    kf_node = arcs[cand_arcs, 0]
    kt_node = arcs[cand_arcs, 1]
    k_ax, k_ay = nx[kf_node], ny[kf_node]
    k_bx, k_by = nx[kt_node], ny[kt_node]
    k_dx = k_bx - k_ax
    k_dy = k_by - k_ay
    k_len = np.hypot(k_dx, k_dy)

    perp_dist, u_curr = point_to_segments_dist(
        tx[:, None], ty[:, None], k_ax, k_ay, k_bx, k_by
    )

    # Distances at shifted timestamps t-2, t-1, t+1, t+2 to candidate segments at step t
    idx_m2 = np.maximum(0, np.arange(T) - 2)
    idx_m1 = np.maximum(0, np.arange(T) - 1)
    idx_p1 = np.minimum(T - 1, np.arange(T) + 1)
    idx_p2 = np.minimum(T - 1, np.arange(T) + 2)

    dist_m2, _ = point_to_segments_dist(tx[idx_m2, None], ty[idx_m2, None], k_ax, k_ay, k_bx, k_by)
    dist_m1, u_m1 = point_to_segments_dist(tx[idx_m1, None], ty[idx_m1, None], k_ax, k_ay, k_bx, k_by)
    dist_p1, u_p1 = point_to_segments_dist(tx[idx_p1, None], ty[idx_p1, None], k_ax, k_ay, k_bx, k_by)
    dist_p2, _ = point_to_segments_dist(tx[idx_p2, None], ty[idx_p2, None], k_ax, k_ay, k_bx, k_by)

    dist_win5 = (dist_m2 + dist_m1 + perp_dist + dist_p1 + dist_p2) / 5.0
    proj_progress = u_p1 - u_m1

    # Endpoint distances
    dist_start = np.hypot(tx[:, None] - k_ax, ty[:, None] - k_ay)
    dist_end = np.hypot(tx[:, None] - k_bx, ty[:, None] - k_by)
    interior_margin = np.minimum(u_curr, 1.0 - u_curr)

    # Heading alignments
    k_len_safe = k_len + 1e-9
    cos_inst = (vx_inst[:, None] * k_dx + vy_inst[:, None] * k_dy) / ((speed_inst[:, None] + 1e-9) * k_len_safe)
    sin_inst = np.abs(vx_inst[:, None] * k_dy - vy_inst[:, None] * k_dx) / ((speed_inst[:, None] + 1e-9) * k_len_safe)
    cos_s5 = (vx_s5[:, None] * k_dx + vy_s5[:, None] * k_dy) / ((speed_s5[:, None] + 1e-9) * k_len_safe)
    cos_disp = (ux_disp[:, None] * k_dx + uy_disp[:, None] * k_dy) / k_len_safe

    # Relative features across the K candidates at step t
    min_d = np.min(perp_dist, axis=1, keepdims=True)
    rel_diff = perp_dist - min_d
    rel_ratio = perp_dist / (min_d + 1.0)

    # Geometric + heading softmax probability across K candidates
    geom_logit = -0.25 * perp_dist + 1.2 * cos_s5 + 1.2 * cos_disp
    geom_logit = geom_logit - np.max(geom_logit, axis=1, keepdims=True)
    exp_logit = np.exp(geom_logit)
    cand_soft_prob = exp_logit / (np.sum(exp_logit, axis=1, keepdims=True) + 1e-12)

    cand_rank = np.broadcast_to(np.linspace(0.0, 1.0, K)[None, :], (T, K))

    # Local 1-step connectivity features
    prev_cands = cand_arcs[idx_m1, :3]
    next_cands = cand_arcs[idx_p1, :3]
    prev_to_nodes = arcs[prev_cands, 1]
    next_from_nodes = arcs[next_cands, 0]

    same_prev1 = (cand_arcs == prev_cands[:, 0:1]).astype(np.float64)
    conn_prev1 = (kf_node == prev_to_nodes[:, 0:1]).astype(np.float64) * 0.85
    conn_from_prev1 = np.maximum(same_prev1, conn_prev1)

    same_next1 = (cand_arcs == next_cands[:, 0:1]).astype(np.float64)
    conn_next1 = (kt_node == next_from_nodes[:, 0:1]).astype(np.float64) * 0.85
    conn_to_next1 = np.maximum(same_next1, conn_next1)

    # Unsupervised Forward-Backward Topological Chain Score over candidate graph
    dp_f = np.zeros((T, K), dtype=np.float64)
    dp_f[0] = cand_rscore[0]
    for t in range(1, T):
        same = (cand_arcs[t - 1][:, None] == cand_arcs[t][None, :])
        conn = (kt_node[t - 1][:, None] == kf_node[t][None, :])
        pen = np.where(same, 0.0, np.where(conn, 1.2, 12.0))
        dp_f[t] = cand_rscore[t] + np.min(dp_f[t - 1][:, None] + pen, axis=0)

    dp_b = np.zeros((T, K), dtype=np.float64)
    dp_b[-1] = cand_rscore[-1]
    for t in range(T - 2, -1, -1):
        same = (cand_arcs[t][:, None] == cand_arcs[t + 1][None, :])
        conn = (kt_node[t][:, None] == kf_node[t + 1][None, :])
        pen = np.where(same, 0.0, np.where(conn, 1.2, 12.0))
        dp_b[t] = cand_rscore[t] + np.min(dp_b[t + 1][None, :] + pen, axis=1)

    fb_total = dp_f + dp_b - cand_rscore
    fb_rel_cost = np.clip(fb_total - np.min(fb_total, axis=1, keepdims=True), 0.0, 50.0)
    fb_exp = np.exp(-0.35 * fb_rel_cost)
    fb_soft_prob = fb_exp / (np.sum(fb_exp, axis=1, keepdims=True) + 1e-12)

    # Stack all 24 features: shape (T, K, 24)
    X_trip = np.stack([
        perp_dist,
        -0.5 * np.square(perp_dist / 8.0),
        u_curr,
        interior_margin,
        dist_start,
        dist_end,
        k_len,
        cos_inst,
        sin_inst,
        cos_s5,
        cos_disp,
        dist_m1,
        dist_p1,
        dist_win5,
        proj_progress,
        rel_diff,
        rel_ratio,
        cand_soft_prob,
        cand_rank,
        conn_from_prev1,
        conn_to_next1,
        fb_rel_cost,
        fb_soft_prob,
        np.broadcast_to(speed_s5[:, None], (T, K)),
    ], axis=-1).astype(np.float32)

    gt_arcs = None
    y_trip = None
    miss_mask = None
    if route is not None:
        gt_arcs = align_ground_truth_route(tx, ty, nx, ny, arcs, route)
        y_trip = (cand_arcs == gt_arcs[:, None]).astype(np.int64)
        miss_mask = ~np.any(y_trip == 1, axis=1)

    return {
        "trip_id": proc_trip["trip_id"],
        "X": X_trip,              # (T, K, 24)
        "y": y_trip,              # (T, K) binary 0/1
        "cand_arcs": cand_arcs,   # (T, K) arc_ids
        "gt_arcs": gt_arcs,       # (T,) ground-truth arc_id per step
        "miss_mask": miss_mask,   # (T,) bool where true arc not in top-K
        "dt": dt,
        "arcs": arcs,
    }


def build_or_load_partition_dataset(
    data_root: str = "Student folder",
    partition: str = "train",
    K: int = 5,
    cache_dir: str = "results/cache",
):
    """Build or load cached feature bundles for a dataset partition."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{partition}_k{K}_v2_bundles.joblib")
    if os.path.exists(cache_path):
        return joblib.load(cache_path)

    part_dir = os.path.join(data_root, partition)
    trip_ids = sorted(os.listdir(part_dir))
    trip_bundles = []
    for tid in trip_ids:
        raw = load_trip_raw(part_dir, tid)
        proc = clean_and_project_trip(raw)
        bundle = extract_trip_candidates_and_features(proc, K=K)
        trip_bundles.append(bundle)
    joblib.dump(trip_bundles, cache_path, compress=1)
    return trip_bundles

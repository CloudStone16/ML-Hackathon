import math
import numpy as np
import pandas as pd
from src.candidate_generator import SpatialCandidateGenerator
from src.config import FEATURE_COLUMNS

def compute_kinematics(track_m):
    """
    Compute vehicle instantaneous headings (degrees) and speeds (m/s).
    """
    dx = np.gradient(track_m[:, 0])
    dy = np.gradient(track_m[:, 1])
    headings = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    speeds = np.hypot(dx, dy)
    return headings, speeds

def match_ground_truth_route(track_m, route, nodes_m, arcs):
    """
    Projects GPS track points onto the chronological route sequence to assign
    the exact true arc_id for each GPS observation.
    """
    route_arcs = np.array(route, dtype=int)
    u = arcs[route_arcs, 0]
    v = arcs[route_arcs, 1]

    ax, ay = nodes_m[u, 0], nodes_m[u, 1]
    bx, by = nodes_m[v, 0], nodes_m[v, 1]
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy

    true_arc_per_point = []
    # Monotonic forward window search along route
    curr_idx = 0
    window_size = 15

    for gx, gy in track_m:
        start_w = max(0, curr_idx - 2)
        end_w = min(len(route_arcs), curr_idx + window_size)

        w_ax, w_ay = ax[start_w:end_w], ay[start_w:end_w]
        w_dx, w_dy = dx[start_w:end_w], dy[start_w:end_w]
        w_l2 = l2[start_w:end_w]

        t = np.where(w_l2 > 0, ((gx - w_ax) * w_dx + (gy - w_ay) * w_dy) / w_l2, 0.0)
        tc = np.clip(t, 0.0, 1.0)
        d = np.hypot(gx - (w_ax + tc * w_dx), gy - (w_ay + tc * w_dy))

        best_local = np.argmin(d)
        best_global = start_w + best_local
        curr_idx = best_global
        true_arc_per_point.append(route_arcs[best_global])

    return true_arc_per_point

def extract_features_for_trip(trip_data):
    """
    Extracts candidate feature table for a single trip.
    """
    trip_id = trip_data["trip_id"]
    track_m = trip_data["track_m"]
    nodes_m = trip_data["nodes_m"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]

    veh_headings, veh_speeds = compute_kinematics(track_m)

    generator = SpatialCandidateGenerator(nodes_m, arcs)
    candidates_per_obs = generator.get_candidates(track_m, veh_headings, veh_speeds)

    true_arcs = None
    if route is not None:
        true_arcs = match_ground_truth_route(track_m, route, nodes_m, arcs)

    # Pre-compute speed changes (acceleration proxy)
    speeds = veh_speeds
    speed_changes = np.abs(np.gradient(speeds))

    rows = []
    for obs_id, cands in enumerate(candidates_per_obs):
        true_a = true_arcs[obs_id] if true_arcs is not None else -1
        n_cands = len(cands)

        # Speed change at this observation
        spd_change = float(speed_changes[obs_id]) if obs_id < len(speed_changes) else 0.0

        for c in cands:
            is_positive = 1 if (c["arc_id"] == true_a) else 0

            # Normalized perpendicular distance (0=best, 1=worst among candidates)
            perp_dist_norm = c["dist_rank"] / max(n_cands, 1)

            rows.append({
                "trip_id": trip_id,
                "obs_id": obs_id,
                "arc_id": c["arc_id"],
                "perp_dist": c["perp_dist"],
                "heading_diff": c["heading_diff"],
                "seg_len": c["seg_len"],
                "proj_t": c["proj_t"],
                "dist_rank": c["dist_rank"],
                "veh_spd": c["veh_spd"],
                "spd_change": spd_change,
                "n_candidates": float(n_cands),
                "perp_dist_norm": perp_dist_norm,
                "label": is_positive
            })

    return pd.DataFrame(rows)

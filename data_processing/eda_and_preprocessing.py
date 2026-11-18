"""
PathSnap: Data Processing, Cleaning, Projection & Exploratory Data Analysis (EDA)
=================================================================================
Reads, inspects, cleans, and preprocesses the raw GPS trajectory (.track) and
road network (.nodes, .arcs, .route) files across the Training (80 trips) and
Student-Testing (10 trips) partitions.
"""

import os
import json
import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6371000.0


def load_trip_raw(partition_dir: str, trip_id: str) -> dict:
    """Load raw .track, .nodes, .arcs, and optional .route files for a trip."""
    trip_dir = os.path.join(partition_dir, trip_id)
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    track = np.loadtxt(track_path, dtype=np.float64)
    nodes = np.loadtxt(nodes_path, dtype=np.float64)
    arcs = np.loadtxt(arcs_path, dtype=np.int32)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    route = None
    if os.path.exists(route_path):
        route = np.loadtxt(route_path, dtype=np.int32)
        if route.ndim == 0:
            route = np.array([int(route)], dtype=np.int32)

    return {
        "trip_id": trip_id,
        "track": track,   # (T, 3): lon, lat, timestamp
        "nodes": nodes,   # (N, 2): lon, lat
        "arcs": arcs,     # (A, 2): from_node_id, to_node_id
        "route": route,   # (M,): chronological ground-truth arc_ids
    }


def clean_and_project_trip(trip_data: dict) -> dict:
    """
    Clean raw GPS observations (check NaNs, sort by timestamp, flag time gaps)
    and project angular (lon, lat) coordinates into a local metric tangent plane (meters).
    """
    track = trip_data["track"].copy()
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]

    # 1. Remove any NaN/Inf GPS rows if present
    valid_mask = np.isfinite(track).all(axis=1)
    n_invalid = int(np.sum(~valid_mask))
    track = track[valid_mask]

    # 2. Ensure chronological timestamp ordering
    timestamps = track[:, 2]
    if not np.all(timestamps[1:] >= timestamps[:-1]):
        order = np.argsort(timestamps, kind="mergesort")
        track = track[order]
        timestamps = track[:, 2]

    # 3. Local Equirectangular / Tangent-Plane Projection to Meters
    # Justified in EDA: 1 deg longitude scales by cos(lat), so raw degree Euclidean
    # distance distorts East-West vs North-South distances by ~45% at lat=55 deg.
    ref_lon = float(np.mean(nodes[:, 0]))
    ref_lat = float(np.mean(nodes[:, 1]))
    cos_lat = np.cos(np.radians(ref_lat))

    tx = np.radians(track[:, 0] - ref_lon) * EARTH_RADIUS_M * cos_lat
    ty = np.radians(track[:, 1] - ref_lat) * EARTH_RADIUS_M

    nx = np.radians(nodes[:, 0] - ref_lon) * EARTH_RADIUS_M * cos_lat
    ny = np.radians(nodes[:, 1] - ref_lat) * EARTH_RADIUS_M

    # Time intervals dt
    dt = np.ones(len(track), dtype=np.float64)
    if len(track) > 1:
        dt[1:] = np.maximum(1e-3, timestamps[1:] - timestamps[:-1])
        dt[0] = dt[1]

    return {
        "trip_id": trip_data["trip_id"],
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
        "tx": tx,
        "ty": ty,
        "timestamps": timestamps,
        "dt": dt,
        "nx": nx,
        "ny": ny,
        "ref_lon": ref_lon,
        "ref_lat": ref_lat,
        "n_invalid": n_invalid,
    }


def run_dataset_inventory(data_root: str = "Student folder", output_dir: str = "results") -> pd.DataFrame:
    """
    Generate full EDA inventory and summary statistics for all 80 training trips
    and 10 student-testing trips, plus identify the 10 reserved evaluator trip IDs.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for part_name, sub_dir in [("train", "train"), ("student_test", "student_test")]:
        part_path = os.path.join(data_root, sub_dir)
        trip_ids = sorted(os.listdir(part_path))
        for tid in trip_ids:
            raw = load_trip_raw(part_path, tid)
            proc = clean_and_project_trip(raw)
            track = proc["track"]
            nodes = proc["nodes"]
            arcs = proc["arcs"]
            route = proc["route"]
            dt = proc["dt"][1:] if len(track) > 1 else np.array([1.0])

            # Step distances
            step_d = np.hypot(np.diff(proc["tx"]), np.diff(proc["ty"])) if len(track) > 1 else np.array([0.0])
            traj_len_m = float(np.sum(step_d))

            # Road network connectivity stats
            active_nodes = np.unique(arcs.ravel())
            n_isolated_nodes = int(len(nodes) - len(active_nodes))

            # Reference route continuity
            route_conn = 0
            if route is not None and len(route) > 1:
                for i in range(len(route) - 1):
                    if arcs[route[i], 1] == arcs[route[i + 1], 0] or route[i] == route[i + 1]:
                        route_conn += 1
                route_conn_ratio = route_conn / (len(route) - 1)
            else:
                route_conn_ratio = 1.0

            records.append({
                "partition": part_name,
                "trip_id": tid,
                "n_gps_points": len(track),
                "n_nodes": len(nodes),
                "n_arcs": len(arcs),
                "n_route_arcs": len(route) if route is not None else 0,
                "n_invalid_gps": proc["n_invalid"],
                "duration_sec": float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0,
                "traj_length_m": round(traj_len_m, 2),
                "pct_1hz_steps": round(float(np.mean(np.isclose(dt, 1.0))) * 100.0, 2),
                "max_time_gap_s": float(np.max(dt)),
                "n_gaps_gt_1s": int(np.sum(dt > 1.05)),
                "mean_speed_mps": round(float(np.mean(step_d / dt)), 2),
                "n_isolated_nodes": n_isolated_nodes,
                "route_connectivity": round(route_conn_ratio, 4),
                "mean_lon": round(proc["ref_lon"], 4),
                "mean_lat": round(proc["ref_lat"], 4),
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "dataset_inventory.csv"), index=False)

    # Summary statistics by partition
    summary = df.groupby("partition").agg(
        num_trips=("trip_id", "count"),
        total_gps_points=("n_gps_points", "sum"),
        mean_gps_points=("n_gps_points", "mean"),
        min_gps_points=("n_gps_points", "min"),
        max_gps_points=("n_gps_points", "max"),
        mean_nodes=("n_nodes", "mean"),
        mean_arcs=("n_arcs", "mean"),
        mean_route_arcs=("n_route_arcs", "mean"),
        total_missing_invalid=("n_invalid_gps", "sum"),
        mean_pct_1hz=("pct_1hz_steps", "mean"),
        max_time_gap_s=("max_time_gap_s", "max"),
        total_gaps_gt_1s=("n_gaps_gt_1s", "sum"),
        mean_traj_length_m=("traj_length_m", "mean"),
        mean_isolated_nodes=("n_isolated_nodes", "mean"),
        mean_route_connectivity=("route_connectivity", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "eda_partition_summary.csv"), index=False)

    # Identify 10 reserved evaluator trips
    all_100 = {f"{i:08d}" for i in range(100)}
    known = set(df["trip_id"].unique())
    evaluator_trips = sorted(all_100 - known)
    with open(os.path.join(output_dir, "evaluator_trips.json"), "w") as f:
        json.dump({"evaluator_trips": evaluator_trips}, f, indent=2)

    return df, summary, evaluator_trips


if __name__ == "__main__":
    df, summary, eval_trips = run_dataset_inventory()
    print("=== EDA Partition Summary ===")
    print(summary.to_string(index=False))
    print("\nReserved Evaluator Trips (10):", eval_trips)

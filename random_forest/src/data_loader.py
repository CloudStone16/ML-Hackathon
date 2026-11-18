import os
import math
import numpy as np
from src.config import REF_LAT, EARTH_RADIUS

COS_LAT = math.cos(math.radians(REF_LAT))
SCALE_FACTOR = EARTH_RADIUS * math.pi / 180.0

def project_coords(lon_arr, lat_arr):
    """
    Project angular degrees (WGS84 lon, lat) into local metric coordinates (x, y in meters).
    Uses equirectangular projection centered around Darmstadt, Germany.
    """
    x = lon_arr * COS_LAT * SCALE_FACTOR
    y = lat_arr * SCALE_FACTOR
    return x, y

def unproject_coords(x_arr, y_arr):
    """
    Inverse projection from local metric coordinates back to (lon, lat) degrees.
    """
    lon = x_arr / (COS_LAT * SCALE_FACTOR)
    lat = y_arr / SCALE_FACTOR
    return lon, lat

def load_trip_data(trip_dir, trip_id):
    """
    Load raw files for a given trip ID.
    Returns:
        nodes: np.ndarray shape (N, 2) [lon, lat]
        nodes_m: np.ndarray shape (N, 2) [x_m, y_m]
        arcs: np.ndarray shape (M, 2) [from_node, to_node]
        track: np.ndarray shape (T, 3) [lon, lat, timestamp]
        track_m: np.ndarray shape (T, 2) [x_m, y_m]
        route: list of int (ground truth arc_ids), or None if not present
    """
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    nodes = np.fromfile(nodes_path, sep=" ").reshape(-1, 2)
    arcs = np.fromfile(arcs_path, sep=" ", dtype=int).reshape(-1, 2)
    track = np.fromfile(track_path, sep=" ").reshape(-1, 3)

    x_nodes, y_nodes = project_coords(nodes[:, 0], nodes[:, 1])
    nodes_m = np.column_stack([x_nodes, y_nodes])

    x_track, y_track = project_coords(track[:, 0], track[:, 1])
    track_m = np.column_stack([x_track, y_track])

    route = None
    if os.path.exists(route_path):
        with open(route_path, "r") as f:
            route = [int(line.strip()) for line in f if line.strip()]

    return {
        "trip_id": trip_id,
        "nodes": nodes,
        "nodes_m": nodes_m,
        "arcs": arcs,
        "track": track,
        "track_m": track_m,
        "route": route
    }

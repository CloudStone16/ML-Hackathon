import os
import sys

# Ensure random_forest root is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if RF_ROOT not in sys.path:
    sys.path.insert(0, RF_ROOT)

import json
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins

from src.config import TRAIN_DIR, TEST_DIR, OUTPUTS_DIR
from src.data_loader import load_trip_data

def run_eda():
    print("=" * 60)
    print("RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    train_trips = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    test_trips = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])

    print(f"Found {len(train_trips)} training trips and {len(test_trips)} student test trips.")

    # Inspect first 10 trips to gather statistics
    stats = []
    sample_trip_id = train_trips[0]
    sample_data = None

    for t_id in train_trips[:15]:
        t_dir = os.path.join(TRAIN_DIR, t_id)
        d = load_trip_data(t_dir, t_id)
        if t_id == sample_trip_id:
            sample_data = d

        track = d["track"]
        dt = np.diff(track[:, 2]) if len(track) > 1 else np.array([1.0])
        nominal_1hz = np.mean((dt >= 0.9) & (dt <= 1.1)) * 100.0

        stats.append({
            "trip_id": t_id,
            "num_gps_points": len(track),
            "num_nodes": len(d["nodes"]),
            "num_arcs": len(d["arcs"]),
            "route_len": len(d["route"]) if d["route"] else 0,
            "nominal_1hz_pct": round(float(nominal_1hz), 2),
            "duration_sec": round(float(track[-1, 2] - track[0, 2]), 2)
        })

    summary_file = os.path.join(OUTPUTS_DIR, "eda_summary.json")
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved EDA summary statistics to {summary_file}")

    # Generate Matplotlib plots
    plots_dir = os.path.join(OUTPUTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. GPS points distribution
    pts_counts = [s["num_gps_points"] for s in stats]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(pts_counts)), pts_counts, color="royalblue", edgecolor="black")
    plt.title("GPS Observations per Trip (Sample Trips)", fontsize=12, fontweight="bold")
    plt.xlabel("Trip Index")
    plt.ylabel("Number of Observations")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_gps_counts.png"), dpi=200)
    plt.close()

    # 2. Generate Interactive Folium Map for Sample Trip
    print(f"Generating interactive Folium map for Trip {sample_trip_id}...")
    track = sample_data["track"]
    nodes = sample_data["nodes"]
    arcs = sample_data["arcs"]
    route = sample_data["route"]

    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    # Layer 1: Ground Truth Route
    route_fg = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
    for arc_id in route:
        u = arcs[arc_id, 0]
        v = arcs[arc_id, 1]
        p1 = [nodes[u, 1], nodes[u, 0]]
        p2 = [nodes[v, 1], nodes[v, 0]]
        folium.PolyLine([p1, p2], color="forestgreen", weight=5, opacity=0.8).add_to(route_fg)
    route_fg.add_to(fmap)

    # Layer 2: Raw GPS Points
    gps_fg = folium.FeatureGroup(name="Raw Noisy GPS Points (Blue Dots)", show=True)
    for i in range(0, len(track), 2):  # plot every 2nd point for visual clarity
        folium.CircleMarker(
            location=[track[i, 1], track[i, 0]],
            radius=3,
            color="dodgerblue",
            fill=True,
            fill_color="dodgerblue",
            fill_opacity=0.9,
            popup=f"GPS #{i} (t={track[i, 2]:.0f}s)"
        ).add_to(gps_fg)
    gps_fg.add_to(fmap)

    folium.LayerControl().add_to(fmap)

    map_path = os.path.join(OUTPUTS_DIR, f"trip_{sample_trip_id}_folium_map.html")
    fmap.save(map_path)
    print(f"Saved interactive Folium map to {map_path}")
    print("EDA and Visualizations Complete!\n")

if __name__ == "__main__":
    run_eda()

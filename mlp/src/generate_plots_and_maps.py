"""
PathSnap: Comprehensive Visualisation & Interactive Folium/Leaflet Map Generator
================================================================================
Generates all required visualisations for the Technical Report & GitHub Repository:
1. EDA plots (trajectory length, sampling gaps, road network topology, anomaly analysis)
2. Cross-validation Macro F1-score comparison across all 6 models
3. Student-testing Top-1 Accuracy, Macro F1-score, and Route Continuity comparison
4. Log Loss and Inference Time comparison across all 6 models
5. Confusion matrices for all 6 models on the student-testing partition
6. Epoch-wise training & validation loss history curves exclusively for PathSnap Deep MLP
7. Bayesian Hyperparameter Optimisation trial progression and Untuned vs Tuned comparison
8. HMM + Viterbi Route Reconstruction vs Classifier-Only comparison
9. Interactive Folium/Leaflet HTML maps with layer controls + high-res map figures
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib

from data_processing.eda_and_preprocessing import load_trip_raw, clean_and_project_trip
from feature_engineering.candidate_features import build_or_load_partition_dataset
from models.mlp_model import PathSnapMLPNet
from hmm_viterbi.viterbi_decoder import PathSnapViterbiDecoder


plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def generate_all_plots(
    data_root: str = "Student folder",
    results_dir: str = "results",
    save_dir: str = "saved_models",
    out_dir: str = "visualisations/outputs",
):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    # -------------------------------------------------------------------------
    # 1. EDA Overview Plot (4 panels)
    # -------------------------------------------------------------------------
    df_inv = pd.read_csv(os.path.join(results_dir, "dataset_inventory.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)

    # Panel A: GPS points per trip distribution
    ax = axes[0, 0]
    sns.histplot(
        data=df_inv, x="n_gps_points", hue="partition", bins=25, kde=True, ax=ax,
        palette={"train": "#2b5c8f", "student_test": "#d95f02"}
    )
    ax.set_title("(a) Distribution of GPS Observations per Trip", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of GPS Points (.track rows)")
    ax.set_ylabel("Number of Trips")

    # Panel B: Road Network Nodes vs Directed Arcs
    ax = axes[0, 1]
    sns.scatterplot(
        data=df_inv, x="n_nodes", y="n_arcs", hue="partition", style="partition",
        s=75, alpha=0.85, ax=ax, palette={"train": "#2b5c8f", "student_test": "#d95f02"}
    )
    ax.set_title("(b) Road Network Scale: Nodes vs. Directed Road Segments", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Road Nodes (.nodes)")
    ax.set_ylabel("Number of Directed Arcs (.arcs)")

    # Panel C: Sampling Interval Regularity (% 1Hz steps vs Max Gap)
    ax = axes[1, 0]
    sns.scatterplot(
        data=df_inv, x="pct_1hz_steps", y="n_gaps_gt_1s", hue="partition",
        s=75, alpha=0.85, ax=ax, palette={"train": "#2b5c8f", "student_test": "#d95f02"}
    )
    ax.set_title("(c) Sampling Regularity (% 1Hz Steps vs. Irregular Gaps > 1s)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Percentage of Nominally 1.0s Steps (%)")
    ax.set_ylabel("Count of Irregular Time Gaps (dt > 1s)")

    # Panel D: Trajectory Length vs Traversed Ground-Truth Route Segments
    ax = axes[1, 1]
    sns.scatterplot(
        data=df_inv, x="traj_length_m", y="n_route_arcs", hue="partition",
        s=75, alpha=0.85, ax=ax, palette={"train": "#2b5c8f", "student_test": "#d95f02"}
    )
    ax.set_title("(d) Trajectory Length (m) vs. Ground-Truth Route Arcs (.route)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Trajectory Distance (meters)")
    ax.set_ylabel("Number of Ground-Truth Arcs in .route")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_eda_overview.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Model Comparison Plots (CV Macro F1, Student-Test Top-1 & F1, Log Loss)
    # -------------------------------------------------------------------------
    df_comp = pd.read_csv(os.path.join(results_dir, "model_comparison_table.csv"))
    with open(os.path.join(results_dir, "model_comparison_detailed.json"), "r") as f:
        comp_detail = json.load(f)

    # Sort with PathSnap Deep MLP first, then by Student-Test Macro F1
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)

    models_short = [
        m.replace(" (HistGradientBoosting)", "\n(HistGBDT)").replace(" (Ours)", "\n[OURS - SELECTED]")
        for m in df_comp["Model"]
    ]
    colors = ["#1b7837" if "MLP" in m else "#4575b4" for m in df_comp["Model"]]

    # Left: 5-Fold GroupKFold CV Macro F1-score with error bars
    ax = axes[0]
    y_pos = np.arange(len(df_comp))
    bars = ax.barh(
        y_pos, df_comp["CV Macro F1-score"], xerr=df_comp["CV Macro F1 Std"],
        color=colors, alpha=0.9, capsize=4, edgecolor="#222222"
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_short, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.65, min(1.0, float(df_comp["CV Macro F1-score"].max()) + 0.06))
    ax.set_xlabel("5-Fold GroupKFold Cross-Validation Macro F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Validation Macro F1-Score (80 Train Trips)", fontsize=12, fontweight="bold")
    for bar, val, std in zip(bars, df_comp["CV Macro F1-score"], df_comp["CV Macro F1 Std"]):
        ax.text(val + std + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9.5, fontweight="bold")

    # Right: Student-Test Macro F1 & Top-1 Accuracy
    ax = axes[1]
    width = 0.36
    b1 = ax.barh(y_pos - width / 2, df_comp["Student-Test Macro F1-score"], height=width, label="Student-Test Macro F1", color="#2b5c8f")
    b2 = ax.barh(y_pos + width / 2, df_comp["Top-1 Accuracy"], height=width, label="Observation Top-1 Accuracy", color="#e08214")
    # Highlight our MLP bars
    b1[0].set_color("#1b7837")
    b2[0].set_color("#66bd63")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_short, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.65, 1.0)
    ax.set_xlabel("Score on Student-Testing Partition (10 Trips)", fontsize=11, fontweight="bold")
    ax.set_title("Student-Testing Macro F1 & Top-1 Accuracy", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    for bar in b1:
        w = bar.get_width()
        ax.text(w + 0.004, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.5)
    for bar in b2:
        w = bar.get_width()
        ax.text(w + 0.004, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_model_f1_top1_comparison.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Log Loss, Route Continuity & Inference Time Comparison
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=180)

    # Panel A: Log Loss (lower is better)
    ax = axes[0]
    bars = ax.barh(y_pos, df_comp["Log Loss"], color=colors, edgecolor="#222222")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_short, fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Binary Log Loss (Lower is Better)", fontweight="bold")
    ax.set_title("(a) Probability Calibration (Log Loss)", fontweight="bold")
    for bar, v in zip(bars, df_comp["Log Loss"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)

    # Panel B: Classifier-Only Route Continuity
    ax = axes[1]
    bars = ax.barh(y_pos, df_comp["Route Continuity"], color=colors, edgecolor="#222222")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("Route Continuity Ratio (Higher is Better)", fontweight="bold")
    ax.set_title("(b) Classifier-Only Route Continuity", fontweight="bold")
    for bar, v in zip(bars, df_comp["Route Continuity"]):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)

    # Panel C: Inference Time on 10 Student-Test Trips (s)
    ax = axes[2]
    bars = ax.barh(y_pos, df_comp["Inference Time (s)"], color=colors, edgecolor="#222222")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("Total Inference Time for 10 Trips (seconds)", fontweight="bold")
    ax.set_title("(c) Computational Efficiency (Inference)", fontweight="bold")
    for bar, v in zip(bars, df_comp["Inference Time (s)"]):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2, f"{v:.3f}s", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_logloss_continuity_runtime.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 4. Confusion Matrices for All 6 Models
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=180)
    for idx, m_name in enumerate(df_comp["Model"]):
        ax = axes[idx // 3, idx % 3]
        cm = np.array(comp_detail[m_name]["student_test_metrics"]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        annot = np.empty_like(cm, dtype=object)
        for r in range(2):
            for c in range(2):
                annot[r, c] = f"{cm[r, c]:,}\n({cm_norm[r, c]*100:.1f}%)"
        cmap = "Greens" if "MLP" in m_name else "Blues"
        sns.heatmap(
            cm_norm, annot=annot, fmt="", cmap=cmap, cbar=False, ax=ax,
            xticklabels=["Pred 0 (Non-Route)", "Pred 1 (True Arc)"],
            yticklabels=["Actual 0", "Actual 1"],
            annot_kws={"size": 10, "weight": "bold"},
        )
        f1_v = comp_detail[m_name]["summary"]["Student-Test Macro F1-score"]
        t1_v = comp_detail[m_name]["summary"]["Top-1 Accuracy"]
        ax.set_title(f"{m_name}\nMacro F1={f1_v:.4f} | Top-1={t1_v:.4f}", fontsize=10.5, fontweight="bold")

    plt.suptitle("Student-Test Confusion Matrices Across All 6 Evaluated Models (95,800 Candidate Pairs)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_confusion_matrices_all_models.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 5. Epoch-Wise Training & Validation Loss Plots (Exclusively for Deep MLP)
    # -------------------------------------------------------------------------
    hist_untuned_path = os.path.join(results_dir, "mlp_training_history.csv")
    hist_tuned_path = os.path.join(results_dir, "tuned_mlp_training_history.csv")
    if os.path.exists(hist_untuned_path):
        df_hu = pd.read_csv(hist_untuned_path)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)

        ax = axes[0]
        epochs_u = np.arange(1, len(df_hu) + 1)
        ax.plot(epochs_u, df_hu["train_loss"], "o-", color="#2b5c8f", lw=2, label="Training Loss (Hybrid BCE+ListCE)")
        ax.plot(epochs_u, df_hu["val_loss"], "s--", color="#d95f02", lw=2, label="Validation Loss (Held-out Trips)")
        ax.set_title("(a) Untuned PathSnap Deep MLP Learning Curve", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Hybrid Loss (BCE + Listwise CE)")
        ax.legend(frameon=True)

        ax = axes[1]
        if os.path.exists(hist_tuned_path):
            df_ht = pd.read_csv(hist_tuned_path)
            epochs_t = np.arange(1, len(df_ht) + 1)
            ax.plot(epochs_t, df_ht["train_loss"], "o-", color="#1b7837", lw=2, label="Tuned MLP Training Loss")
            ax.plot(epochs_t, df_ht["val_loss"], "s--", color="#e08214", lw=2, label="Tuned MLP Validation Loss")
            best_ep = int(np.argmin(df_ht["val_loss"])) + 1
            ax.axvline(best_ep, color="#b2182b", linestyle=":", lw=1.5, label=f"Best Checkpoint (Epoch {best_ep})")
        ax.set_title("(b) Bayesian-Tuned PathSnap Deep MLP Learning Curve", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Hybrid Loss (BCE + Listwise CE)")
        ax.legend(frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig5_mlp_epoch_loss_curves.png"), bbox_inches="tight")
        plt.close()

    # -------------------------------------------------------------------------
    # 6. Bayesian Optimisation Progression & HMM/Viterbi Route Reconstruction Plot
    # -------------------------------------------------------------------------
    bo_trials_path = os.path.join(results_dir, "bayesian_optimisation_trials.csv")
    hmm_trips_path = os.path.join(results_dir, "hmm_viterbi_per_trip_results.csv")
    if os.path.exists(bo_trials_path) and os.path.exists(hmm_trips_path):
        df_bo = pd.read_csv(bo_trials_path)
        df_hmm = pd.read_csv(hmm_trips_path)

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=180)

        # Panel A: Bayesian Optimisation Convergence
        ax = axes[0]
        trials = df_bo["trial"]
        scores = df_bo["cv_macro_f1"]
        best_so_far = np.maximum.accumulate(scores)
        ax.scatter(trials, scores, color="#2b5c8f", s=65, zorder=3, label="Trial CV Macro F1")
        ax.plot(trials, best_so_far, color="#1b7837", lw=2.5, label="Best-So-Far CV Macro F1")
        ax.set_title("(a) Bayesian Hyperparameter Optimisation (GP-EI) Progression", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Bayesian Optimisation Trial")
        ax.set_ylabel("GroupKFold CV Macro F1-Score")
        ax.legend(loc="lower right", frameon=True)

        # Panel B: Per-Trip Classifier-Only vs Classifier + HMM/Viterbi
        ax = axes[1]
        x_idx = np.arange(len(df_hmm))
        w = 0.25
        ax.bar(x_idx - w, df_hmm["clf_top1_accuracy"], width=w, label="MLP Classifier-Only Top-1 Acc", color="#74add1")
        ax.bar(x_idx, df_hmm["hmm_viterbi_top1_accuracy"], width=w, label="MLP + HMM/Viterbi Top-1 Acc", color="#1b7837")
        ax.bar(x_idx + w, df_hmm["hmm_viterbi_route_continuity"], width=w, label="MLP + HMM/Viterbi Continuity", color="#fdae61")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(df_hmm["trip_id"], rotation=35, fontsize=9)
        ax.set_ylim(0.70, 1.03)
        ax.set_title("(b) Student-Test Trips: Classifier-Only vs. MLP + HMM/Viterbi", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Student-Testing Trip ID")
        ax.set_ylabel("Accuracy / Continuity Ratio")
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig6_bayesian_and_hmm_viterbi.png"), bbox_inches="tight")
        plt.close()

    # -------------------------------------------------------------------------
    # 7. Interactive Folium/Leaflet HTML Map & High-Res Geographical Plot
    # -------------------------------------------------------------------------
    generate_geographical_maps(data_root, save_dir, out_dir)
    print(f"All figures and interactive maps saved to '{out_dir}/'")


def generate_geographical_maps(data_root: str, save_dir: str, out_dir: str):
    """
    Generate interactive Leaflet/Folium HTML maps and static high-res comparison maps
    for representative trips (e.g., 00000000 and 00000007) showing:
    1. Raw GPS observations
    2. Road network geometry
    3. Ground-truth reference route
    4. PathSnap Deep MLP + HMM/Viterbi predicted route
    """
    test_bundles = build_or_load_partition_dataset(data_root, "student_test", K=5)
    with open(os.path.join(save_dir, "model_config.json"), "r") as f:
        cfg = json.load(f)

    best_hp = cfg.get("best_hyperparameters") or cfg.get("best_bayesian_hyperparameters")
    scaler = joblib.load(os.path.join(save_dir, "feature_scaler.joblib"))
    net = PathSnapMLPNet(
        in_features=24,
        hidden_dim=best_hp["hidden_dim"],
        head_dim=best_hp["head_dim"],
        dropout=best_hp["dropout"],
    )
    net.load_state_dict(torch.load(os.path.join(save_dir, "pathsnap_mlp_weights.pt"), map_location="cpu"))
    net.eval()

    decoder = PathSnapViterbiDecoder(**cfg["hmm_viterbi_config"])

    # Static 2-panel geographical figure for Trip 00000000 and Trip 00000007
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=180)

    for plot_i, target_tid in enumerate(["00000000", "00000007"]):
        bundle = next(b for b in test_bundles if b["trip_id"] == target_tid)
        raw = load_trip_raw(os.path.join(data_root, "student_test"), target_tid)
        track, nodes, arcs, route = raw["track"], raw["nodes"], raw["arcs"], raw["route"]

        T, K, D = bundle["X"].shape
        xs = scaler.transform(bundle["X"].reshape(-1, D)).reshape(T, K, D).astype(np.float32)
        with torch.no_grad():
            logits = net(torch.from_numpy(xs))
            probs = (0.55 * torch.sigmoid(logits) + 0.45 * torch.softmax(logits, dim=1)).numpy()

        dec = decoder.decode_trip(bundle, probs)
        vit_arcs = dec["viterbi_arcs"]
        clf_arcs = dec["classifier_arcs"]

        # Create Interactive Folium/Leaflet HTML Map
        html_path = os.path.join(out_dir, f"folium_map_trip_{target_tid}.html")
        write_interactive_folium_html(html_path, target_tid, track, nodes, arcs, route, clf_arcs, vit_arcs)

        # Plot on static Matplotlib subplot for PDF report inclusion
        ax = axes[plot_i]
        # Filter road network segments near trajectory bbox for clean plotting
        lon_min, lat_min = track[:, 0].min() - 0.003, track[:, 1].min() - 0.003
        lon_max, lat_max = track[:, 0].max() + 0.003, track[:, 1].max() + 0.003

        # Plot road network segments in grey
        u_nodes = nodes[arcs[:, 0]]
        v_nodes = nodes[arcs[:, 1]]
        in_box = (
            (u_nodes[:, 0] >= lon_min) & (u_nodes[:, 0] <= lon_max) &
            (u_nodes[:, 1] >= lat_min) & (u_nodes[:, 1] <= lat_max)
        )
        for u_pt, v_pt in zip(u_nodes[in_box][::2], v_nodes[in_box][::2]):
            ax.plot([u_pt[0], v_pt[0]], [u_pt[1], v_pt[1]], color="#cccccc", lw=0.6, zorder=1)

        # Plot Reference Route in thick green
        for idx_r, a_id in enumerate(route):
            u_pt, v_pt = nodes[arcs[a_id, 0]], nodes[arcs[a_id, 1]]
            ax.plot(
                [u_pt[0], v_pt[0]], [u_pt[1], v_pt[1]],
                color="#2ca25f", lw=3.5, alpha=0.85, zorder=2,
                label="Ground-Truth Reference Route (.route)" if idx_r == 0 else None
            )

        # Plot Predicted HMM/Viterbi Route in dashed red/magenta
        unique_vit = []
        for a_id in vit_arcs:
            if not unique_vit or unique_vit[-1] != a_id:
                unique_vit.append(int(a_id))
        for idx_p, a_id in enumerate(unique_vit):
            u_pt, v_pt = nodes[arcs[a_id, 0]], nodes[arcs[a_id, 1]]
            ax.plot(
                [u_pt[0], v_pt[0]], [u_pt[1], v_pt[1]],
                color="#d7191c", lw=1.8, linestyle="--", zorder=3,
                label="PathSnap MLP + HMM/Viterbi Route" if idx_p == 0 else None
            )

        # Plot Raw Noisy GPS points in blue
        ax.scatter(
            track[::3, 0], track[::3, 1], color="#2b83ba", s=8, alpha=0.75, zorder=4,
            label="Raw Noisy GPS Observations (.track)"
        )

        top1 = float(np.mean(vit_arcs == bundle["gt_arcs"]))
        cont = float(np.mean((vit_arcs[:-1] == vit_arcs[1:]) | (arcs[vit_arcs[:-1], 1] == arcs[vit_arcs[1:], 0])))
        ax.set_title(
            f"Trip {target_tid}: Geographical Map-Matching Reconstruction\n"
            f"Top-1 Accuracy = {top1*100:.2f}% | Route Continuity = {cont*100:.2f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.legend(loc="best", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_geographical_route_reconstruction.png"), bbox_inches="tight")
    plt.close()


def write_interactive_folium_html(
    html_path: str,
    trip_id: str,
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
    clf_arcs: np.ndarray,
    vit_arcs: np.ndarray,
):
    """
    Generate a self-contained Folium/Leaflet interactive HTML map with:
    - OpenStreetMap tiles
    - Toggleable Layer Control for:
      1. Road Network Segments (nearby)
      2. Raw GPS Observations (Blue CircleMarkers)
      3. Ground-Truth Reference Route (Green PolyLine)
      4. Classifier-Only Predicted Route (Orange Dashed)
      5. PathSnap MLP + HMM/Viterbi Route (Red PolyLine)
    - Custom Legend & Interactive Popups
    """
    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))

    # Collect nearby road segments (within 150m of GPS points) to keep HTML fast and interactive
    cand_set = set(np.unique(clf_arcs).tolist()) | set(np.unique(vit_arcs).tolist())
    if route is not None:
        cand_set |= set(np.unique(route).tolist())

    # Add sample of local road segments around trajectory
    u_lon, u_lat = nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1]
    in_bbox = (
        (u_lon >= track[:, 0].min() - 0.005) & (u_lon <= track[:, 0].max() + 0.005) &
        (u_lat >= track[:, 1].min() - 0.005) & (u_lat <= track[:, 1].max() + 0.005)
    )
    nearby_arc_ids = np.where(in_bbox)[0][::3][:1200]

    def arcs_to_latlon_segments(arc_id_list):
        segs = []
        for aid in arc_id_list:
            u, v = arcs[int(aid)]
            segs.append([[round(float(nodes[u, 1]), 6), round(float(nodes[u, 0]), 6)],
                         [round(float(nodes[v, 1]), 6), round(float(nodes[v, 0]), 6)]])
        return segs

    road_segs = arcs_to_latlon_segments(nearby_arc_ids)
    gt_segs = arcs_to_latlon_segments(route) if route is not None else []

    dedup_clf = [int(clf_arcs[0])]
    for a in clf_arcs[1:]:
        if int(a) != dedup_clf[-1]:
            dedup_clf.append(int(a))
    clf_segs = arcs_to_latlon_segments(dedup_clf)

    dedup_vit = [int(vit_arcs[0])]
    for a in vit_arcs[1:]:
        if int(a) != dedup_vit[-1]:
            dedup_vit.append(int(a))
    vit_segs = arcs_to_latlon_segments(dedup_vit)

    # Sample GPS points (every 2nd point if >1000 for smooth rendering)
    step = 2 if len(track) > 1200 else 1
    gps_pts = [
        [round(float(track[i, 1]), 6), round(float(track[i, 0]), 6), float(track[i, 2])]
        for i in range(0, len(track), step)
    ]

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>PathSnap Folium Map - Trip {trip_id}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body, #map {{ height: 100%; width: 100%; margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        .legend {{
            background: white; padding: 12px 15px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2); line-height: 22px; font-size: 13px;
        }}
        .legend h4 {{ margin: 0 0 8px 0; font-size: 14px; color: #1a1a1a; }}
        .swatch {{ display: inline-block; width: 22px; height: 4px; margin-right: 8px; vertical-align: middle; }}
        .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 14px; margin-left: 6px; vertical-align: middle; }}
    </style>
</head>
<body>
<div id="map"></div>
<script>
    var map = L.map('map').setView([{center_lat}, {center_lon}], 14);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors | PathSnap Folium Visualisation'
    }}).addTo(map);

    var roadLayer = L.layerGroup();
    var gpsLayer = L.layerGroup();
    var gtLayer = L.layerGroup();
    var clfLayer = L.layerGroup();
    var vitLayer = L.layerGroup();

    var roadSegs = {json.dumps(road_segs)};
    roadSegs.forEach(function(s) {{
        L.polyline(s, {{color: '#888888', weight: 1.5, opacity: 0.45}}).addTo(roadLayer);
    }});

    var gtSegs = {json.dumps(gt_segs)};
    gtSegs.forEach(function(s, idx) {{
        L.polyline(s, {{color: '#1b9e77', weight: 6, opacity: 0.85}})
         .bindPopup('Ground-Truth Route Segment #' + idx).addTo(gtLayer);
    }});

    var clfSegs = {json.dumps(clf_segs)};
    clfSegs.forEach(function(s, idx) {{
        L.polyline(s, {{color: '#e6ab02', weight: 3.5, opacity: 0.85, dashArray: '6, 6'}})
         .bindPopup('MLP Classifier-Only Segment #' + idx).addTo(clfLayer);
    }});

    var vitSegs = {json.dumps(vit_segs)};
    vitSegs.forEach(function(s, idx) {{
        L.polyline(s, {{color: '#e41a1c', weight: 3.5, opacity: 0.95}})
         .bindPopup('PathSnap MLP + HMM/Viterbi Segment #' + idx).addTo(vitLayer);
    }});

    var gpsPts = {json.dumps(gps_pts)};
    gpsPts.forEach(function(pt) {{
        L.circleMarker([pt[0], pt[1]], {{
            radius: 3.5, color: '#1f78b4', fillColor: '#1f78b4', fillOpacity: 0.85, weight: 1
        }}).bindPopup('GPS Observation (t=' + pt[2] + 's)<br>Lat: ' + pt[0] + ', Lon: ' + pt[1]).addTo(gpsLayer);
    }});

    roadLayer.addTo(map);
    gtLayer.addTo(map);
    vitLayer.addTo(map);
    gpsLayer.addTo(map);

    var overlays = {{
        "Road Network Geometry (.nodes / .arcs)": roadLayer,
        "Ground-Truth Reference Route (.route)": gtLayer,
        "PathSnap Deep MLP (Classifier-Only)": clfLayer,
        "PathSnap Deep MLP + HMM/Viterbi Route": vitLayer,
        "Raw Noisy GPS Observations (.track)": gpsLayer
    }};
    L.control.layers(null, overlays, {{collapsed: false}}).addTo(map);

    if (gtSegs.length > 0) {{
        var bounds = L.latLngBounds(gtSegs.map(function(s) {{ return s[0]; }}));
        map.fitBounds(bounds.pad(0.1));
    }}

    var legend = L.control({{position: 'bottomright'}});
    legend.onAdd = function(map) {{
        var div = L.DomUtil.create('div', 'legend');
        div.innerHTML = '<h4>PathSnap — Trip {trip_id}</h4>' +
            '<div><span class="dot" style="background:#1f78b4;"></span>Raw GPS Observations (.track)</div>' +
            '<div><span class="swatch" style="background:#1b9e77; height:6px;"></span>Reference Ground-Truth Route (.route)</div>' +
            '<div><span class="swatch" style="background:#e41a1c;"></span>Predicted Route (MLP + HMM/Viterbi)</div>' +
            '<div><span class="swatch" style="background:#e6ab02; border-bottom: 2px dashed #e6ab02;"></span>Classifier-Only Route (MLP)</div>' +
            '<div><span class="swatch" style="background:#888888; height:2px;"></span>Road Network Segments (.arcs)</div>';
        return div;
    }};
    legend.addTo(map);
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    generate_all_plots()

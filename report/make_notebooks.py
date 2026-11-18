"""Generate the four visualisation notebooks for the report (report/notebooks/*.ipynb).

Each notebook reads only files that already exist in the repo (raw trips, results/ CSV and JSON files) and writes its
figures to report/figures/ and its tables to report/tables/. Execute them with report/run_notebooks.sh.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB_DIR = HERE / "notebooks"
NB_DIR.mkdir(exist_ok=True)

SETUP = '''
from pathlib import Path
import json, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

ROOT = Path.cwd().resolve().parents[1]          # report/notebooks -> repository root
sys.path.insert(0, str(ROOT / "src"))
FIG, TAB = ROOT / "report" / "figures", ROOT / "report" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"
DATA = {"train": ROOT / "student_dataset" / "train", "student_test": ROOT / "student_dataset" / "student_test",
        "evaluator_test": ROOT / "evaluator_test"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold"})
PAL = {"train": "#2b6cb0", "student_test": "#dd6b20", "evaluator_test": "#2f855a"}

def save(name):
    plt.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.show()

def read_whitespace(path, dtype=float):
    return pd.read_csv(path, sep=r"\\s+", header=None, dtype=dtype).to_numpy()

def load_trip(part, tid):
    d = DATA[part] / tid
    return {"track": read_whitespace(d / f"{tid}.track"), "nodes": read_whitespace(d / f"{tid}.nodes"),
            "arcs": read_whitespace(d / f"{tid}.arcs", dtype=np.int64),
            "route": read_whitespace(d / f"{tid}.route", dtype=np.int64).ravel()}

def trips(part):
    return sorted(p.name for p in DATA[part].iterdir() if p.is_dir() and not p.name.startswith("."))
'''


def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src.strip("\n"))
               for kind, src in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, NB_DIR / name)
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------------------
# 01 EDA
# ---------------------------------------------------------------------------------------------------------------
eda = [
    ("md", "# 01 Exploratory data analysis\n\nPartitions used: **train** (80 trips), **student_test** (10 trips) and **evaluator_test** (10 trips, "
           "inspected here only for inventory and data quality, never for training or feature selection). Every figure is saved to `report/figures/`."),
    ("code", SETUP),
    ("md", "## 1. Dataset inventory\nOne pass over all 100 trips. Counts, sampling statistics, validity checks and road-network connectivity."),
    ("code", '''
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap.geo import LocalProjection

rows = []
for part in DATA:
    for tid in trips(part):
        d = DATA[part] / tid
        tr = read_whitespace(d / f"{tid}.track"); nodes = read_whitespace(d / f"{tid}.nodes")
        arcs = read_whitespace(d / f"{tid}.arcs", dtype=np.int64); route = read_whitespace(d / f"{tid}.route", dtype=np.int64).ravel()
        lon, lat, t = tr[:, 0], tr[:, 1], tr[:, 2]
        proj = LocalProjection(lon.mean(), lat.mean()); xy = proj.to_xy(tr[:, :2])
        dt = np.diff(t)
        n_nodes = len(nodes)
        g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n_nodes, n_nodes))
        n_comp, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab)
        deg = np.bincount(arcs.ravel(), minlength=n_nodes)
        ra = arcs[route]
        rows.append(dict(partition=part, trip=tid, n_gps=len(tr), n_nan=int(np.isnan(tr).sum()),
                         n_bad_coord=int(((np.abs(lon) > 180) | (np.abs(lat) > 90)).sum()),
                         n_dup_time=int((dt == 0).sum()), n_non_monotonic=int((dt < 0).sum()),
                         duration_s=float(t[-1] - t[0]), length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                         pct_1hz=float(100 * np.mean(dt == 1)), max_gap_s=float(dt.max()), n_gaps_gt_5s=int((dt > 5).sum()),
                         n_gaps_gt_60s=int((dt > 60).sum()), n_nodes=n_nodes, n_arcs=len(arcs), n_components=int(n_comp),
                         largest_component_pct=float(100 * sizes.max() / n_nodes), n_isolated_nodes=int((deg == 0).sum()),
                         n_route_arcs=len(route), route_connectivity=float(np.mean(ra[:-1, 1] == ra[1:, 0])) if len(route) > 1 else 1.0,
                         mean_lon=float(lon.mean()), mean_lat=float(lat.mean())))
inv = pd.DataFrame(rows)
inv.to_csv(TAB / "inventory_all_trips.csv", index=False)
summary = inv.groupby("partition").agg(trips=("trip", "count"), gps_points=("n_gps", "sum"), mean_points=("n_gps", "mean"),
    min_points=("n_gps", "min"), max_points=("n_gps", "max"), mean_nodes=("n_nodes", "mean"), mean_arcs=("n_arcs", "mean"),
    mean_route_arcs=("n_route_arcs", "mean"), mean_length_km=("length_m", lambda s: s.mean() / 1000), mean_pct_1hz=("pct_1hz", "mean"),
    max_gap_s=("max_gap_s", "max"), trips_with_gap_gt_60s=("n_gaps_gt_60s", lambda s: int((s > 0).sum()))).round(2)
summary.to_csv(TAB / "partition_summary.csv")
summary.T
'''),
    ("md", "## 2. Trip size and length"),
    ("code", '''
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for part in DATA:
    s = inv[inv.partition == part]
    ax[0].hist(np.log10(s.n_gps), bins=14, alpha=0.6, label=f"{part} (n={len(s)})", color=PAL[part])
    ax[1].hist(s.length_m / 1000, bins=14, alpha=0.6, color=PAL[part], label=part)
    ax[2].scatter(s.duration_s / 60, s.length_m / 1000, s=22, alpha=0.8, color=PAL[part], label=part)
ax[0].set(xlabel="log10(GPS points per trip)", ylabel="trips", title="GPS observations per trip")
ax[1].set(xlabel="trajectory length (km)", ylabel="trips", title="Trajectory length")
ax[2].set(xlabel="duration (min)", ylabel="length (km)", title="Duration against length")
ax[0].legend(); plt.tight_layout(); save("eda_01_trip_size.png")
'''),
    ("md", "## 3. Where the trips are\nMean longitude and latitude of every trip. The data spans several continents, so a single global projection is not appropriate."),
    ("code", '''
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.6, 1]})
for part in DATA:
    s = inv[inv.partition == part]
    ax[0].scatter(s.mean_lon, s.mean_lat, s=40, alpha=0.8, color=PAL[part], label=part, edgecolor="white", linewidth=0.5)
ax[0].set(xlabel="mean longitude", ylabel="mean latitude", title="Trip centroids, all 100 trips"); ax[0].legend()
for part in DATA:
    s = inv[inv.partition == part]
    ax[1].hist(s.mean_lat, bins=np.arange(-40, 70, 10), alpha=0.6, color=PAL[part], label=part)
ax[1].set(xlabel="mean latitude", ylabel="trips", title="Latitude spread"); ax[1].legend()
plt.tight_layout(); save("eda_02_trip_locations.png")
'''),
    ("md", "## 4. Sampling rate and time gaps"),
    ("code", '''
dts = {p: np.concatenate([np.diff(read_whitespace(DATA[p] / t / f"{t}.track")[:, 2]) for t in trips(p)]) for p in DATA}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for p, v in dts.items():
    vals, cnt = np.unique(np.clip(v, 0, 12), return_counts=True)
    ax[0].plot(vals, cnt / cnt.sum() * 100, marker="o", ms=3, color=PAL[p], label=p)
ax[0].set(yscale="log", xlabel="time step between observations (s, clipped at 12)", ylabel="% of steps", title="Step size distribution"); ax[0].legend()
for p in DATA:
    s = inv[inv.partition == p]; ax[1].hist(s.pct_1hz, bins=15, alpha=0.6, color=PAL[p], label=p)
ax[1].set(xlabel="% of steps exactly 1 s", ylabel="trips", title="Trips by share of 1 Hz steps")
ax[2].scatter(inv.n_gps, inv.max_gap_s, s=20, c=inv.partition.map(PAL), alpha=0.8)
ax[2].set(xscale="log", yscale="log", xlabel="GPS points in trip", ylabel="largest time gap (s)", title="Largest gap per trip")
plt.tight_layout(); save("eda_03_sampling_gaps.png")
gap_tab = pd.DataFrame({p: {"steps": len(v), "% exactly 1 s": 100 * np.mean(v == 1), "% > 5 s": 100 * np.mean(v > 5),
                            "steps > 60 s": int((v > 60).sum()), "largest gap (s)": v.max()} for p, v in dts.items()}).T.round(3)
gap_tab.to_csv(TAB / "sampling_gap_summary.csv"); gap_tab
'''),
    ("md", "## 5. Missing and invalid values\nChecks on every raw `.track` file: NaN or infinite values, coordinates outside the valid range, duplicate timestamps and timestamps that go backwards."),
    ("code", '''
chk = inv.groupby("partition")[["n_nan", "n_bad_coord", "n_dup_time", "n_non_monotonic", "n_isolated_nodes"]].sum()
chk["trips_with_gap_gt_60s"] = inv.groupby("partition")["n_gaps_gt_60s"].apply(lambda s: int((s > 0).sum()))
chk["trips"] = inv.groupby("partition")["trip"].count()
chk.columns = ["NaN or inf values", "invalid coordinates", "duplicate timestamps", "timestamps going back", "isolated road nodes", "trips with a gap over 60 s", "trips"]
chk.to_csv(TAB / "missing_value_checks.csv")
fig, ax = plt.subplots(figsize=(11, 3.2))
sns.heatmap(chk.drop(columns="trips"), annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, ax=ax)
ax.set_title("Data validity checks by partition (counts)"); ax.set_ylabel(""); ax.grid(False)
plt.xticks(rotation=25, ha="right"); plt.tight_layout(); save("eda_04_missing_value_checks.png")
chk
'''),
    ("md", "## 6. Road network size and connectivity"),
    ("code", '''
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for p in DATA:
    s = inv[inv.partition == p]; ax[0].scatter(s.n_nodes, s.n_arcs, s=22, alpha=0.8, color=PAL[p], label=p)
lim = [inv.n_nodes.min(), inv.n_nodes.max()]; ax[0].plot(lim, [2 * x for x in lim], "k--", lw=0.8, label="arcs = 2 x nodes")
ax[0].set(xscale="log", yscale="log", xlabel="road nodes", ylabel="directed arcs", title="Network size per trip"); ax[0].legend(fontsize=8)
ax[1].hist(inv.n_components, bins=range(1, int(inv.n_components.max()) + 2), color="#2b6cb0")
ax[1].set(xlabel="weakly connected components", ylabel="trips", title="Components per network")
ax[2].hist(inv.largest_component_pct, bins=20, color="#2f855a"); ax[2].set(xlabel="% of nodes in largest component", ylabel="trips", title="Largest component")
plt.tight_layout(); save("eda_05_network_connectivity.png")
net = inv.groupby("partition").agg(mean_nodes=("n_nodes", "mean"), mean_arcs=("n_arcs", "mean"), mean_components=("n_components", "mean"),
      min_largest_component_pct=("largest_component_pct", "min"), isolated_nodes=("n_isolated_nodes", "sum"), route_connectivity=("route_connectivity", "mean")).round(4)
net.to_csv(TAB / "network_connectivity.csv"); net
'''),
    ("md", "## 7. Trajectories on the road network\nGPS observations (coloured by time) with the reference route in green and the surrounding road network in grey."),
    ("code", '''
from matplotlib.collections import LineCollection
def draw_trip(ax, part, tid, pad=0.0, window=None, route=True, network=True):
    tr = load_trip(part, tid); proj = LocalProjection(tr["nodes"][:, 0].mean(), tr["nodes"][:, 1].mean())
    N = proj.to_xy(tr["nodes"]); P = proj.to_xy(tr["track"][:, :2]); A, B = N[tr["arcs"][:, 0]], N[tr["arcs"][:, 1]]
    if window is not None:
        cx, cy, r = window; keep = (np.abs(A[:, 0] - cx) < r) & (np.abs(A[:, 1] - cy) < r)
    else:
        x0, x1, y0, y1 = P[:, 0].min() - pad, P[:, 0].max() + pad, P[:, 1].min() - pad, P[:, 1].max() + pad
        keep = (A[:, 0] > x0) & (A[:, 0] < x1) & (A[:, 1] > y0) & (A[:, 1] < y1)
    if network:
        ax.add_collection(LineCollection(np.stack([A[keep], B[keep]], 1), colors="#b8bec6", linewidths=0.5, zorder=1))
    if route:
        ra = tr["route"]; ax.add_collection(LineCollection(np.stack([N[tr["arcs"][ra, 0]], N[tr["arcs"][ra, 1]]], 1), colors="#2f855a", linewidths=2.2, zorder=2))
    ax.scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), s=4, cmap="viridis", zorder=3)
    ax.set_aspect("equal"); ax.set_title(f"{part} trip {tid}: {len(P)} GPS points, {len(tr['route'])} route arcs"); ax.grid(False)
    return P

fig, axs = plt.subplots(2, 2, figsize=(13, 11))
for ax, (part, tid) in zip(axs.ravel(), [("train", "00000001"), ("train", "00000046"), ("train", "00000075"), ("student_test", "00000007")]):
    draw_trip(ax, part, tid, pad=150); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
plt.tight_layout(); save("eda_06_trajectories.png")
'''),
    ("code", '''
tr = load_trip("train", "00000001"); proj = LocalProjection(tr["nodes"][:, 0].mean(), tr["nodes"][:, 1].mean()); P = proj.to_xy(tr["track"][:, :2])
fig, ax = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1, 2]})
draw_trip(ax[0], "train", "00000001", pad=1500); ax[0].set_title("Whole network around trip 00000001")
c = P[len(P) // 3]; draw_trip(ax[1], "train", "00000001", window=(c[0], c[1], 350)); ax[1].set_xlim(c[0] - 350, c[0] + 350); ax[1].set_ylim(c[1] - 350, c[1] + 350)
ax[1].set_title("Zoom: GPS noise next to parallel and cross streets")
plt.tight_layout(); save("eda_07_road_network.png")
'''),
    ("md", "## 8. Quality of the reference route\nThe `.route` file lists arcs but gives no per-point label. We align every GPS point to the route with a monotone dynamic programme. The distance from a point to its aligned arc shows how far the reference can be trusted."),
    ("code", '''
rep = pd.read_csv(RES / "preprocess_report_train.csv", dtype={"trip": str}); rep_s = pd.read_csv(RES / "preprocess_report_student_test.csv", dtype={"trip": str})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
r = rep.sort_values("frac_label_invalid", ascending=False).reset_index(drop=True)
ax[0].bar(range(len(r)), 100 * r.frac_label_invalid, color=["#c53030" if v > 0.5 else "#2b6cb0" for v in r.frac_label_invalid])
for i in range(3): ax[0].annotate(r.trip[i], (i, 100 * r.frac_label_invalid[i]), fontsize=8, ha="left", va="bottom")
ax[0].set(xlabel="training trips, sorted", ylabel="% points more than 60 m from aligned arc", title="Reference route mismatch per trip")
ax[1].hist(rep.label_dist_med, bins=25, color="#2b6cb0"); ax[1].set(xlabel="median distance GPS to aligned arc (m)", ylabel="trips", title="Median alignment distance")
ax[2].bar(["train", "student_test"], [100 * rep.miss_rate.mean(), 100 * rep_s.miss_rate.mean()], color=[PAL["train"], PAL["student_test"]])
ax[2].set(ylabel="candidate miss rate (%)", title="Aligned arc missing from the 50 m candidate set")
plt.tight_layout(); save("eda_08_label_quality.png")
worst = rep.sort_values("frac_label_invalid", ascending=False).head(6)[["trip", "n_pts", "label_dist_med", "label_dist_p95", "frac_label_invalid", "miss_rate"]].round(4)
worst.to_csv(TAB / "worst_label_quality_trips.csv", index=False); worst
'''),
    ("md", "## 9. Choosing the candidate radius"),
    ("code", '''
ft = pd.read_parquet(ROOT / "artifacts" / "features_train.parquet", columns=["trip", "d", "label", "label_valid", "n_cands"])
good = ft[(ft.label == 1) & (ft.label_valid == 1) & (~ft.trip.isin(["00000033", "00000054"]))]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(good.d, bins=60, color="#2b6cb0"); ax[0].set(xlabel="distance GPS point to its true arc (m)", ylabel="points", title="Distance to the true arc (train)")
xs = np.linspace(0, 60, 121); ax[1].plot(xs, [100 * (good.d <= x).mean() for x in xs], color="#2f855a")
for rr in (30, 50): ax[1].axvline(rr, ls="--", color="grey"); ax[1].annotate(f"{rr} m: {100 * (good.d <= rr).mean():.1f}%", (rr, 60), fontsize=8)
ax[1].set(xlabel="search radius (m)", ylabel="% of points whose true arc is inside", title="Candidate recall against radius")
ax[2].hist(ft.drop_duplicates("trip").set_index("trip").n_cands, bins=20, color="#dd6b20")
ax[2].set(xlabel="candidate arcs at a typical point, per trip", ylabel="trips", title="Candidates per GPS point")
plt.tight_layout(); save("eda_09_candidate_radius.png")
'''),
    ("md", "## 10. Feature correlations\nSample of 150,000 candidate rows from the training partition (36 engineered features)."),
    ("code", '''
from pathsnap.features import FEATURES
samp = pd.read_parquet(ROOT / "artifacts" / "features_train.parquet", columns=FEATURES + ["label"]).sample(150_000, random_state=42)
corr = samp.corr()
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": 0.7}, linewidths=0.2)
ax.set_title("Pearson correlation of engineered features and the label"); ax.grid(False)
plt.tight_layout(); save("eda_10_feature_correlation.png")
corr["label"].drop("label").sort_values(key=abs, ascending=False).head(10).round(3).to_frame("corr with label")
'''),
]
nb(eda, "01_eda.ipynb")

# ---------------------------------------------------------------------------------------------------------------
# 02 model comparison
# ---------------------------------------------------------------------------------------------------------------
cmp_cells = [
    ("md", "# 02 Model experimentation and comparison\n\nFive models were trained and compared: LightGBM, Random Forest (RFC), Deep MLP, Logistic Regression and XGBoost. CV numbers come from 5-fold GroupKFold by trip on the training trips. "
           "Student-test and evaluator-test Top-1 are scored for every model against one shared per-point truth (route arc aligned with the same labeller)."),
    ("code", SETUP),
    ("code", '''
git = lambda p: json.load(open(p))
oth = git(RES / "other_models" / "metrics.json"); folds_main = pd.read_csv(RES / "cv_folds.csv"); folds_oth = pd.read_csv(RES / "other_models" / "cv_folds.csv")
st = pd.read_csv(RES / "student_test" / "student_test_summary.csv"); ev = pd.read_csv(RES / "evaluator_test" / "evaluator_test_summary.csv")
KEY = {"LightGBM": "lightgbm", "Logistic Regression": "logreg", "Random Forest": "rf", "XGBoost": "xgboost", "Deep MLP": "mlp"}
tim = {k: git(ROOT / "artifacts" / "student_test_preds" / f"{v}_time.json") for k, v in KEY.items()}
rf_t = git(ROOT / "random_forest" / "outputs" / "baseline_metrics.json")
xg_t = git(ROOT / "xg boost" / "results" / "xgb_tuned.json"); xg_b = git(ROOT / "xg boost" / "results" / "xgb_baseline.json")
ml = git(ROOT / "mlp" / "results" / "mlp_complete_report_metrics.json"); ml_cv = pd.read_csv(ROOT / "mlp" / "results" / "mlp_5fold_cv_results.csv")
ov = pd.read_csv(RES / "mlp_overfit_check" / "mlp_overfit_check.csv"); mlp_clean_cv = ov[ov.run == "clean 5-fold CV"].top1
fold_rf = [0.8237, 0.7682, 0.9524, 0.9817, 0.7553]       # per-fold Top-1 from the RFC's baseline_metrics README table
def shared(df, label, stage="classifier only"): return df[(df.model == label) & (df.stage == stage)].iloc[0]
def fs(df, name, col): v = df[df.model == name][col]; return v.mean(), v.std()
cfg = {"LightGBM": ("Boosting", "400 trees, lr 0.05, 63 leaves, 36 features"),
       "Random Forest": ("Ensemble (bagging)", "150 trees, depth 15, balanced weights, 9 features"),
       "Deep MLP": ("Neural network", "128/64 hidden units, dropout 0.10, 24 features, K=5"),
       "Logistic Regression": ("Linear", "C=1.0, standardised, 36 features"),
       "XGBoost": ("Boosting", "487 trees, depth 8, lr 0.023, 15 features")}
cvv = {"LightGBM": (*fs(folds_main, "lightgbm", "top1_acc"), fs(folds_main, "lightgbm", "cand_macro_f1")[0], fs(folds_main, "lightgbm", "log_loss")[0]),
       "Logistic Regression": (oth["logreg"]["cv_top1_acc_mean"], oth["logreg"]["cv_top1_acc_std"], oth["logreg"]["cv_cand_macro_f1_mean"], oth["logreg"]["cv_log_loss_mean"]),
       "Random Forest": (np.mean(fold_rf), np.std(fold_rf, ddof=1), rf_t["cv_macro_f1"], rf_t["cv_log_loss"]),
       "XGBoost": (xg_t["cv_summary"]["top1_acc"]["mean"], xg_t["cv_summary"]["top1_acc"]["std"], xg_t["cv_summary"]["f1_macro"]["mean"], xg_t["cv_summary"]["log_loss"]["mean"]),
       "Deep MLP": (mlp_clean_cv.mean(), mlp_clean_cv.std(), ml["cv_5fold_mean_macro_f1"], ml_cv.cv_log_loss.mean())}
train = {"LightGBM": oth["lightgbm"]["train_top1"], "Logistic Regression": oth["logreg"]["train_top1"], "Random Forest": np.nan,
         "XGBoost": float(np.mean([f["train_top1_acc"] for f in xg_b["cv_folds"]])), "Deep MLP": float(ov[ov.partition == "fit trips (72)"].top1.mean())}
rows = []
for name in KEY:
    s, e = shared(st, name), shared(ev, name)
    rows.append(dict(model=name, key=KEY[name], family=cfg[name][0], config=cfg[name][1], cv_top1=cvv[name][0], cv_top1_sd=cvv[name][1], cv_macro_f1=cvv[name][2], cv_logloss=cvv[name][3],
        train_top1=train[name], test_top1=s.top1_acc, test_macro_f1=s.macro_f1, test_logloss=s.log_loss, test_continuity=s.route_continuity, test_miss=s.candidate_miss_rate,
        eval_top1=e.top1_acc, eval_macro_f1=e.macro_f1, eval_logloss=e.log_loss, eval_continuity=e.route_continuity, infer_s=tim[name]["inference_s"]))
T = pd.DataFrame(rows); T.to_csv(TAB / "all_models_table.csv", index=False)
T[["model", "family", "cv_top1", "cv_top1_sd", "cv_macro_f1", "test_top1", "test_macro_f1", "test_logloss", "test_continuity", "eval_top1", "infer_s"]].round(4)
'''),
    ("md", "## Cross-validation comparison"),
    ("code", '''
C = {"LightGBM": "#2f855a", "Random Forest": "#2b6cb0", "Deep MLP": "#c05621", "Logistic Regression": "#805ad5", "XGBoost": "#b7791f"}
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
Tm = T.sort_values("cv_macro_f1"); ax[0].barh(Tm.model, Tm.cv_macro_f1, color=[C[m] for m in Tm.model]); ax[0].set(xlabel="5-fold CV macro F1 (mean over folds)", title="Cross-validation macro F1"); ax[0].set_xlim(0.8, 1)
for i, v in enumerate(Tm.cv_macro_f1): ax[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
Tt = T.sort_values("cv_top1"); ax[1].barh(Tt.model, Tt.cv_top1, xerr=Tt.cv_top1_sd, color=[C[m] for m in Tt.model], capsize=3); ax[1].set(xlabel="5-fold CV Top-1 (mean, bar = std over folds)", title="Cross-validation Top-1 accuracy"); ax[1].set_xlim(0.7, 1)
for i, v in enumerate(Tt.cv_top1): ax[1].text(0.705, i, f"{v:.3f}", va="center", fontsize=9, color="white")
plt.tight_layout(); save("model_01_cv_comparison.png")
'''),
    ("md", "## Student-test comparison"),
    ("code", '''
Tt = T.sort_values("test_top1"); fig, ax = plt.subplots(figsize=(11, 4.4)); y = np.arange(len(Tt)); h = 0.38
ax.barh(y - h / 2, Tt.test_top1, h, label="Top-1 accuracy", color="#2b6cb0"); ax.barh(y + h / 2, Tt.test_macro_f1, h, label="macro F1", color="#dd6b20")
ax.set_yticks(y); ax.set_yticklabels(Tt.model); ax.set_xlim(0.75, 1); ax.legend(loc="lower right"); ax.set_title("Student-test (10 trips): Top-1 accuracy and macro F1")
for i, (a, b) in enumerate(zip(Tt.test_top1, Tt.test_macro_f1)): ax.text(a + 0.003, i - h / 2, f"{a:.3f}", va="center", fontsize=8); ax.text(b + 0.003, i + h / 2, f"{b:.3f}", va="center", fontsize=8)
plt.tight_layout(); save("model_02_student_test.png")
Tl = T.sort_values("test_logloss"); fig, ax = plt.subplots(figsize=(10, 3.8)); ax.barh(Tl.model, Tl.test_logloss, color=[C[m] for m in Tl.model]); ax.set(xlabel="student-test log loss (lower is better)", title="Log loss comparison")
for i, v in enumerate(Tl.test_logloss): ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); save("model_03_log_loss.png")
'''),
    ("md", "## Evaluator-test comparison (final check, never used for selection)"),
    ("code", '''
E = ev.copy(); E["label"] = E.model + np.where(E.stage.str.contains("HMM"), " + HMM", ""); E = E.sort_values("top1_acc")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
ax[0].barh(E.label, E.top1_acc, color=["#c05621" if "HMM" in l else "#2b6cb0" for l in E.label]); ax[0].set_xlim(0.7, 1); ax[0].set(xlabel="Top-1 accuracy", title="evaluator_test Top-1 (shared truth)")
for i, v in enumerate(E.top1_acc): ax[0].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
ax[1].barh(E.label, E.route_continuity, color=["#c05621" if "HMM" in l else "#2b6cb0" for l in E.label]); ax[1].set_xlim(0.85, 1); ax[1].set(xlabel="route continuity", title="evaluator_test route continuity"); ax[1].set_yticklabels([])
for i, v in enumerate(E.route_continuity): ax[1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); save("model_04_evaluator_test.png")
'''),
    ("md", "## Confusion matrices (candidate level, student-test, threshold 0.5, shared truth)"),
    ("code", '''
truth = pd.read_parquet(ROOT / "artifacts" / "student_test_preds" / "truth.parquet")[["trip", "pt", "truth_arc"]]
cms = {}
for name, k in KEY.items():
    c = pd.read_parquet(ROOT / "artifacts" / "student_test_preds" / f"{k}_cand.parquet"); c["trip"] = c.trip.astype(str).str.zfill(8)
    c = c.merge(truth, on=["trip", "pt"]); y = (c.arc == c.truth_arc).to_numpy(); yhat = (c.p >= 0.5).to_numpy()
    cms[name] = np.array([[((~y) & (~yhat)).sum(), ((~y) & yhat).sum()], [(y & (~yhat)).sum(), (y & yhat).sum()]])
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for ax, (nm, cm) in zip(axs.ravel(), cms.items()):
    cmn = cm / cm.sum(1, keepdims=True)
    sns.heatmap(cmn, annot=[[f"{cmn[i, j]:.2f}\\n({int(cm[i, j]):,})" for j in range(2)] for i in range(2)], fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax, xticklabels=["not true arc", "true arc"], yticklabels=["not true arc", "true arc"])
    ax.set_title(nm); ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.grid(False)
axs.ravel()[-1].axis("off"); plt.suptitle("Row-normalised confusion matrices on student_test (counts in brackets)", y=1.0)
plt.tight_layout(); save("model_05_confusion_matrices.png")
pd.DataFrame({k: dict(zip(["TN", "FP", "FN", "TP"], v.ravel())) for k, v in cms.items()}).T.to_csv(TAB / "confusion_counts.csv")
'''),
    ("md", "## Training and validation loss (only for models that produce epoch histories)"),
    ("code", '''
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for i, (nm, f) in enumerate([("untuned", "mlp_untuned_epoch_history.csv"), ("tuned", "mlp_tuned_epoch_history.csv")]):
    h = pd.read_csv(ROOT / "mlp" / "results" / f); ax[i].plot(h.train_loss, label="training loss"); ax[i].plot(h.val_loss, label="validation loss")
    ax[i].set(title=f"Deep MLP, {nm} configuration", xlabel="epoch", ylabel="hybrid BCE + listwise loss"); ax[i].legend()
lh = xg_b["loss_history"]
def curve(v):
    if isinstance(v, dict): v = list(v.values())[0]
    return list(v)
ax[2].plot(curve(lh["train"]), label="training log loss"); ax[2].plot(curve(lh["student_test"]), label="student-test log loss")
ax[2].set(title="XGBoost, boosting rounds", xlabel="boosting round", ylabel="log loss"); ax[2].legend()
plt.tight_layout(); save("model_06_training_curves.png")
'''),
    ("md", "## Feature importance"),
    ("code", '''
imp = pd.read_csv(RES / "lightgbm_feature_importance.csv").head(15).iloc[::-1]; rfi = pd.Series(rf_t["feature_importances"]).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].barh(imp.feature, imp.gain / imp.gain.sum(), color="#2f855a"); ax[0].set(title="LightGBM: top 15 features (share of total gain)", xlabel="share of gain")
ax[1].barh(rfi.index, rfi.values, color="#2b6cb0"); ax[1].set(title="Random Forest: impurity importance (9 features)", xlabel="importance")
plt.tight_layout(); save("model_07_feature_importance.png")
'''),
    ("md", "## Over and underfitting: train against CV against held-out"),
    ("code", '''
G = T.dropna(subset=["train_top1"]).copy()
fig, ax = plt.subplots(figsize=(12, 4.8)); x = np.arange(len(G)); w = 0.2
ax.bar(x - 1.5 * w, G.train_top1, w, label="train (fit trips)", color="#718096"); ax.bar(x - 0.5 * w, G.cv_top1, w, label="5-fold CV mean", color="#2b6cb0")
ax.bar(x + 0.5 * w, G.test_top1, w, label="student-test", color="#dd6b20"); ax.bar(x + 1.5 * w, G.eval_top1, w, label="evaluator-test", color="#2f855a")
ax.set_xticks(x); ax.set_xticklabels(G.model); ax.set_ylim(0.75, 1.03); ax.set_ylabel("Top-1 accuracy"); ax.legend(ncol=4, loc="upper center")
ax.set_title("Top-1 on train, CV and held-out trips (classifier only)"); plt.tight_layout(); save("model_08_fit_gap.png")
G[["model", "train_top1", "cv_top1", "test_top1", "eval_top1"]].round(4)
'''),
    ("md", "## Per-trip accuracy on evaluator_test"),
    ("code", '''
pt = pd.read_csv(RES / "evaluator_test" / "evaluator_test_per_trip_top1.csv", index_col=0)
fig, ax = plt.subplots(figsize=(13, 4.8)); sns.heatmap(pt.T, annot=True, fmt=".3f", cmap="YlGn", vmin=0.55, vmax=1, ax=ax, cbar_kws={"label": "Top-1"}); ax.grid(False)
ax.set_title("evaluator_test: Top-1 per trip and model (shared truth)"); ax.set_xlabel("trip"); plt.tight_layout(); save("model_09_per_trip_evaluator.png")
'''),
    ("md", "## Model selection\nSelection uses only the data we are allowed to select on: student_test (weight 0.6) and the 5-fold CV mean (weight 0.4). evaluator_test is shown next to it as a one-time final check and is not part of the score."),
    ("code", '''
cvm = T.set_index("model").cv_top1; rows = []
for label in KEY:
    for stage in ["classifier only", "classifier + HMM/Viterbi"]:
        a = st[(st.model == label) & (st.stage == stage)]; b = ev[(ev.model == label) & (ev.stage == stage)]
        if len(a) == 0: continue
        a, b = a.iloc[0], b.iloc[0]
        rows.append(dict(model=label, stage=stage, cv_top1=cvm[label], student_top1=a.top1_acc, evaluator_top1=b.top1_acc, student_continuity=a.route_continuity, evaluator_continuity=b.route_continuity))
R = pd.DataFrame(rows); R["selection_score"] = 0.6 * R.student_top1 + 0.4 * R.cv_top1
R = R.sort_values("selection_score", ascending=False).reset_index(drop=True); R.insert(0, "rank", R.index + 1)
(RES / "ranking").mkdir(exist_ok=True); R.to_csv(RES / "ranking" / "model_ranking_final.csv", index=False); R.to_csv(TAB / "final_ranking.csv", index=False)
R["label"] = R.model + np.where(R.stage.str.contains("HMM"), " + HMM", "")
fig, ax = plt.subplots(figsize=(11, 4.8)); r = R.iloc[::-1]
ax.barh(r.label, r.selection_score, color=["#c05621" if "HMM" in l else "#2b6cb0" for l in r.label], alpha=0.9)
ax.scatter(r.evaluator_top1, r.label, color="black", zorder=3, s=26, label="evaluator_test Top-1 (final check, not in score)")
ax.set_xlim(0.84, 0.96); ax.set(xlabel="selection score = 0.6 x student-test Top-1 + 0.4 x CV Top-1", title="Model selection and final evaluator check"); ax.legend(loc="lower right", fontsize=8)
for i, v in enumerate(r.selection_score): ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); save("model_10_selection.png"); R.drop(columns="label").round(4)
'''),
    ("md", "## Does the ranking depend on who labelled the points?\nEach final pipeline is scored against three independent per-point labellers (the shared one and the labellers built into the XGBoost and Deep MLP pipelines)."),
    ("code", '''
lr = pd.read_csv(RES / "lightgbm_hmm" / "labeller_robustness.csv"); lr = lr[lr.pipeline.str.contains("HMM")]
lr["pipeline"] = lr.pipeline.str.replace("MLP + HMM", "Deep MLP + HMM")
hm = lr.melt(id_vars=["split", "pipeline"], var_name="labeller", value_name="top1"); hm["labeller"] = hm.labeller.str.replace("my labeller", "shared labeller").str.replace(" labeller", "")
hm["col"] = hm.split.str.replace("_test", "") + " / " + hm.labeller; piv = hm.pivot(index="pipeline", columns="col", values="top1")
fig, ax = plt.subplots(figsize=(11, 3.2)); sns.heatmap(piv, annot=True, fmt=".4f", cmap="YlGn", vmin=0.90, vmax=0.96, ax=ax, cbar=False); ax.grid(False)
ax.set_title("Top-1 of the three HMM pipelines under three labellers"); ax.set_xlabel(""); ax.set_ylabel(""); plt.xticks(rotation=25, ha="right")
plt.tight_layout(); save("model_11_labeller_robustness.png"); piv.round(4)
'''),
]
nb(cmp_cells, "02_model_comparison.ipynb")

# ---------------------------------------------------------------------------------------------------------------
# 03 tuning and HMM
# ---------------------------------------------------------------------------------------------------------------
tune_cells = [
    ("md", "# 03 Bayesian optimisation and HMM / Viterbi decoding"),
    ("code", SETUP),
    ("code", '''
TUNE = RES / "tuning"
git = lambda p: json.load(open(p))
tl, tx, tm = (pd.read_csv(TUNE / f) for f in ["lightgbm_trials.csv", "xgboost_trials.csv", "mlp_trials.csv"])
sl, sx, sm = (git(TUNE / f) for f in ["lightgbm_tuning_summary.json", "xgboost_tuning_summary.json", "mlp_tuning_summary.json"])
gp = pd.read_csv(ROOT / "mlp" / "results" / "bayesian_optimisation_trials.csv"); ml = git(ROOT / "mlp" / "results" / "mlp_complete_report_metrics.json")

def history(ax, df, title):
    done = df[df.state == "COMPLETE"]; pruned = df[df.state == "PRUNED"]
    ax.scatter(done.number, done.value, s=30, color="#2b6cb0", label="completed trial", zorder=3)
    ax.scatter(pruned.number, pruned.value, s=18, color="#a0aec0", marker="x", label="pruned (partial mean)", zorder=2)
    ax.step(done.number, done.value.cummax(), where="post", color="#c05621", label="best so far"); ax.axhline(done.value.iloc[0], ls="--", color="grey", lw=0.8)
    ax.set(title=title, xlabel="trial", ylabel="mean 5-fold CV Top-1"); ax.legend(fontsize=8)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
history(ax[0], tl, "LightGBM (TPE, 30 trials)"); history(ax[1], tx, "XGBoost (TPE, 30 trials)"); history(ax[2], tm, "Deep MLP (TPE, 15 trials)")
plt.tight_layout(); save("tune_01_trial_history.png")
print({k: (len(v), v.state.value_counts().to_dict()) for k, v in {"lightgbm": tl, "xgboost": tx, "mlp": tm}.items()})
'''),
    ("code", '''
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].plot(gp.trial, gp.objective, marker="o", color="#2b6cb0", label="objective (0.85 macro F1 + 0.15 Top-1)"); ax[0].plot(gp.trial, gp.cv_top1_accuracy, marker="s", color="#dd6b20", label="CV Top-1")
ax[0].set(xlabel="trial", title="GP-EI search for the Deep MLP (10 trials)"); ax[0].legend(fontsize=8)
c = tl[tl.state == "COMPLETE"]; s = ax[1].scatter(c.params_learning_rate, c.params_num_leaves, c=c.value, cmap="viridis", s=60); ax[1].set(xscale="log", yscale="log", xlabel="learning rate", ylabel="num_leaves", title="LightGBM: completed trials")
plt.colorbar(s, ax=ax[1], label="CV Top-1"); plt.tight_layout(); save("tune_02_gpei_and_lightgbm_space.png")
'''),
    ("code", '''
rows = [
 ("LightGBM", "untuned", sl["untuned_cv_top1_mean"], sl["student_test_untuned"]["top1_acc"], sl["student_test_untuned"]["cand_macro_f1"], sl["student_test_untuned"]["log_loss"], sl["student_test_untuned"]["inference_s"]),
 ("LightGBM", "tuned", sl["best_cv_top1_mean"], sl["student_test_tuned"]["top1_acc"], sl["student_test_tuned"]["cand_macro_f1"], sl["student_test_tuned"]["log_loss"], sl["student_test_tuned"]["inference_s"]),
 ("XGBoost", "untuned", np.nan, sx["student_test_their_untuned"]["top1_acc"], sx["student_test_their_untuned"]["f1_macro"], sx["student_test_their_untuned"]["log_loss"], sx["student_test_their_untuned"]["inference_s"]),
 ("XGBoost", "tuned, search 1", sx["their_tuned_cv_top1_mean"], sx["student_test_their_tuned"]["top1_acc"], sx["student_test_their_tuned"]["f1_macro"], sx["student_test_their_tuned"]["log_loss"], sx["student_test_their_tuned"]["inference_s"]),
 ("XGBoost", "re-tuned, search 2", sx["best_cv_top1_mean"], sx["student_test_retuned"]["top1_acc"], sx["student_test_retuned"]["f1_macro"], sx["student_test_retuned"]["log_loss"], sx["student_test_retuned"]["inference_s"]),
]
tab = pd.DataFrame(rows, columns=["model", "config", "cv_top1_mean", "student_top1", "student_macro_f1", "student_logloss", "inference_s"])
tab.to_csv(TAB / "tuning_comparison_trees.csv", index=False)
mlp_rows = pd.DataFrame([
 ("Deep MLP", "untuned", ml["cv_5fold_mean_macro_f1"], 0.9041, 0.9271, 0.9576, 0.0888, 0.1821),
 ("Deep MLP", "tuned, GP-EI search", 0.9558, 0.9050, 0.9329, 0.9625, 0.0684, 0.2229),
 ("Deep MLP", "re-tuned, TPE search", np.nan, sm["best_cv_top1_mean"], sm["student_test_retuned"]["top1_mean"], np.nan, np.nan, np.nan)],
 columns=["model", "config", "cv_macro_f1", "cv_top1_mean", "student_top1", "student_macro_f1", "student_logloss", "inference_s"])
mlp_rows.to_csv(TAB / "tuning_comparison_mlp.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
lab = [f"{a}\\n{b}" for a, b in zip(tab.model, tab.config)]
ax[0].bar(range(len(tab)), tab.cv_top1_mean.fillna(0), color="#2b6cb0"); ax[0].set_ylim(0.8, 0.92); ax[0].set_xticks(range(len(tab))); ax[0].set_xticklabels(lab, fontsize=7); ax[0].set_title("Mean CV Top-1, untuned and tuned (tree models)")
ax[1].bar(range(len(tab)), tab.student_top1, color="#dd6b20"); ax[1].set_ylim(0.8, 0.93); ax[1].set_xticks(range(len(tab))); ax[1].set_xticklabels(lab, fontsize=7); ax[1].set_title("Student-test Top-1, untuned and tuned (tree models)")
for a, col in zip(ax, ["cv_top1_mean", "student_top1"]):
    for i, v in enumerate(tab[col]):
        if not np.isnan(v): a.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
plt.tight_layout(); save("tune_03_untuned_vs_tuned_trees.png")
mlp_rows.round(4)
'''),
    ("md", "## HMM / Viterbi against classifier only"),
    ("code", '''
ev = pd.read_csv(RES / "evaluator_test" / "evaluator_test_summary.csv"); st = pd.read_csv(RES / "student_test" / "student_test_summary.csv")
MOD = ["LightGBM", "Deep MLP", "XGBoost"]
def pair(df, model):
    return df[(df.model == model) & (df.stage == "classifier only")].iloc[0], df[(df.model == model) & (df.stage.str.contains("HMM"))].iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8)); w = 0.2; x = np.arange(6)
labels = [f"{m}\\n{sp}" for m in MOD for sp in ("student", "evaluator")]
a_top, b_top, a_c, b_c = [], [], [], []
for m in MOD:
    for df in (st, ev):
        a, b = pair(df, m); a_top.append(a.top1_acc); b_top.append(b.top1_acc); a_c.append(a.route_continuity); b_c.append(b.route_continuity)
ax[0].bar(x - w / 2, a_top, w, label="classifier only", color="#2b6cb0"); ax[0].bar(x + w / 2, b_top, w, label="classifier + HMM/Viterbi", color="#c05621")
ax[1].bar(x - w / 2, a_c, w, color="#2b6cb0"); ax[1].bar(x + w / 2, b_c, w, color="#c05621")
for a_, vv in ((ax[0], (a_top, b_top)), (ax[1], (a_c, b_c))):
    for i in range(6): a_.text(i - w / 2, vv[0][i] + 0.002, f"{vv[0][i]:.3f}", ha="center", fontsize=7); a_.text(i + w / 2, vv[1][i] + 0.002, f"{vv[1][i]:.3f}", ha="center", fontsize=7)
for a_, t_ in zip(ax, ["Top-1 accuracy", "Route continuity"]): a_.set_xticks(x); a_.set_xticklabels(labels, fontsize=8); a_.set_title(t_)
ax[0].set_ylim(0.75, 1); ax[1].set_ylim(0.85, 1.01); ax[0].legend(fontsize=8); plt.tight_layout(); save("hmm_01_classifier_vs_hmm.png")
'''),
    ("code", '''
pt = pd.read_csv(RES / "evaluator_test" / "evaluator_test_per_trip_top1.csv", index_col=0)
fig, ax = plt.subplots(1, 3, figsize=(19, 4.6)); x = np.arange(len(pt))
for a_, m in zip(ax, MOD):
    a_.bar(x - 0.2, pt[m], 0.4, label="classifier only", color="#2b6cb0"); a_.bar(x + 0.2, pt[m + " + HMM"], 0.4, label="+ HMM/Viterbi", color="#c05621")
    a_.set_xticks(x); a_.set_xticklabels(pt.index, rotation=35, ha="right", fontsize=8); a_.set_ylim(0.5, 1); a_.set_title(f"{m}: per-trip Top-1, evaluator_test"); a_.legend(fontsize=8)
plt.tight_layout(); save("hmm_02_per_trip_gain.png")
pd.DataFrame({m + " gain": pt[m + " + HMM"] - pt[m] for m in MOD}).round(4).T
'''),
    ("code", '''
ht = pd.read_csv(ROOT / "xg boost" / "results" / "hmm_trials.csv")
print(list(ht.columns))
col = [c for c in ht.columns if any(k in c.lower() for k in ("top1", "value", "objective", "acc"))][0]
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(ht.index, ht[col], marker="o", color="#2b6cb0", label=col); ax.plot(ht.index, ht[col].cummax(), color="#c05621", label="best so far")
ax.set(xlabel="trial", ylabel=col, title="HMM parameter search (40 trials)"); ax.legend(); plt.tight_layout(); save("hmm_03_xgb_hmm_search.png")
'''),
]
nb(tune_cells, "03_tuning_and_hmm.ipynb")

# ---------------------------------------------------------------------------------------------------------------
# 04 Folium maps and route comparison
# ---------------------------------------------------------------------------------------------------------------
fol = [
    ("md", "# 04 Geographical visualisation with Folium\n\nInteractive maps for evaluator_test trips: raw GPS, road network, reference route, classifier-only errors and the classifier + HMM/Viterbi route. "
           "Predictions come from `artifacts/evaluator_preds/` (written by `scripts/eval_evaluator_test.py`). The reference route is used only for display and scoring."),
    ("code", SETUP),
    ("code", '''
import folium
from matplotlib.collections import LineCollection
from pathsnap.geo import LocalProjection
PRED = ROOT / "artifacts" / "evaluator_preds"; MAPS = ROOT / "report" / "maps"; MAPS.mkdir(exist_ok=True)
truth = pd.read_parquet(PRED / "truth.parquet"); cand = pd.read_parquet(PRED / "lightgbm_cand.parquet"); hmm = pd.read_parquet(PRED / "lightgbm_point.parquet")
for d in (truth, cand, hmm): d["trip"] = d["trip"].astype(str).str.zfill(8)
top = cand.sort_values(["trip", "pt", "p"], ascending=[True, True, False]).drop_duplicates(["trip", "pt"]).rename(columns={"arc": "clf_arc"})[["trip", "pt", "clf_arc"]]
M = truth.merge(top, on=["trip", "pt"], how="left").merge(hmm.rename(columns={"pred_arc": "hmm_arc"}), on=["trip", "pt"], how="left")
M["clf_ok"] = M.clf_arc == M.truth_arc; M["hmm_ok"] = M.hmm_arc == M.truth_arc
M.groupby("trip")[["clf_ok", "hmm_ok"]].mean().round(4)
'''),
    ("code", '''
def arc_line(nodes, arcs, a): return [(nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]), (nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0])]
def route_line(nodes, arcs, seq):
    out = []
    for a in seq:
        p = arc_line(nodes, arcs, a)
        if not out or out[-1] != p[0]: out.append(p[0])
        out.append(p[1])
    return out
def dedupe(seq):
    s = np.asarray(seq); return s[np.r_[True, s[1:] != s[:-1]]]

def build_map(tid, name):
    tr = load_trip("evaluator_test", tid); nodes, arcs, track, route = tr["nodes"], tr["arcs"], tr["track"], tr["route"]
    m = M[M.trip == tid].sort_values("pt")
    lat, lon = track[:, 1], track[:, 0]; c = [lat.mean(), lon.mean()]
    fmap = folium.Map(location=c, zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    fmap.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    pad = 0.004; keep = ((nodes[arcs[:, 0], 1] > lat.min() - pad) & (nodes[arcs[:, 0], 1] < lat.max() + pad) & (nodes[arcs[:, 0], 0] > lon.min() - pad) & (nodes[arcs[:, 0], 0] < lon.max() + pad))
    ids = np.flatnonzero(keep)
    if len(ids) > 6000: ids = ids[:: int(np.ceil(len(ids) / 6000))]
    g_net = folium.FeatureGroup("Road network (grey)", show=True)
    for a in ids: folium.PolyLine(arc_line(nodes, arcs, a), color="#8a94a0", weight=1, opacity=0.7).add_to(g_net)
    g_net.add_to(fmap)
    g_ref = folium.FeatureGroup("Reference route (green)", show=True)
    folium.PolyLine(route_line(nodes, arcs, route), color="#2f855a", weight=7, opacity=0.65, tooltip="Reference route").add_to(g_ref); g_ref.add_to(fmap)
    g_hmm = folium.FeatureGroup("Predicted route: LightGBM + HMM/Viterbi (red)", show=True)
    folium.PolyLine(route_line(nodes, arcs, dedupe(m.hmm_arc.dropna().astype(int))), color="#c53030", weight=3, opacity=0.95, tooltip="LightGBM + HMM route").add_to(g_hmm); g_hmm.add_to(fmap)
    g_gps = folium.FeatureGroup("Raw GPS observations (blue)", show=True)
    for (la, lo, t) in zip(lat, lon, track[:, 2]): folium.CircleMarker((la, lo), radius=2, color="#2b6cb0", fill=True, fill_opacity=0.9, weight=0, tooltip=f"t={t:.0f}s").add_to(g_gps)
    g_gps.add_to(fmap)
    g_wrong = folium.FeatureGroup("Classifier-only errors (orange)", show=True)
    for _, r in m[~m.clf_ok].iterrows():
        folium.CircleMarker((lat[int(r.pt)], lon[int(r.pt)]), radius=4, color="#dd6b20", fill=True, fill_opacity=0.95, weight=1, tooltip=f"point {int(r.pt)}: classifier-only arc is wrong").add_to(g_wrong)
    g_wrong.add_to(fmap)
    folium.Marker((lat[0], lon[0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(fmap); folium.Marker((lat[-1], lon[-1]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    legend = f"""<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; padding: 10px 12px; border: 1px solid #888; font-size: 13px; line-height: 1.5">
    <b>Trip {tid} (evaluator_test)</b><br>{len(track)} GPS points<br>
    <span style="color:#2b6cb0">&#9679;</span> raw GPS observation<br><span style="color:#8a94a0">&#9632;</span> road network<br>
    <span style="color:#2f855a">&#9632;</span> reference route<br><span style="color:#c53030">&#9632;</span> predicted route (LightGBM + HMM)<br>
    <span style="color:#dd6b20">&#9679;</span> point wrong without the HMM<br>Top-1: classifier {m.clf_ok.mean():.3f}, with HMM {m.hmm_ok.mean():.3f}</div>"""
    fmap.get_root().html.add_child(folium.Element(legend)); folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(MAPS / name); return fmap
gain = (M.groupby("trip").hmm_ok.mean() - M.groupby("trip").clf_ok.mean()); size = M.groupby("trip").size()
big = gain[size > 500].idxmax(); hard = M.groupby("trip").hmm_ok.mean().idxmin()
print("largest HMM gain:", big, round(gain[big], 4), "| hardest trip:", hard)
build_map(big, f"evaluator_trip_{big}.html"); build_map(hard, f"evaluator_trip_{hard}.html")
MAP_TRIPS = {"gain": big, "hard": hard}
MAP_TRIPS
sorted(p.name for p in MAPS.iterdir())
'''),
    ("md", "## EDA map on a training trip\nTraining-partition trip 00000046: raw GPS, road network and reference route only (no predictions)."),
    ("code", '''
def eda_map(part, tid, name):
    tr = load_trip(part, tid); nodes, arcs, track, route = tr["nodes"], tr["arcs"], tr["track"], tr["route"]; lat, lon = track[:, 1], track[:, 0]
    fmap = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    fmap.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    pad = 0.003; keep = ((nodes[arcs[:, 0], 1] > lat.min() - pad) & (nodes[arcs[:, 0], 1] < lat.max() + pad) & (nodes[arcs[:, 0], 0] > lon.min() - pad) & (nodes[arcs[:, 0], 0] < lon.max() + pad))
    g1 = folium.FeatureGroup("Road network (grey)"); [folium.PolyLine(arc_line(nodes, arcs, a), color="#8a94a0", weight=1, opacity=0.7).add_to(g1) for a in np.flatnonzero(keep)[:6000]]; g1.add_to(fmap)
    g2 = folium.FeatureGroup("Reference route (green)"); folium.PolyLine(route_line(nodes, arcs, route), color="#2f855a", weight=7, opacity=0.65).add_to(g2); g2.add_to(fmap)
    g3 = folium.FeatureGroup("Raw GPS observations (blue)"); [folium.CircleMarker((la, lo), radius=2, color="#2b6cb0", fill=True, fill_opacity=0.9, weight=0).add_to(g3) for la, lo in zip(lat, lon)]; g3.add_to(fmap)
    legend = f"""<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; padding: 10px 12px; border: 1px solid #888; font-size: 13px; line-height: 1.5">
    <b>Trip {tid} ({part} partition)</b><br>{len(track)} GPS points, {len(route)} reference route arcs<br>
    <span style="color:#2b6cb0">&#9679;</span> raw GPS observation<br><span style="color:#8a94a0">&#9632;</span> road network<br><span style="color:#2f855a">&#9632;</span> reference route</div>"""
    fmap.get_root().html.add_child(folium.Element(legend)); folium.LayerControl(collapsed=False).add_to(fmap); fmap.save(MAPS / name)
eda_map("train", "00000046", "eda_train_00000046.html")
sorted(p.name for p in MAPS.iterdir())
'''),
    ("md", "## Static comparison (same data, no map tiles)\nReference route, classifier-only route and HMM route for one trip."),
    ("code", '''
tid = MAP_TRIPS["gain"]; tr = load_trip("evaluator_test", tid); proj = LocalProjection(tr["nodes"][:, 0].mean(), tr["nodes"][:, 1].mean())
N = proj.to_xy(tr["nodes"]); P = proj.to_xy(tr["track"][:, :2]); arcs = tr["arcs"]; m = M[M.trip == tid].sort_values("pt")
def seg(seq): s = dedupe(np.asarray(seq).astype(int)); return np.stack([N[arcs[s, 0]], N[arcs[s, 1]]], 1)
fig, ax = plt.subplots(1, 3, figsize=(18, 6.2), sharex=True, sharey=True)
x0, x1, y0, y1 = P[:, 0].min() - 80, P[:, 0].max() + 80, P[:, 1].min() - 80, P[:, 1].max() + 80
A, B = N[arcs[:, 0]], N[arcs[:, 1]]; kp = (A[:, 0] > x0) & (A[:, 0] < x1) & (A[:, 1] > y0) & (A[:, 1] < y1)
for a_, (title, seq, col) in zip(ax, [("Reference route", tr["route"], "#2f855a"), ("Classifier only (LightGBM)", m.clf_arc.dropna(), "#dd6b20"), ("Classifier + HMM/Viterbi", m.hmm_arc.dropna(), "#c53030")]):
    a_.add_collection(LineCollection(np.stack([A[kp], B[kp]], 1), colors="#d0d5db", linewidths=0.5)); a_.add_collection(LineCollection(seg(seq), colors=col, linewidths=2.4))
    a_.scatter(P[:, 0], P[:, 1], s=3, color="#2b6cb0", zorder=3); a_.set_aspect("equal"); a_.set_title(title); a_.grid(False)
ax[0].set_xlim(x0, x1); ax[0].set_ylim(y0, y1); plt.suptitle(f"evaluator_test trip {tid}: GPS (blue) and routes", y=1.0)
plt.tight_layout(); save("map_01_route_comparison_static.png")
'''),
]
nb(fol, "04_folium_maps.ipynb")
print("done")

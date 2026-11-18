# PathSnap: Random Forest Classifier (Bagging Ensemble)

Random Forest candidate-arc classifier for GPS map matching on noisy urban and highway vehicle trajectories.

**Author:** Sanjeev Murthy  
**Algorithm Family:** Ensemble (Bagging)  
**Task:** Binary classification / ranking over candidate road segments for each observed GPS point.

---

## 1. Model Configuration & Architecture

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | **150** | Balance between ensemble variance reduction and low inference latency |
| `max_depth` | **15** | Captures non-linear geometric interactions without memorizing noise |
| `min_samples_split` | **5** | Allows fine decision boundaries for ambiguous intersection candidates |
| `min_samples_leaf` | **2** | Leaves with at least 2 samples to prevent single-outlier leaf creation |
| `max_features` | `sqrt` | Selects $\sqrt{9} = 3$ random features per split to decorrelate individual trees |
| `class_weight` | `balanced` | Compensates for the class imbalance (~1 true positive candidate per 4-5 negatives) |
| `random_state` | 42 | Full reproducibility |

---

## 2. Engineered Features (9 Spatial & Kinematic Attributes)

1. `dist_rank`: Distance rank (0 to $K-1$) among spatial candidates for the observation. (32.30% importance)
2. `heading_diff`: Absolute angular difference ($0^\circ$ to $180^\circ$) between vehicle trajectory heading and road segment direction. (28.05% importance)
3. `proj_t`: Normalized projection point parameter along the segment $[0, 1]$. (12.30% importance)
4. `perp_dist_norm`: Normalized perpendicular distance relative to candidate count. (8.29% importance)
5. `perp_dist`: Orthogonal Euclidean distance in meters from GPS point to segment. (7.71% importance)
6. `veh_spd`: Estimated vehicle instantaneous speed from successive GPS observations (m/s). (4.90% importance)
7. `seg_len`: Road segment length in meters. (3.30% importance)
8. `n_candidates`: Number of competing candidate arcs in the 50m search radius. (2.17% importance)
9. `spd_change`: Rate of speed change (acceleration proxy) between GPS observations. (0.98% importance)

---

## 3. Experimental Results

### 5-Fold GroupKFold Cross-Validation (80 Training Trips)
*Splits grouped strictly by `trip_id` to guarantee zero data leakage across trajectories.*

| Fold | Top-1 Accuracy | Macro F1 | Log Loss |
|---|---|---|---|
| Fold 1 | 82.37% | 0.8622 | 0.2358 |
| Fold 2 | 76.82% | 0.8585 | 0.2719 |
| Fold 3 | 95.24% | 0.8952 | 0.1853 |
| Fold 4 | 98.17% | 0.9102 | 0.1664 |
| Fold 5 | 75.53% | 0.8441 | 0.2449 |
| **Mean CV** | **85.63%** | **0.8740** | **0.2209** |

### Student-Test Evaluation (10 Unseen Trips, 19,159 Observations)

| Metric | Score |
|---|---|
| **Top-1 Accuracy** | **61.52%** |
| **Macro F1** | **0.8160** |
| **Precision** | **0.7846** |
| **Recall** | **0.8775** |
| **Log Loss** | **0.3429** |
| **Candidate Miss Rate** | 20.21% |
| **Inference Time** | **0.447s for 10 trips (0.005 ms/candidate)** |

---

## 4. Directory Structure

```
random_forest/
├── README.md                          # Model documentation and instructions
├── requirements.txt                   # Dependencies
├── .gitignore                         # Ignores large parquet files and caches
├── src/
│   ├── config.py                      # Dynamic paths and hyperparameter constants
│   ├── data_loader.py                 # Parser for .track, .nodes, .arcs, .route
│   ├── candidate_generator.py         # 25m-sampled spatial KD-Tree candidate search
│   ├── feature_extraction.py          # 9 spatial & kinematic features extractor
│   └── metrics.py                     # Top-1 accuracy, Macro F1, Log Loss calculator
├── scripts/
│   ├── 01_eda_and_maps.py             # Trajectory summary & Folium interactive map
│   ├── 02_extract_features.py         # Batch feature extraction for train & test
│   └── 03_train_random_forest.py      # 5-fold CV & final model training script
├── models/
│   └── random_forest_baseline.joblib  # Trained model weights (compressed)
└── outputs/
    ├── baseline_metrics.json          # Complete JSON results
    ├── eda_summary.json               # Dataset exploration stats
    ├── trip_00000001_folium_map.html  # Interactive Folium map
    └── plots/
        └── eda_gps_counts.png         # Trajectory length distribution plot
```

---

## 5. Execution Instructions

From the `random_forest/` directory:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run EDA and generate Folium map
python scripts/01_eda_and_maps.py

# 3. Extract features (if not already cached)
python scripts/02_extract_features.py

# 4. Train model & run 5-Fold GroupKFold CV + student_test evaluation
python scripts/03_train_random_forest.py
```

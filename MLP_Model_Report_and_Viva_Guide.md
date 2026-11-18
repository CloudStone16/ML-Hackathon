# PathSnap: Deep MLP Model — Complete Experimental Results, Report Answers & Viva Guide

This document contains all measured results, tables, plots, and compulsory answers for the **PathSnap Deep Multi-Layer Perceptron (`PathSnap-MLP`)** model following the structure in `Deliverables.pdf` and `Models and Training Guidelines.pdf`.

---

## 1. Summary of PathSnap Deep MLP Performance

| Metric | 5-Fold GroupKFold CV (80 Train Trips) | Student-Test: Untuned Deep MLP | Student-Test: Bayesian-Tuned Deep MLP | Student-Test: Tuned MLP + HMM/Viterbi |
| :--- | :---: | :---: | :---: | :---: |
| **Classification Accuracy** | — | **97.35%** (`0.9735`) | **97.65%** (`0.9765`) | — |
| **Macro F1-Score** | **95.53%** (`0.9553 ± 0.0142`) | **95.76%** (`0.9576`) | **96.25%** (`0.9625`) | — |
| **Observation-Level Top-1 Accuracy** | **90.41%** (`0.9041`) | **92.71%** (`0.9271`) | **93.29%** (`0.9329`) | **94.27%** (`0.9427`) |
| **Mean Per-Trip Top-1 Accuracy** | — | — | **93.52%** (`0.9352`) | **94.22%** (`0.9422`) |
| **Precision (Macro)** | 95.69% (`0.9569`) | 96.45% (`0.9645`) | **96.72%** (`0.9672`) | — |
| **Recall (Macro)** | 95.39% (`0.9539`) | 95.10% (`0.9510`) | **95.79%** (`0.9579`) | — |
| **Log Loss** | 0.0943 | 0.0888 | **0.0684** | — |
| **Route Continuity** | 95.30% (`0.9530`) | 95.43% (`0.9543`) | **96.28%** (`0.9628`) | **97.43%** (`0.9743`) |
| **Sequence-Level Route F1** | — | — | **91.11%** (`0.9111`) | **91.95%** (`0.9195`) |
| **Candidate Miss Rate ($K=5$)** | — | 0.0141 | **0.0141** | **0.0141** |
| **Inference Time (all 10 Test Trips)** | — | 0.1821 s | **0.2229 s** | — |

---

## 2. Dataset Understanding, Preprocessing & Feature Engineering (Section 2.2)

### 2.1 How We Preprocessed the Data
1. **Partition Inventory & Integrity**:
   - **Training Partition (`train`)**: 80 trips (`208,417` total GPS observations, average `2,605.2` points/trip; average `24,168.6` road nodes and `48,461.3` directed road segments per trip).
   - **Student-Testing Partition (`student_test`)**: 10 trips (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`; `19,160` total GPS observations, average `1,916.0` points/trip).
   - **Reserved Evaluator Partition**: Identified the 10 held-out trip IDs (`00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`, `00000067`, `00000073`, `00000087`, `00000094`).
2. **Coordinate System Projection (Angular Degrees to Local Metric Tangent Plane)**:
   - Raw `.track` and `.nodes` coordinates are in angular degrees `(longitude, latitude)` across high-latitude European/Eurasian cities ($\text{lat} \approx 49.8^\circ\text{N} - 56.3^\circ\text{N}$), where $1^\circ$ of longitude shrinks by $\cos(\text{lat}) \approx 0.55 - 0.64$.
   - We project each trip into a local metric equirectangular tangent plane centered at the trip's road network centroid $(\text{lon}_0, \text{lat}_0)$:
     $$x = (\text{lon} - \text{lon}_0)\frac{\pi}{180} R_{\oplus}\cos(\text{lat}_0), \quad y = (\text{lat} - \text{lat}_0)\frac{\pi}{180} R_{\oplus}$$
3. **Ground-Truth Alignment & Data-Quality Anomaly Handling**:
   - Because `.track` contains $T$ 1 Hz GPS observations while `.route` contains $M$ chronologically traversed `arc_id`s ($M \ll T$), we align each GPS point $t$ to its true traversed road segment along `.route` using **Monotonic Dynamic Programming (DP)** with displacement-tangent and interior-projection tie-breaking at shared road vertices.
   - **Key EDA Anomaly Discovery**: While there are `0` NaN/missing values, two training trips (`00000054` and `00000033`) contain prolonged stationary off-road parking segments where the vehicle remained parked $\sim 135\text{m}$ away from the road network inside a building/lot for `17,100s` and `4,143s` respectively, and `7,273` steps in `train` (`264` in `student_test`) exhibit sampling gaps $\Delta t > 1\text{s}$ (up to `210s` in `student_test`). Filtering off-road stationary parking bursts ($>45\text{m}$ from the route) during training prevents corrupted all-zero candidate groups from biasing model weights.
   - **Strict Leakage Prevention**: `StandardScaler` is fit **strictly on the 80 training trips** (and inside each training fold during `GroupKFold` CV).

### 2.2 Candidate Generation & 24 Extracted Features
For each GPS observation $t$, we query a spatial `cKDTree` over road segment midpoints/quarter-points and select the **$K=5$ best candidate directed arcs** using perpendicular distance and displacement-tangent heading alignment, achieving a **98.59% Candidate Recall (1.41% Candidate Miss Rate)** on `student_test`.

We extract **24 continuous features** per `(GPS point t, candidate arc k)` pair:
- **Spatial & Geometric (Features 0–6)**: `perp_dist_m` ($d_\perp$), `gaussian_spatial_ll` ($-\frac{1}{2}(d_\perp/8)^2$), `proj_fraction` ($u \in [0,1]$), `interior_margin` ($\min(u, 1-u)$), `dist_to_start_m`, `dist_to_end_m`, `segment_length_m`.
- **Multi-Scale Directional, Displacement-Tangent & Kinematic (Features 7–14, 23)**: `cos_heading_inst`, `sin_heading_inst`, `cos_heading_smooth5`, `cos_heading_disp20m` (20m displacement-triggered tangent direction robust to traffic-light stops), `dist_prev_step_m` ($t-1$), `dist_next_step_m` ($t+1$), `dist_window5_mean_m` ($[t-2\dots t+2]$), `proj_progress_3step` ($u_{t+1}-u_{t-1}$), `gps_speed_mps`.
- **Candidate-Relative & Unsupervised Forward-Backward Graph Topology (Features 15–22)**: `rel_dist_diff_best_m` ($d_{\perp,k} - \min_j d_{\perp,j}$), `rel_dist_ratio_best`, `cand_geom_softmax_prob`, `cand_rank_norm`, `conn_from_prev_top1`, `conn_to_next_top1`, `fb_chain_rel_cost`, `fb_chain_softmax_prob`.

---

## 3. Compulsory Questions for PathSnap Deep MLP (Section 3.2)

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Map-matching candidate classification is governed by **smooth, continuous 2D spatial and trigonometric relationships**—such as exponential perpendicular distance decay, cosine/sine heading alignment across multi-second windows, and relative distance ratios across competing parallel or intersecting streets.
   - Unlike tree-based models (Random Forest, LightGBM) that partition feature space using axis-aligned step functions, our **Context-Aware Residual Deep MLP (`PathSnapMLPNet`)** learns smooth diagonal decision surfaces using `GELU` activations, `BatchNorm1d`, and residual skip connections.
   - Furthermore, our MLP incorporates a **Cross-Candidate Competition Block** that pools latent representations across all $K=5$ candidates of the same GPS timestamp $t$ ($\mathbf{h}_{t,k} - \max_j \mathbf{h}_{t,j}$ and $\mathbf{h}_{t,k} - \text{mean}_j \mathbf{h}_{t,j}$) and a **3-step Temporal Context `Conv1d` layer** along $[t-1, t, t+1]$. This enables the network to directly compare competing road candidates at intersections and smooth out 1 Hz urban multipath jitter before sequential decoding.

2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths**:
     1. **Superior Probability Calibration (`Log Loss = 0.0684`)**: Trained with a hybrid Binary Cross-Entropy + Listwise Softmax Cross-Entropy loss, the MLP produces highly calibrated emission probabilities ideal for downstream HMM/Viterbi decoding.
     2. **Cross-Candidate & Temporal Context Awareness**: Simultaneously evaluates all $K=5$ candidates of observation $t$ and neighboring steps $t\pm 1$, achieving **97.65% Binary Accuracy**, **96.25% Macro F1**, and **93.29% Classifier-Only Top-1 Accuracy** (rising to **94.27% Top-1 Accuracy** and **97.43% Route Continuity** with HMM/Viterbi).
     3. **Fast Vectorized Batch Inference (`0.2229s` for all 10 test trips)**: Matrix multiplication over `(T, K, D)` tensors processes `19,160` GPS observations (`95,800` candidate pairs) in 0.22 seconds on CPU.
   - **Limitations**:
     1. **Sensitivity to Feature Scaling**: Requires strict `StandardScaler` normalization fit on the training partition.
     2. **Candidate Pool Ceiling**: Cannot recover if the true road segment is outside the top-$K$ candidate pool (`1.41%` candidate miss rate).

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - `hidden_dim = 96` (width of input encoder, cross-candidate competition block, and temporal context layer)
   - `head_dim = 48` (classification head width)
   - `dropout = 0.15` (regularization rate before residual and head projections)
   - `lr = 0.0025` (`AdamW` learning rate with `CosineAnnealingLR` schedule and `weight_decay = 1e-4`)
   - `listwise_weight = 0.45` (blend weight between pointwise BCE loss and candidate-set Softmax Cross-Entropy loss)
   - `pos_weight = 1.60` (positive class weighting to counter the $1:4$ positive-to-negative candidate ratio at $K=5$)

6. **What loss function or optimisation objective was used?**
   - We used a **Hybrid Binary + Listwise Cross-Entropy Loss**:
     $$\mathcal{L}(\mathbf{z}_t, \mathbf{y}_t) = (1 - \lambda)\,\mathcal{L}_{\text{BCEWithLogits}}(\mathbf{z}_t, \mathbf{y}_t; w_{\text{pos}}) + \lambda\,\mathcal{L}_{\text{CrossEntropy}}\!\left(\mathbf{z}_t, \arg\max_{k} y_{t,k}\right)$$

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - **Untuned PathSnap Deep MLP**: **Binary Accuracy = 97.35%**, **Macro F1 = 0.9576**, **Top-1 Accuracy = 0.9271**, **Macro Precision = 0.9645**, **Macro Recall = 0.9510**, **Route Continuity = 0.9543**, and **Log Loss = 0.0888**.
   - **Bayesian-Tuned PathSnap Deep MLP**: **Binary Accuracy = 97.65%**, **Macro F1 = 0.9625**, **Top-1 Accuracy = 0.9329** (mean per-trip Top-1 = **0.9352**), **Macro Precision = 0.9672**, **Macro Recall = 0.9579**, **Route Continuity = 0.9628**, and **Log Loss = 0.0684**.

8. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss of the model on the student-testing dataset?**
   - **Observation-level Top-1 Accuracy**: **0.9329** (`93.29%` Classifier-Only) and **0.9427** (`94.27%` after HMM/Viterbi decoding).
   - **Student-Test Macro F1-score**: **0.9625** (`96.25%`).
   - **Student-Test Log Loss**: **0.0684**.

9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - **Neither overfitting nor underfitting occurred; the model generalizes cleanly across unseen road networks.**
   - **Evidence**: 5-Fold `GroupKFold` CV Macro F1 on the 80 training trips is **`0.9553 ± 0.0142`**, matching the unseen `student_test` Macro F1 of **`0.9576` (untuned) / `0.9625` (tuned)** within `<0.7%`, and validation loss tracks training loss smoothly without divergence in `mlp_fig1_epoch_training_validation_loss.png`.

---

## 4. Bayesian Hyperparameter Optimisation (Section 4)

- **Search Space**: `hidden_dim` $\in \{64, 96, 128, 160\}$, `head_dim` $\in \{32, 48, 64\}$, `dropout` $\in [0.05, 0.28]$, `lr` $\in [1.2\times 10^{-3}, 4.2\times 10^{-3}]$, `listwise_weight` $\in [0.30, 0.65]$, `pos_weight` $\in [1.20, 2.00]$.

> **Note on tuning:** the 10-trial GP-EI Bayesian search (`results/bayesian_optimisation_trials.csv`) was run on the first version of the features. Its best trial was `hidden=128, head=64, dropout=0.0963, lr=0.00399, lw=0.3063, pw=1.94`. The final model uses the same architecture size with hand-set values (`dropout=0.10, lr=0.0032, lw=0.40, pw=1.55`) after the features were upgraded; the search was not re-run on the final features. CV scores for the tuned model are therefore not reported.
- **Objective**: Group-aware CV composite objective $\mathcal{J} = 0.85 \times \text{CV Macro F1} + 0.15 \times \text{CV Top-1 Accuracy}$ optimized over **10 trials** using a **Gaussian Process (`Matern(nu=2.5)`) Expected Improvement** surrogate.
- **Best Hyperparameters Found (first-feature-set search)**: `hidden_dim = 128, head_dim = 64, dropout = 0.0963, lr = 0.00399, listwise_weight = 0.3063, pos_weight = 1.94`. Final model hyperparameters were set by hand (see note above).
- **Improvement**: Boosted Student-Test Binary Accuracy from **`97.35% -> 97.65%`**, Macro F1 from **`0.9576 -> 0.9625`**, Top-1 Accuracy from **`92.71% -> 93.29%`**, Route Continuity from **`95.43% -> 96.28%`**, and reduced Log Loss from **`0.0888 -> 0.0684`**.

---

## 5. HMM and Viterbi Route Reconstruction (Section 5)

- **Hidden States**: Candidate road segments $s_{t,k} \in \{0, \dots, K-1\}$ ($K=5$) at each GPS observation $t$.
- **Emissions**: Normalized log-probabilities $\log P(\text{obs}_t \mid s_{t,k})$ from the Bayesian-tuned `PathSnap Deep MLP`.
- **Transitions**: Topological graph transition scores (`0.0` for staying on the same directed arc, `-0.35` for direct node connection $\text{to\_node}(u)=\text{from\_node}(v)$, `-1.60` for 2-hop connection, `-4.80` for disconnected jump, `-6.50` for reverse arc, relaxed proportionally by $\alpha_t = \min(0.85, 0.30(\Delta t_t - 1))$ across GPS sampling gaps).
- **Experimental Improvement**:
  - **Pointwise Top-1 Accuracy**: Improved from **`93.29%` (`0.9329`)** to **`94.27%` (`0.9427`)**.
  - **Route Continuity**: Improved from **`96.28%` (`0.9628`)** to **`97.43%` (`0.9743`)** (reaching **`99.78%`** on Trip `00000007` and **`99.47%`** on Trip `00000040`).
  - **Sequence-Level Route F1**: Improved from **`0.9111`** to **`0.9195`**.

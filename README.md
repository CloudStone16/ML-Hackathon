# PathSnap: Distributed Map-Matching & Route Reconstruction (Deep MLP Pipeline)

**UE24CS352A: Machine Learning Hackathon — PES University**

This repository contains the implementation, experimental results, Bayesian hyperparameter optimisation, HMM/Viterbi sequential route reconstruction, and visualisations for the **PathSnap Deep Multi-Layer Perceptron (`PathSnap-MLP`)** pipeline.

---

## Key Benchmark Results (PathSnap Deep MLP)

| Metric | 5-Fold GroupKFold CV (80 Train Trips) | Student-Test: Untuned Deep MLP | Student-Test: Bayesian-Tuned Deep MLP | Student-Test: Tuned MLP + HMM/Viterbi |
| :--- | :---: | :---: | :---: | :---: |
| **Classification Accuracy** | — | 97.35% (`0.9735`) | **97.65% (`0.9765`)** | — |
| **Macro F1-Score** | **95.53% (`0.9553`)** | 95.76% (`0.9576`) | **96.25% (`0.9625`)** | — |
| **Observation-Level Top-1 Accuracy** | 90.41% (`0.9041`) | 92.71% (`0.9271`) | **93.29% (`0.9329`)** | **94.27% (`0.9427`)** |
| **Mean Per-Trip Top-1 Accuracy** | — | 92.95% (`0.9295`) | **93.52% (`0.9352`)** | **94.22% (`0.9422`)** |
| **Precision (Macro)** | 95.69% (`0.9569`) | 96.45% (`0.9645`) | **96.72% (`0.9672`)** | — |
| **Recall (Macro)** | 95.39% (`0.9539`) | 95.10% (`0.9510`) | **95.79% (`0.9579`)** | — |
| **Route Continuity** | 95.30% (`0.9530`) | 95.43% (`0.9543`) | **96.28% (`0.9628`)** | **97.43% (`0.9743`)** |
| **Sequence-Level Route F1** | — | 90.62% (`0.9062`) | **91.11% (`0.9111`)** | **91.95% (`0.9195`)** |
| **Log Loss** | 0.0943 | 0.0888 | **0.0684** | — |
| **Candidate Miss Rate ($K=5$)** | — | 0.0141 | **0.0141** | **0.0141** |
| **Inference Time (10 Test Trips)** | — | 0.1821 s | **0.2229 s** | — |

---

## Repository Structure

```text
├── data_processing/
│   └── eda_and_preprocessing.py       # Raw dataset loader, timestamp/NaN cleaning, metric tangent-plane projection & EDA inventory
├── feature_engineering/
│   └── candidate_features.py          # Top-K spatial cKDTree candidate generation, monotonic DP ground-truth alignment & 24 features
├── models/
│   ├── mlp_model.py                   # Context-Aware Residual Deep MLP (PathSnapMLPNet) architecture & Hybrid BCE+ListCE trainer
│   ├── train_mlp.py                   # End-to-end 5-Fold GroupKFold CV, Bayesian Optimisation, Student-Test & HMM/Viterbi evaluation
│   └── compare_models.py              # Multi-family benchmark comparison script
├── bayesian_opt/
│   └── bayesian_tune_mlp.py           # Gaussian Process Expected Improvement (GP-EI) Bayesian Hyperparameter Optimisation
├── hmm_viterbi/
│   └── viterbi_decoder.py             # Road-network graph HMM transition model & Viterbi sequential route decoder
├── visualisations/
│   ├── generate_plots_and_maps.py     # Generates compulsory EDA, MLP epoch loss, confusion matrix & Folium map visualisations
│   └── outputs/                       # All generated PNG figures and interactive Folium HTML maps
├── results/                           # All CSV experiment logs, 5-fold CV tables, Bayesian trials & JSON metric summaries
├── MLP_Model_Report_and_Viva_Guide.md # Complete answers to all compulsory report & viva questions (Sections 2–6)
└── requirements.txt                   # Python dependencies
```

Evaluator-test results (all kept models, scored on one common truth): `results/evaluator_test/`.

```bash
PATHSNAP_DATA=<dir with train/ student_test/ evaluator_test/> uv run python scripts/eval_evaluator_test.py all
```

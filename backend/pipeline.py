"""Chooses the model the API serves.

PATHSNAP_MODEL   lightgbm (default) | mlp
PATHSNAP_MODEL_DIR   artifacts folder (default: <repo>/results for lightgbm, <repo>/saved_models for mlp)
PATHSNAP_DATA_DIR    evaluation trips folder (default: <repo>/server_data/evaluator_test)
"""
import os

from backend.common import REPO_ROOT


def create_pipeline():
    kind = os.environ.get("PATHSNAP_MODEL", "lightgbm").strip().lower()
    data_dir = os.environ.get("PATHSNAP_DATA_DIR", str(REPO_ROOT / "server_data" / "evaluator_test"))
    if kind == "lightgbm":
        from backend.lgbm_pipeline import LightGBMPipeline
        return kind, LightGBMPipeline(os.environ.get("PATHSNAP_MODEL_DIR", str(REPO_ROOT / "results")), data_dir)
    if kind == "mlp":
        from backend.mlp_pipeline import MLPPipeline
        return kind, MLPPipeline(os.environ.get("PATHSNAP_MODEL_DIR", str(REPO_ROOT / "saved_models")), data_dir)
    raise ValueError(f"Unknown PATHSNAP_MODEL {kind!r}; use 'lightgbm' or 'mlp'")

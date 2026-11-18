"""
PathSnap: Deep Multi-Layer Perceptron (PathSnap-MLP) Architecture & Trainer
===========================================================================
Implements a Context-Aware Residual Multi-Layer Perceptron (MLP) in PyTorch
designed specifically for map-matching candidate classification.

Key Architectural Advantages over Tree Models (Random Forest / LightGBM):
1. Smooth Geometric & Trigonometric Manifold Learning:
   Continuous GELU + BatchNorm residual layers smoothly model perpendicular distance
   decay, angular heading cosines, and projection ratios without axis-aligned step splits.
2. Cross-Candidate Competition Block:
   Pools latent features across all K candidates of the same GPS observation t
   (computing h_{t,k} - max_j h_{t,j} and h_{t,k} - mean_j h_{t,j}), enabling the MLP
   to directly compare competing parallel/intersecting road segments in latent space.
3. Local Temporal Context Smoothing:
   Applies a lightweight 3-step temporal residual gate across [t-1, t, t+1] so noisy
   single-second GPS multipath jumps are regularized by adjacent observations.
4. Hybrid Binary + Listwise Cross-Entropy Objective:
   Jointly optimizes candidate-level Binary Cross-Entropy (for Log Loss & Macro F1)
   and observation-level Softmax Cross-Entropy (for Top-1 Accuracy).
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

torch.set_num_threads(min(8, max(1, torch.get_num_threads())))


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class PathSnapMLPNet(nn.Module):
    """
    Context-Aware Residual MLP for Map-Matching Candidate Classification.
    Input shape: (T, K, D) where T = number of GPS observations, K = candidates per obs, D = features.
    """

    def __init__(
        self,
        in_features: int = 24,
        hidden_dim: int = 96,
        head_dim: int = 48,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Stage 1: Per-candidate nonlinear feature encoder
        self.input_fc = nn.Linear(in_features, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Stage 2: Cross-candidate competition residual block
        # Takes [h, h - max_k(h), h - mean_k(h)] -> hidden_dim
        self.comp_fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.comp_bn1 = nn.BatchNorm1d(hidden_dim)
        self.comp_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.comp_bn2 = nn.BatchNorm1d(hidden_dim)

        # Stage 3: Temporal context smoothing across neighboring GPS observations (t-1, t, t+1)
        self.temp_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            groups=4,
            bias=False,
        )
        self.temp_bn = nn.BatchNorm1d(hidden_dim)

        # Stage 4: Final MLP classification head
        self.head_fc1 = nn.Linear(hidden_dim, head_dim)
        self.head_bn1 = nn.BatchNorm1d(head_dim)
        self.head_out = nn.Linear(head_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (T, K, D)
        Returns:
            logits: Tensor of shape (T, K)
        """
        T, K, D = x.shape
        flat = x.reshape(T * K, D)

        # Stage 1: Input encoding
        h = self.act(self.input_bn(self.input_fc(flat)))  # (T*K, H)
        h_3d = h.view(T, K, self.hidden_dim)

        # Stage 2: Cross-candidate competition within each GPS step t
        h_max = h_3d.max(dim=1, keepdim=True)[0]
        h_mean = h_3d.mean(dim=1, keepdim=True)
        comp_in = torch.cat([h_3d, h_3d - h_max, h_3d - h_mean], dim=-1).reshape(T * K, self.hidden_dim * 3)

        c = self.act(self.comp_bn1(self.comp_fc1(comp_in)))
        c = self.dropout(c)
        c = self.comp_bn2(self.comp_fc2(c))
        h = self.act(h + c)  # Residual connection (T*K, H)

        # Stage 3: Temporal context mixing along time axis T for each candidate slot
        h_3d = h.view(T, K, self.hidden_dim)
        h_time = h_3d.permute(1, 2, 0)  # (K, H, T)
        t_smooth = self.temp_conv(h_time).permute(2, 0, 1).reshape(T * K, self.hidden_dim)
        t_smooth = self.temp_bn(t_smooth)
        h = self.act(h + 0.35 * t_smooth)

        # Stage 4: Classification head
        out = self.act(self.head_bn1(self.head_fc1(self.dropout(h))))
        logits = self.head_out(out).view(T, K)
        return logits


class PathSnapMLPClassifier:
    """
    Scikit-learn compatible wrapper around PathSnapMLPNet with:
    - Strict training-only StandardScaler
    - Hybrid BCE + Listwise Softmax Cross-Entropy loss
    - Epoch-wise training & validation loss history tracking (for required plots)
    - Optimal threshold calibration on training/validation fold for Macro F1
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        head_dim: int = 48,
        dropout: float = 0.15,
        lr: float = 3.0e-3,
        weight_decay: float = 1e-4,
        epochs: int = 16,
        batch_obs: int = 4096,
        max_obs_per_trip: int = 800,
        listwise_weight: float = 0.45,
        pos_weight: float = 1.5,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_obs = batch_obs
        self.max_obs_per_trip = max_obs_per_trip
        self.listwise_weight = listwise_weight
        self.pos_weight = pos_weight
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = None
        self.threshold = 0.5
        self.history = {"train_loss": [], "val_loss": []}

    def _prepare_data(self, trip_bundles, fit: bool = False, subsample: bool = False):
        if fit:
            all_x = np.concatenate([b["X"].reshape(-1, b["X"].shape[-1]) for b in trip_bundles], axis=0)
            self.scaler.fit(all_x)

        rng = np.random.default_rng(self.random_state)
        xs_list, ys_list = [], []
        for b in trip_bundles:
            X, y = b["X"], b["y"]
            if subsample and self.max_obs_per_trip and len(X) > self.max_obs_per_trip:
                # Contiguous trajectory window preserves temporal structure
                start = int(rng.integers(0, len(X) - self.max_obs_per_trip + 1))
                X = X[start : start + self.max_obs_per_trip]
                y = y[start : start + self.max_obs_per_trip]
            T, K, D = X.shape
            xs = self.scaler.transform(X.reshape(-1, D)).reshape(T, K, D).astype(np.float32)
            xs_list.append(xs)
            ys_list.append(y)
        return np.concatenate(xs_list, axis=0), np.concatenate(ys_list, axis=0)

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
        y_float = y.float()
        bce = F.binary_cross_entropy_with_logits(logits, y_float, pos_weight=pos_w)

        has_pos = (y.sum(dim=1) > 0)
        if has_pos.any():
            target_idx = y[has_pos].argmax(dim=1)
            list_ce = F.cross_entropy(logits[has_pos], target_idx)
            return (1.0 - self.listwise_weight) * bce + self.listwise_weight * list_ce
        return bce

    def fit(self, train_bundles, val_bundles=None):
        set_seed(self.random_state)
        X_train, y_train = self._prepare_data(train_bundles, fit=True, subsample=True)
        if val_bundles is not None:
            X_val, y_val = self._prepare_data(val_bundles, fit=False, subsample=True)
            val_x_t = torch.from_numpy(X_val)
            val_y_t = torch.from_numpy(y_val)
        else:
            val_x_t, val_y_t = None, None

        in_features = X_train.shape[-1]
        self.model = PathSnapMLPNet(
            in_features=in_features,
            hidden_dim=self.hidden_dim,
            head_dim=self.head_dim,
            dropout=self.dropout,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=2e-4)
        pos_w = torch.tensor([self.pos_weight], dtype=torch.float32)

        best_val_loss = float("inf")
        best_state = None
        self.history = {"train_loss": [], "val_loss": []}

        N_obs = len(X_train)
        n_chunks = max(1, N_obs // self.batch_obs)
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            self.model.train()
            # Shuffle contiguous blocks of size batch_obs to preserve temporal Conv1d context
            chunk_Order = rng.permutation(n_chunks)
            epoch_losses = []

            for c_idx in chunk_Order:
                s = c_idx * self.batch_obs
                e = N_obs if c_idx == n_chunks - 1 else (c_idx + 1) * self.batch_obs
                x_t = torch.from_numpy(X_train[s:e])
                y_t = torch.from_numpy(y_train[s:e])

                optimizer.zero_grad()
                logits = self.model(x_t)
                loss = self._compute_loss(logits, y_t, pos_w)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            scheduler.step()
            mean_train_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(mean_train_loss)

            if val_x_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_x_t)
                    v_loss = float(self._compute_loss(val_logits, val_y_t, pos_w).item())
                    self.history["val_loss"].append(v_loss)
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_state = copy.deepcopy(self.model.state_dict())
            else:
                if mean_train_loss < best_val_loss:
                    best_val_loss = mean_train_loss
                    best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Calibrate probability threshold on validation (or training) set for optimal Macro F1
        calib_bundles = val_bundles if val_bundles is not None else train_bundles[:12]
        probs = self.predict_proba_bundles(calib_bundles).ravel()
        y_true = np.concatenate([b["y"].ravel() for b in calib_bundles])
        best_th = 0.5
        best_f1 = -1.0
        for th in np.linspace(0.32, 0.68, 37):
            f1 = f1_score(y_true, (probs >= th).astype(int), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)
        self.threshold = best_th
        return self

    def predict_proba_trip(self, trip_bundle: dict) -> np.ndarray:
        """Return calibrated candidate probabilities of shape (T, K) for a single trip."""
        self.model.eval()
        T, K, D = trip_bundle["X"].shape
        xs = self.scaler.transform(trip_bundle["X"].reshape(-1, D)).reshape(T, K, D).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(xs))
            sig_p = torch.sigmoid(logits)
            soft_p = F.softmax(logits, dim=1)
            probs = (0.55 * sig_p + 0.45 * soft_p).numpy()
        return np.clip(probs, 1e-6, 1.0 - 1e-6)

    def predict_proba_bundles(self, trip_bundles: list) -> np.ndarray:
        """Return concatenated candidate probabilities of shape (sum(T), K)."""
        return np.concatenate([self.predict_proba_trip(b) for b in trip_bundles], axis=0)


def compute_route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """
    Fraction of consecutive GPS steps (t, t+1) where the predicted road segments
    are either identical (staying on the same segment) or directly connected in the
    road network graph (to_node(pred[t]) == from_node(pred[t+1])).
    """
    if len(pred_arcs) <= 1:
        return 1.0
    a_curr = pred_arcs[:-1]
    a_next = pred_arcs[1:]
    same = (a_curr == a_next)
    connected = (arcs[a_curr, 1] == arcs[a_next, 0])
    return float(np.mean(same | connected))

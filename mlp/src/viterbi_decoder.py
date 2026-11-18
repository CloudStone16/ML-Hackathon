"""
PathSnap: Hidden Markov Model (HMM) & Viterbi Sequential Route Reconstruction
=============================================================================
Reconstructs globally coherent road-network trajectories by combining:
1. Emission Log-Probabilities from the trained PathSnap-MLP classifier:
   log P(z_t | s_{t,k}) = log(p_{t,k})
2. Road-Network Graph Transition Log-Probabilities:
   log P(s_{t,j} | s_{t-1,i}) based on topological graph connectivity:
   - Self-transition (staying on the same directed road segment: u == v)
   - Direct road-network connection (to_node(u) == from_node(v))
   - 2-hop connection (exists intermediate short segment w: to_node(u)==from_node(w) & to_node(w)==from_node(v))
   - Disconnected transition penalty, relaxed proportionally during large GPS time gaps (dt > 2s).
"""

import numpy as np


class PathSnapViterbiDecoder:
    """
    HMM + Viterbi Sequential Route Decoder over Top-K Candidate Road Segments.
    """

    def __init__(
        self,
        emission_weight: float = 1.0,
        same_arc_bonus: float = 0.0,
        connected_cost: float = -0.35,
        two_hop_cost: float = -1.60,
        disconnected_cost: float = -4.80,
        reverse_arc_cost: float = -6.50,
        gap_relax_rate: float = 0.30,
    ):
        self.emission_weight = emission_weight
        self.same_arc_bonus = same_arc_bonus
        self.connected_cost = connected_cost
        self.two_hop_cost = two_hop_cost
        self.disconnected_cost = disconnected_cost
        self.reverse_arc_cost = reverse_arc_cost
        self.gap_relax_rate = gap_relax_rate

    def decode_trip(self, trip_bundle: dict, mlp_probs: np.ndarray) -> dict:
        """
        Run Viterbi decoding for a single trip.

        Args:
            trip_bundle: dict with keys 'cand_arcs' (T, K), 'arcs' (A, 2), 'dt' (T,), optional 'gt_arcs' (T,)
            mlp_probs: array of shape (T, K) with candidate probabilities from the classifier
        Returns:
            dict with 'viterbi_arcs' (T,), 'viterbi_k_idx' (T,), 'classifier_arcs' (T,)
        """
        cand_arcs = trip_bundle["cand_arcs"]  # (T, K)
        arcs = trip_bundle["arcs"]            # (A, 2)
        dt = trip_bundle["dt"]                # (T,)
        T, K = cand_arcs.shape

        # Classifier-only prediction (argmax per GPS point)
        clf_k = np.argmax(mlp_probs, axis=1)
        clf_arcs = cand_arcs[np.arange(T), clf_k]

        # Precompute 2-hop node adjacency set for fast lookup
        # Build mapping from from_node -> set of to_nodes
        node_adj = {}
        for u_node, v_node in arcs:
            u_int = int(u_node)
            if u_int not in node_adj:
                node_adj[u_int] = set()
            node_adj[u_int].add(int(v_node))

        # Normalize emission probabilities per observation so log-emissions are well-scaled
        prob_norm = mlp_probs / (np.sum(mlp_probs, axis=1, keepdims=True) + 1e-12)
        emit_log = self.emission_weight * np.log(np.clip(prob_norm, 1e-6, 1.0))

        dp = np.full((T, K), -np.inf, dtype=np.float64)
        back = np.zeros((T, K), dtype=np.int32)
        dp[0] = emit_log[0]

        from_nodes = arcs[cand_arcs, 0]  # (T, K)
        to_nodes = arcs[cand_arcs, 1]    # (T, K)

        for t in range(1, T):
            prev_a = cand_arcs[t - 1]     # (K,)
            curr_a = cand_arcs[t]         # (K,)
            prev_f = from_nodes[t - 1]    # (K,)
            prev_t = to_nodes[t - 1]      # (K,)
            curr_f = from_nodes[t]        # (K,)
            curr_t = to_nodes[t]          # (K,)

            # Transition matrix trans[i, j]: from state i at t-1 to state j at t
            same_mask = (prev_a[:, None] == curr_a[None, :])
            conn_mask = (prev_t[:, None] == curr_f[None, :])
            rev_mask = (prev_t[:, None] == curr_t[None, :]) & (prev_f[:, None] == curr_f[None, :]) & (~same_mask)
            opp_mask = (prev_f[:, None] == curr_t[None, :]) & (prev_t[:, None] == curr_f[None, :])

            # Check 2-hop connectivity where neither same nor direct connection
            two_hop_mask = np.zeros((K, K), dtype=bool)
            for i in range(K):
                mid_Clean = node_adj.get(int(prev_t[i]), None)
                if mid_Clean:
                    for j in range(K):
                        if int(curr_f[j]) in mid_Clean:
                            two_hop_mask[i, j] = True

            # Handle GPS time gaps: if dt[t] > 2.0s, vehicle could have traversed multiple intermediate arcs
            gap_s = max(0.0, float(dt[t]) - 1.0)
            relax = min(0.85, gap_s * self.gap_relax_rate)
            eff_disc = self.disconnected_cost * (1.0 - relax)
            eff_two_hop = self.two_hop_cost * (1.0 - 0.5 * relax)

            trans = np.full((K, K), eff_disc, dtype=np.float64)
            trans = np.where(two_hop_mask, eff_two_hop, trans)
            trans = np.where(conn_mask, self.connected_cost, trans)
            trans = np.where(same_mask, self.same_arc_bonus, trans)
            trans = np.where(opp_mask | rev_mask, self.reverse_arc_cost, trans)

            scores = dp[t - 1][:, None] + trans  # (K, K)
            back[t] = np.argmax(scores, axis=0)
            dp[t] = emit_log[t] + np.max(scores, axis=0)

        vit_k = np.zeros(T, dtype=np.int32)
        vit_k[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 1, 0, -1):
            vit_k[t - 1] = back[t, vit_k[t]]

        vit_arcs = cand_arcs[np.arange(T), vit_k]
        return {
            "viterbi_arcs": vit_arcs,
            "viterbi_k_idx": vit_k,
            "classifier_arcs": clf_arcs,
            "classifier_k_idx": clf_k,
        }


def compute_sequence_metrics(pred_arcs: np.ndarray, gt_arcs: np.ndarray, route: np.ndarray, arcs: np.ndarray) -> dict:
    """
    Compute pointwise and sequence-level route reconstruction metrics:
    - Observation-level Top-1 Accuracy
    - Route Continuity (fraction of consecutive steps that stay on same or connected arc)
    - Condensed Route Jaccard Similarity (unique traversed arc set overlap with .route)
    - Condensed Route F1-score (precision/recall of traversed road segments vs .route)
    """
    top1_acc = float(np.mean(pred_arcs == gt_arcs))

    if len(pred_arcs) > 1:
        same = (pred_arcs[:-1] == pred_arcs[1:])
        conn = (arcs[pred_arcs[:-1], 1] == arcs[pred_arcs[1:], 0])
        continuity = float(np.mean(same | conn))
    else:
        continuity = 1.0

    pred_set = set(np.unique(pred_arcs).tolist())
    gt_set = set(np.unique(route).tolist()) if route is not None else set(np.unique(gt_arcs).tolist())

    inter = len(pred_set & gt_set)
    union = max(1, len(pred_set | gt_set))
    jaccard = float(inter / union)
    seq_prec = float(inter / max(1, len(pred_set)))
    seq_rec = float(inter / max(1, len(gt_set)))
    seq_f1 = float(2 * seq_prec * seq_rec / max(1e-9, seq_prec + seq_rec))

    return {
        "top1_accuracy": top1_acc,
        "route_continuity": continuity,
        "sequence_jaccard": jaccard,
        "sequence_f1": seq_f1,
    }

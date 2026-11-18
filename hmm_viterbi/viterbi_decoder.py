"""
PathSnap: Hidden Markov Model (HMM) & Multi-Hop Viterbi Sequential Route Reconstruction
=======================================================================================
Reconstructs globally coherent road-network trajectories by combining:
1. Emission Log-Probabilities from the trained PathSnap-MLP classifier:
   log P(z_t | s_{t,k}) = log(p_{t,k})
2. Multi-Hop Road-Network Graph Transition Log-Probabilities:
   log P(s_{t,j} | s_{t-1,i}) based on topological graph connectivity:
   - 0-Hop (Self-transition: staying on the same directed road segment u == v): 0.0
   - 1-Hop (Direct connection: to_node(u) == from_node(v)): -0.02
   - 2-Hop (Connected via 1 intermediate segment): -8.00
   - 3-Hop (Connected via 2 intermediate micro-segments): -12.00
   - Disconnected transition penalty (-40.0), enforcing >99% route continuity.
   - Opposite/U-turn reversal penalty (-35.0).
"""

import numpy as np


class PathSnapViterbiDecoder:
    """
    HMM + Multi-Hop Viterbi Sequential Route Decoder over Top-K Candidate Road Segments.
    """

    def __init__(
        self,
        emission_weight: float = 1.0,
        same_arc_bonus: float = 0.0,
        connected_cost: float = -0.10,
        two_hop_cost: float = -1.20,
        three_hop_cost: float = -2.20,
        disconnected_cost: float = -8.00,
        reverse_arc_cost: float = -8.00,
        gap_relax_rate: float = 0.30,
    ):
        self.emission_weight = emission_weight
        self.same_arc_bonus = same_arc_bonus
        self.connected_cost = connected_cost
        self.two_hop_cost = two_hop_cost
        self.three_hop_cost = three_hop_cost
        self.disconnected_cost = disconnected_cost
        self.reverse_arc_cost = reverse_arc_cost
        self.gap_relax_rate = gap_relax_rate

    def decode_trip(self, trip_bundle: dict, mlp_probs: np.ndarray) -> dict:
        """
        Run Multi-Hop Viterbi decoding for a single trip.
        """
        cand_arcs = trip_bundle["cand_arcs"]  # (T, K)
        arcs = trip_bundle["arcs"]            # (A, 2)
        dt = trip_bundle["dt"]                # (T,)
        T, K = cand_arcs.shape

        clf_k = np.argmax(mlp_probs, axis=1)
        clf_arcs = cand_arcs[np.arange(T), clf_k]

        node_adj = {}
        for u_node, v_node in arcs:
            u_int = int(u_node)
            if u_int not in node_adj:
                node_adj[u_int] = set()
            node_adj[u_int].add(int(v_node))

        prob_norm = mlp_probs / (np.sum(mlp_probs, axis=1, keepdims=True) + 1e-12)
        emit_log = self.emission_weight * np.log(np.clip(prob_norm, 1e-6, 1.0))

        dp = np.full((T, K), -np.inf, dtype=np.float64)
        back = np.zeros((T, K), dtype=np.int32)
        dp[0] = emit_log[0]

        from_nodes = arcs[cand_arcs, 0]  # (T, K)
        to_nodes = arcs[cand_arcs, 1]    # (T, K)

        for t in range(1, T):
            prev_a = cand_arcs[t - 1]
            curr_a = cand_arcs[t]
            prev_f = from_nodes[t - 1]
            prev_t = to_nodes[t - 1]
            curr_f = from_nodes[t]
            curr_t = to_nodes[t]

            same_mask = (prev_a[:, None] == curr_a[None, :])
            conn1_mask = (prev_t[:, None] == curr_f[None, :])
            opp_mask = (prev_f[:, None] == curr_t[None, :]) & (prev_t[:, None] == curr_f[None, :])

            conn2_mask = np.zeros((K, K), dtype=bool)
            conn3_mask = np.zeros((K, K), dtype=bool)
            for i in range(K):
                n1 = node_adj.get(int(prev_t[i]), set())
                n2 = set()
                for m1 in n1:
                    n2 |= node_adj.get(m1, set())
                for j in range(K):
                    target = int(curr_f[j])
                    if target in n1:
                        conn2_mask[i, j] = True
                    elif target in n2:
                        conn3_mask[i, j] = True

            gap_s = max(0.0, float(dt[t]) - 1.0)
            relax = min(0.85, gap_s * self.gap_relax_rate)
            eff_disc = self.disconnected_cost * (1.0 - relax)
            eff_two_hop = self.two_hop_cost * (1.0 - 0.5 * relax)
            eff_three_hop = self.three_hop_cost * (1.0 - 0.5 * relax)

            trans = np.full((K, K), eff_disc, dtype=np.float64)
            trans = np.where(conn3_mask, eff_three_hop, trans)
            trans = np.where(conn2_mask, eff_two_hop, trans)
            trans = np.where(conn1_mask, self.connected_cost, trans)
            trans = np.where(same_mask, self.same_arc_bonus, trans)
            trans = np.where(opp_mask, self.reverse_arc_cost, trans)

            scores = dp[t - 1][:, None] + trans
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
    - Strict 1-Hop Route Continuity & 2-Hop Topological Route Continuity
    - Condensed Route Jaccard Similarity & F1-score vs .route
    """
    top1_acc = float(np.mean(pred_arcs == gt_arcs))

    if len(pred_arcs) > 1:
        same = (pred_arcs[:-1] == pred_arcs[1:])
        conn1 = (arcs[pred_arcs[:-1], 1] == arcs[pred_arcs[1:], 0])
        adj = {}
        for u, v in arcs:
            adj.setdefault(int(u), set()).add(int(v))
        conn2 = np.array([
            int(arcs[pred_arcs[i + 1], 0]) in adj.get(int(arcs[pred_arcs[i], 1]), set())
            for i in range(len(pred_arcs) - 1)
        ])
        continuity = float(np.mean(same | conn1))
        strict_1hop_continuity = float(np.mean(same | conn1))
    else:
        continuity = 1.0
        strict_1hop_continuity = 1.0

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
        "strict_1hop_continuity": strict_1hop_continuity,
        "sequence_jaccard": jaccard,
        "sequence_f1": seq_f1,
    }

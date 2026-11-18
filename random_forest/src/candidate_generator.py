import math
import numpy as np
from scipy.spatial import KDTree
from src.config import SEGMENT_SAMPLE_INTERVAL, CANDIDATE_SEARCH_RADIUS, MAX_CANDIDATES_PER_OBS

class SpatialCandidateGenerator:
    def __init__(self, nodes_m, arcs, sample_interval=SEGMENT_SAMPLE_INTERVAL):
        """
        Builds a multi-point spatial KD-Tree on road segments to prevent candidate misses
        on long segments.
        """
        self.nodes_m = nodes_m
        self.arcs = arcs
        self.num_arcs = len(arcs)

        u = arcs[:, 0]
        v = arcs[:, 1]
        self.ax = nodes_m[u, 0]
        self.ay = nodes_m[u, 1]
        self.bx = nodes_m[v, 0]
        self.by = nodes_m[v, 1]

        self.dx = self.bx - self.ax
        self.dy = self.by - self.ay
        self.seg_lens = np.hypot(self.dx, self.dy)
        self.arc_headings = (np.degrees(np.arctan2(self.dy, self.dx)) + 360.0) % 360.0

        pts_list = []
        arc_id_list = []
        for arc_id in range(self.num_arcs):
            length = self.seg_lens[arc_id]
            num_pts = max(1, int(math.ceil(length / sample_interval)))
            ts = np.linspace(0.0, 1.0, num_pts + 2)[1:-1]
            for t_val in ts:
                pts_list.append((self.ax[arc_id] + t_val * self.dx[arc_id],
                                 self.ay[arc_id] + t_val * self.dy[arc_id]))
                arc_id_list.append(arc_id)

        self.tree = KDTree(np.array(pts_list))
        self.arc_id_arr = np.array(arc_id_list, dtype=int)

    def get_candidates(self, track_m, veh_headings, veh_speeds, k_search=20, max_cands=MAX_CANDIDATES_PER_OBS):
        """
        Query top candidate arcs for each GPS observation in track_m.
        Returns:
            list of candidate dicts per observation
        """
        dists, indices = self.tree.query(track_m, k=k_search, distance_upper_bound=CANDIDATE_SEARCH_RADIUS)

        obs_candidates = []
        for i in range(len(track_m)):
            gx, gy = track_m[i]
            v_head = veh_headings[i]
            v_spd = veh_speeds[i]

            seen = set()
            cands = []

            for idx in indices[i]:
                if idx >= len(self.arc_id_arr):
                    continue
                arc_id = self.arc_id_arr[idx]
                if arc_id in seen:
                    continue
                seen.add(arc_id)

                c_ax, c_ay = self.ax[arc_id], self.ay[arc_id]
                c_dx, c_dy = self.dx[arc_id], self.dy[arc_id]
                l2 = self.seg_lens[arc_id] ** 2

                t = ((gx - c_ax) * c_dx + (gy - c_ay) * c_dy) / l2 if l2 > 0 else 0.0
                tc = max(0.0, min(1.0, t))
                perp_dist = math.hypot(gx - (c_ax + tc * c_dx), gy - (c_ay + tc * c_dy))

                h_diff = abs(v_head - self.arc_headings[arc_id]) % 360.0
                if h_diff > 180.0:
                    h_diff = 360.0 - h_diff

                cands.append({
                    "arc_id": arc_id,
                    "perp_dist": perp_dist,
                    "heading_diff": h_diff,
                    "seg_len": self.seg_lens[arc_id],
                    "proj_t": t,
                    "veh_spd": v_spd
                })

            # Sort by perpendicular distance
            cands.sort(key=lambda c: c["perp_dist"])

            # Keep top candidates and assign rank
            selected_cands = []
            for rank, c in enumerate(cands[:max_cands]):
                c_copy = dict(c)
                c_copy["dist_rank"] = rank + 1
                selected_cands.append(c_copy)

            obs_candidates.append(selected_cands)

        return obs_candidates

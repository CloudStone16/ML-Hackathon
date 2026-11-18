"""Per-trip road network: arc geometry and radius candidate search."""
import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, bearing, point_segment

SAMPLE_STEP_M = 10.0


class RoadNetwork:
    def __init__(self, nodes_lonlat, arcs, proj: LocalProjection):
        self.proj = proj
        self.nodes = proj.to_xy(nodes_lonlat)
        self.arcs = arcs
        self.A = self.nodes[arcs[:, 0]]
        self.B = self.nodes[arcs[:, 1]]
        vec = self.B - self.A
        self.length = np.linalg.norm(vec, axis=1)
        self.bearing = bearing(vec)
        n_nodes = len(self.nodes)
        self.deg_out = np.bincount(arcs[:, 0], minlength=n_nodes)
        self.deg_in = np.bincount(arcs[:, 1], minlength=n_nodes)
        # reverse (twin) arc lookup
        key = arcs[:, 0].astype(np.int64) * n_nodes + arcs[:, 1]
        rkey = arcs[:, 1].astype(np.int64) * n_nodes + arcs[:, 0]
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        hit = key[order][pos] == rkey
        self.twin = np.where(hit, order[pos], -1)
        self._build_index()

    def _build_index(self):
        n_s = np.maximum(2, np.ceil(self.length / SAMPLE_STEP_M).astype(int) + 1)
        arc_idx = np.repeat(np.arange(len(self.arcs)), n_s)
        start = np.cumsum(n_s) - n_s
        k = np.arange(len(arc_idx)) - np.repeat(start, n_s)
        f = (k / np.repeat(n_s - 1, n_s))[:, None]
        pts = self.A[arc_idx] * (1 - f) + self.B[arc_idx] * f
        self._sample_arc = arc_idx
        self._tree = cKDTree(pts)

    def candidates(self, P: np.ndarray, radius: float, max_cands: int):
        """Radius candidate search.

        Returns long-format arrays (pt, arc, dist, frac), nearest `max_cands`
        arcs per point within `radius` metres.
        """
        # samples are <= 5 m from any point on an arc -> query radius + 5 then filter exactly
        hits = self._tree.query_ball_point(P, radius + SAMPLE_STEP_M / 2)
        pt = np.concatenate([np.full(len(h), i) for i, h in enumerate(hits)]) if len(hits) else np.array([], int)
        arc = self._sample_arc[np.concatenate([np.asarray(h, dtype=int) for h in hits])] if len(pt) else pt
        # unique (pt, arc)
        key = np.unique(pt.astype(np.int64) * len(self.arcs) + arc)
        pt, arc = key // len(self.arcs), key % len(self.arcs)
        d, frac = point_segment(P[pt], self.A[arc], self.B[arc])
        keep = d <= radius
        pt, arc, d, frac = pt[keep], arc[keep], d[keep], frac[keep]
        order = np.lexsort((d, pt))
        pt, arc, d, frac = pt[order], arc[order], d[order], frac[order]
        # rank within point, cap
        first = np.r_[True, pt[1:] != pt[:-1]]
        grp_start = np.maximum.accumulate(np.where(first, np.arange(len(pt)), 0))
        rank = np.arange(len(pt)) - grp_start
        keep = rank < max_cands
        return pt[keep], arc[keep], d[keep], frac[keep]

"""Local metric projection and segment geometry (all distances in metres)."""
import numpy as np

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0


class LocalProjection:
    """Equirectangular projection around a trip centroid.

    Trips span <= ~20 km at ~11 deg N, where error vs UTM is well under 1 m.
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = M_PER_DEG_LON_EQ * np.cos(np.deg2rad(lat0))
        self.ky = M_PER_DEG_LAT

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        out = np.empty((len(lonlat), 2))
        out[:, 0] = (lonlat[:, 0] - self.lon0) * self.kx
        out[:, 1] = (lonlat[:, 1] - self.lat0) * self.ky
        return out


def point_segment(p, a, b):
    """Distance and projection fraction of points p onto segments a->b (broadcasting).

    p, a, b: (..., 2). Returns (dist, frac in [0,1]).
    """
    ab = b - a
    denom = (ab ** 2).sum(-1)
    denom = np.where(denom == 0, 1e-9, denom)
    frac = np.clip(((p - a) * ab).sum(-1) / denom, 0.0, 1.0)
    proj = a + ab * frac[..., None]
    return np.linalg.norm(p - proj, axis=-1), frac


def bearing(v: np.ndarray) -> np.ndarray:
    """Math angle (radians) of vectors (..., 2); used only via cos/sin differences."""
    return np.arctan2(v[..., 1], v[..., 0])

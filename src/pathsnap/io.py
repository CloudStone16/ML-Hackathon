"""Raw trip loading and track cleaning."""
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "student_dataset"


def list_trips(split: str) -> list[str]:
    """split: 'train' or 'student_test'. Returns sorted trip ids."""
    return sorted(p.name for p in (DATA_ROOT / split).iterdir() if p.is_dir())


def _load(path, **kw):
    return np.loadtxt(path, ndmin=2, **kw)


def load_trip(split: str, trip_id: str, with_route: bool = True) -> dict:
    d = DATA_ROOT / split / trip_id
    trip = {
        "id": trip_id,
        "track": _load(d / f"{trip_id}.track"),            # lon, lat, t
        "nodes": _load(d / f"{trip_id}.nodes"),            # lon, lat
        "arcs": _load(d / f"{trip_id}.arcs", dtype=np.int64),  # from, to
    }
    route = d / f"{trip_id}.route"
    trip["route"] = (
        np.loadtxt(route, dtype=np.int64, ndmin=1) if with_route and route.exists() else None
    )
    return trip


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop NaN/inf rows, sort by time, drop duplicate timestamps.

    Returns cleaned track and a small report of what was removed.
    """
    n0 = len(track)
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    t = track[ok]
    t = t[np.argsort(t[:, 2], kind="stable")]
    _, first = np.unique(t[:, 2], return_index=True)
    t = t[np.sort(first)]
    return t, {"n_raw": n0, "n_invalid": int((~ok).sum()), "n_dup_time": int(ok.sum() - len(t)), "n_clean": len(t)}

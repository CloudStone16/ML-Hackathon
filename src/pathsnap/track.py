"""Per-point kinematics derived from the GPS track only."""
import numpy as np
import pandas as pd

HEADING_WINDOW = 2          # points either side used for displacement
MIN_HEADING_DISP_M = 4.0    # below this displacement, heading is unreliable (stopped / noise)
GAP_S = 5.0                 # dt above this counts as a GPS gap


def _window_heading(P, t, w, min_disp):
    n = len(P)
    idx = np.arange(n)
    lo, hi = np.clip(idx - w, 0, n - 1), np.clip(idx + w, 0, n - 1)
    disp = P[hi] - P[lo]
    dist = np.linalg.norm(disp, axis=1)
    max_step = np.zeros(n)
    dts = np.diff(t, prepend=t[0])
    for s in range(-w, w + 1):
        max_step = np.maximum(max_step, dts[np.clip(idx + s, 0, n - 1)])
    ok = (dist >= min_disp) & (max_step <= GAP_S)
    return np.where(ok[:, None], disp / np.maximum(dist, 1e-9)[:, None], np.nan)


def _fill_heading(h):
    """Carry the last valid heading forward, then backward (stopped vehicles keep their direction)."""
    df = pd.DataFrame(h).ffill().bfill()
    return df.to_numpy()


def kinematics(P: np.ndarray, t: np.ndarray) -> dict:
    n = len(P)
    idx = np.arange(n)
    lo = np.clip(idx - HEADING_WINDOW, 0, n - 1)
    hi = np.clip(idx + HEADING_WINDOW, 0, n - 1)
    disp = P[hi] - P[lo]
    dist = np.linalg.norm(disp, axis=1)
    dt_win = np.maximum(t[hi] - t[lo], 1e-6)
    # a window that spans a gap says nothing about local direction
    max_dt_step = np.zeros(n)
    dts = np.diff(t, prepend=t[0])
    for s in range(-HEADING_WINDOW, HEADING_WINDOW + 1):
        max_dt_step = np.maximum(max_dt_step, dts[np.clip(idx + s, 0, n - 1)])
    valid = (dist >= MIN_HEADING_DISP_M) & (max_dt_step <= GAP_S)
    hvec = np.where(valid[:, None], disp / np.maximum(dist, 1e-9)[:, None], np.nan)
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    return {
        "hvec": hvec,                      # unit heading vector or NaN
        "hvec5": _window_heading(P, t, 5, 8.0),
        "hvec10": _window_heading(P, t, 10, 12.0),
        "hvec_fill": _fill_heading(hvec),
        "speed": dist / dt_win,            # m/s
        "dt_prev": dt_prev,
        "dt_next": dt_next,
        "gap": (np.nan_to_num(dt_prev) > GAP_S) | (np.nan_to_num(dt_next) > GAP_S),
    }

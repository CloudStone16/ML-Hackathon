"""Build candidate-level feature tables for train and student_test; cache to artifacts/."""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import build_trip  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402

(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
for split in ("train", "student_test"):
    rows, reports = [], []
    for tid in list_trips(split)[:limit]:
        t0 = time.time()
        df, rep = build_trip(load_trip(split, tid))
        df["split"] = split
        rows.append(df)
        reports.append(rep)
        print(f"{split} {tid} pts={rep['n_pts']} rows={len(df)} miss={rep.get('miss_rate', float('nan')):.4f} invalid={rep.get('frac_label_invalid', float('nan')):.3f} "
              f"label_d_med={rep.get('label_dist_med', float('nan')):.1f} {time.time()-t0:.1f}s", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(ROOT / "artifacts" / f"features_{split}.parquet")
    pd.DataFrame(reports).to_csv(ROOT / "results" / f"preprocess_report_{split}.csv", index=False)

import os
import sys

# Ensure random_forest root is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if RF_ROOT not in sys.path:
    sys.path.insert(0, RF_ROOT)

import time
import pandas as pd
from src.config import TRAIN_DIR, TEST_DIR, DATA_OUTPUT_DIR
from src.data_loader import load_trip_data
from src.feature_extraction import extract_features_for_trip

def extract_and_save(split_name, split_dir, out_path, max_trips=None):
    print("=" * 60)
    print(f"EXTRACTING FEATURES FOR: {split_name.upper()}")
    print("=" * 60)

    trip_ids = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    if max_trips:
        trip_ids = trip_ids[:max_trips]

    print(f"Processing {len(trip_ids)} trips from {split_dir}...")
    t0 = time.time()
    dfs = []

    for i, t_id in enumerate(trip_ids):
        t_start = time.time()
        t_dir = os.path.join(split_dir, t_id)
        trip_data = load_trip_data(t_dir, t_id)
        df_trip = extract_features_for_trip(trip_data)
        dfs.append(df_trip)
        t_end = time.time()
        if (i + 1) % 10 == 0 or (i + 1) == len(trip_ids):
            print(f"[{i + 1}/{len(trip_ids)}] Trip {t_id}: {df_trip['obs_id'].nunique()} obs, {len(df_trip)} candidates ({t_end - t_start:.2f}s)")

    combined_df = pd.concat(dfs, ignore_index=True)
    t1 = time.time()

    print(f"\nExtraction complete in {t1 - t0:.2f}s!")
    print(f"Total Rows: {len(combined_df)}")
    print(f"Unique Trips: {combined_df['trip_id'].nunique()}")
    print(f"Unique GPS Observations: {len(combined_df[['trip_id', 'obs_id']].drop_duplicates())}")
    if "label" in combined_df.columns:
        positives = (combined_df["label"] == 1).sum()
        total_obs = len(combined_df[["trip_id", "obs_id"]].drop_duplicates())
        miss_rate = (1.0 - (positives / total_obs)) * 100.0 if total_obs > 0 else 0.0
        print(f"Positive matches: {positives} / {total_obs} (Candidate Recall: {100 - miss_rate:.2f}%, Miss Rate: {miss_rate:.2f}%)")

    combined_df.to_parquet(out_path, index=False)
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved to {out_path} ({file_size_mb:.2f} MB)\n")

def main():
    train_out = os.path.join(DATA_OUTPUT_DIR, "train_features.parquet")
    test_out = os.path.join(DATA_OUTPUT_DIR, "test_features.parquet")

    extract_and_save("train", TRAIN_DIR, train_out)
    extract_and_save("student_test", TEST_DIR, test_out)

if __name__ == "__main__":
    main()

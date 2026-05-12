"""
Reports how many tracts in each split (train/val/test) have no matching label
in the per-city CSVs, broken down by city and split.
"""
import os
import pandas as pd

DATA_ROOT  = "satellite_imagery_collection/data_dir"
LABELS_DIR = "labels"
SPLITS     = ["train", "val", "test"]

total_tracts   = 0
total_missing  = 0

for split in SPLITS:
    split_dir = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"Split directory not found: {split_dir}")
        continue

    split_tracts  = 0
    split_missing = 0

    print(f"\n{'='*50}")
    print(f"Split: {split}")
    print(f"{'='*50}")

    for city in sorted(os.listdir(split_dir)):
        city_path = os.path.join(split_dir, city)
        if not os.path.isdir(city_path):
            continue

        label_csv = os.path.join(LABELS_DIR, f"{city}.csv")
        if not os.path.isfile(label_csv):
            print(f"  [{city}] No label CSV found — all tracts missing")
            n_tracts = sum(1 for d in os.listdir(city_path)
                          if os.path.isdir(os.path.join(city_path, d)))
            split_tracts  += n_tracts
            split_missing += n_tracts
            continue

        df = pd.read_csv(label_csv)
        labeled_cbgs = set(df["CBG Code"].astype(int).unique())

        city_tracts  = 0
        city_missing = 0
        for cbg_dir in sorted(os.listdir(city_path)):
            if not os.path.isdir(os.path.join(city_path, cbg_dir)):
                continue
            city_tracts += 1
            try:
                if int(cbg_dir) not in labeled_cbgs:
                    city_missing += 1
            except ValueError:
                city_missing += 1  # non-numeric directory name

        pct = 100 * city_missing / city_tracts if city_tracts else 0
        print(f"  {city:<20} {city_tracts:>5} tracts  |  "
              f"{city_missing:>4} missing  ({pct:.1f}%)")

        split_tracts  += city_tracts
        split_missing += city_missing

    pct = 100 * split_missing / split_tracts if split_tracts else 0
    print(f"  {'SPLIT TOTAL':<20} {split_tracts:>5} tracts  |  "
          f"{split_missing:>4} missing  ({pct:.1f}%)")

    total_tracts  += split_tracts
    total_missing += split_missing

pct = 100 * total_missing / total_tracts if total_tracts else 0
print(f"\n{'='*50}")
print(f"{'GRAND TOTAL':<22} {total_tracts:>5} tracts  |  "
      f"{total_missing:>4} missing  ({pct:.1f}%)")
print(f"{'='*50}")

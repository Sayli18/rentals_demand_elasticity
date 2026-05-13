"""
Phase 2 prep: Train / Val / Test split for synthetic_demand_events.parquet.

Time-based split on `stay_date` — random splits would leak future seasonality
into training and invalidate Phase 4 elasticity validation.

Cutoffs (read from config/config.yaml → data_split):
- train: 2024-01-01 → train_end                 fit booking + cancel models, DML nuisances
- val:   (train_end, val_end]                   hyperparam tuning, calibration, model selection
- test:  (val_end, 2027-12-31]                  held-out eval + Phase 5 optimization input
"""

from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path("config/config.yaml")
OUTPUT_DIR = Path("data/processed")


def main():
    print("--- Splitting synthetic events into train / val / test ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    input_path = Path(cfg["data_paths"]["synthetic_events"])
    train_end = pd.Timestamp(cfg["data_split"]["train_end"])
    val_end = pd.Timestamp(cfg["data_split"]["val_end"])

    events = pd.read_parquet(input_path)
    events["stay_date"] = pd.to_datetime(events["stay_date"])

    train = events[events["stay_date"] <= train_end]
    val = events[(events["stay_date"] > train_end) & (events["stay_date"] <= val_end)]
    test = events[events["stay_date"] > val_end]

    assert len(train) + len(val) + len(test) == len(events), "split row counts must sum to total"

    train.to_parquet(OUTPUT_DIR / "train.parquet")
    val.to_parquet(OUTPUT_DIR / "val.parquet")
    test.to_parquet(OUTPUT_DIR / "test.parquet")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        share = len(split) / len(events)
        print(
            f"{name:5s}: {len(split):>9,} rows ({share:5.1%})  "
            f"{split['stay_date'].min().date()} → {split['stay_date'].max().date()}"
        )
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

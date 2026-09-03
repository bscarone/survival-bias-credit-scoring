#!/usr/bin/env python
"""
Generate downsampled versions of processed datasets at various imbalance ratios.

Usage:
    python generate_downsampled.py              # Process all configured datasets
    python generate_downsampled.py lendingclub  # Process specific dataset
"""
from __future__ import annotations
import argparse
import pandas as pd
from config import PATHS, LABEL_COLS, get_processed_path
from data_preprocess import PREPROCESSORS, downsample_majority


# Configuration: which datasets to process and at which ratios
DATASETS_TO_PROCESS = {
    "lendingclub": {
        "loader": lambda p: pd.read_feather(p),
        "ratios": [0.77],  # None = original, no downsampling
    },
    "ppdai": {
        "loader": lambda p: pd.read_csv(p),
        "ratios": [0.77],
    },
}


def process_dataset(name: str, ratios: list[float | None], loader) -> None:
    """Process a dataset and save at multiple imbalance ratios."""
    print(f"\n=== {name} ===")

    # Load raw data
    raw_path = PATHS[name]["raw"]
    df_raw = loader(raw_path)

    # Get preprocessor (disable built-in downsampling for ppdai)
    preprocess_fn = PREPROCESSORS[name]
    if name == "ppdai":
        df_processed = preprocess_fn(df_raw, downsample=False)
    else:
        df_processed = preprocess_fn(df_raw)

    print(f"Loaded: {len(df_processed)} rows")
    print(f"Original ratio: {df_processed[LABEL_COLS[name]].value_counts(normalize=True).to_dict()}")

    # Save each ratio version
    for ratio in ratios:
        if ratio is None:
            df_out = df_processed
        else:
            df_out = downsample_majority(
                df_processed,
                label_col=LABEL_COLS[name],
                target_ratio=ratio,
                random_state=42,
            )

        out_path = get_processed_path(name, ratio)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"Saved {out_path.name}: {len(df_out)} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Generate downsampled dataset versions at various imbalance ratios"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Datasets to process (default: all configured)",
    )
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DATASETS_TO_PROCESS.keys()

    for name in datasets:
        if name not in DATASETS_TO_PROCESS:
            print(f"Unknown or unconfigured dataset: {name}")
            continue
        config = DATASETS_TO_PROCESS[name]
        process_dataset(name, config["ratios"], config["loader"])


if __name__ == "__main__":
    main()
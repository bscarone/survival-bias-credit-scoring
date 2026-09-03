#!/usr/bin/env python
"""
Preprocess datasets for credit scoring experiments.

Usage:
    python run_preprocess.py              # Process all available datasets
    python run_preprocess.py default      # Process specific dataset
    python run_preprocess.py --list       # Show available datasets
"""
import argparse
import sys

from config import PATHS, LABEL_COLS, get_processed_path
from data_loaders import LOADERS
from data_preprocess import PREPROCESSORS


def process_dataset(name: str) -> bool:
    """
    Load, preprocess, and save a single dataset.
    
    Returns True if successful, False if skipped/failed.
    """
    print(f"Processing {name}...")
    
    # Load raw data
    try:
        df = LOADERS[name]()
    except FileNotFoundError as e:
        print(f"  Skipped: {e}\n")
        return False
    except Exception as e:
        print(f"  Error loading: {e}\n", file=sys.stderr)
        return False
    
    # Preprocess
    try:
        df = PREPROCESSORS[name](df)
    except Exception as e:
        print(f"  Error preprocessing: {e}\n", file=sys.stderr)
        return False
    
    # Save
    out_path = get_processed_path(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    # Report
    label_col = LABEL_COLS[name]
    print(f"  Shape: {df.shape}")
    if label_col in df.columns:
        dist = df[label_col].value_counts().sort_index()
        print("  Label distribution:")
        for val, count in dist.items():
            pct = 100 * count / len(df)
            print(f"    {val}: {count:,} ({pct:.1f}%)")
    print(f"  Saved to: {out_path}\n")
    return True


def list_datasets():
    """Print available datasets."""
    print("Available datasets:\n")
    for name in PREPROCESSORS.keys():
        raw_path = PATHS[name]["raw"]
        status = "✓" if raw_path.exists() else "✗ (not downloaded)"
        print(f"  {name}: {status}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess credit scoring datasets")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Datasets to process (default: all available)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets and exit",
    )
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    # Determine which datasets to process
    if args.datasets:
        names = args.datasets
        for name in names:
            if name not in PREPROCESSORS:
                print(f"Unknown dataset: {name}", file=sys.stderr)
                print(f"Available: {', '.join(PREPROCESSORS.keys())}", file=sys.stderr)
                sys.exit(1)
    else:
        # Process all datasets that have raw data available
        names = [n for n in PREPROCESSORS.keys() if PATHS[n]["raw"].exists()]
        if not names:
            print("No datasets found. Run download_data.py first.")
            sys.exit(1)
    
    # Process each dataset
    success = 0
    for name in names:
        if process_dataset(name):
            success += 1
    
    print(f"Done. Processed {success}/{len(names)} datasets.")


if __name__ == "__main__":
    main()
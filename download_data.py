#!/usr/bin/env python
"""
Download datasets for survival bias credit scoring experiments.

Usage:
    python download_data.py              # Download all
    python download_data.py default      # Download specific dataset
    python download_data.py --list       # List available datasets

Note: Some datasets (lendingclub, home_credit, gmsc) require manual download from Kaggle. See data/README.md for instructions.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


# === Dataset Registry ===

DATASETS = {
    # --- Automatic downloads ---
    "default": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        "filename": "default_raw.xls",
        "description": "UCI Default of Credit Card Clients (Taiwan, 30K rows)",
        "source": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "auto": True,
    },
    # --- Manual downloads (Figshare) ---
    "ppdai": {
        "filename": "ppdaiData.csv",
        "description": "PPDai P2P lending (China, 56K rows)",
        "source": "https://figshare.com/articles/dataset/DefaultData_and_PPDaiData/13573871",
        "auto": False,
    },
    # --- Manual downloads (Kaggle) ---
    "lendingclub": {
        "filename": "lending_club_clean.feather",
        "description": "LendingClub loans pre-cleaned (USA, ~2.9M rows)",
        "source": "https://www.kaggle.com/datasets/marcusos/lending-club-clean",
        "auto": False,
    },
}


# === Download Functions ===

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True
    
    print(f"  Downloading {desc}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to: {dest}")
        return True
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return False


def download_dataset(name: str) -> bool:
    """Download a single dataset by name."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}", file=sys.stderr)
        return False
    
    info = DATASETS[name]
    
    if not info.get("auto", False):
        print(f"  Manual download required: {info['source']}")
        return False
    
    dest_dir = DATA_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = dest_dir / info["filename"]
    return download_file(info["url"], dest_path, info["description"])


def download_all(auto_only: bool = True):
    """Download all available datasets."""
    print("Downloading datasets...\n")
    
    for name, info in DATASETS.items():
        print(f"{name}:")
        if auto_only and not info.get("auto", False):
            print(f"  Skipped (manual download): {info['source']}\n")
            continue
        download_dataset(name)
        print()
    
    manual = [n for n, i in DATASETS.items() if not i.get("auto", False)]
    if manual:
        print(f"Manual downloads needed: {', '.join(manual)}")
        print("See data/README.md for instructions.")


def list_datasets():
    """Print available datasets."""
    print("Available datasets:\n")
    print("  Auto-download:")
    for name, info in DATASETS.items():
        if info.get("auto", False):
            print(f"    {name}: {info['description']}")
    
    print("\n  Manual download (Kaggle):")
    for name, info in DATASETS.items():
        if not info.get("auto", False):
            print(f"    {name}: {info['description']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download credit scoring datasets"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Datasets to download (default: all auto-downloadable)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets",
    )
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if not args.datasets:
        download_all(auto_only=True)
    else:
        for name in args.datasets:
            if name not in DATASETS:
                print(f"Unknown: {name}. Use --list to see options.")
                sys.exit(1)
            print(f"{name}:")
            download_dataset(name)
            print()


if __name__ == "__main__":
    main()
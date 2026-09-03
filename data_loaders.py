"""
Dataset loading functions for credit scoring experiments.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from config import PATHS, GERMAN_COLUMNS, get_processed_path


def load_default(path: Path | None = None) -> pd.DataFrame:
    """
    Load UCI Default of Credit Card Clients dataset.
    
    Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
    30,000 instances, Taiwan credit card data.
    
    Note: Requires xlrd for .xls files (pip install xlrd).
    """
    path = path or PATHS["default"]["raw"]
    df = pd.read_excel(path, header=1)  # Skip duplicate header row
    return df


def load_ppdai(path: Path | None = None) -> pd.DataFrame:
    """
    Load PPDai credit scoring dataset.
    
    Source: https://figshare.com/articles/dataset/DefaultData_and_PPDaiData/13573871
    55,596 instances from Chinese P2P lending platform.
    """
    path = path or PATHS["ppdai"]["raw"]
    df = pd.read_csv(path)
    return df


def load_lendingclub(path: Path | None = None) -> pd.DataFrame:
    """
    Load LendingClub loan dataset (pre-cleaned version).
    
    Source: https://www.kaggle.com/datasets/marcusos/lending-club-clean
    Requires manual download from Kaggle.
    """
    path = path or PATHS["lendingclub"]["raw"]
    if not path.exists():
        raise FileNotFoundError(
            f"LendingClub data not found at {path}. "
            "See data/README.md for download instructions."
        )
    df = pd.read_feather(path)
    return df


# Registry for convenience
LOADERS = {
    "default": load_default,
    "ppdai": load_ppdai,
    "lendingclub": load_lendingclub,
}


def load_raw(dataset: str) -> pd.DataFrame:
    """Load a raw dataset by name."""
    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(LOADERS.keys())}")
    return LOADERS[dataset]()


def load_processed(dataset: str, imbalance_ratio: float | None = None) -> pd.DataFrame:
    """
    Load a preprocessed dataset by name.
    
    Args:
        dataset: Dataset name
        imbalance_ratio: Majority class ratio (e.g., 0.77, 0.50).
                         None for original (no downsampling).
    """
    if dataset not in PATHS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(PATHS.keys())}")
    
    path = get_processed_path(dataset, imbalance_ratio)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run generate_downsampled.py first."
        )
    return pd.read_csv(path)
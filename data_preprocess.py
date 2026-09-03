"""
Preprocessing utilities for credit scoring datasets.
"""
import numpy as np 
import pandas as pd
from config import LC_DEFAULT_STATUSES, LC_DROP_COLS, LABEL_COLS, DEMOGRAPHIC_RENAME


# === General Utilities ===

def filter_by_col(df: pd.DataFrame, col: str, regex: str) -> pd.DataFrame:
    """Filter dataframe rows where column matches regex pattern."""
    mask = df[col].astype(str).str.contains(regex, regex=True)
    return df.loc[mask]  # type: ignore[return-value]


def downsample_majority(
    df: pd.DataFrame,
    label_col: str,
    target_ratio: float = 0.77,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Downsample majority class to achieve target imbalance ratio.
    
    Args:
        df: Input dataframe
        label_col: Name of binary label column
        target_ratio: Desired ratio of majority class
        random_state: Random seed for reproducibility
    """
    counts = df[label_col].value_counts()
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()
    
    n_minority = counts[minority_label]
    n_majority_new = int(target_ratio * n_minority / (1 - target_ratio))
    
    df_minority = df.loc[df[label_col] == minority_label]  # type: ignore[assignment]
    df_majority = df.loc[df[label_col] == majority_label].sample(  # type: ignore[union-attr]
        n=n_majority_new, random_state=random_state
    )
    
    return pd.concat([df_minority, df_majority], ignore_index=True) # type: ignore


def imbalance_ratio(df: pd.DataFrame, label_col: str) -> float:
    """
    Measure the imbalance ratio of a binary label as the proportion of the majority class.

    Args:
        df: Input dataframe
        label_col: Name of binary label column

    Returns:
        Ratio of majority class count to total count, in [0.5, 1.0).
    """
    counts = df[label_col].value_counts()
    return counts.iloc[0] / counts.sum()


# === Demographic Standardization ===  

def standardize_demographics(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Rename raw demographic columns to standardized names defined in
    DEMOGRAPHIC_RENAME (config.py).

    This enables consistent downstream analysis across datasets using
    column names like 'sex', 'age', 'education', 'occupation', 'income'.

    Args:
        df: DataFrame with raw or preprocessed column names.
        dataset: One of 'default', 'ppdai', 'home_credit'.

    Returns:
        DataFrame with renamed demographic columns. All other columns
        are preserved unchanged.
    """
    rename_map = DEMOGRAPHIC_RENAME.get(dataset, {})
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing)


# === Dataset-specific preprocessing ===

def preprocess_default(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess UCI Default of Credit Card Clients dataset.
    
    - Standardizes label column name
    - Drops ID column
    """
    df = df.copy()
    
    # Standardize label column name (varies by source)
    label_variants = ["default payment next month", "default.payment.next.month", "Y"]
    for variant in label_variants:
        if variant in df.columns:
            df = df.rename(columns={variant: LABEL_COLS["default"]})
            break
    
    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    
    return df


def preprocess_ppdai(df: pd.DataFrame, downsample: bool = True) -> pd.DataFrame:
    """
    Preprocess PPDai dataset.
    
    - Optionally downsamples to match typical imbalance ratio
    - Standardizes label column name
    """
    df = df.copy()
    
    # Rename label column
    if "label" in df.columns:
        df = df.rename(columns={"label": LABEL_COLS["ppdai"]})
    
    if downsample:
        df = downsample_majority(df, label_col=LABEL_COLS["ppdai"])
    
    return df


def preprocess_lendingclub(
    df: pd.DataFrame,
    year_filter: str = "2017",
    drop_top_null_cols: int = 10,
    null_threshold: int = 314212,
    drop_na: bool = False,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """
    Preprocess LendingClub dataset.
    
    - Filters by issue year
    - Drops top N columns with highest null counts (above threshold)
    - Optionally drops data leakage columns (post-hoc info)
    - Creates binary default label from loan_status
    - Optionally drops rows with remaining missing values
    """
    df = df.copy()
    
    if year_filter and "issue_d" in df.columns:
        if df["issue_d"].dtype == "object":
            mask = df["issue_d"].str.contains(year_filter, na=False)
        else:
            mask = df["issue_d"].astype(str).str.contains(year_filter)
        df = df.loc[mask]  # type: ignore[assignment]
    
    null_counts = df.isna().sum().sort_values(ascending=False)
    high_null_cols = null_counts[null_counts > null_threshold].head(drop_top_null_cols).index.tolist()
    df = df.drop(columns=high_null_cols)
    
    df[LABEL_COLS["lendingclub"]] = df["loan_status"].isin(LC_DEFAULT_STATUSES).astype(int)
    
    if drop_leakage:
        leakage_cols = [c for c in LC_DROP_COLS if c in df.columns]
        df = df.drop(columns=leakage_cols)
    
    if drop_na:
        df = df.dropna()
    
    return df


# Registry for convenience
PREPROCESSORS = {
    "default": preprocess_default,
    "ppdai": preprocess_ppdai,
    "lendingclub": preprocess_lendingclub,
}
"""
Configuration for dataset paths and column mappings.
"""
from __future__ import annotations
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# Dataset-specific paths
PATHS = {
    "default": {
        "raw": DATA_DIR / "default" / "default_raw.xls",
        "processed_dir": DATA_DIR / "default",
    },
    "german": {
        "raw": DATA_DIR / "german" / "german_raw.data",
        "processed_dir": DATA_DIR / "german",
    },
    "australian": {
        "raw": DATA_DIR / "australian" / "australian_raw.dat",
        "processed_dir": DATA_DIR / "australian",
    },
    "ppdai": {
        "raw": DATA_DIR / "ppdai" / "ppdaiData.csv",
        "processed_dir": DATA_DIR / "ppdai",
    },
    "lendingclub": {
        "raw": DATA_DIR / "lendingclub" / "lending_club_clean.feather",
        "processed_dir": DATA_DIR / "lendingclub",
    },
    "gmsc": {
        "raw": DATA_DIR / "gmsc" / "cs-training.csv",
        "processed_dir": DATA_DIR / "gmsc",
    },
    "home_credit": {
        "raw": DATA_DIR / "home_credit" / "application_train.csv",
        "processed_dir": DATA_DIR / "home_credit",
    },
}


def get_processed_path(dataset: str, imbalance_ratio: float | None = None) -> Path:
    """
    Get path for processed dataset file.

    Args:
        dataset: Dataset name
        imbalance_ratio: Majority class ratio (e.g., 0.77, 0.50).
                         None for original (no downsampling).
    """
    base_dir = PATHS[dataset]["processed_dir"]

    if imbalance_ratio is None:
        return base_dir / f"{dataset}_processed.csv"

    ratio_str = f"{imbalance_ratio * 100:02.0f}"
    return base_dir / f"{dataset}_processed_imb{ratio_str}.csv"


# Standardized label column names (after preprocessing)
LABEL_COLS = {
    "default": "default",
    "german": "default",
    "australian": "default",
    "ppdai": "default",
    "lendingclub": "default",
    "gmsc": "default",
    "home_credit": "default",
}


# Mapping from raw column names to standardized demographic names, per dataset.
# Used by standardize_demographics() in data_preprocess.py.
DEMOGRAPHIC_RENAME = {
    "default": {
        "SEX": "sex",
        "AGE": "age",
        "EDUCATION": "education",
        "MARRIAGE": "marriage",
    },
    "ppdai": {
        # ppdai_processed.csv already uses lowercase names
        "sex": "sex",
        "occupation": "occupation",
        "education": "education",
        "marriage": "marriage",
        "household": "household",
        "income": "income",
    },
    "home_credit": {
        "CODE_GENDER": "sex",       # decoded M/F -> male/female in preprocessor
        "DAYS_BIRTH": "age",        # converted to years (positive) in preprocessor
        "OCCUPATION_TYPE": "occupation",
        "AMT_INCOME_TOTAL": "income",
    },
}


# LendingClub columns to drop (data leakage - post-hoc outcome info)
LC_DROP_COLS = [
    # Identifiers
    "id", "url",
    # Post-origination funding
    "funded_amnt", "funded_amnt_inv",
    # Payment outcomes
    "total_pymnt", "total_pymnt_inv",
    # Recovery
    "total_rec_int", "total_rec_late_fee", "total_rec_prncp",
    # Post-default
    "recoveries", "collection_recovery_fee",
    # Outstanding principal
    "out_prncp", "out_prncp_inv",
    # Last payment info
    "last_pymnt_d", "last_pymnt_amnt",
    # Post-hoc credit pulls
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    # Future payment
    "next_pymnt_d",
    # Target variable (derived from this)
    "loan_status",
    # Current delinquency status
    "acc_now_delinq", "delinq_amnt",
    "num_tl_30dpd", "num_tl_120dpd_2m",
    # Post-hoc chargeoffs
    "chargeoff_within_12_mths", "sec_app_chargeoff_within_12_mths",
    # Hardship program (post-hoc)
    "hardship_flag", "deferral_term", "hardship_amount",
    "hardship_length", "hardship_dpd",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "orig_projected_additional_accrued_interest",
    # Debt settlement (post-hoc)
    "debt_settlement_flag",
    "pymnt_plan"
]

# LendingClub loan statuses considered as default
LC_DEFAULT_STATUSES = [
    "Default",
    "Charged Off",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Does not meet the credit policy. Status:Charged Off",
]

# German Credit column names (space-separated file has no header)
GERMAN_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
    "own_telephone", "foreign_worker", "class",
]


# n0 values per dataset-imbalance 
# Key format: "{dataset}" for original, "{dataset}_imb{ratio}" for downsampled
# Value can be int (same for all models) or dict (per-model)
N0_VALUES = {
    "default": 5000,
    "ppdai_imb77": 5000,
    "lendingclub_imb77": 60000,
}
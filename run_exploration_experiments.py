"""
Run exploration-rate sweep experiments.

For each exploration rate r in [0, 1], runs the temporal evolution loop and
records standard classification metrics (accuracy, precision, recall), the
two lending-specific quantities of Section 3 (Kickout and the training set
default rate), and exploration-specific diagnostics (cost, default rate among
explored rejects).

r=0.0 is equivalent to Biased/Filtered.
r=1.0 is equivalent to Oracle (all rejects approved with true labels).

Usage:
    python run_exploration_experiments.py --datasets default ppdai --models rf gbdt
    python run_exploration_experiments.py --datasets lendingclub --granularity fine
    python run_exploration_experiments.py --datasets default --rankings least_risky uncertainty random
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import models
from metrics import compute_kickout_from_cm, get_confusion_matrix
from data_loaders import load_processed
from config import LABEL_COLS, N0_VALUES
from models import get_model_name
from exploration_strategy import (
    ExplorationStrategy,
    ExplorationInfo,
    RankingStrategy,
    get_exploration_rates,
)


# =============================================================================
# Data Preparation (mirrors run_temporal_experiments.py)
# =============================================================================

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def prepare_data(
    dataset: str, imbalance_ratio: float | None = None,
) -> tuple[pd.DataFrame, str]:
    df = load_processed(dataset, imbalance_ratio)
    label_col = LABEL_COLS[dataset]
    df = df.dropna()
    df = encode_categoricals(df)

    features = df.columns.drop(label_col)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])  # type: ignore

    return df, label_col


def compute_n0(
    n: int,
    dataset: str,
    model_id: str,
    imbalance_ratio: float | None = None,
    n0_fraction: float = 0.2,
    n0_min: int = 100,
) -> tuple[int, str]:
    key = dataset if imbalance_ratio is None else f"{dataset}_imb{imbalance_ratio * 100:02.0f}"

    if N0_VALUES is not None and key in N0_VALUES:
        value = N0_VALUES[key]
        if isinstance(value, dict):
            if model_id in value:
                return value[model_id], f"config[{key}][{model_id}]"
        else:
            return value, f"config[{key}]"

    n0 = max(n0_min, int(n * n0_fraction))
    return n0, f"fraction ({n0_fraction})"


def get_imbalance_suffix(imbalance_ratio: float | None) -> str:
    if imbalance_ratio is None:
        return ""
    return f"_imb{imbalance_ratio * 100:02.0f}"


# =============================================================================
# Single Experiment
# =============================================================================

def run_single_exploration_experiment(
    data: pd.DataFrame,
    model_id: str,
    label_col: str,
    exploration_rate: float,
    ranking: RankingStrategy = RankingStrategy.LEAST_RISKY,
    T: int = 10,
    n0: int = 1000,
    tau: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run one temporal evolution experiment with a given exploration rate and ranking.

    Returns a DataFrame with one row per iteration (0..T), containing:
    - Standard classification metrics (accuracy, precision, recall), computed on
      a held-out split of the accumulated training set
    - Kickout, the rejection quality metric of Section 3 (equal to
      2 * precision - 1 on that split)
    - Training set statistics: size, and training_default_rate (the TDR)
    - Exploration diagnostics: n_explored, n_explored_defaulted,
      exploration_cost, reject_default_rate (over all rejects at that cycle),
      and exploration_default_rate (over the rejects approved for exploration)
    """
    n = len(data)
    n_per_iter = (n - n0) // T

    # Initialize: sample C[0]
    C = data.sample(n=n0, replace=False, random_state=random_state)
    remaining = data.drop(C.index)

    strategy = ExplorationStrategy(
        exploration_rate=exploration_rate,
        ranking=ranking,
        random_state=random_state,
    )
    current_model = None
    records = []

    for i in range(T + 1):
        if i > 0:
            batch_size = min(n_per_iter, len(remaining))
            if batch_size == 0:
                break
            new_sample = remaining.sample(
                n=batch_size, replace=False, random_state=random_state + i,
            )
            remaining = remaining.drop(new_sample.index)

            result = strategy.apply(C, new_sample, current_model, label_col, tau)
            C = result.augmented_data
            info = result.exploration_info
        else:
            info = ExplorationInfo()

        # Train and evaluate
        X = C.drop(columns=[label_col])
        y = C[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )

        current_model = models.train_model(model_id, X_train, y_train)
        y_pred = models.predict_with_threshold(current_model, X_test, tau)
        eval_metrics = models.evaluate_predictions(y_test, y_pred)
        cm = get_confusion_matrix(y_test, y_pred)
        kickout = compute_kickout_from_cm(cm)

        train_default_rate = float(C[label_col].mean())

        record = {
            "iteration": i,
            "exploration_rate": exploration_rate,
            "ranking_strategy": ranking.value,
            "accuracy": eval_metrics.accuracy,
            "precision": eval_metrics.precision,
            "recall": eval_metrics.recall,
            "kickout": kickout.kickout if hasattr(kickout, "kickout") else kickout,
            "training_set_size": len(C),
            "training_default_rate": train_default_rate,
            # Exploration diagnostics
            "n_rejects_total": info.n_rejects_total,
            "n_explored": info.n_explored,
            "n_explored_defaulted": info.n_explored_defaulted,
            "n_explored_repaid": info.n_explored_repaid,
            "exploration_default_rate": info.exploration_default_rate,
            "reject_default_rate": info.reject_default_rate,
            "exploration_cost": info.exploration_cost,
            "explored_risk_mean": info.explored_risk_mean,
        }
        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# Sweep
# =============================================================================

def run_exploration_sweep(
    data: pd.DataFrame,
    dataset_name: str,
    model_id: str,
    label_col: str,
    exploration_rates: list[float],
    rankings: list[RankingStrategy] | None = None,
    n_runs: int = 5,
    T: int = 10,
    n0: int = 1000,
    tau: float = 0.5,
    random_state: int = 42,
    output_dir: Path = Path("experiments/exploration/results"),
) -> pd.DataFrame:
    """Sweep over exploration rates and ranking strategies, with multiple runs each."""
    if rankings is None:
        rankings = [RankingStrategy.LEAST_RISKY]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    total = len(rankings) * len(exploration_rates) * n_runs
    count = 0

    for ranking in rankings:
        for r in exploration_rates:
            for run in range(n_runs):
                count += 1
                print(
                    f"  [{count}/{total}] ranking={ranking.value} "
                    f"r={r:.2f} run={run+1}/{n_runs}",
                    end="",
                )

                run_start = time.time()
                df = run_single_exploration_experiment(
                    data=data,
                    model_id=model_id,
                    label_col=label_col,
                    exploration_rate=r,
                    ranking=ranking,
                    T=T,
                    n0=n0,
                    tau=tau,
                    random_state=random_state + run,
                )
                elapsed = time.time() - run_start
                print(f" ({elapsed:.1f}s)")

                df["run"] = run
                df["dataset"] = dataset_name
                df["model"] = model_id
                all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)

    out_path = output_dir / f"exploration_{dataset_name}_{model_id}.csv"
    combined.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return combined


# =============================================================================
# Summary & Pareto
# =============================================================================

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics across runs for each (ranking, rate, iteration)."""
    df = df.copy()
    df["cumulative_cost"] = df.groupby(
        ["dataset", "model", "ranking_strategy", "exploration_rate", "run"]
    )["exploration_cost"].cumsum()

    group_cols = ["dataset", "model", "ranking_strategy", "exploration_rate", "iteration"]
    numeric_cols = [
        "accuracy", "precision", "recall", "kickout",
        "training_set_size", "training_default_rate",
        "n_explored", "n_explored_defaulted", "exploration_cost",
        "cumulative_cost", "exploration_default_rate",
        "reject_default_rate", "explored_risk_mean",
    ]

    agg_dict = {}
    for col in numeric_cols:
        if col in df.columns:
            agg_dict[col] = ["mean", "std"]

    summary = df.groupby(group_cols).agg(agg_dict)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    return summary


def compute_pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    """Extract final-iteration accuracy, Kickout, TDR and cumulative cost per
    (ranking strategy, exploration rate). All combinations are retained; no
    dominance filtering is applied."""
    # [TODO] Rename function to extract_final_iteration or similar, since it doesn't actually compute a Pareto frontier.
    final = summary[summary["iteration"] == summary["iteration"].max()].copy()

    rows = []
    for _, row in final.iterrows():
        rows.append({
            "ranking_strategy": row["ranking_strategy"],
            "exploration_rate": row["exploration_rate"],
            "final_accuracy": row.get("accuracy_mean", 0),
            "final_kickout": row.get("kickout_mean", 0),
            "cumulative_cost": row.get("cumulative_cost_mean", 0),
            "final_training_default_rate": row.get("training_default_rate_mean", 0),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run exploration-rate sweep experiments."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["default"],
    )
    parser.add_argument(
        "--models", nargs="+", default=["rf"],
        choices=["lr", "rf", "gbdt"],
    )
    parser.add_argument(
        "--rankings", nargs="+", default=["least_risky"],
        choices=[r.value for r in RankingStrategy],
        help="Ranking strategies to compare",
    )
    parser.add_argument(
        "--granularity", default="standard",
        choices=["coarse", "standard", "fine"],
    )
    parser.add_argument("--imbalance-ratio", type=float, default=None,
                        help="Down-sample the majority class to this non-default fraction "
                        "(paper uses 0.77 for ppdai and lendingclub; omit for default).",)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("experiments/exploration/results"),
    )
    args = parser.parse_args()

    rates = get_exploration_rates(args.granularity)
    rankings = [RankingStrategy(r) for r in args.rankings]

    print("=" * 70)
    print("EXPLORATION-RATE SWEEP EXPERIMENTS")
    print(f"Datasets:   {args.datasets}")
    print(f"Models:     {args.models}")
    print(f"Rankings:   {[r.value for r in rankings]}")
    print(f"Rates:      {rates}")
    print(f"T={args.T}, tau={args.tau}, n_runs={args.n_runs}")
    print("=" * 70)

    for dataset_name in args.datasets:
        suffix = get_imbalance_suffix(args.imbalance_ratio)
        output_name = f"{dataset_name}{suffix}"

        print(f"\n{'='*60}")
        print(f"Dataset: {output_name}")
        print(f"{'='*60}")

        try:
            df, label_col = prepare_data(dataset_name, args.imbalance_ratio)
            print(f"  Shape: {df.shape}, Label: {label_col}")
        except FileNotFoundError as e:
            print(f"  Skipping {dataset_name}: {e}")
            continue
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            continue

        for model_id in args.models:
            n = len(df)
            n0, n0_source = compute_n0(
                n, dataset_name, model_id, args.imbalance_ratio,
            )
            print(f"\n--- {get_model_name(model_id)} ({model_id}) ---")
            print(f"  n={n}, n0={n0} ({n0_source})")

            results = run_exploration_sweep(
                data=df,
                dataset_name=output_name,
                model_id=model_id,
                label_col=label_col,
                exploration_rates=rates,
                rankings=rankings,
                n_runs=args.n_runs,
                T=args.T,
                n0=n0,
                tau=args.tau,
                random_state=args.random_state,
                output_dir=args.results_dir,
            )

            summary = compute_summary(results)
            summary.to_csv(
                args.results_dir / f"summary_{output_name}_{model_id}.csv",
                index=False,
            )

            pareto = compute_pareto_frontier(summary)
            pareto.to_csv(
                args.results_dir / f"pareto_{output_name}_{model_id}.csv",
                index=False,
            )
            print(f"\n  Pareto frontier ({output_name}, {model_id}):")
            print(pareto.to_string(index=False))


if __name__ == "__main__":
    main()
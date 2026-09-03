"""
Run temporal evolution experiments across multiple datasets and models.

Supports both:
- Basic comparison: Biased vs Oracle (original)
- Full comparison: All reject inference strategies

Usage:
    python run_temporal_experiments.py
    python run_temporal_experiments.py --datasets default --models rf --n-runs 5
    python run_temporal_experiments.py --datasets default --models rf --full-ri
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from data_loaders import load_processed
from config import LABEL_COLS, N0_VALUES
from models import get_model_name
from sklearn.preprocessing import StandardScaler
from temporal_experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_comparison_experiment,
    save_comparison_results,
)
from temporal_experiment_ri import (
    run_ri_comparison_experiment,
)
from reject_inference import (
    StrategyID,
    OracleStrategy,
    FilteredStrategy,
    ParcelingStrategy,
    FuzzyAugmentationStrategy,
    SimpleExtrapolationStrategy,
    ShallowSelfLearningStrategy,
    TwinsStrategy,
)


STRATEGY_LABELS: dict[StrategyID, str] = {
    StrategyID.FILTERED: "Biased",
    StrategyID.ORACLE: "Oracle",
    StrategyID.PARCELING: "Parceling",
    StrategyID.FUZZY_AUGMENTATION: "Fuzzy Aug.",
    StrategyID.SIMPLE_EXTRAPOLATION: "Extrapolation",
    StrategyID.SHALLOW_SELF_LEARNING: "Shallow SL",
    StrategyID.TWINS: "Twins",
}


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class TemporalExperimentSpec:
    """Specification for temporal evolution experiments."""
    datasets: list[str]
    model_ids: list[str]
    imbalance_ratio: float | None = None
    T: int = 10
    tau: float = 0.5
    test_size: float = 0.2
    random_state: int = 42
    n_runs: int = 1
    positive_is_good: bool = False
    
    # RI-specific settings
    full_ri: bool = False  # If True, run all RI strategies
    parceling_n_bins: int = 10
    parceling_min_bin_size: int = 30
    
    # n0 configuration
    n0_values: dict[str, dict[str, int] | int] | None = None
    n0_fraction: float = 0.2
    n0_min: int = 100
    
    # Output
    results_dir: Path = Path("experiments/temporal_evolution/results")
    figures_dir: Path = Path("experiments/temporal_evolution/figures")
    save_figures: bool = True
    show_figures: bool = False


DEFAULT_SPEC = TemporalExperimentSpec(
    datasets=["default", "ppdai", "gmsc", "home_credit"],
    model_ids=["rf", "gbdt"],
    n0_values=N0_VALUES,  # type: ignore
    T=10,
    tau=0.5,
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_n0_key(dataset: str, imbalance_ratio: float | None) -> str:
    if imbalance_ratio is None:
        return dataset
    return f"{dataset}_imb{imbalance_ratio * 100:02.0f}"


def get_imbalance_suffix(imbalance_ratio: float | None) -> str:
    if imbalance_ratio is None:
        return ""
    return f"_imb{imbalance_ratio * 100:02.0f}"


def compute_n0(n: int, dataset: str, model_id: str, spec: TemporalExperimentSpec)\
    -> tuple[int, str]:
    key = get_n0_key(dataset, spec.imbalance_ratio)
    
    if spec.n0_values is not None and key in spec.n0_values:
        value = spec.n0_values[key]
        if isinstance(value, dict):
            if model_id in value:
                return value[model_id], f"config[{key}][{model_id}]"
            else:
                print(f"  Warning: {model_id} not in N0_VALUES[{key}], using fraction")
        else:
            return value, f"config[{key}]"
    
    n0 = max(spec.n0_min, int(n * spec.n0_fraction))
    return n0, f"fraction ({spec.n0_fraction})"


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def prepare_data(dataset: str, spec: TemporalExperimentSpec) -> tuple[pd.DataFrame, str]:
    df = load_processed(dataset, spec.imbalance_ratio)
    label_col = LABEL_COLS[dataset]
    df = df.dropna()
    df = encode_categoricals(df)

    # Scale features (helps LR/SVM/MLP convergence, no effect on trees)
    features = df.columns.drop(label_col)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features]) # type: ignore

    return df, label_col


def generate_base_filename(dataset_name: str, config: ExperimentConfig) -> str:
    return f"{dataset_name}_{config.model_id}_T{config.T}_n0{config.n0}_tau{config.tau:.2f}"


# =============================================================================
# Aggregated Results Saving
# =============================================================================

def save_aggregated_results(
    filtered_runs: list[ExperimentResult],
    oracle_runs: list[ExperimentResult],
    dataset_name: str,
    output_dir: Path,
    positive_is_good: bool = False,
) -> Path:
    """Save aggregated biased vs oracle results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_runs = len(filtered_runs)
    config = filtered_runs[0].config
    T = config.T
    
    # Collect all metrics
    data_sources = {
        "filtered": filtered_runs,
        "oracle": oracle_runs,
    }
    
    rows = []
    for t in range(T + 1):
        row = {"iteration": t, "n_runs": n_runs}
        for name, runs in data_sources.items():
            accs = np.array([r.accuracy_history[t] for r in runs])
            precs = np.array([r.precision_history[t] for r in runs])
            recs = np.array([r.recall_history[t] for r in runs])
            sizes = np.array([r.training_sizes[t] for r in runs])
            kickouts = np.array([r.kickout_history(positive_is_good)[t] for r in runs])
            default_rates = np.array([r.default_rate_history[t] for r in runs])
            
            row.update({
                f"{name}_accuracy_mean": accs.mean(),
                f"{name}_accuracy_std": accs.std(),
                f"{name}_precision_mean": precs.mean(),
                f"{name}_precision_std": precs.std(),
                f"{name}_recall_mean": recs.mean(),
                f"{name}_recall_std": recs.std(),
                f"{name}_train_size_mean": sizes.mean(),
                f"{name}_train_size_std": sizes.std(),
                f"{name}_kickout_mean": kickouts.mean(),
                f"{name}_kickout_std": kickouts.std(),
                f"{name}_default_rate_mean": default_rates.mean(),
                f"{name}_default_rate_std": default_rates.std(),
            })
        rows.append(row)
    
    df = pd.DataFrame(rows)
    filename = f"{generate_base_filename(dataset_name, config)}_aggregated.csv"
    path = output_dir / filename
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


def save_ri_aggregated_results(
    results: dict[StrategyID, list[ExperimentResult]],
    dataset_name: str,
    output_dir: Path,
    positive_is_good: bool = False,
) -> Path:
    """Save aggregated results for all RI strategies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    first_result = next(iter(results.values()))[0]
    config = first_result.config
    T = config.T
    n_runs = len(next(iter(results.values())))
    
    rows = []
    for t in range(T + 1):
        row = {"iteration": t, "n_runs": n_runs}
        for strategy_id, runs in results.items():
            prefix = strategy_id.name.lower()
            accs = np.array([r.accuracy_history[t] for r in runs])
            precs = np.array([r.precision_history[t] for r in runs])
            recs = np.array([r.recall_history[t] for r in runs])
            sizes = np.array([r.training_sizes[t] for r in runs])
            kickouts = np.array([r.kickout_history(positive_is_good)[t] for r in runs])
            default_rates = np.array([r.default_rate_history[t] for r in runs])
            
            row.update({
                f"{prefix}_accuracy_mean": accs.mean(),
                f"{prefix}_accuracy_std": accs.std(),
                f"{prefix}_precision_mean": precs.mean(),
                f"{prefix}_precision_std": precs.std(),
                f"{prefix}_recall_mean": recs.mean(),
                f"{prefix}_recall_std": recs.std(),
                f"{prefix}_train_size_mean": sizes.mean(),
                f"{prefix}_train_size_std": sizes.std(),
                f"{prefix}_kickout_mean": kickouts.mean(),
                f"{prefix}_kickout_std": kickouts.std(),
                f"{prefix}_default_rate_mean": default_rates.mean(),
                f"{prefix}_default_rate_std": default_rates.std(),
            })
        rows.append(row)
    
    df = pd.DataFrame(rows)
    filename = f"{generate_base_filename(dataset_name, config)}_ri_aggregated.csv"
    path = output_dir / filename
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_single_experiment(
    dataset: str,
    model_id: str,
    df: pd.DataFrame,
    label_col: str,
    spec: TemporalExperimentSpec,
) -> bool:
    """Run experiment for a single dataset/model combination."""
    n = len(df)
    n0, n0_source = compute_n0(n, dataset, model_id, spec)
    
    print(f"  n={n}, n0={n0} ({n0_source}), T={spec.T}, tau={spec.tau}, n_runs={spec.n_runs}")
    if spec.full_ri:
        print("  Running FULL RI comparison (7 strategies)")
    
    suffix = get_imbalance_suffix(spec.imbalance_ratio)
    output_dataset_name = f"{dataset}{suffix}"
    
    if spec.full_ri:
        # Run all RI strategies
        all_results: dict[StrategyID, list[ExperimentResult]] = {
            StrategyID.FILTERED: [],
            StrategyID.ORACLE: [],
            StrategyID.PARCELING: [],
            StrategyID.FUZZY_AUGMENTATION: [],
            StrategyID.SIMPLE_EXTRAPOLATION: [],
            StrategyID.SHALLOW_SELF_LEARNING: [],
            StrategyID.TWINS: [],
        }
        
        for run_idx in range(spec.n_runs):
            run_config = ExperimentConfig(
                model_id=model_id,
                label_col=label_col,
                n0=n0,
                T=spec.T,
                tau=spec.tau,
                test_size=spec.test_size,
                random_state=spec.random_state + run_idx,
            )
            
            run_seed = spec.random_state + run_idx
            strategies = [
                FilteredStrategy(),
                OracleStrategy(),
                ParcelingStrategy(
                    n_bins=spec.parceling_n_bins,
                    min_bin_size=spec.parceling_min_bin_size,
                    random_state=run_seed,
                ),
                FuzzyAugmentationStrategy(use_probabilistic=True, random_state=run_seed),
                SimpleExtrapolationStrategy(),
                ShallowSelfLearningStrategy(random_state=run_seed),
                TwinsStrategy(),
            ]
            
            run_start = time.time()
            run_results = run_ri_comparison_experiment(df, run_config, strategies, progress_callback=None)
            run_elapsed = time.time() - run_start
            
            for strategy_id, result in run_results.items():
                all_results[strategy_id].append(result)
            
            if spec.n_runs > 1:
                biased_acc = run_results[StrategyID.FILTERED].accuracy_history[-1]
                oracle_acc = run_results[StrategyID.ORACLE].accuracy_history[-1]
                print(f"    Run {run_idx + 1}/{spec.n_runs}: Biased={biased_acc:.3f}, Oracle={oracle_acc:.3f} ({run_elapsed:.1f}s)")
        
        # Print summary
        print("  Final accuracy:")
        for strategy_id, runs in all_results.items():
            accs = np.array([r.accuracy_history[-1] for r in runs])
            label = STRATEGY_LABELS.get(strategy_id, strategy_id.name)
            print(f"    {label}: {accs.mean():.3f} ± {accs.std():.3f}")
        
        # Save results
        save_ri_aggregated_results(
            all_results, output_dataset_name, spec.results_dir, spec.positive_is_good
        )
    
    else:
        # Original biased vs oracle only
        filtered_runs: list[ExperimentResult] = []
        oracle_runs: list[ExperimentResult] = []
        
        for run_idx in range(spec.n_runs):
            run_config = ExperimentConfig(
                model_id=model_id,
                label_col=label_col,
                n0=n0,
                T=spec.T,
                tau=spec.tau,
                test_size=spec.test_size,
                random_state=spec.random_state + run_idx,
            )
            
            run_start = time.time()
            filtered, oracle = run_comparison_experiment(df, run_config, progress_callback=None)
            run_elapsed = time.time() - run_start
            filtered_runs.append(filtered)
            oracle_runs.append(oracle)
            
            if spec.n_runs > 1:
                print(f"    Run {run_idx + 1}/{spec.n_runs}: "
                      f"Biased={filtered.accuracy_history[-1]:.3f}, "
                      f"Oracle={oracle.accuracy_history[-1]:.3f} ({run_elapsed:.1f}s)")
        
        # Print summary
        filtered_accs = np.array([r.accuracy_history[-1] for r in filtered_runs])
        oracle_accs = np.array([r.accuracy_history[-1] for r in oracle_runs])
        
        if spec.n_runs > 1:
            print(f"  Final accuracy - Biased: {filtered_accs.mean():.3f} ± {filtered_accs.std():.3f}, "
                  f"Oracle: {oracle_accs.mean():.3f} ± {oracle_accs.std():.3f}")
        else:
            print(f"  Final accuracy - Biased: {filtered_accs[0]:.3f}, Oracle: {oracle_accs[0]:.3f}")
        
        # Save results
        if spec.n_runs == 1:
            save_comparison_results(
                filtered_runs[0], oracle_runs[0],
                output_dataset_name, spec.results_dir, spec.positive_is_good
            )
        else:
            save_aggregated_results(
                filtered_runs, oracle_runs,
                output_dataset_name, spec.results_dir, spec.positive_is_good
            )
    
    return True


def run_all_experiments(spec: TemporalExperimentSpec | None = None) -> dict:
    """Run experiments for all dataset/model combinations."""
    spec = spec or DEFAULT_SPEC
    results = {}
    
    print("=" * 70)
    print("TEMPORAL EVOLUTION EXPERIMENTS")
    print(f"Datasets: {spec.datasets}")
    print(f"Models: {spec.model_ids}")
    print(f"T={spec.T}, tau={spec.tau}, n_runs={spec.n_runs}")
    print(f"Full RI comparison: {spec.full_ri}")
    print("=" * 70)
    
    for dataset in spec.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}{get_imbalance_suffix(spec.imbalance_ratio)}")
        print(f"{'='*60}")
        
        try:
            df, label_col = prepare_data(dataset, spec)
            print(f"Shape: {df.shape}, Label: {label_col}")
        except FileNotFoundError as e:
            print(f"Skipping {dataset}: {e}")
            continue
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
            continue
        
        results[dataset] = {}
        
        for model_id in spec.model_ids:
            print(f"\n--- {get_model_name(model_id)} ({model_id}) ---")
            
            try:
                success = run_single_experiment(dataset, model_id, df, label_col, spec)
                results[dataset][model_id] = success
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                results[dataset][model_id] = False
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for dataset, model_results in results.items():
        successes = sum(model_results.values())
        total = len(model_results)
        print(f"{dataset}: {successes}/{total} successful")
    
    return results


# =============================================================================
# Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run temporal evolution experiments")
    parser.add_argument("--datasets", nargs="+", default=["default"])
    parser.add_argument("--models", nargs="+", default=["rf"])
    parser.add_argument("--imbalance-ratio", type=float, default=None)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--full-ri", action="store_true", help="Run all RI strategies")
    parser.add_argument("--parceling-n-bins", type=int, default=10)
    parser.add_argument("--parceling-min-bin-size", type=int, default=30)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/temporal_evolution/results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("experiments/temporal_evolution/figures"))
    parser.add_argument("--no-save-figures", action="store_true")
    parser.add_argument("--show-figures", action="store_true")
    
    args = parser.parse_args()
    
    spec = TemporalExperimentSpec(
        datasets=args.datasets,
        model_ids=args.models,
        imbalance_ratio=args.imbalance_ratio,
        T=args.T,
        tau=args.tau,
        random_state=args.random_state,
        n_runs=args.n_runs,
        full_ri=args.full_ri,
        parceling_n_bins=args.parceling_n_bins,
        parceling_min_bin_size=args.parceling_min_bin_size,
        n0_values=N0_VALUES,  # type: ignore
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        save_figures=not args.no_save_figures,
        show_figures=args.show_figures,
        positive_is_good=False,
    )
    
    run_all_experiments(spec)


if __name__ == "__main__":
    main()
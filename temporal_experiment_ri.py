"""
Temporal evolution experiments with reject inference strategies.

This module extends temporal_experiment.py to support multiple RI strategies
(Parceling, FuzzyAugmentation, SimpleExtrapolation, etc.) in a single run.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import models
from metrics import (
    evaluate_predictions,
    get_confusion_matrix,
)
from temporal_experiment import (
    ExperimentConfig,
    ExperimentResult,
    IterationResult,
    ProgressCallback,
)
from reject_inference import (
    RejectInferenceStrategy,
    StrategyID,
)


class TemporalEvolutionExperimentRI:
    """
    Temporal evolution experiment with pluggable reject inference strategies.
    
    Unlike the base TemporalEvolutionExperiment which uses is_oracle boolean,
    this version accepts any RejectInferenceStrategy instance.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: ExperimentConfig,
        progress_callback: ProgressCallback | None = None,
    ):
        self.data = data.copy()
        self.config = config
        self.progress = progress_callback or (lambda *_: None)
        
        self._validate_data()
        self._compute_sizes()
    
    def _validate_data(self) -> None:
        if self.config.label_col not in self.data.columns:
            raise ValueError(f"Label column '{self.config.label_col}' not in data")
        if self.data.isna().any().any():
            raise ValueError("Data contains NaN values")
    
    def _compute_sizes(self) -> None:
        n = len(self.data)
        self.n_per_iteration = math.floor((n - self.config.n0) / self.config.T)
        if self.n_per_iteration <= 0:
            raise ValueError(
                f"Not enough data: n={n}, n0={self.config.n0}, T={self.config.T}"
            )
    
    def run(self, strategy: RejectInferenceStrategy) -> ExperimentResult:
        """
        Run temporal evolution with a specific reject inference strategy.
        
        Args:
            strategy: RejectInferenceStrategy instance.
        
        Returns:
            ExperimentResult with metrics for all iterations.
        """
        cfg = self.config
        results: list[IterationResult] = []
        
        # Initialize: sample C[0] uniformly
        C = self.data.sample(n=cfg.n0, replace=False, random_state=cfg.random_state)
        remaining = self.data.drop(C.index)
        sample_weights: np.ndarray | None = None
        
        current_model: models.Classifier | None = None
        
        for i in range(cfg.T + 1):
            self.progress(i, cfg.T)
            
            if i > 0:
                # Sample new data
                new_sample = remaining.sample(
                    n=min(self.n_per_iteration, len(remaining)),
                    replace=False,
                    random_state=cfg.random_state + i,
                )
                remaining = remaining.drop(new_sample.index)
                
                # Apply reject inference strategy
                inference_result = strategy.apply(
                    current_data=C,
                    new_sample=new_sample,
                    model=current_model,
                    label_col=cfg.label_col,
                    tau=cfg.tau,
                )
                C = inference_result.augmented_data
                sample_weights = inference_result.sample_weights
            
            # Split data (and weights if present)
            X = C.drop(columns=[cfg.label_col])
            y = C[cfg.label_col]
            
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(
                indices,
                test_size=cfg.test_size,
                random_state=cfg.random_state,
            )
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            train_weights = sample_weights[train_idx] if sample_weights is not None else None
            
            # Train model
            current_model = models.train_model(
                cfg.model_id, X_train, y_train,
                sample_weight=train_weights,
            )
            
            # Evaluate
            y_pred = models.predict_with_threshold(current_model, X_test, cfg.tau)
            metrics = evaluate_predictions(y_test, y_pred)
            cm = get_confusion_matrix(y_test, y_pred)
            
            # Compute default rate in training set (label=1 means default)
            default_rate = (C[cfg.label_col] == 1).mean()
            
            results.append(IterationResult(
                iteration=i,
                metrics=metrics,
                confusion_matrix=cm,
                training_set_size=len(C),
                default_rate=default_rate,
                model=current_model if cfg.store_models else None,
            ))
        
        return ExperimentResult(iterations=results, config=cfg)


def run_ri_comparison_experiment(
    data: pd.DataFrame,
    config: ExperimentConfig,
    strategies: list[RejectInferenceStrategy],
    progress_callback: ProgressCallback | None = None,
) -> dict[StrategyID, ExperimentResult]:
    """
    Run experiments with multiple reject inference strategies.
    
    Args:
        data: Full dataset.
        config: Experiment configuration.
        strategies: List of RI strategies to compare.
        progress_callback: Optional progress reporter.
    
    Returns:
        Dictionary mapping StrategyID to ExperimentResult.
    """
    experiment = TemporalEvolutionExperimentRI(data, config, progress_callback)
    results = {}
    
    for strategy in strategies:
        if progress_callback:
            print(f"  Running {strategy.id.name}...", end=" ", flush=True)
        
        result = experiment.run(strategy)
        results[strategy.id] = result
        
        if progress_callback:
            print(f"acc={result.accuracy_history[-1]:.3f}")
    
    return results


def save_ri_comparison_results(
    results: dict[StrategyID, ExperimentResult],
    dataset_name: str,
    output_dir: Path | str = "experiments/temporal_evolution/results",
    positive_is_good: bool = False,
) -> Path:
    """Save results from all RI strategies to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    first_result = next(iter(results.values()))
    config = first_result.config
    
    def result_to_dict(result: ExperimentResult) -> dict:
        return {
            "accuracy": result.accuracy_history,
            "precision": result.precision_history,
            "recall": result.recall_history,
            "training_sizes": result.training_sizes,
            "default_rate": result.default_rate_history,
            "confusion_matrices": [cm.tolist() for cm in result.confusion_matrices],
            "kickout": result.kickout_history(positive_is_good),
        }
    
    data = {
        "config": asdict(config),
        "dataset": dataset_name,
        "positive_is_good": positive_is_good,
        "strategies": {
            strategy_id.name: result_to_dict(result)
            for strategy_id, result in results.items()
        },
    }
    
    filename = f"{dataset_name}_{config.model_id}_T{config.T}_n0{config.n0}_tau{config.tau:.2f}_ri.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    return filepath
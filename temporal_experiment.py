"""
Temporal evolution experiments for studying selection bias in ML pipelines.

This module simulates how a classifier evolves over time when trained on data 
that may be filtered by previous model predictions (selection bias) versus having 
access to all data (oracle baseline).

Label convention: 1 = default (bad), 0 = no default (good)
Accept/Reject: Accept clients predicted to repay (y_pred=0), reject those 
predicted to default (y_pred=1)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import models
from metrics import (
    EvalMetrics,
    evaluate_predictions,
    get_confusion_matrix,
    compute_kickout_from_cm,
    KickoutResult,
)


@dataclass
class ExperimentConfig:
    """Configuration for temporal evolution experiment."""
    model_id: str
    label_col: str
    n0: int  # Initial sample size
    T: int   # Number of iterations
    tau: float = 0.5  # Classification threshold
    test_size: float = 0.2
    random_state: int = 42
    store_models: bool = False  # Whether to keep trained models in results
    
    def __post_init__(self):
        if not 0 < self.tau < 1:
            raise ValueError(f"tau must be in (0,1), got {self.tau}")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be in (0,1), got {self.test_size}")


@dataclass
class IterationResult:
    """Results from a single iteration of temporal evolution."""
    iteration: int
    metrics: EvalMetrics
    confusion_matrix: np.ndarray
    training_set_size: int
    default_rate: float = 0.0  # Fraction of defaulters (label=1) in training set
    model: models.Classifier | None = None  # Optionally store the model


@dataclass
class ExperimentResult:
    """Complete results from a temporal evolution experiment."""
    iterations: list[IterationResult]
    config: ExperimentConfig
    
    @property
    def accuracy_history(self) -> list[float]:
        return [r.metrics.accuracy for r in self.iterations]
    
    @property
    def precision_history(self) -> list[float]:
        return [r.metrics.precision for r in self.iterations]
    
    @property
    def recall_history(self) -> list[float]:
        return [r.metrics.recall for r in self.iterations]
    
    @property
    def confusion_matrices(self) -> list[np.ndarray]:
        return [r.confusion_matrix for r in self.iterations]
    
    @property
    def training_sizes(self) -> list[int]:
        return [r.training_set_size for r in self.iterations]
    
    @property
    def default_rate_history(self) -> list[float]:
        """Fraction of defaulters (label=1) in training set over time."""
        return [r.default_rate for r in self.iterations]
    
    def extract_cm_component(self, row: int, col: int) -> list[int]:
        """Extract a specific cell from all confusion matrices (e.g., FP, FN, TP, TN)."""
        return [cm[row, col] for cm in self.confusion_matrices]
    
    @property
    def false_positives(self) -> list[int]:
        return self.extract_cm_component(0, 1)
    
    @property
    def false_negatives(self) -> list[int]:
        return self.extract_cm_component(1, 0)
    
    @property
    def true_positives(self) -> list[int]:
        return self.extract_cm_component(1, 1)
    
    @property
    def true_negatives(self) -> list[int]:
        return self.extract_cm_component(0, 0)
    
    def normalized_cm_component(self, row: int, col: int) -> list[float]:
        """Get confusion matrix component as fraction of total predictions."""
        return [cm[row, col] / cm.sum() for cm in self.confusion_matrices]
    
    # -------------------------------------------------------------------------
    # Kickout metrics
    # -------------------------------------------------------------------------
    
    def kickout_history(self, positive_is_good: bool = False) -> list[float]:
        """
        Extract kickout metric from each iteration's confusion matrix.
        
        Args:
            positive_is_good: Label convention. For credit scoring datasets
                where 1=default, use False (the default).
        """
        return [
            compute_kickout_from_cm(r.confusion_matrix, positive_is_good).kickout 
            for r in self.iterations
        ]
    
    def kickout_results(self, positive_is_good: bool = False) -> list[KickoutResult]:
        """Get full kickout results for each iteration."""
        return [
            compute_kickout_from_cm(r.confusion_matrix, positive_is_good) 
            for r in self.iterations
        ]
    
    def bad_rejection_rate_history(self, positive_is_good: bool = False) -> list[float]:
        """P(rejected | bad) over time - want this high."""
        return [
            compute_kickout_from_cm(r.confusion_matrix, positive_is_good).bad_rejection_rate
            for r in self.iterations
        ]
    
    def good_rejection_rate_history(self, positive_is_good: bool = False) -> list[float]:
        """P(rejected | good) over time - want this low."""
        return [
            compute_kickout_from_cm(r.confusion_matrix, positive_is_good).good_rejection_rate
            for r in self.iterations
        ]


ProgressCallback = Callable[[int, int], None]  # (current_iter, total_iters) -> None


def _result_to_dict(result: ExperimentResult, positive_is_good: bool = False) -> dict:
    """Convert ExperimentResult to JSON-serializable dict."""
    return {
        "accuracy": result.accuracy_history,
        "precision": result.precision_history,
        "recall": result.recall_history,
        "training_sizes": result.training_sizes,
        "default_rate": result.default_rate_history,
        "confusion_matrices": [cm.tolist() for cm in result.confusion_matrices],
        # Kickout metrics
        "kickout": result.kickout_history(positive_is_good),
        "bad_rejection_rate": result.bad_rejection_rate_history(positive_is_good),
        "good_rejection_rate": result.good_rejection_rate_history(positive_is_good),
    }


def _dict_to_result(data: dict, config: ExperimentConfig) -> ExperimentResult:
    """Reconstruct ExperimentResult from dict."""
    iterations = []
    # Handle backward compatibility for old results without default_rate
    default_rates = data.get("default_rate", [0.0] * len(data["accuracy"]))
    
    for i, (acc, prec, rec, size, cm, dr) in enumerate(zip(
        data["accuracy"],
        data["precision"],
        data["recall"],
        data["training_sizes"],
        data["confusion_matrices"],
        default_rates,
    )):
        iterations.append(IterationResult(
            iteration=i,
            metrics=EvalMetrics(accuracy=acc, precision=prec, recall=rec),
            confusion_matrix=np.array(cm),
            training_set_size=size,
            default_rate=dr,
            model=None,
        ))
    return ExperimentResult(iterations=iterations, config=config)


def generate_filename(dataset_name: str, config: ExperimentConfig) -> str:
    """Generate standardized filename (without extension)."""
    return f"{dataset_name}_{config.model_id}_T{config.T}_n0{config.n0}_tau{config.tau:.2f}"


def save_comparison_results(
    filtered: ExperimentResult,
    oracle: ExperimentResult,
    dataset_name: str,
    output_dir: Path | str = "experiments/temporal_evolution/results",
    positive_is_good: bool = False,
) -> Path:
    """
    Save both filtered and oracle results to a single JSON file.
    
    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = generate_filename(dataset_name, filtered.config) + ".json"
    filepath = output_dir / filename
    
    data = {
        "config": asdict(filtered.config),
        "dataset": dataset_name,
        "positive_is_good": positive_is_good,
        "biased": _result_to_dict(filtered, positive_is_good),
        "oracle": _result_to_dict(oracle, positive_is_good),
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    return filepath


def load_comparison_results(
    filepath: Path | str,
) -> tuple[ExperimentResult, ExperimentResult, str]:
    """
    Load comparison results from JSON file.
    
    Returns:
        Tuple of (filtered_results, oracle_results, dataset_name).
    """
    with open(filepath) as f:
        data = json.load(f)
    
    config = ExperimentConfig(**data["config"])
    filtered = _dict_to_result(data["biased"], config)
    oracle = _dict_to_result(data["oracle"], config)
    
    return filtered, oracle, data["dataset"]


def _default_progress(current: int, total: int) -> None:
    """Default progress reporter."""
    end = "\n" if current == total else ", "
    print(current, end=end)


class TemporalEvolutionExperiment:
    """
    Simulates temporal evolution of a classifier with potential selection bias.
    
    In each iteration:
    1. New data arrives from the population
    2. In 'oracle' mode: all new data is added to training set
       In 'filtered' mode: only data predicted to repay (y_pred=0) is added
    3. Model is retrained on updated training set
    4. Model performance is evaluated
    
    Label convention: 1 = default (bad), 0 = no default (good)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: ExperimentConfig,
        progress_callback: ProgressCallback | None = _default_progress,
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
            raise ValueError("Data contains NaN values. Preprocess before running experiment.")
    
    def _compute_sizes(self) -> None:
        n = len(self.data)
        self.n_per_iteration = math.floor((n - self.config.n0) / self.config.T)
        if self.n_per_iteration <= 0:
            raise ValueError(
                f"Not enough data: n={n}, n0={self.config.n0}, T={self.config.T}. "
                f"Need (n - n0) / T > 0."
            )
    
    def run(self, is_oracle: bool) -> ExperimentResult:
        """
        Run the temporal evolution experiment.
        
        At each iteration t ∈ {1,...,T}, a fresh batch S_t is drawn uniformly
        without replacement from applicants in D not yet seen by the model.
        Once drawn, applicants are removed from the remaining pool, so each
        applicant appears at most once across the entire simulation.
        
        Args:
            is_oracle: If True, all applicants in S_t are added to the training
                       set with their true labels (infeasible upper bound in practice).
                       If False, only applicants predicted to repay (y_pred == 0)
                       are added; rejected applicants are discarded (status-quo
                       baseline with accumulated survival bias).

        
        Returns:
            ExperimentResult containing metrics for all iterations.
        """
        cfg = self.config
        results: list[IterationResult] = []
        
        # Initialize: sample C[0] uniformly
        C = self.data.sample(n=cfg.n0, replace=False, random_state=cfg.random_state)
        remaining = self.data.drop(C.index)
        
        current_model: models.Classifier | None = None
        
        for i in range(cfg.T + 1):
            self.progress(i, cfg.T)
            
            if i > 0:
                # Sample new data from remaining pool
                new_sample = remaining.sample(
                    n=min(self.n_per_iteration, len(remaining)),
                    replace=False,
                    random_state=cfg.random_state + i,  # Vary seed per iteration
                )
                remaining = remaining.drop(new_sample.index)
                
                if is_oracle:
                    # Oracle: add all new samples
                    C = pd.concat([C, new_sample], ignore_index=True)
                else:
                    # Filtered: add only samples predicted to repay (not default)
                    X_new = new_sample.drop(columns=[cfg.label_col])
                    y_pred = models.predict_with_threshold(current_model, X_new, cfg.tau) # type: ignore
                    
                    # Accept clients predicted NOT to default (y_pred == 0)
                    # This models the bank only giving loans to clients believed to repay
                    accepted_mask = y_pred == 0
                    filtered_sample = new_sample.iloc[accepted_mask]
                    C = pd.concat([C, filtered_sample], ignore_index=True)
            
            # Train model on current training set C
            X = C.drop(columns=[cfg.label_col])
            y = C[cfg.label_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=cfg.test_size,
                random_state=cfg.random_state,
            )
            
            current_model = models.train_model(cfg.model_id, X_train, y_train)
            
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


def run_comparison_experiment(
    data: pd.DataFrame,
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = _default_progress,
) -> tuple[ExperimentResult, ExperimentResult]:
    """
    Run both oracle and filtered experiments for comparison.
    
    Returns:
        Tuple of (filtered_results, oracle_results)
    """
    experiment = TemporalEvolutionExperiment(data, config, progress_callback)
    
    if progress_callback:
        print("Running filtered (biased) experiment...")
    filtered = experiment.run(is_oracle=False)
    
    if progress_callback:
        print("Running oracle experiment...")
    oracle = experiment.run(is_oracle=True)
    
    return filtered, oracle
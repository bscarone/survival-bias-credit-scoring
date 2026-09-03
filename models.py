"""
Model definitions and training utilities for credit scoring experiments.

Note: Evaluation metrics are in metrics.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Re-export metrics for backward compatibility
from metrics import (
    EvalMetrics,
    evaluate_predictions,
)


@runtime_checkable
class Classifier(Protocol):
    """Protocol for sklearn-compatible classifiers."""
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Classifier: ...
    def predict(self, X: ArrayLike) -> np.ndarray: ...
    def predict_proba(self, X: ArrayLike) -> np.ndarray: ...


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    random_state: int = 42


DEFAULT_CONFIGS: dict[str, ModelConfig] = {
    "lr": ModelConfig(name="Logistic Regression", params={"max_iter": 1000}),
    "dt": ModelConfig(name="Decision Tree", params={"max_depth": 10}),
    "rf": ModelConfig(name="Random Forest", params={"n_estimators": 100, "n_jobs": -1}),
    "gbdt": ModelConfig(name="Gradient Boosting", params={"n_estimators": 100}),
    "svm": ModelConfig(name="SVM", params={"kernel": "rbf", "probability": True}),
    "mlp": ModelConfig(name="MLP", params={"hidden_layer_sizes": (100, 50), "max_iter": 500}),
}


def get_model(
    model_id: str,
    config: ModelConfig | None = None,
    random_state: int | None = None,
) -> Classifier:
    """Get a model instance by ID."""
    if config is None:
        if model_id not in DEFAULT_CONFIGS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(DEFAULT_CONFIGS.keys())}")
        config = DEFAULT_CONFIGS[model_id]

    rs = random_state if random_state is not None else config.random_state
    params = {**config.params, "random_state": rs}

    model_classes: dict[str, type] = {
        "lr": LogisticRegression,
        "dt": DecisionTreeClassifier,
        "rf": RandomForestClassifier,
        "gbdt": GradientBoostingClassifier,
        "svm": SVC,
        "mlp": MLPClassifier,
    }

    if model_id not in model_classes:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(model_classes.keys())}")

    if model_id == "svm":
        params.pop("random_state", None)

    return model_classes[model_id](**params)


def train_model(
    model_id: str,
    X_train: ArrayLike,
    y_train: ArrayLike,
    random_state: int | None = None,
    sample_weight: ArrayLike | None = None,
) -> Classifier:
    """
    Instantiate and fit a model.
    
    Args:
        model_id: Model identifier (e.g., 'lr', 'rf', 'gbdt').
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed for reproducibility.
        sample_weight: Per-sample weights for weighted training.
    
    Returns:
        Fitted classifier.
    """
    model = get_model(model_id, random_state=random_state)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def predict_with_threshold(
    model: Classifier,
    X: ArrayLike,
    tau: float = 0.5,
) -> np.ndarray:
    """
    Predict using custom probability threshold.
    
    Args:
        model: Fitted classifier with predict_proba method.
        X: Features to predict.
        tau: Threshold in [0,1]; predict 1 if P(y=1) > tau.
    
    Returns:
        Binary predictions as numpy array.
    
    Note:
        With label convention 1=default, returns 1 for high-risk clients.
    """
    probas = model.predict_proba(X)[:, 1]
    return (probas > tau).astype(int)


def evaluate_model(
    model: Classifier,
    X_test: ArrayLike,
    y_test: ArrayLike,
    tau: float = 0.5,
) -> EvalMetrics:
    """Evaluate a fitted model on test data."""
    y_pred = predict_with_threshold(model, X_test, tau)
    return evaluate_predictions(y_test, y_pred)


# Convenience functions
def list_models() -> list[str]:
    return list(DEFAULT_CONFIGS.keys())


def get_model_name(model_id: str) -> str:
    return DEFAULT_CONFIGS.get(model_id, ModelConfig(name=model_id.upper())).name
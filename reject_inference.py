"""
Reject inference strategies for handling selection bias in credit scoring.

These methods attempt to recover information about rejected applicants
who have no observed outcomes (labels).

Label convention: 1 = default (bad), 0 = no default (good)
Accept = y_pred == 0 (predicted to repay)
Reject = y_pred == 1 (predicted to default)

References:
    - Parceling: Industry standard, Banasik & Crook 2007
    - Fuzzy Augmentation: Weighted label assignment
    - Simple Extrapolation: Label imputation from accepts-only model
    - Shallow Self-Learning: Kozodoi et al. 2019 (ECML-PKDD)
    - Twins Analysis: Nearest-neighbor label transfer, Crook & Banasik 2004
"""
from __future__ import annotations

from abc import ABC, abstractmethod # ABC is Python Abstract Base Class
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from models import Classifier, predict_with_threshold, train_model


class StrategyID(Enum):
    """Defines identifiers for reject inference strategies."""
    # auto() auto-assigns integer values for the enum class
    ORACLE = auto()
    FILTERED = auto()
    PARCELING = auto()
    FUZZY_AUGMENTATION = auto()
    SIMPLE_EXTRAPOLATION = auto()
    SHALLOW_SELF_LEARNING = auto()
    TWINS = auto()


@dataclass
class InferenceResult:
    """Result of applying a reject inference strategy."""
    augmented_data: pd.DataFrame # original data + imputed rejects
    sample_weights: np.ndarray | None = None


class RejectInferenceStrategy(ABC):
    """Base class for reject inference strategies."""
    
    @property
    @abstractmethod
    def id(self) -> StrategyID:
        pass
    
    @abstractmethod
    def apply(
        self,
        current_data: pd.DataFrame,
        new_sample: pd.DataFrame,
        model: Classifier | None,
        label_col: str,
        tau: float, # classification threshold 
    ) -> InferenceResult:
        pass

# The six strategy classes — Each one inherits from RejectInferenceStrategy 
# and implements apply differently

class OracleStrategy(RejectInferenceStrategy):
    """Baseline: full access to all labels (no selection bias)."""
    
    @property
    def id(self) -> StrategyID:
        return StrategyID.ORACLE
    
    def apply(self, current_data, new_sample, model, label_col, tau) -> InferenceResult:
        augmented = pd.concat([current_data, new_sample], ignore_index=True)
        return InferenceResult(augmented_data=augmented)


class FilteredStrategy(RejectInferenceStrategy):
    """
    Selection bias without correction: only clients predicted to repay are added.
    
    This models the real-world scenario where banks only give loans to clients 
    the model believes will repay, so we only observe outcomes for those clients.
    """
    
    @property
    def id(self) -> StrategyID:
        return StrategyID.FILTERED
    
    def apply(self, current_data, new_sample, model, label_col, tau) -> InferenceResult:
        if model is None:
            augmented = pd.concat([current_data, new_sample], ignore_index=True)
            return InferenceResult(augmented_data=augmented)
        
        X_new = new_sample.drop(columns=[label_col])
        y_pred = predict_with_threshold(model, X_new, tau)
        
        # Accept clients predicted NOT to default (y_pred == 0)
        accepted_mask = y_pred == 0
        accepted = new_sample.iloc[accepted_mask]
        augmented = pd.concat([current_data, accepted], ignore_index=True)
        return InferenceResult(augmented_data=augmented)


class ParcelingStrategy(RejectInferenceStrategy):
    """
    Parceling: Bin-based label assignment for rejects.

    Industry standard approach (Banasik & Crook 2007). All new applicants 
    (accepts and rejects) are scored and binned into parcels by risk score.
    The observed bad rate (defaults / total accepts) among accepts is computed 
    within each parcel, and that same rate is applied to the rejects in the same 
    parcel. For example, if parcel k has a 30% bad rate among accepted applicants 
    and contains 100 rejects, then 30 of those rejects are randomly labeled Bad 
    and 70 are labeled Good.

    For parcels with fewer than `min_bin_size` accepts, the global accept bad rate 
    is used as a fallback to avoid noisy estimates.

    Args:
        n_bins: Number of risk score bins (parcels). Default is 10 (deciles).
        min_bin_size: Minimum number of accepts in a bin to trust its bad rate. 
        Bins below this threshold use the global accept bad rate.
        random_state: Seed for reproducible label assignment within bins.
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_bin_size: int = 30,
        random_state: int = 42,
    ):
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.random_state = random_state

    @property
    def id(self) -> StrategyID:
        return StrategyID.PARCELING

    def apply(self, current_data, new_sample, model, label_col, tau) -> InferenceResult:
        if model is None:
            augmented = pd.concat([current_data, new_sample], ignore_index=True)
            return InferenceResult(augmented_data=augmented)

        X_new = new_sample.drop(columns=[label_col])
        y_pred = predict_with_threshold(model, X_new, tau)

        # Score all new applicants
        risk_scores = model.predict_proba(X_new)[:, 1]

        # Split into accepts and rejects
        accepted_mask = y_pred == 0
        accepted = new_sample.iloc[accepted_mask].copy()
        rejected = new_sample.iloc[~accepted_mask].copy()
        accepted_scores = risk_scores[accepted_mask]
        rejected_scores = risk_scores[~accepted_mask]

        if len(rejected) > 0:
            # Bin all new applicants together by risk score quantiles
            n_bins = min(self.n_bins, len(new_sample))
            bin_edges = np.quantile(risk_scores, np.linspace(0, 1, n_bins + 1))
            bin_edges[-1] += 1e-9
            accepted_bins = np.clip(np.digitize(accepted_scores, bin_edges) - 1, 0, n_bins - 1)
            rejected_bins = np.clip(np.digitize(rejected_scores, bin_edges) - 1, 0, n_bins - 1)

            # Global accept bad rate as fallback for sparse bins
            global_bad_rate = (accepted[label_col] == 1).mean() if len(accepted) > 0 else (
                (current_data[label_col] == 1).mean()
            )

            # Compute observed bad rate among accepts in each bin
            accepted_labels = accepted[label_col].values
            bin_bad_rates = np.full(n_bins, global_bad_rate)
            for b in range(n_bins):
                bin_mask = accepted_bins == b
                n_accepts_in_bin = bin_mask.sum()
                if n_accepts_in_bin >= self.min_bin_size:
                    # Bad rate = defaults / total accepts in this bin
                    bin_bad_rates[b] = (accepted_labels[bin_mask] == 1).mean()

            # Assign labels to rejects using their bin's observed accept bad rate
            rng = np.random.default_rng(self.random_state)
            imputed_labels = np.zeros(len(rejected), dtype=int)

            for b in range(n_bins):
                mask = rejected_bins == b
                n_in_bin = mask.sum()
                if n_in_bin == 0:
                    continue
                n_bad = int(n_in_bin * bin_bad_rates[b])
                bad_positions = rng.choice(n_in_bin, size=n_bad, replace=False)
                bin_labels = np.zeros(n_in_bin, dtype=int)
                bin_labels[bad_positions] = 1
                imputed_labels[mask] = bin_labels

            rejected[label_col] = imputed_labels

        augmented = pd.concat([current_data, accepted, rejected], ignore_index=True)
        return InferenceResult(augmented_data=augmented)



class FuzzyAugmentationStrategy(RejectInferenceStrategy):
    """
    Fuzzy Augmentation: Probabilistic label assignment for rejects.
    
    Original method duplicates each reject with complementary weights, but this 
    doesn't work well with tree-based models (RF, GBDT) that can't handle 
    identical features with opposite labels.
    
    This implementation uses probabilistic label assignment instead:
    each rejected sample is assigned a hard label by sampling from a Bernoulli 
    distribution parameterized by the model's predicted P(default).
    
    Args:
        use_probabilistic: If True (default), use probabilistic assignment.
            If False, use original weighted duplication (for LR, neural nets).
        random_state: Seed for reproducible label assignment.
    """
    
    def __init__(self, use_probabilistic: bool = True, random_state: int = 42):
        self.use_probabilistic = use_probabilistic
        self.random_state = random_state
    
    @property
    def id(self) -> StrategyID:
        return StrategyID.FUZZY_AUGMENTATION
    
    def apply(self, current_data, new_sample, model, label_col, tau) -> InferenceResult:
        if model is None:
            augmented = pd.concat([current_data, new_sample], ignore_index=True)
            return InferenceResult(augmented_data=augmented)
        
        X_new = new_sample.drop(columns=[label_col])
        y_pred = predict_with_threshold(model, X_new, tau)
        proba_default = model.predict_proba(X_new)[:, 1]
        
        accepted_mask = y_pred == 0
        accepted = new_sample.iloc[accepted_mask].copy()
        rejected = new_sample.iloc[~accepted_mask].copy()
        rejected_proba_default = proba_default[~accepted_mask]
        
        if len(rejected) > 0:
            if self.use_probabilistic:
                # Probabilistic label assignment (works with RF/GBDT)
                # Sample label from Bernoulli(P(default)) per reject:
                # if random draw < P(default), label as defaulter (1), 
                # otherwise label as non-defaulter (0)
                rng = np.random.default_rng(self.random_state) # Creates a random number generator with a fixed seed
                # rng.random(len(rejected)) generates one uniform random number 
                # between 0 and 1 for each reject
                imputed_labels = (rng.random(len(rejected)) < rejected_proba_default).astype(int) # element-wise comparison
                rejected[label_col] = imputed_labels
                
                augmented = pd.concat([current_data, accepted, rejected], ignore_index=True)
                return InferenceResult(augmented_data=augmented)
            else:
                # Original weighted duplication (works with LR, neural nets)
                current_weights = np.ones(len(current_data))
                accepted_weights = np.ones(len(accepted))
                
                rejected_good = rejected.copy()
                rejected_good[label_col] = 0
                good_weights = 1 - rejected_proba_default
                
                rejected_bad = rejected.copy()
                rejected_bad[label_col] = 1
                bad_weights = rejected_proba_default
                
                augmented = pd.concat(
                    [current_data, accepted, rejected_good, rejected_bad],
                    ignore_index=True
                )
                weights = np.concatenate([
                    current_weights, accepted_weights, good_weights, bad_weights
                ])
                return InferenceResult(augmented_data=augmented, sample_weights=weights)
        else:
            augmented = pd.concat([current_data, accepted], ignore_index=True)
            return InferenceResult(augmented_data=augmented)


class SimpleExtrapolationStrategy(RejectInferenceStrategy):
    """
    Simple Extrapolation: impute reject labels from the current model's own
    hard predictions.

    No new model is fitted here. The model passed in — already trained on the
    accumulated training set at this cycle — is re-applied to the rejected
    applicants and its thresholded output is recorded as their label.

    Because the rejects are exactly the applicants for which this model
    predicted 1 at this threshold, and both RF and GBDT are deterministic at
    inference time, re-applying it necessarily returns 1 for every reject.
    Every rejected applicant therefore receives the label y = 1 (see paper,
    Sec. 3 and footnote 9).
    """
    
    @property
    def id(self) -> StrategyID:
        return StrategyID.SIMPLE_EXTRAPOLATION
    
    def apply(self, current_data, new_sample, model, label_col, tau) -> InferenceResult:
        if model is None:
            augmented = pd.concat([current_data, new_sample], ignore_index=True)
            return InferenceResult(augmented_data=augmented)
        
        X_new = new_sample.drop(columns=[label_col])
        y_pred = predict_with_threshold(model, X_new, tau)
        
        accepted_mask = y_pred == 0
        accepted = new_sample.iloc[accepted_mask].copy()
        rejected = new_sample.iloc[~accepted_mask].copy()
        
        if len(rejected) > 0:
            X_rejected = rejected.drop(columns=[label_col])
            imputed_labels = predict_with_threshold(model, X_rejected, tau)
            rejected[label_col] = imputed_labels
        
        augmented = pd.concat([current_data, accepted, rejected], ignore_index=True)
        return InferenceResult(augmented_data=augmented)


class ShallowSelfLearningStrategy(RejectInferenceStrategy):
    """
    Shallow Self-Learning (Kozodoi et al. 2019, ECML-PKDD).

    Three-stage reject inference framework:

    Stage 1 — Filtering: An Isolation Forest trained on accepts estimates
        novelty scores for rejects. Rejects too different from accepts
        (bottom beta_bottom percentile) are removed because the model
        can't reliably predict them. Rejects too similar to accepts
        (top beta_top percentile) are also removed because they add
        little new information. Only the middle band is kept.

    Stage 2 — Labeling: An L1-regularized logistic regression (shallow/weak
        learner) iteratively labels high-confidence rejects. In the first
        iteration, confidence is determined by a percentile threshold alpha
        on predicted probabilities. After the first iteration, the
        thresholds are fixed to the absolute probability values from
        iteration 1, preventing threshold drift and error propagation.

    Stage 3 — Model training: The final model is trained on the augmented
        dataset (accepts + pseudo-labeled rejects). This is handled
        externally by the caller.

    Args:
        alpha: Percentile threshold for confident predictions in the first
            iteration. E.g., alpha=0.1 selects the top/bottom 10% of
            predicted probabilities as confident.
        beta_bottom: Percentile of rejects to remove as too different from
            accepts (low novelty score). Default is 0.1.
        beta_top: Percentile of rejects to remove as too similar to accepts
            (high novelty score). Default is 0.1.
        imbalance_ratio: Expected ratio of reject bad rate to accept bad
            rate. Used to adjust class weights during weak learner training
            to account for the higher default rate among rejects.
        max_iterations: Maximum number of labeling iterations.
        weak_model_id: Model identifier for the weak learner used in
            labeling. Default is logistic regression ("lr").
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta_bottom: float = 0.1,
        beta_top: float = 0.1,
        imbalance_ratio: float = 2.0,
        max_iterations: int = 5,
        weak_model_id: str = "lr",
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.beta_bottom = beta_bottom
        self.beta_top = beta_top
        self.imbalance_ratio = imbalance_ratio
        self.max_iterations = max_iterations
        self.weak_model_id = weak_model_id
        self.random_state = random_state

    @property
    def id(self) -> StrategyID:
        return StrategyID.SHALLOW_SELF_LEARNING

    def _filter_rejects(self, X_accepted: pd.DataFrame, X_rejected: pd.DataFrame) -> np.ndarray:
        """
        Stage 1: Filter rejects using Isolation Forest novelty detection.

        Returns a boolean mask over rejected samples indicating which to keep.
        """
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(random_state=self.random_state)
        iso.fit(X_accepted)

        # Higher score = more similar to accepts
        reject_scores = iso.decision_function(X_rejected)

        lower_bound = np.percentile(reject_scores, self.beta_bottom * 100)
        upper_bound = np.percentile(reject_scores, (1 - self.beta_top) * 100)

        return (reject_scores >= lower_bound) & (reject_scores <= upper_bound)

    def apply(self, current_data, new_sample, model, label_col, tau) -> InferenceResult:
        if model is None:
            augmented = pd.concat([current_data, new_sample], ignore_index=True)
            return InferenceResult(augmented_data=augmented)

        X_new = new_sample.drop(columns=[label_col])
        y_pred = predict_with_threshold(model, X_new, tau)

        accepted_mask = y_pred == 0
        accepted = new_sample.iloc[accepted_mask].copy()
        rejected = new_sample.iloc[~accepted_mask].copy()

        # Stage 1: Filter rejects via Isolation Forest
        if len(rejected) > 0:
            X_accepted = accepted.drop(columns=[label_col])
            X_rejected = rejected.drop(columns=[label_col])

            # Use all accepts (current + new) to fit the novelty detector
            X_all_accepted = pd.concat(
                [current_data.drop(columns=[label_col]), X_accepted],
                ignore_index=True,
            )
            keep_mask = self._filter_rejects(X_all_accepted, X_rejected)
            unlabeled = rejected.iloc[keep_mask].copy()
        else:
            unlabeled = rejected.copy()

        labeled_pool = pd.concat([current_data, accepted], ignore_index=True)

        # Compute per-class weight adjusted for expected reject imbalance
        accept_bad_rate = (labeled_pool[label_col] == 1).mean()
        adjusted_bad_rate = min(accept_bad_rate * self.imbalance_ratio, 1.0)
        # Weight defaulters higher to reflect expected reject default rate
        weight_bad = adjusted_bad_rate / max(accept_bad_rate, 1e-9)

        # Absolute probability thresholds, set after the first iteration
        threshold_bad = None
        threshold_good = None

        # Stage 2: Iterative labeling with shallow learner
        for iteration in range(self.max_iterations):
            if len(unlabeled) == 0:
                break

            X_labeled = labeled_pool.drop(columns=[label_col])
            y_labeled = labeled_pool[label_col]
            weak_model = train_model(
                self.weak_model_id, X_labeled, y_labeled,
                sample_weight=np.where(y_labeled == 1, weight_bad, 1.0),
            )

            X_unlabeled = unlabeled.drop(columns=[label_col])
            proba_default = weak_model.predict_proba(X_unlabeled)[:, 1]

            if iteration == 0:
                # First iteration: use percentile-based thresholds
                threshold_bad = np.percentile(proba_default, (1 - self.alpha) * 100)
                threshold_good = np.percentile(proba_default, self.alpha * 100)

            # Apply fixed absolute thresholds from iteration 1 onwards
            confident_bad = proba_default >= threshold_bad
            confident_good = proba_default <= threshold_good
            confident_mask = confident_good | confident_bad

            if not confident_mask.any():
                break

            newly_labeled = unlabeled.iloc[confident_mask].copy()
            newly_labeled[label_col] = (proba_default[confident_mask] > 0.5).astype(int)

            labeled_pool = pd.concat([labeled_pool, newly_labeled], ignore_index=True)
            unlabeled = unlabeled.iloc[~confident_mask]

        return InferenceResult(augmented_data=labeled_pool)


class TwinsStrategy(RejectInferenceStrategy):
    """
    Twins Analysis: Nearest-neighbor label transfer for rejects.

    For each rejected applicant, finds the k closest accepted applicants
    in feature space and assigns the majority label among those neighbors.
    Features are standardized before computing distances.

    Args:
        n_neighbors: Number of accepted neighbors to consider per reject.
        metric: Distance metric for nearest-neighbor search.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    @property
    def id(self) -> StrategyID:
        return StrategyID.TWINS

    def apply(self, current_data, new_sample, model, label_col, tau) -> InferenceResult:
        if model is None:
            augmented = pd.concat([current_data, new_sample], ignore_index=True)
            return InferenceResult(augmented_data=augmented)

        X_new = new_sample.drop(columns=[label_col])
        y_pred = predict_with_threshold(model, X_new, tau)

        accepted_mask = y_pred == 0
        accepted = new_sample.iloc[accepted_mask].copy()
        rejected = new_sample.iloc[~accepted_mask].copy()

        if len(rejected) > 0:
            # Build neighbor pool from all accepted applicants (current + newly accepted)
            accepted_pool = pd.concat([current_data, accepted], ignore_index=True)
            X_pool = accepted_pool.drop(columns=[label_col])
            y_pool = accepted_pool[label_col].values

            X_rejected = rejected.drop(columns=[label_col])

            # Standardize features for meaningful distance computation
            scaler = StandardScaler()
            X_pool_scaled = scaler.fit_transform(X_pool)
            X_rejected_scaled = scaler.transform(X_rejected)

            # Find nearest accepted neighbors for each reject
            k = min(self.n_neighbors, len(accepted_pool))
            nn = NearestNeighbors(n_neighbors=k, metric=self.metric)
            nn.fit(X_pool_scaled)
            neighbor_indices = nn.kneighbors(X_rejected_scaled, return_distance=False)

            # Assign majority label from neighbors
            imputed_labels = np.array([
                int(y_pool[idx].mean() >= 0.5)
                for idx in neighbor_indices
            ])
            rejected[label_col] = imputed_labels

        augmented = pd.concat([current_data, accepted, rejected], ignore_index=True)
        return InferenceResult(augmented_data=augmented)


# Factory
STRATEGY_REGISTRY: dict[StrategyID, type[RejectInferenceStrategy]] = {
    StrategyID.ORACLE: OracleStrategy,
    StrategyID.FILTERED: FilteredStrategy,
    StrategyID.PARCELING: ParcelingStrategy,
    StrategyID.FUZZY_AUGMENTATION: FuzzyAugmentationStrategy,
    StrategyID.SIMPLE_EXTRAPOLATION: SimpleExtrapolationStrategy,
    StrategyID.SHALLOW_SELF_LEARNING: ShallowSelfLearningStrategy,
    StrategyID.TWINS: TwinsStrategy,
}


def get_strategy(strategy_id: StrategyID, **kwargs) -> RejectInferenceStrategy:
    """Factory function to create strategy instances."""
    strategy_class = STRATEGY_REGISTRY.get(strategy_id)
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_id}")
    return strategy_class(**kwargs)
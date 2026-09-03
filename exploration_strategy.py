"""
Exploration strategy for studying the cost-of-learning trade-off in credit scoring 
under selection bias.

Unlike reject inference strategies that *impute* labels for rejects, this strategy 
*approves* a fraction of rejects so the bank observes their true outcomes. 
This breaks the feedback loop at a known cost.

Multiple ranking strategies determine *which* rejects to explore:
- LEAST_RISKY:   lowest P(default) first (cheapest for the bank)
- MOST_RISKY:    highest P(default) first (most expensive, theoretical bound)
- RANDOM:        uniform random sample (unbiased baseline)

Usage:
    strategy = ExplorationStrategy(
        exploration_rate=0.1,
        ranking=RankingStrategy.RANDOM,
    )
    result = strategy.apply(current_data, new_sample, model, label_col, tau)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# These imports match the existing codebase
from models import Classifier, predict_with_threshold #, train_model


# ---------------------------------------------------------------------------
# Ranking strategies
# ---------------------------------------------------------------------------

class RankingStrategy(Enum):
    LEAST_RISKY = "least_risky"
    MOST_RISKY = "most_risky"
    RANDOM = "random"


def _select_least_risky(
    risk_scores: np.ndarray, n: int, rng: np.random.Generator, **kwargs
) -> np.ndarray:
    """Select the n rejects with lowest P(default)."""
    order = np.argsort(risk_scores) # sorts risk_scores ascending
    return order[:n]


def _select_most_risky(
    risk_scores: np.ndarray, n: int, rng: np.random.Generator, **kwargs
) -> np.ndarray:
    """Select the n rejects with highest P(default)."""
    order = np.argsort(risk_scores)[::-1] # [::-1] reverses the array
    return order[:n]


def _select_random(
    risk_scores: np.ndarray, n: int, rng: np.random.Generator, **kwargs
) -> np.ndarray:
    """Select n rejects uniformly at random."""
    indices = np.arange(len(risk_scores))
    return rng.choice(indices, size=n, replace=False)


RANKING_FUNCTIONS = {
    RankingStrategy.LEAST_RISKY: _select_least_risky,
    RankingStrategy.MOST_RISKY: _select_most_risky,
    RankingStrategy.RANDOM: _select_random,
}


# ---------------------------------------------------------------------------
# Exploration diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ExplorationInfo:
    """Diagnostics for the exploration step at a single iteration."""
    n_rejects_total: int = 0
    n_explored: int = 0
    n_explored_defaulted: int = 0
    n_explored_repaid: int = 0
    exploration_default_rate: float = 0.0
    reject_default_rate: float = 0.0
    exploration_cost: float = 0.0
    explored_risk_min: float = 0.0
    explored_risk_max: float = 0.0
    explored_risk_mean: float = 0.0
    ranking_strategy: str = ""


@dataclass
class ExplorationResult:
    """Result of applying the exploration strategy."""
    augmented_data: pd.DataFrame
    sample_weights: np.ndarray | None = None
    exploration_info: ExplorationInfo = field(default_factory=ExplorationInfo)


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------

class ExplorationStrategy:
    """
    Controlled Exploration: approve a fraction of rejects to observe true outcomes.

    At each iteration:
    1. Score all new applicants with the current model.
    2. Accept those predicted to repay (y_pred == 0), as usual.
    3. Among rejects (y_pred == 1), select r% according to the ranking strategy.
    4. Add ALL approved applicants (regular + explored) with their TRUE labels.

    Args:
        exploration_rate: Fraction of rejects to approve (0.0 = Biased, 1.0 = Oracle).
        ranking: Which ranking strategy to use for selecting rejects to explore.
        n_bins: Number of bins for STRATIFIED ranking (default 10).
        random_state: Seed for reproducibility (affects RANDOM, DIVERSITY, STRATIFIED).
    """

    def __init__(
        self,
        exploration_rate: float = 0.1,
        ranking: RankingStrategy = RankingStrategy.LEAST_RISKY,
        n_bins: int = 10,
        random_state: int = 42,
    ):
        if not 0.0 <= exploration_rate <= 1.0:
            raise ValueError(f"exploration_rate must be in [0, 1], got {exploration_rate}")
        self.exploration_rate = exploration_rate
        self.ranking = ranking
        self.n_bins = n_bins
        self.rng = np.random.default_rng(random_state)
        self._select_fn = RANKING_FUNCTIONS[ranking] # select ranking function

    @property
    def id(self) -> str:
        return f"exploration_{self.ranking.value}_r{self.exploration_rate:.2f}"

    def apply(
        self,
        current_data: pd.DataFrame,
        new_sample: pd.DataFrame,
        model: Classifier | None,
        label_col: str,
        tau: float,
    ) -> ExplorationResult:
        """
        Apply the exploration strategy to a new batch of applicants.

        Args:
            current_data: Training set from previous iterations.
            new_sample: New applicants (with true labels, for simulation).
            model: Current credit scoring model (None at iteration 0).
            label_col: Name of the label column (1=default, 0=repay).
            tau: Decision threshold for accept/reject.

        Returns:
            ExplorationResult with augmented data and exploration diagnostics.
        """
        info = ExplorationInfo(ranking_strategy=self.ranking.value)

        # First iteration: no model yet, add everything
        if model is None:
            augmented = pd.concat([current_data, new_sample], ignore_index=True)
            return ExplorationResult(augmented_data=augmented, exploration_info=info)

        X_new = new_sample.drop(columns=[label_col])
        y_true = new_sample[label_col].values

        # Step 1: Get predictions and risk scores
        y_pred = predict_with_threshold(model, X_new, tau)
        proba_default = model.predict_proba(X_new)[:, 1]

        # Step 2: Separate accepts and rejects
        accepted_mask = y_pred == 0
        rejected_mask = ~accepted_mask

        accepted = new_sample.iloc[accepted_mask].copy()
        rejected = new_sample.iloc[rejected_mask].copy()

        info.n_rejects_total = len(rejected)
        # If r=0%, this is exactly the Biased/Filtered strategy — only accepts are added
        if len(rejected) == 0 or self.exploration_rate == 0.0:
            augmented = pd.concat([current_data, accepted], ignore_index=True)
            return ExplorationResult(augmented_data=augmented, exploration_info=info)

        # Step 3: Select which rejects to explore
        rejected_risk_scores = proba_default[rejected_mask]
        n_explore = max(1, int(np.ceil(self.exploration_rate * len(rejected))))
        n_explore = min(n_explore, len(rejected))

        # Prepare kwargs for ranking functions that need extra context
        select_kwargs = {"tau": tau, "n_bins": self.n_bins}

        # Apply ranking function to get indices into the rejected array
        local_indices = self._select_fn( # local to reject array
            rejected_risk_scores, n_explore, self.rng, **select_kwargs,
        )

        # Map back to new_sample positions
        rejected_positions = np.where(rejected_mask)[0]
        explore_global_indices = rejected_positions[local_indices]

        explored = new_sample.iloc[explore_global_indices].copy()

        # Step 4: Record exploration diagnostics
        explored_true_labels = y_true[explore_global_indices]
        explored_scores = proba_default[explore_global_indices]

        info.n_explored = len(explored)
        info.n_explored_defaulted = int(explored_true_labels.sum())
        info.n_explored_repaid = len(explored) - info.n_explored_defaulted
        info.exploration_default_rate = (
            explored_true_labels.mean() if len(explored) > 0 else 0.0
        )
        info.exploration_cost = info.n_explored_defaulted

        all_reject_true_labels = y_true[rejected_mask]
        info.reject_default_rate = (
            all_reject_true_labels.mean() if len(all_reject_true_labels) > 0 else 0.0
        )

        info.explored_risk_min = float(explored_scores.min())
        info.explored_risk_max = float(explored_scores.max())
        info.explored_risk_mean = float(explored_scores.mean())

        # Step 5: Augment training set
        augmented = pd.concat(
            [current_data, accepted, explored],
            ignore_index=True,
        )

        return ExplorationResult(augmented_data=augmented, exploration_info=info)


# ---------------------------------------------------------------------------
# Sweep over exploration rates
# ---------------------------------------------------------------------------

def get_exploration_rates(granularity: str = "standard") -> list[float]:
    """
    Return a list of exploration rates to sweep.

    Args:
        granularity: 'coarse' (6 values), 'standard' (11 values), 'fine' (21 values).
    """
    if granularity == "coarse":
        return [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]
    elif granularity == "standard":
        return [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
    elif granularity == "fine":
        return [i / 20 for i in range(21)]
    else:
        raise ValueError(f"Unknown granularity: {granularity}")
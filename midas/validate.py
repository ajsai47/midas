"""Validation engine — prove your formula actually works.

Scores every post in your dataset against your config and measures how well
MIDAS scores predict actual engagement.  Reports Spearman rank correlation,
per-tier calibration, and holdout cross-validation.

Usage:
    from midas.validate import validate, holdout_validate

    result = validate(posts, config)
    print(result)                          # Spearman rho, p-value, tier table

    cv = holdout_validate(posts, n_splits=5)
    print(cv)                              # Cross-validated correlation
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analyze import _engagement, _norm_cdf, analyze_posts
from .config import MidasConfig, load_config
from .scorer import score


# ---------------------------------------------------------------------------
# Spearman rank correlation (pure Python)
# ---------------------------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    """Assign average ranks to values, handling ties."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j

    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman rank correlation coefficient with p-value.

    Returns (rho, p_value).  Uses the t-distribution approximation for
    the p-value, which is standard for n >= 10.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    rx = _rank(x)
    ry = _rank(y)

    # Pearson correlation of ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    std_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if std_x == 0 or std_y == 0:
        return 0.0, 1.0

    rho = cov / (std_x * std_y)

    # t-test for significance
    if abs(rho) >= 1.0:
        return rho, 0.0

    t_stat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    # Approximate p-value using normal distribution (good for n >= 30)
    p_value = 2 * (1 - _norm_cdf(abs(t_stat)))

    return rho, p_value


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class TierCalibration:
    """How well a single tier predicts actual engagement."""

    tier_name: str
    count: int
    predicted_min_score: float
    actual_median_engagement: float
    actual_mean_engagement: float
    engagement_range: tuple[float, float]  # (min, max)


@dataclass
class ValidationResult:
    """Complete validation output."""

    total_posts: int
    spearman_rho: float
    spearman_p: float
    tier_calibration: list[TierCalibration]
    # Per-post details for analysis
    post_scores: list[float] = field(default_factory=list)
    post_engagements: list[float] = field(default_factory=list)

    @property
    def correlation_strength(self) -> str:
        """Human-readable interpretation of the correlation."""
        rho = abs(self.spearman_rho)
        if rho >= 0.7:
            return "STRONG"
        elif rho >= 0.5:
            return "MODERATE"
        elif rho >= 0.3:
            return "WEAK"
        else:
            return "NEGLIGIBLE"

    @property
    def is_significant(self) -> bool:
        return self.spearman_p < 0.05

    def __str__(self) -> str:
        parts = [
            "MIDAS Validation Report",
            "=" * 50,
            f"Posts scored: {self.total_posts}",
            "",
            f"Spearman rho:  {self.spearman_rho:+.4f}  ({self.correlation_strength})",
            f"p-value:       {self.spearman_p:.6f}  "
            f"({'SIGNIFICANT' if self.is_significant else 'NOT SIGNIFICANT'})",
            "",
        ]

        if self.spearman_rho > 0 and self.is_significant:
            parts.append(
                "Your formula positively correlates with actual engagement."
            )
        elif self.spearman_rho <= 0:
            parts.append(
                "WARNING: No positive correlation found. Your formula may "
                "not be predictive. Try re-analyzing with more data."
            )

        if self.tier_calibration:
            parts.append("")
            parts.append("Tier Calibration:")
            parts.append(f"  {'Tier':<20} {'Count':>5}  {'Med. Eng.':>10}  {'Range':>16}")
            parts.append(f"  {'-'*20} {'-'*5}  {'-'*10}  {'-'*16}")
            for tc in self.tier_calibration:
                parts.append(
                    f"  {tc.tier_name:<20} {tc.count:>5}  "
                    f"{tc.actual_median_engagement:>10.1f}  "
                    f"{tc.engagement_range[0]:>7.0f}-{tc.engagement_range[1]:>7.0f}"
                )

        return "\n".join(parts)


@dataclass
class CrossValidationResult:
    """K-fold holdout validation results."""

    n_splits: int
    fold_results: list[ValidationResult]
    mean_rho: float
    std_rho: float
    all_significant: bool

    def __str__(self) -> str:
        parts = [
            f"MIDAS {self.n_splits}-Fold Cross-Validation",
            "=" * 50,
        ]

        for i, fold in enumerate(self.fold_results, 1):
            sig = "*" if fold.is_significant else ""
            parts.append(
                f"  Fold {i}: rho={fold.spearman_rho:+.4f}  "
                f"p={fold.spearman_p:.4f}{sig}  "
                f"(n={fold.total_posts})"
            )

        parts.append("")
        parts.append(f"Mean rho:  {self.mean_rho:+.4f} +/- {self.std_rho:.4f}")
        parts.append(
            f"Result:    {'ALL FOLDS SIGNIFICANT' if self.all_significant else 'Some folds not significant'}"
        )

        if self.mean_rho > 0.3 and self.all_significant:
            parts.append("\nYour formula generalizes. It is predictive on unseen data.")
        elif self.mean_rho > 0:
            parts.append("\nWeak but positive signal. Consider collecting more data.")
        else:
            parts.append("\nYour formula does not generalize. Re-analyze with more data.")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def validate(
    posts: list[dict],
    config: MidasConfig,
) -> ValidationResult:
    """Score every post and measure correlation with actual engagement.

    Parameters
    ----------
    posts : list[dict]
        Posts with text and engagement data.
    config : MidasConfig
        The scoring config to validate.

    Returns
    -------
    ValidationResult
        Spearman correlation, p-value, and per-tier calibration.
    """
    if len(posts) < 5:
        raise ValueError("Need at least 5 posts to validate. Got {}.".format(len(posts)))

    midas_scores: list[float] = []
    actual_engs: list[float] = []

    for post in posts:
        text = post.get("text", "")
        if not text.strip():
            continue
        result = score(text, config)
        midas_scores.append(result.score)
        actual_engs.append(_engagement(post))

    # Spearman rank correlation
    rho, p_value = spearman_correlation(midas_scores, actual_engs)

    # Per-tier calibration
    tier_data: dict[str, list[float]] = {}
    for ms, eng in zip(midas_scores, actual_engs):
        tier_name = "UNRANKED"
        for t in config.sorted_tiers:
            if ms >= t.min_score:
                tier_name = t.name
                break
        tier_data.setdefault(tier_name, []).append(eng)

    calibration: list[TierCalibration] = []
    for t in config.sorted_tiers:
        engs = tier_data.get(t.name, [])
        if engs:
            calibration.append(TierCalibration(
                tier_name=t.name,
                count=len(engs),
                predicted_min_score=t.min_score,
                actual_median_engagement=statistics.median(engs),
                actual_mean_engagement=statistics.mean(engs),
                engagement_range=(min(engs), max(engs)),
            ))

    return ValidationResult(
        total_posts=len(midas_scores),
        spearman_rho=rho,
        spearman_p=p_value,
        tier_calibration=calibration,
        post_scores=midas_scores,
        post_engagements=actual_engs,
    )


def holdout_validate(
    posts: list[dict],
    *,
    n_splits: int = 5,
    seed: int = 42,
    min_frequency: float = 0.02,
) -> CrossValidationResult:
    """K-fold holdout validation: analyze on train set, validate on test set.

    For each fold:
    1. Split posts into train (80%) and test (20%)
    2. Run analyze_posts on the train set to generate a config
    3. Score the test set with that config
    4. Measure Spearman correlation on the test set

    This proves the formula generalizes to unseen data.

    Parameters
    ----------
    posts : list[dict]
        All posts with engagement data.
    n_splits : int
        Number of folds (default: 5).
    seed : int
        Random seed for reproducibility.
    min_frequency : float
        Minimum signal frequency for analysis.

    Returns
    -------
    CrossValidationResult
        Per-fold and aggregate correlation statistics.
    """
    if len(posts) < n_splits * 5:
        raise ValueError(
            f"Need at least {n_splits * 5} posts for {n_splits}-fold CV. "
            f"Got {len(posts)}."
        )

    rng = random.Random(seed)
    indices = list(range(len(posts)))
    rng.shuffle(indices)

    fold_size = len(indices) // n_splits
    fold_results: list[ValidationResult] = []

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_splits - 1 else len(indices)
        test_indices = set(indices[test_start:test_end])
        train_indices = [i for i in indices if i not in test_indices]

        train_posts = [posts[i] for i in train_indices]
        test_posts = [posts[i] for i in test_indices]

        # Analyze train set to produce a config
        analysis = analyze_posts(train_posts, min_frequency=min_frequency)
        fold_config = analysis.config

        # Validate on test set
        result = validate(test_posts, fold_config)
        fold_results.append(result)

    rhos = [r.spearman_rho for r in fold_results]
    mean_rho = statistics.mean(rhos)
    std_rho = statistics.stdev(rhos) if len(rhos) > 1 else 0.0
    all_sig = all(r.is_significant for r in fold_results)

    return CrossValidationResult(
        n_splits=n_splits,
        fold_results=fold_results,
        mean_rho=mean_rho,
        std_rho=std_rho,
        all_significant=all_sig,
    )


# ---------------------------------------------------------------------------
# File I/O convenience
# ---------------------------------------------------------------------------

def validate_file(
    data_path: str,
    config_path: str,
) -> ValidationResult:
    """Load a data file and config, then run validation."""
    filepath = Path(data_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    posts: list[dict] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                posts.append(json.loads(line))

    config = load_config(config_path)
    return validate(posts, config)

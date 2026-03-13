"""Tests for the MIDAS validation engine and statistical functions."""

import json
from pathlib import Path

import pytest

from midas.analyze import (
    _mann_whitney_u,
    _bootstrap_ci,
    _benjamini_hochberg,
    _norm_cdf,
    analyze_posts,
    SignalAnalysis,
)
from midas.validate import spearman_correlation, validate, ValidationResult
from midas.config import load_config

SAMPLE_CONFIG = Path(__file__).parent.parent / "examples" / "sample_config.yaml"
SAMPLE_DATA = Path(__file__).parent.parent / "examples" / "sample_data.jsonl"


def _load_sample_posts() -> list[dict]:
    posts = []
    with open(SAMPLE_DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                posts.append(json.loads(line))
    return posts


# ---------------------------------------------------------------------------
# Statistical function tests
# ---------------------------------------------------------------------------

class TestMannWhitneyU:
    def test_identical_groups(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        u, p = _mann_whitney_u(a, b)
        assert p > 0.05  # Not significant

    def test_clearly_different_groups(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        u, p = _mann_whitney_u(a, b)
        assert p < 0.05  # Should be significant

    def test_empty_group(self):
        u, p = _mann_whitney_u([], [1.0, 2.0, 3.0])
        assert p == 1.0

    def test_returns_valid_range(self):
        a = [5.0, 10.0, 15.0, 20.0]
        b = [1.0, 2.0, 3.0, 4.0]
        u, p = _mann_whitney_u(a, b)
        assert 0.0 <= p <= 1.0


class TestBootstrapCI:
    def test_returns_valid_interval(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [5.0, 10.0, 15.0, 20.0, 25.0]
        lo, hi = _bootstrap_ci(a, b)
        assert lo <= hi
        assert lo > 0

    def test_ci_contains_point_estimate(self):
        import statistics
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [5.0, 10.0, 15.0, 20.0, 25.0]
        lo, hi = _bootstrap_ci(a, b)
        point_est = statistics.median(a) / statistics.median(b)
        assert lo <= point_est <= hi

    def test_identical_groups_ci_around_one(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [10.0, 20.0, 30.0, 40.0, 50.0]
        lo, hi = _bootstrap_ci(a, b)
        assert lo <= 1.0 <= hi


class TestBenjaminiHochberg:
    def test_no_significant(self):
        p_values = [0.5, 0.6, 0.7, 0.8]
        rejected = _benjamini_hochberg(p_values, alpha=0.05)
        assert not any(rejected)

    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003, 0.004]
        rejected = _benjamini_hochberg(p_values, alpha=0.05)
        assert all(rejected)

    def test_mixed(self):
        p_values = [0.001, 0.5, 0.8, 0.9]
        rejected = _benjamini_hochberg(p_values, alpha=0.05)
        assert rejected[0]  # First should be rejected
        assert not rejected[-1]  # Last should not

    def test_empty(self):
        assert _benjamini_hochberg([]) == []


class TestNormCDF:
    def test_zero(self):
        assert abs(_norm_cdf(0) - 0.5) < 0.001

    def test_large_positive(self):
        assert _norm_cdf(5.0) > 0.999

    def test_large_negative(self):
        assert _norm_cdf(-5.0) < 0.001

    def test_standard_value(self):
        # P(Z < 1.96) ~= 0.975
        assert abs(_norm_cdf(1.96) - 0.975) < 0.005


# ---------------------------------------------------------------------------
# Spearman correlation tests
# ---------------------------------------------------------------------------

class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        rho, p = spearman_correlation(x, y)
        assert abs(rho - 1.0) < 0.001

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        rho, p = spearman_correlation(x, y)
        assert abs(rho - (-1.0)) < 0.001

    def test_no_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 1.0, 5.0, 2.0, 4.0]
        rho, p = spearman_correlation(x, y)
        assert abs(rho) < 0.5  # Should be weak

    def test_too_few_points(self):
        rho, p = spearman_correlation([1.0, 2.0], [3.0, 4.0])
        assert p == 1.0  # Not enough data

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            spearman_correlation([1.0, 2.0], [3.0])


# ---------------------------------------------------------------------------
# Validation engine tests
# ---------------------------------------------------------------------------

class TestValidate:
    def test_validate_sample_data(self):
        posts = _load_sample_posts()
        config = load_config(str(SAMPLE_CONFIG))
        result = validate(posts, config)
        assert isinstance(result, ValidationResult)
        assert result.total_posts == len(posts)
        assert -1.0 <= result.spearman_rho <= 1.0

    def test_validate_has_tier_calibration(self):
        posts = _load_sample_posts()
        config = load_config(str(SAMPLE_CONFIG))
        result = validate(posts, config)
        assert len(result.tier_calibration) > 0

    def test_validate_correlation_strength(self):
        posts = _load_sample_posts()
        config = load_config(str(SAMPLE_CONFIG))
        result = validate(posts, config)
        assert result.correlation_strength in [
            "STRONG", "MODERATE", "WEAK", "NEGLIGIBLE"
        ]

    def test_validate_too_few_posts(self):
        with pytest.raises(ValueError, match="at least 5"):
            validate([{"text": "hi", "reactions": 1}], load_config(str(SAMPLE_CONFIG)))


# ---------------------------------------------------------------------------
# Analysis with statistics tests
# ---------------------------------------------------------------------------

class TestAnalysisStatistics:
    def test_signals_have_p_values(self):
        posts = _load_sample_posts()
        result = analyze_posts(posts)
        for signal in result.signals + result.anti_patterns:
            assert hasattr(signal, "p_value")
            assert 0.0 <= signal.p_value <= 1.0

    def test_signals_have_ci(self):
        posts = _load_sample_posts()
        result = analyze_posts(posts)
        for signal in result.signals + result.anti_patterns:
            assert signal.ci_lower <= signal.ci_upper

    def test_signals_have_median_lift(self):
        posts = _load_sample_posts()
        result = analyze_posts(posts)
        for signal in result.signals:
            assert signal.median_lift > 0
        for anti in result.anti_patterns:
            assert anti.median_lift < 1.0

    def test_fdr_correction_applied(self):
        posts = _load_sample_posts()
        result = analyze_posts(posts)
        # At least check that significant is a bool
        for signal in result.signals + result.anti_patterns:
            assert isinstance(signal.significant, bool)

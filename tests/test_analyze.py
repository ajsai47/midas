"""Tests for the MIDAS analysis engine."""

import json
import tempfile
from pathlib import Path

import pytest

SAMPLE_DATA = Path(__file__).parent.parent / "examples" / "sample_data.jsonl"


@pytest.fixture
def posts():
    from midas.export import load_jsonl
    return load_jsonl(str(SAMPLE_DATA))


@pytest.fixture
def small_posts():
    """Minimal synthetic dataset for testing."""
    return [
        {"text": "I built something.\n\nIt was amazing.\n\n→ Result one\n→ Result two\n\nComment YES below.", "reactions": 50, "comments": 20, "reposts": 10, "date": "2026-01-01", "has_image": False},
        {"text": "Quick update #AI #ML", "reactions": 3, "comments": 0, "reposts": 0, "date": "2026-01-02", "has_image": False},
        {"text": "I remember when nobody cared about AI.\n\nBack in 2018, I was the weird one.\n\nNow everyone's an AI expert.\n\nBut here's the thing → most still don't understand the fundamentals.", "reactions": 35, "comments": 15, "reposts": 5, "date": "2026-01-03", "has_image": False},
        {"text": "Here's my newsletter about trends this week.\n\n#trending #tech #newsletter", "reactions": 4, "comments": 1, "reposts": 0, "date": "2026-01-04", "has_image": False},
        {"text": "Whoa. Just saw the latest benchmark results.\n\n95% accuracy on a task that was impossible 2 years ago.\n\n$0.001 per inference.\n\nThe future is here.", "reactions": 28, "comments": 10, "reposts": 3, "date": "2026-01-05", "has_image": False},
    ]


class TestAnalyzePosts:
    def test_basic_analysis(self, small_posts):
        from midas.analyze import analyze_posts
        result = analyze_posts(small_posts, min_frequency=0.0)
        assert result.total_posts == 5
        assert len(result.signals) > 0

    def test_signals_have_positive_lift(self, small_posts):
        from midas.analyze import analyze_posts
        result = analyze_posts(small_posts, min_frequency=0.0)
        for s in result.signals:
            assert s.lift > 1.0, f"Signal {s.name} has lift {s.lift}"

    def test_penalties_have_negative_lift(self, small_posts):
        from midas.analyze import analyze_posts
        result = analyze_posts(small_posts, min_frequency=0.0)
        for p in result.anti_patterns:
            assert p.lift < 1.0, f"Penalty {p.name} has lift {p.lift}"

    def test_frequency_range(self, small_posts):
        from midas.analyze import analyze_posts
        result = analyze_posts(small_posts, min_frequency=0.01)
        for s in result.signals + result.anti_patterns:
            assert 0 < s.frequency <= 1.0

    def test_analysis_from_file(self):
        from midas.analyze import analyze_file
        result = analyze_file(str(SAMPLE_DATA), min_frequency=0.0)
        assert result.total_posts == 10


class TestExportConfig:
    def test_export_produces_valid_yaml(self, small_posts):
        from midas.analyze import analyze_posts, export_config
        from midas.config import load_config

        result = analyze_posts(small_posts, min_frequency=0.0)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            export_config(result, f.name)
            config = load_config(f.name)

        assert len(config.signals) > 0
        assert len(config.tiers) > 0

    def test_roundtrip_scoring(self, small_posts):
        """Config from analysis should be usable for scoring."""
        from midas.analyze import analyze_posts, export_config
        from midas.config import load_config
        from midas.scorer import score

        result = analyze_posts(small_posts, min_frequency=0.0)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            export_config(result, f.name)
            config = load_config(f.name)

        # Score the highest-engagement post — should score higher than the lowest
        high_score = score(small_posts[0]["text"], config).score
        low_score = score(small_posts[1]["text"], config).score
        assert high_score > low_score

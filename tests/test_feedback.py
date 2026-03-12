"""Tests for the MIDAS feedback system."""

import json
import tempfile
from pathlib import Path

import pytest

from midas.config import MidasConfig, SignalDef, PenaltyDef, ScoreTier


@pytest.fixture
def config():
    return MidasConfig(
        signals=[
            SignalDef(name="has_arrow", weight=100, regex="→"),
            SignalDef(name="hook_personal", weight=80, scope="hook", regex="^I[' ]"),
            SignalDef(name="cta_comment", weight=300, scope="close", keywords=["comment"]),
        ],
        penalties=[
            PenaltyDef(name="has_hashtag", weight=-60, regex=r"#\w+"),
        ],
        tiers=[
            ScoreTier(name="HIGH", min_score=200),
            ScoreTier(name="LOW", min_score=0),
        ],
        close_lines=3,
    )


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "test_feedback.jsonl")


class TestLogEdit:
    def test_basic_logging(self, config, log_path):
        from midas.feedback import log_edit

        original = "Check out my article. #AI"
        edited = "I built something new.\n\n→ It works\n\nComment YES below."

        entry = log_edit(original, edited, config, log_path=log_path)

        assert entry.edited_score > entry.original_score
        # Removing a penalty counts as "signal added" (improvement)
        assert "has_hashtag" in entry.signals_added
        assert "has_arrow" in entry.signals_added
        assert Path(log_path).exists()

    def test_log_appends(self, config, log_path):
        from midas.feedback import log_edit

        log_edit("Post one", "Post one edited", config, log_path=log_path)
        log_edit("Post two", "Post two edited", config, log_path=log_path)

        lines = Path(log_path).read_text().strip().split("\n")
        assert len(lines) == 2

    def test_log_format(self, config, log_path):
        from midas.feedback import log_edit

        log_edit("Original", "Edited", config, log_path=log_path)

        entry = json.loads(Path(log_path).read_text().strip())
        assert "timestamp" in entry
        assert "original" in entry
        assert "edited" in entry
        assert "original_score" in entry
        assert "edited_score" in entry
        assert "score_delta" in entry


class TestGetStats:
    def test_empty_log(self, log_path):
        from midas.feedback import get_stats

        Path(log_path).write_text("")
        stats = get_stats(log_path)
        assert stats.total_edits == 0

    def test_stats_from_edits(self, config, log_path):
        from midas.feedback import log_edit, get_stats

        # Log several edits that add arrows
        for i in range(3):
            log_edit(f"Post {i}", f"I wrote → something {i}\n\nComment YES.", config, log_path=log_path)

        stats = get_stats(log_path)
        assert stats.total_edits == 3
        assert stats.avg_score_improvement >= 0


class TestExportDpo:
    def test_export_with_edits(self, config, log_path, tmp_path):
        from midas.feedback import log_edit, export_dpo

        # Create edits with large score deltas
        for i in range(5):
            log_edit(
                f"Boring post {i}. #AI",
                f"I built something amazing.\n\n→ Point {i}\n\nComment YES.",
                config,
                log_path=log_path,
            )

        dpo_path = str(tmp_path / "dpo.jsonl")
        count = export_dpo(log_path=log_path, output_path=dpo_path, min_score_delta=0)

        assert count > 0
        assert Path(dpo_path).exists()

        # Verify format
        line = json.loads(Path(dpo_path).read_text().strip().split("\n")[0])
        assert "prompt" in line
        assert "chosen" in line
        assert "rejected" in line

    def test_export_filters_by_delta(self, config, log_path, tmp_path):
        from midas.feedback import log_edit, export_dpo

        # Create an edit with minimal change
        log_edit("Hello", "Hello there", config, log_path=log_path)

        dpo_path = str(tmp_path / "dpo.jsonl")
        count = export_dpo(log_path=log_path, output_path=dpo_path, min_score_delta=9999)

        assert count == 0

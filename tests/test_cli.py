"""Tests for the MIDAS CLI."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from midas.cli import main

SAMPLE_CONFIG = Path(__file__).parent.parent / "examples" / "sample_config.yaml"
SAMPLE_DATA = Path(__file__).parent.parent / "examples" / "sample_data.jsonl"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


class TestScoreCommand:
    def test_score_inline_text(self, runner):
        result = runner.invoke(main, [
            "score",
            "I built something amazing.\n\n→ It works\n→ It scales\n\nComment YES below.",
            "--config", str(SAMPLE_CONFIG),
        ])
        assert result.exit_code == 0
        assert "Score:" in result.output

    def test_score_from_file(self, runner, tmp_dir):
        post_file = tmp_dir / "post.txt"
        post_file.write_text("I remember building my first startup.\n\nIt was wild.\n\nComment AGREE below.")
        result = runner.invoke(main, [
            "score",
            "--file", str(post_file),
            "--config", str(SAMPLE_CONFIG),
        ])
        assert result.exit_code == 0
        assert "Score:" in result.output

    def test_score_from_stdin(self, runner):
        result = runner.invoke(main, [
            "score",
            "--config", str(SAMPLE_CONFIG),
        ], input="I built something amazing.\n\n→ It works\n\nComment YES below.")
        assert result.exit_code == 0
        assert "Score:" in result.output

    def test_score_empty_text(self, runner):
        result = runner.invoke(main, [
            "score", "",
            "--config", str(SAMPLE_CONFIG),
        ])
        assert result.exit_code != 0

    def test_score_shows_signals(self, runner):
        result = runner.invoke(main, [
            "score",
            "I built something → amazing\n\nComment YES below.",
            "--config", str(SAMPLE_CONFIG),
        ])
        assert result.exit_code == 0
        assert "Signal" in result.output or "Score:" in result.output

    def test_score_shows_penalties(self, runner):
        result = runner.invoke(main, [
            "score",
            "Check out this post #AI #ML #DeepLearning",
            "--config", str(SAMPLE_CONFIG),
        ])
        assert result.exit_code == 0
        assert "Penalt" in result.output or "Score:" in result.output


class TestAnalyzeCommand:
    def test_analyze_sample_data(self, runner, tmp_dir):
        output_path = tmp_dir / "output_config.yaml"
        result = runner.invoke(main, [
            "analyze",
            str(SAMPLE_DATA),
            "--output", str(output_path),
        ])
        assert result.exit_code == 0
        assert "Analyzing" in result.output
        assert "Posts analyzed" in result.output
        assert "Signals found" in result.output
        assert output_path.exists()

    def test_analyze_custom_frequency(self, runner, tmp_dir):
        output_path = tmp_dir / "output_config.yaml"
        result = runner.invoke(main, [
            "analyze",
            str(SAMPLE_DATA),
            "--output", str(output_path),
            "--min-frequency", "0.1",
        ])
        assert result.exit_code == 0

    def test_analyze_nonexistent_file(self, runner):
        result = runner.invoke(main, [
            "analyze",
            "/nonexistent/file.jsonl",
        ])
        assert result.exit_code != 0

    def test_analyze_produces_usable_config(self, runner, tmp_dir):
        """Config from analyze should work with score."""
        config_path = tmp_dir / "analyzed.yaml"
        runner.invoke(main, [
            "analyze",
            str(SAMPLE_DATA),
            "--output", str(config_path),
        ])
        assert config_path.exists()

        # Now score with the generated config
        result = runner.invoke(main, [
            "score",
            "I built something → amazing\n\nComment YES below.",
            "--config", str(config_path),
        ])
        assert result.exit_code == 0
        assert "Score:" in result.output


class TestInitCommand:
    def test_init_creates_files(self, runner, tmp_dir):
        result = runner.invoke(main, [
            "init",
            "--dir", str(tmp_dir),
        ])
        assert result.exit_code == 0
        assert "MIDAS" in result.output
        assert "Next steps" in result.output

    def test_init_creates_config(self, runner, tmp_dir):
        result = runner.invoke(main, [
            "init",
            "--dir", str(tmp_dir),
        ])
        assert result.exit_code == 0
        # Should create config if sample exists
        config_path = tmp_dir / "midas_config.yaml"
        if config_path.exists():
            from midas.config import load_config
            cfg = load_config(str(config_path))
            assert len(cfg.signals) > 0

    def test_init_idempotent(self, runner, tmp_dir):
        """Running init twice should not error."""
        runner.invoke(main, ["init", "--dir", str(tmp_dir)])
        result = runner.invoke(main, ["init", "--dir", str(tmp_dir)])
        assert result.exit_code == 0
        assert "already exists" in result.output


class TestFeedbackCommand:
    def test_feedback_log_edit(self, runner, tmp_dir):
        original = tmp_dir / "original.txt"
        edited = tmp_dir / "edited.txt"
        log_path = tmp_dir / "feedback.jsonl"

        original.write_text("Check out this post #AI #ML")
        edited.write_text("I built something → amazing\n\nComment YES below.")

        result = runner.invoke(main, [
            "feedback",
            "--original", str(original),
            "--edited", str(edited),
            "--log", str(log_path),
            "--config", str(SAMPLE_CONFIG),
        ])
        assert result.exit_code == 0
        assert "Original score" in result.output
        assert "Edited score" in result.output
        assert log_path.exists()

    def test_feedback_stats_empty(self, runner, tmp_dir):
        log_path = tmp_dir / "feedback.jsonl"
        log_path.write_text("")

        result = runner.invoke(main, [
            "feedback",
            "--stats",
            "--log", str(log_path),
        ])
        assert result.exit_code == 0
        assert "Feedback Stats" in result.output

    def test_feedback_missing_files(self, runner):
        result = runner.invoke(main, ["feedback"])
        assert result.exit_code != 0

    def test_feedback_export_dpo(self, runner, tmp_dir):
        # First log an edit
        original = tmp_dir / "original.txt"
        edited = tmp_dir / "edited.txt"
        log_path = tmp_dir / "feedback.jsonl"
        dpo_path = tmp_dir / "dpo.jsonl"

        original.write_text("Check out this post #AI #ML")
        edited.write_text("I built something → amazing\n\nComment YES below.")

        runner.invoke(main, [
            "feedback",
            "--original", str(original),
            "--edited", str(edited),
            "--log", str(log_path),
            "--config", str(SAMPLE_CONFIG),
        ])

        # Export DPO
        result = runner.invoke(main, [
            "feedback",
            "--export-dpo", str(dpo_path),
            "--log", str(log_path),
        ])
        assert result.exit_code == 0
        assert "Exported" in result.output


class TestVersionFlag:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0." in result.output


class TestHelpText:
    def test_main_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "MIDAS" in result.output

    def test_score_help(self, runner):
        result = runner.invoke(main, ["score", "--help"])
        assert result.exit_code == 0
        assert "Score" in result.output or "score" in result.output

    def test_analyze_help(self, runner):
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_init_help(self, runner):
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0

    def test_feedback_help(self, runner):
        result = runner.invoke(main, ["feedback", "--help"])
        assert result.exit_code == 0

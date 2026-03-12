"""Tests for the MIDAS scoring engine."""

from pathlib import Path

import pytest

from midas.config import load_config, MidasConfig, SignalDef, PenaltyDef, ScoreTier
from midas.scorer import score, score_text

SAMPLE_CONFIG = Path(__file__).parent.parent / "examples" / "sample_config.yaml"


@pytest.fixture
def config():
    return load_config(SAMPLE_CONFIG)


@pytest.fixture
def minimal_config():
    return MidasConfig(
        signals=[
            SignalDef(name="has_arrow", weight=100, regex="→"),
            SignalDef(name="is_long", weight=50, field="char_count", min_value=200),
            SignalDef(name="hook_personal", weight=80, scope="hook", regex="^I[' ]"),
        ],
        penalties=[
            PenaltyDef(name="has_hashtag", weight=-60, regex=r"#\w+"),
        ],
        tiers=[
            ScoreTier(name="HIGH", min_score=150),
            ScoreTier(name="MED", min_score=50),
            ScoreTier(name="LOW", min_score=0),
        ],
        suggestions={"has_arrow": "Use → arrows for better formatting"},
    )


class TestScore:
    def test_empty_post(self, config):
        result = score("", config)
        # Empty string has hook_length=0 which triggers hook_short_teaser
        assert "hook_short_teaser" in result.signals
        assert result.penalties == {}

    def test_basic_scoring(self, minimal_config):
        text = "I built something new.\n\n→ Feature one\n→ Feature two\n" + "x" * 200
        result = score(text, minimal_config)
        assert result.signals["has_arrow"] == 100
        assert result.signals["hook_personal"] == 80
        assert result.signals["is_long"] == 50
        assert result.score == 230

    def test_penalties_subtract(self, minimal_config):
        text = "Check this out!\n\n#AI #ML"
        result = score(text, minimal_config)
        assert "has_hashtag" in result.penalties
        assert result.penalty_total < 0
        assert result.score < 0

    def test_tier_assignment(self, minimal_config):
        # High tier
        text = "I built → something" + "x" * 200
        result = score(text, minimal_config)
        assert result.tier == "HIGH"

        # Low tier
        result2 = score("Hello world", minimal_config)
        assert result2.tier == "LOW"

    def test_suggestions_for_missing_signals(self, minimal_config):
        text = "Hello world"
        result = score(text, minimal_config)
        assert any("arrow" in s.lower() for s in result.suggestions)

    def test_no_suggestions_for_matched_signals(self, minimal_config):
        text = "I love → arrows" + "x" * 200
        result = score(text, minimal_config)
        # has_arrow matched, so its suggestion shouldn't appear
        assert not any("arrow" in s.lower() for s in result.suggestions)

    def test_hook_detection(self, minimal_config):
        text = "I started a company.\n\nIt was wild."
        result = score(text, minimal_config)
        assert "hook_personal" in result.signals

    def test_hook_no_match(self, minimal_config):
        text = "The company started.\n\nIt was wild."
        result = score(text, minimal_config)
        assert "hook_personal" not in result.signals

    def test_close_scope(self):
        config = MidasConfig(
            signals=[
                SignalDef(name="cta_comment", weight=300, scope="close", keywords=["comment"]),
            ],
            close_lines=3,
        )
        text = "Hook line\n\nBody\n\nMore body\n\nComment MIDAS below."
        result = score(text, config)
        assert "cta_comment" in result.signals

    def test_score_text_returns_number(self, config):
        result = score_text("Just a test post", config)
        assert isinstance(result, (int, float))

    def test_stats_populated(self, config):
        text = "Test post with some content\n\nAnd multiple lines"
        result = score(text, config)
        assert result.stats["char_count"] > 0
        assert result.stats["line_count"] >= 2
        assert result.stats["newline_count"] >= 1

    def test_str_output(self, minimal_config):
        text = "I built → something" + "x" * 200
        result = score(text, minimal_config)
        output = str(result)
        assert "Score:" in output
        assert "has_arrow" in output


class TestSampleConfig:
    def test_loads_successfully(self):
        config = load_config(SAMPLE_CONFIG)
        assert config.name == "sample"
        assert len(config.signals) > 0
        assert len(config.penalties) > 0
        assert len(config.tiers) > 0

    def test_high_scoring_post(self, config):
        """A post with many signals should score high."""
        text = (
            "I just made a discovery.\n\n"
            "Back in 2020, I remember thinking AI was overhyped.\n\n"
            "But here's the thing → the data tells a different story.\n\n"
            "→ 73% of companies now use AI daily\n"
            "→ Developer productivity up 2.5x\n"
            "→ Cost per inference down 90%\n\n"
            + "\n" * 20
            + "\nComment AGREE if you've seen this too."
        )
        result = score(text, config)
        assert result.score > 200

    def test_low_scoring_post(self, config):
        """A post with hashtags and no signals should score low."""
        text = "Check out this article about AI. #AI #ML #DeepLearning"
        result = score(text, config)
        assert result.score < 50

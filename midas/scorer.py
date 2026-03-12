"""Config-driven scoring engine for social media posts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import MidasConfig, SignalDef


@dataclass
class ScoreResult:
    """Detailed scoring breakdown."""

    score: float
    tier: str
    tier_description: str
    signals: dict[str, float]  # signal_name -> weight
    penalties: dict[str, float]  # penalty_name -> weight (negative)
    signal_total: float
    penalty_total: float
    suggestions: list[str]
    stats: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"Score: {self.score:.0f} — {self.tier}"]
        if self.signals:
            parts.append("  Signals:")
            for name, w in sorted(self.signals.items(), key=lambda x: -x[1]):
                parts.append(f"    +{w:.0f}  {name}")
        if self.penalties:
            parts.append("  Penalties:")
            for name, w in sorted(self.penalties.items(), key=lambda x: x[1]):
                parts.append(f"    {w:.0f}  {name}")
        if self.suggestions:
            parts.append("  Quick wins:")
            for s in self.suggestions:
                parts.append(f"    → {s}")
        return "\n".join(parts)


def _compute_stats(text: str, hook_max_chars: int) -> dict[str, float]:
    """Extract numeric stats from a post."""
    lines = text.split("\n")
    hook = lines[0] if lines else ""
    return {
        "char_count": len(text),
        "line_count": len(lines),
        "newline_count": text.count("\n"),
        "word_count": len(text.split()),
        "hook_length": len(hook),
    }


def score(text: str, config: MidasConfig) -> ScoreResult:
    """Score a post against a config and return a detailed breakdown."""
    lines = text.strip().split("\n")
    hook = lines[0] if lines else ""
    close_lines = lines[-config.close_lines :] if len(lines) >= config.close_lines else lines
    close = "\n".join(close_lines)
    stats = _compute_stats(text, config.hook_max_chars)

    # Evaluate signals
    matched_signals: dict[str, float] = {}
    unmatched_signals: list[str] = []

    for sig in config.signals:
        # Special case: hook_short_teaser checks < threshold, not >=
        if sig.name == "hook_short_teaser":
            if stats.get("hook_length", 999) < 50:
                matched_signals[sig.name] = sig.weight
            else:
                unmatched_signals.append(sig.name)
            continue

        if sig.matches(text, hook=hook, close=close, stats=stats):
            matched_signals[sig.name] = sig.weight
        else:
            unmatched_signals.append(sig.name)

    # Evaluate penalties
    matched_penalties: dict[str, float] = {}
    for pen in config.penalties:
        if pen.matches(text, hook=hook, close=close):
            matched_penalties[pen.name] = pen.weight

    signal_total = sum(matched_signals.values())
    penalty_total = sum(matched_penalties.values())
    total = signal_total + penalty_total

    # Determine tier
    tier_name = "UNRANKED"
    tier_desc = ""
    for t in config.sorted_tiers:
        if total >= t.min_score:
            tier_name = t.name
            tier_desc = t.description
            break

    # Generate suggestions for missing signals
    suggestions: list[str] = []
    for sig_name in unmatched_signals:
        if sig_name in config.suggestions:
            suggestions.append(config.suggestions[sig_name])

    # Also suggest removing active penalties
    for pen_name in matched_penalties:
        if pen_name in config.suggestions:
            suggestions.append(config.suggestions[pen_name])

    return ScoreResult(
        score=total,
        tier=tier_name,
        tier_description=tier_desc,
        signals=matched_signals,
        penalties=matched_penalties,
        signal_total=signal_total,
        penalty_total=penalty_total,
        suggestions=suggestions,
        stats=stats,
    )


def score_text(text: str, config: MidasConfig) -> float:
    """Score a post and return just the numeric score."""
    return score(text, config).score

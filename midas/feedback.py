"""Feedback loop and edit logging for MIDAS.

Tracks before/after edits so users can learn from their editing patterns,
analyze which signal changes actually improve scores, track editing skill
over time, and export preference pairs in DPO format for fine-tuning.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import MidasConfig
from .scorer import ScoreResult, score


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EditLog:
    """A single logged edit with scoring metadata."""

    timestamp: str
    original: str
    edited: str
    original_score: float
    edited_score: float
    score_delta: float
    signals_added: list[str]
    signals_removed: list[str]

    def __str__(self) -> str:
        direction = "+" if self.score_delta >= 0 else ""
        return (
            f"[{self.timestamp}] {direction}{self.score_delta:.0f} "
            f"({self.original_score:.0f} -> {self.edited_score:.0f})  "
            f"added={self.signals_added}  removed={self.signals_removed}"
        )


@dataclass
class SignalWinRate:
    """How often adding or removing a specific signal improved the score."""

    signal_name: str
    times_added: int
    times_adding_improved: int
    avg_delta_when_added: float
    times_removed: int
    times_removing_improved: int
    avg_delta_when_removed: float

    @property
    def add_win_rate(self) -> float:
        return self.times_adding_improved / self.times_added if self.times_added > 0 else 0.0

    @property
    def remove_win_rate(self) -> float:
        return self.times_removing_improved / self.times_removed if self.times_removed > 0 else 0.0


@dataclass
class FeedbackStats:
    """Aggregate statistics from the feedback log."""

    total_edits: int
    avg_score_improvement: float
    most_commonly_added: list[tuple[str, int]]
    most_commonly_removed: list[tuple[str, int]]
    editing_patterns: str
    # Extended stats
    signal_win_rates: list[SignalWinRate] = field(default_factory=list)
    improvement_rate: float = 0.0  # % of edits that improved score
    streak: int = 0  # Current consecutive improvements
    best_streak: int = 0
    skill_trend: str = ""  # "improving", "stable", "declining"

    def __str__(self) -> str:
        parts = [
            f"Total edits: {self.total_edits}",
            f"Avg score improvement: {self.avg_score_improvement:+.1f}",
            f"Improvement rate: {self.improvement_rate:.0%}",
        ]

        if self.streak > 0:
            parts.append(f"Current streak: {self.streak} consecutive improvements")
        if self.best_streak > 1:
            parts.append(f"Best streak: {self.best_streak}")
        if self.skill_trend:
            parts.append(f"Skill trend: {self.skill_trend}")

        if self.most_commonly_added:
            parts.append("\nMost commonly added signals:")
            for name, count in self.most_commonly_added:
                parts.append(f"  + {name} ({count}x)")

        if self.most_commonly_removed:
            parts.append("Most commonly removed signals:")
            for name, count in self.most_commonly_removed:
                parts.append(f"  - {name} ({count}x)")

        if self.signal_win_rates:
            parts.append("\nSignal win rates (how often each change improved the score):")
            for swr in self.signal_win_rates:
                if swr.times_added > 0:
                    parts.append(
                        f"  + {swr.signal_name}: {swr.add_win_rate:.0%} win rate "
                        f"({swr.times_adding_improved}/{swr.times_added}), "
                        f"avg delta {swr.avg_delta_when_added:+.0f}"
                    )
                if swr.times_removed > 0:
                    parts.append(
                        f"  - {swr.signal_name}: {swr.remove_win_rate:.0%} win rate "
                        f"({swr.times_removing_improved}/{swr.times_removed}), "
                        f"avg delta {swr.avg_delta_when_removed:+.0f}"
                    )

        if self.editing_patterns:
            parts.append(f"\nPatterns: {self.editing_patterns}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _compute_signal_diff(
    original_result: ScoreResult,
    edited_result: ScoreResult,
) -> tuple[list[str], list[str]]:
    """Compare two ScoreResults and return (signals_added, signals_removed).

    Considers both positive signals and penalties.  A penalty removed counts
    as a "signal added" (improvement), and a penalty gained counts as a
    "signal removed" (regression).
    """
    orig_signals = set(original_result.signals.keys())
    edit_signals = set(edited_result.signals.keys())

    orig_penalties = set(original_result.penalties.keys())
    edit_penalties = set(edited_result.penalties.keys())

    # Positive signals gained or penalties eliminated
    signals_added = sorted(
        (edit_signals - orig_signals) | (orig_penalties - edit_penalties)
    )

    # Positive signals lost or penalties introduced
    signals_removed = sorted(
        (orig_signals - edit_signals) | (edit_penalties - orig_penalties)
    )

    return signals_added, signals_removed


def log_edit(
    original: str,
    edited: str,
    config: MidasConfig,
    log_path: str = "midas_feedback.jsonl",
) -> EditLog:
    """Score both versions of a post and append the edit to a JSONL log."""
    original_result = score(original, config)
    edited_result = score(edited, config)

    signals_added, signals_removed = _compute_signal_diff(
        original_result, edited_result
    )

    entry = EditLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        original=original,
        edited=edited,
        original_score=original_result.score,
        edited_score=edited_result.score,
        score_delta=edited_result.score - original_result.score,
        signals_added=signals_added,
        signals_removed=signals_removed,
    )

    # Serialize and append to the log file
    record = {
        "timestamp": entry.timestamp,
        "original": entry.original,
        "edited": entry.edited,
        "original_score": entry.original_score,
        "edited_score": entry.edited_score,
        "score_delta": entry.score_delta,
        "signals_added": entry.signals_added,
        "signals_removed": entry.signals_removed,
    }

    path = Path(log_path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return entry


def _load_entries(log_path: str) -> list[dict]:
    """Load entries from a feedback log file."""
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No feedback log found at '{log_path}'. "
            "Log some edits first with log_edit()."
        )

    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _compute_signal_win_rates(entries: list[dict]) -> list[SignalWinRate]:
    """Compute per-signal win rates from edit history."""
    # Track per-signal: additions and their deltas, removals and their deltas
    add_deltas: dict[str, list[float]] = defaultdict(list)
    remove_deltas: dict[str, list[float]] = defaultdict(list)

    for e in entries:
        delta = e.get("score_delta", 0)
        for sig in e.get("signals_added", []):
            add_deltas[sig].append(delta)
        for sig in e.get("signals_removed", []):
            remove_deltas[sig].append(delta)

    all_signals = set(add_deltas.keys()) | set(remove_deltas.keys())
    win_rates: list[SignalWinRate] = []

    for sig in sorted(all_signals):
        adds = add_deltas.get(sig, [])
        removes = remove_deltas.get(sig, [])

        win_rates.append(SignalWinRate(
            signal_name=sig,
            times_added=len(adds),
            times_adding_improved=sum(1 for d in adds if d > 0),
            avg_delta_when_added=statistics.mean(adds) if adds else 0.0,
            times_removed=len(removes),
            times_removing_improved=sum(1 for d in removes if d > 0),
            avg_delta_when_removed=statistics.mean(removes) if removes else 0.0,
        ))

    # Sort by most impactful (highest add win rate)
    win_rates.sort(key=lambda x: -(x.add_win_rate * x.times_added + x.remove_win_rate * x.times_removed))
    return win_rates


def _compute_streaks(deltas: list[float]) -> tuple[int, int]:
    """Compute current streak and best streak of consecutive improvements."""
    current = 0
    best = 0

    for d in deltas:
        if d > 0:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return current, best


def _compute_skill_trend(deltas: list[float], window: int = 5) -> str:
    """Detect whether editing skill is improving over time.

    Compares the average delta of the most recent `window` edits
    to the average of all prior edits.
    """
    if len(deltas) < window * 2:
        return "not enough data"

    recent = deltas[-window:]
    earlier = deltas[:-window]

    recent_avg = statistics.mean(recent)
    earlier_avg = statistics.mean(earlier)

    diff = recent_avg - earlier_avg
    if diff > 20:
        return "improving (recent edits are stronger)"
    elif diff < -20:
        return "declining (recent edits are weaker)"
    else:
        return "stable"


def get_stats(log_path: str = "midas_feedback.jsonl") -> FeedbackStats:
    """Read the feedback log and compute aggregate editing statistics."""
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No feedback log found at '{log_path}'. "
            "Log some edits first with log_edit()."
        )

    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        return FeedbackStats(
            total_edits=0,
            avg_score_improvement=0.0,
            most_commonly_added=[],
            most_commonly_removed=[],
            editing_patterns="No edits recorded yet.",
        )

    total = len(entries)
    deltas = [e["score_delta"] for e in entries]
    avg_improvement = sum(deltas) / total

    # Count signal additions and removals
    added_counter: Counter[str] = Counter()
    removed_counter: Counter[str] = Counter()
    for e in entries:
        added_counter.update(e.get("signals_added", []))
        removed_counter.update(e.get("signals_removed", []))

    most_added = added_counter.most_common(10)
    most_removed = removed_counter.most_common(10)

    # Extended stats
    positive_edits = sum(1 for d in deltas if d > 0)
    negative_edits = sum(1 for d in deltas if d < 0)
    neutral_edits = sum(1 for d in deltas if d == 0)
    improvement_rate = positive_edits / total if total > 0 else 0.0

    current_streak, best_streak = _compute_streaks(deltas)
    skill_trend = _compute_skill_trend(deltas)
    signal_win_rates = _compute_signal_win_rates(entries)

    # Build patterns summary
    patterns_parts: list[str] = []
    patterns_parts.append(f"{positive_edits}/{total} edits improved the score")
    if negative_edits:
        patterns_parts.append(f"{negative_edits}/{total} edits lowered the score")
    if neutral_edits:
        patterns_parts.append(f"{neutral_edits}/{total} edits had no effect")

    if most_added:
        top_add = most_added[0][0]
        patterns_parts.append(f"You most often add '{top_add}' when editing")
    if most_removed:
        top_rm = most_removed[0][0]
        patterns_parts.append(f"You most often remove '{top_rm}' when editing")

    return FeedbackStats(
        total_edits=total,
        avg_score_improvement=avg_improvement,
        most_commonly_added=most_added,
        most_commonly_removed=most_removed,
        editing_patterns=". ".join(patterns_parts) + ".",
        signal_win_rates=signal_win_rates,
        improvement_rate=improvement_rate,
        streak=current_streak,
        best_streak=best_streak,
        skill_trend=skill_trend,
    )


def export_dpo(
    log_path: str = "midas_feedback.jsonl",
    output_path: str = "dpo_from_edits.jsonl",
    min_score_delta: float = 50,
) -> int:
    """Export edit logs as DPO preference pairs for fine-tuning.

    Uses signal-aware prompts derived from the edit context instead of a
    generic prompt.  Each pair includes the specific signals that were
    added/removed, making DPO training more targeted.

    Parameters
    ----------
    log_path : str
        Path to the JSONL feedback log file.
    output_path : str
        Path for the output DPO JSONL file.
    min_score_delta : float
        Minimum score improvement to include as a preference pair.

    Returns
    -------
    int
        Number of DPO pairs exported.
    """
    entries = _load_entries(log_path)

    count = 0
    out_path = Path(output_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            delta = entry.get("score_delta", 0)
            if delta < min_score_delta:
                continue

            # Build a signal-aware prompt from the edit context
            signals_added = entry.get("signals_added", [])
            signals_removed = entry.get("signals_removed", [])

            prompt_parts = ["Write an engaging LinkedIn post."]
            if signals_added:
                prompt_parts.append(
                    "Include: " + ", ".join(
                        s.replace("_", " ") for s in signals_added
                    ) + "."
                )
            if signals_removed:
                prompt_parts.append(
                    "Avoid: " + ", ".join(
                        s.replace("_", " ") for s in signals_removed
                    ) + "."
                )

            # Extract topic hint from the first line of the edited text
            edited_text = entry.get("edited", "")
            first_line = edited_text.split("\n")[0][:100] if edited_text else ""
            if first_line:
                prompt_parts.append(f"Topic context: \"{first_line}\"")

            dpo_pair = {
                "prompt": [
                    {"role": "system", "content": "You are a LinkedIn post writer."},
                    {"role": "user", "content": " ".join(prompt_parts)},
                ],
                "chosen": [
                    {"role": "assistant", "content": entry["edited"]},
                ],
                "rejected": [
                    {"role": "assistant", "content": entry["original"]},
                ],
                "chosen_score": entry["edited_score"],
                "rejected_score": entry["original_score"],
                "score_delta": delta,
                "signals_added": signals_added,
                "signals_removed": signals_removed,
                "timestamp": entry.get("timestamp", ""),
            }
            f.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")
            count += 1

    return count


def get_signal_impact_matrix(
    log_path: str = "midas_feedback.jsonl",
) -> dict[str, dict[str, float]]:
    """Build a signal impact matrix from editing history.

    Returns a dict mapping each signal to its impact stats:
    {
        "cta_comment": {
            "times_added": 5,
            "avg_delta_when_added": +120.0,
            "add_win_rate": 0.8,
            "times_removed": 1,
            "avg_delta_when_removed": -50.0,
        },
        ...
    }
    """
    entries = _load_entries(log_path)
    win_rates = _compute_signal_win_rates(entries)

    matrix: dict[str, dict[str, float]] = {}
    for swr in win_rates:
        matrix[swr.signal_name] = {
            "times_added": swr.times_added,
            "avg_delta_when_added": swr.avg_delta_when_added,
            "add_win_rate": swr.add_win_rate,
            "times_removed": swr.times_removed,
            "avg_delta_when_removed": swr.avg_delta_when_removed,
            "remove_win_rate": swr.remove_win_rate,
        }

    return matrix

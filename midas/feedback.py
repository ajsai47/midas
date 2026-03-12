"""Feedback loop and edit logging for MIDAS.

Tracks before/after edits so users can learn from their editing patterns
and export preference pairs in DPO format for fine-tuning.
"""

from __future__ import annotations

import json
from collections import Counter
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
class FeedbackStats:
    """Aggregate statistics from the feedback log."""

    total_edits: int
    avg_score_improvement: float
    most_commonly_added: list[tuple[str, int]]
    most_commonly_removed: list[tuple[str, int]]
    editing_patterns: str

    def __str__(self) -> str:
        parts = [
            f"Total edits: {self.total_edits}",
            f"Avg score improvement: {self.avg_score_improvement:+.1f}",
        ]

        if self.most_commonly_added:
            parts.append("Most commonly added signals:")
            for name, count in self.most_commonly_added:
                parts.append(f"  + {name} ({count}x)")

        if self.most_commonly_removed:
            parts.append("Most commonly removed signals:")
            for name, count in self.most_commonly_removed:
                parts.append(f"  - {name} ({count}x)")

        if self.editing_patterns:
            parts.append(f"Patterns: {self.editing_patterns}")

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

    Parameters
    ----------
    original_result : ScoreResult
        Scoring breakdown of the original text.
    edited_result : ScoreResult
        Scoring breakdown of the edited text.

    Returns
    -------
    tuple[list[str], list[str]]
        (signals_added, signals_removed) where "added" means the edited
        version gained a positive signal or lost a penalty, and "removed"
        means the opposite.
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
    """Score both versions of a post and append the edit to a JSONL log.

    Parameters
    ----------
    original : str
        The original post text (before editing).
    edited : str
        The edited post text (after editing).
    config : MidasConfig
        The scoring configuration to evaluate both versions.
    log_path : str
        Path to the JSONL feedback log file.  Created if it does not exist.

    Returns
    -------
    EditLog
        The recorded edit entry with full scoring metadata.
    """
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


def get_stats(log_path: str = "midas_feedback.jsonl") -> FeedbackStats:
    """Read the feedback log and compute aggregate editing statistics.

    Parameters
    ----------
    log_path : str
        Path to the JSONL feedback log file.

    Returns
    -------
    FeedbackStats
        Aggregate statistics about the user's editing behaviour.

    Raises
    ------
    FileNotFoundError
        If the log file does not exist.
    """
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

    # Count signal additions and removals across all edits
    added_counter: Counter[str] = Counter()
    removed_counter: Counter[str] = Counter()
    for e in entries:
        added_counter.update(e.get("signals_added", []))
        removed_counter.update(e.get("signals_removed", []))

    most_added = added_counter.most_common(10)
    most_removed = removed_counter.most_common(10)

    # Build a human-readable patterns summary
    patterns_parts: list[str] = []

    positive_edits = sum(1 for d in deltas if d > 0)
    negative_edits = sum(1 for d in deltas if d < 0)
    neutral_edits = sum(1 for d in deltas if d == 0)

    patterns_parts.append(
        f"{positive_edits}/{total} edits improved the score"
    )
    if negative_edits:
        patterns_parts.append(
            f"{negative_edits}/{total} edits lowered the score"
        )
    if neutral_edits:
        patterns_parts.append(
            f"{neutral_edits}/{total} edits had no effect"
        )

    if most_added:
        top_add = most_added[0][0]
        patterns_parts.append(
            f"You most often add '{top_add}' when editing"
        )
    if most_removed:
        top_rm = most_removed[0][0]
        patterns_parts.append(
            f"You most often remove '{top_rm}' when editing"
        )

    return FeedbackStats(
        total_edits=total,
        avg_score_improvement=avg_improvement,
        most_commonly_added=most_added,
        most_commonly_removed=most_removed,
        editing_patterns=". ".join(patterns_parts) + ".",
    )


def export_dpo(
    log_path: str = "midas_feedback.jsonl",
    output_path: str = "dpo_from_edits.jsonl",
    min_score_delta: float = 50,
) -> int:
    """Export edit logs as DPO preference pairs for fine-tuning.

    Converts each qualifying edit into a preference pair where the edited
    version is "chosen" and the original is "rejected".  Only includes
    pairs where the score improvement meets the ``min_score_delta``
    threshold.

    Parameters
    ----------
    log_path : str
        Path to the JSONL feedback log file.
    output_path : str
        Path for the output DPO JSONL file.
    min_score_delta : float
        Minimum score improvement required to include an edit as a
        preference pair.  Defaults to 50 points.

    Returns
    -------
    int
        Number of DPO pairs exported.

    Raises
    ------
    FileNotFoundError
        If the log file does not exist.
    """
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

    count = 0
    out_path = Path(output_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            delta = entry.get("score_delta", 0)
            if delta < min_score_delta:
                continue

            dpo_pair = {
                "prompt": "Write an engaging LinkedIn post.",
                "chosen": entry["edited"],
                "rejected": entry["original"],
                "chosen_score": entry["edited_score"],
                "rejected_score": entry["original_score"],
                "score_delta": delta,
                "timestamp": entry.get("timestamp", ""),
            }
            f.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")
            count += 1

    return count

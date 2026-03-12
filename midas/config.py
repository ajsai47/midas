"""YAML-based configuration for scoring formulas."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class SignalDef:
    """A named scoring signal with detection logic and weight."""

    name: str
    weight: float
    description: str = ""
    # Detection can be: regex pattern, keyword list, or callable
    regex: str | None = None
    keywords: list[str] | None = None
    scope: str = "full"  # "full", "hook", "close"
    min_value: float | None = None  # For numeric checks (e.g., char_count >= 1000)
    field: str | None = None  # Which numeric field to check

    def matches(self, text: str, *, hook: str = "", close: str = "", stats: dict | None = None) -> bool:
        target = {"full": text, "hook": hook, "close": close}.get(self.scope, text)

        if self.field and self.min_value is not None and stats:
            return stats.get(self.field, 0) >= self.min_value

        if self.regex:
            return bool(re.search(self.regex, target, re.IGNORECASE))

        if self.keywords:
            lower = target.lower()
            return any(kw in lower for kw in self.keywords)

        return False


@dataclass
class PenaltyDef:
    """A penalty rule (negative signal)."""

    name: str
    weight: float  # Should be negative
    description: str = ""
    regex: str | None = None
    keywords: list[str] | None = None
    scope: str = "full"

    def matches(self, text: str, *, hook: str = "", close: str = "") -> bool:
        target = {"full": text, "hook": hook, "close": close}.get(self.scope, text)

        if self.regex:
            return bool(re.search(self.regex, target, re.IGNORECASE))

        if self.keywords:
            lower = target.lower()
            return any(kw in lower for kw in self.keywords)

        return False


@dataclass
class ScoreTier:
    """A named tier with a minimum score threshold."""

    name: str
    min_score: float
    description: str = ""


@dataclass
class MidasConfig:
    """Complete scoring configuration."""

    name: str = "default"
    description: str = ""
    signals: list[SignalDef] = field(default_factory=list)
    penalties: list[PenaltyDef] = field(default_factory=list)
    tiers: list[ScoreTier] = field(default_factory=list)
    hook_max_chars: int = 100
    close_lines: int = 3
    suggestions: dict[str, str] = field(default_factory=dict)

    @property
    def sorted_tiers(self) -> list[ScoreTier]:
        return sorted(self.tiers, key=lambda t: t.min_score, reverse=True)


def _parse_signal(data: dict) -> SignalDef:
    return SignalDef(
        name=data["name"],
        weight=data["weight"],
        description=data.get("description", ""),
        regex=data.get("regex"),
        keywords=data.get("keywords"),
        scope=data.get("scope", "full"),
        min_value=data.get("min_value"),
        field=data.get("field"),
    )


def _parse_penalty(data: dict) -> PenaltyDef:
    return PenaltyDef(
        name=data["name"],
        weight=-abs(data["weight"]),  # Ensure negative
        description=data.get("description", ""),
        regex=data.get("regex"),
        keywords=data.get("keywords"),
        scope=data.get("scope", "full"),
    )


def _parse_tier(data: dict) -> ScoreTier:
    return ScoreTier(
        name=data["name"],
        min_score=data["min_score"],
        description=data.get("description", ""),
    )


def load_config(path: str | Path) -> MidasConfig:
    """Load a MIDAS config from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    suggestions = {}
    for s in raw.get("suggestions", []):
        suggestions[s["signal"]] = s["message"]

    return MidasConfig(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        signals=[_parse_signal(s) for s in raw.get("signals", [])],
        penalties=[_parse_penalty(p) for p in raw.get("penalties", [])],
        tiers=[_parse_tier(t) for t in raw.get("tiers", [])],
        hook_max_chars=raw.get("hook_max_chars", 100),
        close_lines=raw.get("close_lines", 3),
        suggestions=suggestions,
    )


def save_config(config: MidasConfig, path: str | Path) -> None:
    """Save a MIDAS config to a YAML file."""
    path = Path(path)

    data: dict[str, Any] = {
        "name": config.name,
        "description": config.description,
        "hook_max_chars": config.hook_max_chars,
        "close_lines": config.close_lines,
    }

    data["signals"] = []
    for s in config.signals:
        entry: dict[str, Any] = {"name": s.name, "weight": s.weight}
        if s.description:
            entry["description"] = s.description
        if s.regex:
            entry["regex"] = s.regex
        if s.keywords:
            entry["keywords"] = s.keywords
        if s.scope != "full":
            entry["scope"] = s.scope
        if s.min_value is not None:
            entry["min_value"] = s.min_value
        if s.field:
            entry["field"] = s.field
        data["signals"].append(entry)

    data["penalties"] = []
    for p in config.penalties:
        entry = {"name": p.name, "weight": abs(p.weight)}
        if p.description:
            entry["description"] = p.description
        if p.regex:
            entry["regex"] = p.regex
        if p.keywords:
            entry["keywords"] = p.keywords
        if p.scope != "full":
            entry["scope"] = p.scope
        data["penalties"].append(entry)

    data["tiers"] = []
    for t in config.sorted_tiers:
        entry = {"name": t.name, "min_score": t.min_score}
        if t.description:
            entry["description"] = t.description
        data["tiers"].append(entry)

    if config.suggestions:
        data["suggestions"] = [
            {"signal": k, "message": v} for k, v in config.suggestions.items()
        ]

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

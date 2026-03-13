"""Signal analysis engine — derive a scoring formula from your post history.

Given a JSONL file of posts with engagement metrics, this module computes the
empirical lift of each candidate signal (engagement when present vs absent),
tests each signal for statistical significance (Mann-Whitney U), computes
bootstrap confidence intervals, and applies Benjamini-Hochberg FDR correction
to control for multiple comparisons.

Usage:
    from midas.analyze import analyze_file, export_config

    result = analyze_file("my_posts.jsonl")
    print(result)
    export_config(result, "my_config.yaml")
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import MidasConfig, PenaltyDef, ScoreTier, SignalDef, save_config


# ---------------------------------------------------------------------------
# Post engagement helper
# ---------------------------------------------------------------------------

def _engagement(post: dict) -> float:
    """Compute total weighted engagement for a post.

    Formula: reactions + comments*2 + reposts*3
    Comments and reposts are weighted higher because they require more effort
    and indicate stronger resonance.
    """
    return (
        post.get("reactions", 0)
        + post.get("comments", 0) * 2
        + post.get("reposts", 0) * 3
    )


# ---------------------------------------------------------------------------
# Candidate signal definitions
# ---------------------------------------------------------------------------

@dataclass
class _CandidateSignal:
    """Internal representation of a signal to test against the data."""

    name: str
    description: str
    detect: Callable[[str, str, str, dict[str, float]], bool]
    scope: str = "full"
    # These are populated after detection so we can reconstruct SignalDef/PenaltyDef
    regex: str | None = None
    keywords: list[str] | None = None
    field_name: str | None = None
    min_value: float | None = None


def _build_candidates(hook_max_chars: int) -> list[_CandidateSignal]:
    """Build the built-in set of candidate signals to test.

    Each candidate has a detection function that receives
    (text, hook, close, stats) and returns True/False.
    """
    candidates: list[_CandidateSignal] = []

    # ----- Hook patterns -----
    candidates.append(_CandidateSignal(
        name="hook_personal_i",
        description="Starting with 'I' signals personal story",
        scope="hook",
        regex="^I[' ]",
        detect=lambda text, hook, close, stats: bool(re.search(r"^I[' ]", hook)),
    ))

    candidates.append(_CandidateSignal(
        name="hook_short_teaser",
        description="Short punchy hook under 50 chars creates curiosity gap",
        scope="hook",
        field_name="hook_length",
        min_value=0,  # special: triggers when < 50
        detect=lambda text, hook, close, stats: len(hook) < 50 and len(hook) > 0,
    ))

    candidates.append(_CandidateSignal(
        name="hook_number",
        description="Numbers in hook promise specificity",
        scope="hook",
        regex=r"^[\d$]",
        detect=lambda text, hook, close, stats: bool(re.search(r"^[\d$]", hook)),
    ))

    candidates.append(_CandidateSignal(
        name="hook_exclamation",
        description="Emotional opener grabs attention",
        scope="hook",
        regex=r"^(Wow|Well|Wait|Whoa|Holy)",
        detect=lambda text, hook, close, stats: bool(
            re.search(r"^(Wow|Well|Wait|Whoa|Holy)", hook)
        ),
    ))

    candidates.append(_CandidateSignal(
        name="hook_question",
        description="Question hook — may under- or over-perform",
        scope="hook",
        regex=r"\?$",
        detect=lambda text, hook, close, stats: bool(re.search(r"\?$", hook)),
    ))

    candidates.append(_CandidateSignal(
        name="hook_superlative",
        description="Superlatives create intrigue",
        scope="hook",
        keywords=["most", "biggest", "first", "largest", "best", "worst"],
        detect=lambda text, hook, close, stats: any(
            kw in hook.lower() for kw in ["most", "biggest", "first", "largest", "best", "worst"]
        ),
    ))

    # ----- Structure patterns -----
    candidates.append(_CandidateSignal(
        name="uses_arrows",
        description="Arrow formatting improves scannability",
        regex="\u2192",
        detect=lambda text, hook, close, stats: "\u2192" in text,
    ))

    candidates.append(_CandidateSignal(
        name="heavy_linebreaks",
        description="Heavy whitespace keeps readers scrolling",
        field_name="newline_count",
        min_value=25,
        detect=lambda text, hook, close, stats: text.count("\n") >= 25,
    ))

    candidates.append(_CandidateSignal(
        name="has_pivot",
        description="Narrative pivot creates tension",
        keywords=["but ", "however ", "here's the thing", "here's why"],
        detect=lambda text, hook, close, stats: any(
            kw in text.lower() for kw in ["but ", "however ", "here's the thing", "here's why"]
        ),
    ))

    candidates.append(_CandidateSignal(
        name="word_count_long",
        description="Longer posts tend to perform better",
        field_name="char_count",
        min_value=1000,
        detect=lambda text, hook, close, stats: len(text) >= 1000,
    ))

    # ----- Content patterns -----
    candidates.append(_CandidateSignal(
        name="personal_anecdote",
        description="Personal stories outperform information-only posts",
        keywords=["i remember", "years ago", "back in", "my first", "when i"],
        detect=lambda text, hook, close, stats: any(
            kw in text.lower() for kw in ["i remember", "years ago", "back in", "my first", "when i"]
        ),
    ))

    candidates.append(_CandidateSignal(
        name="has_data",
        description="Specific numbers add credibility",
        regex=r"\d+[%x]|\$[\d,]+|\d+\.\d+",
        detect=lambda text, hook, close, stats: bool(
            re.search(r"\d+[%x]|\$[\d,]+|\d+\.\d+", text)
        ),
    ))

    candidates.append(_CandidateSignal(
        name="has_image",
        description="Posts with images tend to get more engagement",
        detect=lambda text, hook, close, stats: stats.get("has_image", 0) > 0,
    ))

    # ----- CTA patterns -----
    candidates.append(_CandidateSignal(
        name="cta_comment",
        description="Asking for comments in closing drives engagement",
        scope="close",
        keywords=["comment"],
        detect=lambda text, hook, close, stats: "comment" in close.lower(),
    ))

    candidates.append(_CandidateSignal(
        name="cta_newsletter",
        description="Newsletter CTA in close — may feel transactional",
        scope="close",
        keywords=["newsletter", "subscribe"],
        detect=lambda text, hook, close, stats: any(
            kw in close.lower() for kw in ["newsletter", "subscribe"]
        ),
    ))

    # ----- Anti-pattern candidates -----
    candidates.append(_CandidateSignal(
        name="has_hashtag",
        description="Hashtags may reduce engagement",
        regex=r"#\w+",
        detect=lambda text, hook, close, stats: bool(re.search(r"#\w+", text)),
    ))

    candidates.append(_CandidateSignal(
        name="has_link",
        description="External links may reduce feed visibility",
        regex=r"https?://",
        detect=lambda text, hook, close, stats: bool(re.search(r"https?://", text)),
    ))

    return candidates


# ---------------------------------------------------------------------------
# Statistical tests (pure Python — no scipy dependency)
# ---------------------------------------------------------------------------

def _mann_whitney_u(
    group_a: list[float],
    group_b: list[float],
) -> tuple[float, float]:
    """Mann-Whitney U test for two independent samples.

    Tests whether the distribution of group_a is stochastically greater than
    group_b.  Uses the normal approximation for n >= 20, which is standard
    in the literature.

    Returns (U_statistic, p_value).  Two-sided test.
    """
    n1, n2 = len(group_a), len(group_b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Combine and rank
    combined = [(v, 0) for v in group_a] + [(v, 1) for v in group_b]
    combined.sort(key=lambda x: x[0])

    # Assign ranks with tie handling (average ranks)
    ranks: list[float] = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-indexed average
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum of ranks for group_a
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)

    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation (valid for n >= 20)
    mu = n1 * n2 / 2
    # Tie correction for variance
    n = n1 + n2
    tie_correction = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        t = j - i
        if t > 1:
            tie_correction += (t ** 3 - t)
        i = j

    sigma_sq = (n1 * n2 / 12) * ((n + 1) - tie_correction / (n * (n - 1)))
    if sigma_sq <= 0:
        return u, 1.0

    sigma = math.sqrt(sigma_sq)
    z = abs(u - mu) / sigma

    # Two-sided p-value from standard normal (approximation via error function)
    p = 2 * (1 - _norm_cdf(z))
    return u, p


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun 7.1.26).

    Maximum error: 1.5e-7.  Good enough for hypothesis testing.
    """
    # Handle negative z
    if z < 0:
        return 1 - _norm_cdf(-z)

    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-z * z / 2) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )
    return 1.0 - p


def _bootstrap_ci(
    group_a: list[float],
    group_b: list[float],
    *,
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the lift ratio (median_a / median_b).

    Returns (lower_bound, upper_bound) for the lift.
    """
    rng = random.Random(seed)
    lifts: list[float] = []

    for _ in range(n_bootstraps):
        sample_a = [rng.choice(group_a) for _ in range(len(group_a))]
        sample_b = [rng.choice(group_b) for _ in range(len(group_b))]
        med_a = statistics.median(sample_a)
        med_b = statistics.median(sample_b)
        if med_b > 0:
            lifts.append(med_a / med_b)
        elif med_a > 0:
            lifts.append(2.0)
        else:
            lifts.append(1.0)

    lifts.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstraps * alpha / 2)
    hi_idx = int(n_bootstraps * (1 - alpha / 2))
    return lifts[lo_idx], lifts[min(hi_idx, len(lifts) - 1)]


def _benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Returns a list of booleans — True if the hypothesis is rejected
    (i.e., the signal is statistically significant after correction).
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n

    # Find the largest k where p_(k) <= k/n * alpha
    max_k = -1
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        threshold = rank / n * alpha
        if p <= threshold:
            max_k = rank

    # Reject all hypotheses with rank <= max_k
    if max_k > 0:
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            if rank <= max_k:
                rejected[orig_idx] = True

    return rejected


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------

@dataclass
class SignalAnalysis:
    """Analysis results for a single candidate signal."""

    name: str
    description: str
    lift: float
    frequency: float
    weight: float
    count_present: int
    count_absent: int
    mean_engagement_present: float
    mean_engagement_absent: float
    is_anti_pattern: bool
    # Statistical rigor fields
    median_engagement_present: float = 0.0
    median_engagement_absent: float = 0.0
    median_lift: float = 1.0
    p_value: float = 1.0
    significant: bool = False  # After FDR correction
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    def __str__(self) -> str:
        direction = "PENALTY" if self.is_anti_pattern else "SIGNAL"
        sig_marker = "*" if self.significant else ""
        p_str = f"p={self.p_value:.4f}" if self.p_value < 1.0 else "p=n/a"
        ci_str = f"CI=[{self.ci_lower:.2f}, {self.ci_upper:.2f}]"
        return (
            f"  {direction}: {self.name}{sig_marker}\n"
            f"    lift={self.median_lift:.2f}x  freq={self.frequency:.1%}  "
            f"weight={self.weight:.0f}  {p_str}  {ci_str}\n"
            f"    present={self.count_present} (med {self.median_engagement_present:.1f})  "
            f"absent={self.count_absent} (med {self.median_engagement_absent:.1f})"
        )


@dataclass
class AnalysisResult:
    """Complete analysis output."""

    total_posts: int
    mean_engagement: float
    median_engagement: float
    signals: list[SignalAnalysis]
    anti_patterns: list[SignalAnalysis]
    config: MidasConfig

    def __str__(self) -> str:
        sig_count = sum(1 for s in self.signals if s.significant)
        anti_sig = sum(1 for a in self.anti_patterns if a.significant)
        parts = [
            f"MIDAS Signal Analysis",
            f"{'=' * 50}",
            f"Posts analyzed: {self.total_posts}",
            f"Mean engagement: {self.mean_engagement:.1f}",
            f"Median engagement: {self.median_engagement:.1f}",
            f"Signals tested: {len(self.signals) + len(self.anti_patterns)}  "
            f"(significant after FDR: {sig_count + anti_sig})",
            "",
            f"Positive signals ({len(self.signals)}, {sig_count} significant):",
        ]
        for s in sorted(self.signals, key=lambda x: -x.median_lift):
            parts.append(str(s))

        if self.anti_patterns:
            parts.append("")
            parts.append(f"Anti-patterns ({len(self.anti_patterns)}, {anti_sig} significant):")
            for a in sorted(self.anti_patterns, key=lambda x: x.median_lift):
                parts.append(str(a))

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core analysis logic
# ---------------------------------------------------------------------------

def _extract_parts(text: str, hook_max_chars: int, close_lines: int = 3) -> tuple[str, str]:
    """Extract hook and close from post text.

    Returns:
        (hook, close) where hook is the first line (up to hook_max_chars)
        and close is the last N lines joined.
    """
    lines = text.strip().split("\n")
    hook = lines[0][:hook_max_chars] if lines else ""
    close_portion = lines[-close_lines:] if len(lines) >= close_lines else lines
    close = "\n".join(close_portion)
    return hook, close


def analyze_posts(
    posts: list[dict],
    hook_max_chars: int = 100,
    close_lines: int = 3,
    min_frequency: float = 0.02,
) -> AnalysisResult:
    """Analyze a list of posts and derive a scoring formula.

    Args:
        posts: List of post dicts with keys: text, reactions, comments,
               reposts, date, has_image.
        hook_max_chars: Number of characters that define the "hook" region.
        close_lines: Number of lines from the end that define the "close" region.
        min_frequency: Minimum frequency (fraction) for a signal to be included.
                       Signals appearing in fewer posts are too rare to be reliable.

    Returns:
        AnalysisResult with signal analyses and a generated MidasConfig.
    """
    if not posts:
        raise ValueError("No posts to analyze. Provide at least one post.")

    # Compute engagement for each post
    engagements = [_engagement(p) for p in posts]
    mean_eng = statistics.mean(engagements)
    median_eng = statistics.median(engagements)

    candidates = _build_candidates(hook_max_chars)
    signal_analyses: list[SignalAnalysis] = []

    # Collect raw p-values for FDR correction later
    raw_analyses: list[tuple[SignalAnalysis, list[float], list[float]]] = []

    for candidate in candidates:
        present_engs: list[float] = []
        absent_engs: list[float] = []

        for post, eng in zip(posts, engagements):
            text = post.get("text", "")
            hook, close = _extract_parts(text, hook_max_chars, close_lines)
            stats: dict[str, float] = {
                "has_image": 1.0 if post.get("has_image", False) else 0.0,
                "hook_length": float(len(hook)),
                "char_count": float(len(text)),
                "newline_count": float(text.count("\n")),
            }

            if candidate.detect(text, hook, close, stats):
                present_engs.append(eng)
            else:
                absent_engs.append(eng)

        count_present = len(present_engs)
        count_absent = len(absent_engs)
        frequency = count_present / len(posts) if posts else 0

        # Skip signals that are too rare or too universal to be meaningful
        if frequency < min_frequency or frequency > 0.98:
            continue

        # Compute both mean and median
        mean_present = statistics.mean(present_engs) if present_engs else 0
        mean_absent = statistics.mean(absent_engs) if absent_engs else 0
        med_present = statistics.median(present_engs) if present_engs else 0
        med_absent = statistics.median(absent_engs) if absent_engs else 0

        # Primary lift uses median (robust to outliers)
        if med_absent > 0:
            median_lift = med_present / med_absent
        elif med_present > 0:
            median_lift = 2.0
        else:
            median_lift = 1.0

        # Mean-based lift kept for backwards compatibility
        if mean_absent > 0:
            mean_lift = mean_present / mean_absent
        elif mean_present > 0:
            mean_lift = 2.0
        else:
            mean_lift = 1.0

        # Mann-Whitney U test: is the engagement distribution significantly
        # different when this signal is present vs absent?
        _, p_value = _mann_whitney_u(present_engs, absent_engs)

        # Bootstrap 95% CI for the median lift
        if count_present >= 3 and count_absent >= 3:
            ci_lo, ci_hi = _bootstrap_ci(present_engs, absent_engs)
        else:
            ci_lo, ci_hi = median_lift, median_lift  # Not enough data

        # Weight is derived from median lift * 100, rounded to nearest 10
        raw_weight = median_lift * 100
        weight = round(raw_weight / 10) * 10

        is_anti = median_lift < 1.0

        sa = SignalAnalysis(
            name=candidate.name,
            description=candidate.description,
            lift=mean_lift,
            frequency=frequency,
            weight=weight if not is_anti else -weight,
            count_present=count_present,
            count_absent=count_absent,
            mean_engagement_present=mean_present,
            mean_engagement_absent=mean_absent,
            is_anti_pattern=is_anti,
            median_engagement_present=med_present,
            median_engagement_absent=med_absent,
            median_lift=median_lift,
            p_value=p_value,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
        )
        raw_analyses.append((sa, present_engs, absent_engs))

    # Benjamini-Hochberg FDR correction across all tested signals
    p_values = [sa.p_value for sa, _, _ in raw_analyses]
    significant_mask = _benjamini_hochberg(p_values, alpha=0.05)

    for i, (sa, _, _) in enumerate(raw_analyses):
        sa.significant = significant_mask[i]
        signal_analyses.append(sa)

    # Split into positive signals and anti-patterns
    positive = [s for s in signal_analyses if not s.is_anti_pattern]
    anti = [s for s in signal_analyses if s.is_anti_pattern]

    # Generate a MidasConfig from the analysis
    config = _build_config(positive, anti, hook_max_chars, close_lines, engagements)

    return AnalysisResult(
        total_posts=len(posts),
        mean_engagement=mean_eng,
        median_engagement=median_eng,
        signals=positive,
        anti_patterns=anti,
        config=config,
    )


def _build_config(
    positive: list[SignalAnalysis],
    anti: list[SignalAnalysis],
    hook_max_chars: int,
    close_lines: int,
    engagements: list[float],
) -> MidasConfig:
    """Build a MidasConfig from analysis results."""
    candidates = _build_candidates(hook_max_chars)
    candidate_map = {c.name: c for c in candidates}

    # Build SignalDef list from positive signals
    signal_defs: list[SignalDef] = []
    for sa in sorted(positive, key=lambda x: -x.lift):
        cand = candidate_map.get(sa.name)
        if cand is None:
            continue

        signal_defs.append(SignalDef(
            name=sa.name,
            weight=sa.weight,
            description=f"{sa.description} (lift={sa.lift:.2f}x, freq={sa.frequency:.0%})",
            regex=cand.regex,
            keywords=cand.keywords,
            scope=cand.scope,
            min_value=cand.min_value,
            field=cand.field_name,
        ))

    # Build PenaltyDef list from anti-patterns
    penalty_defs: list[PenaltyDef] = []
    for sa in sorted(anti, key=lambda x: x.lift):
        cand = candidate_map.get(sa.name)
        if cand is None:
            continue

        penalty_defs.append(PenaltyDef(
            name=sa.name,
            weight=-abs(sa.weight),
            description=f"{sa.description} (lift={sa.lift:.2f}x, freq={sa.frequency:.0%})",
            regex=cand.regex,
            keywords=cand.keywords,
            scope=cand.scope,
        ))

    # Derive tiers from engagement percentiles
    sorted_eng = sorted(engagements)
    n = len(sorted_eng)
    p95 = sorted_eng[int(n * 0.95)] if n > 20 else sorted_eng[-1]
    p85 = sorted_eng[int(n * 0.85)] if n > 20 else sorted_eng[-1]
    p65 = sorted_eng[int(n * 0.65)] if n > 10 else sorted_eng[n // 2]
    p50 = sorted_eng[int(n * 0.50)] if n > 4 else sorted_eng[n // 2]

    tiers = [
        ScoreTier(
            name="VIRAL CANDIDATE",
            min_score=800,
            description=f"Top 5% — engagement >= {p95:.0f}",
        ),
        ScoreTier(
            name="HIGH PERFORMER",
            min_score=500,
            description=f"Top 15% — engagement >= {p85:.0f}",
        ),
        ScoreTier(
            name="ABOVE AVERAGE",
            min_score=250,
            description=f"Top 35% — engagement >= {p65:.0f}",
        ),
        ScoreTier(
            name="AVERAGE",
            min_score=100,
            description=f"Median range — engagement >= {p50:.0f}",
        ),
        ScoreTier(
            name="BELOW AVERAGE",
            min_score=0,
            description="Below median — consider revising",
        ),
    ]

    # Build suggestions for high-lift missing signals
    suggestions: dict[str, str] = {}
    for sa in positive:
        if sa.lift >= 1.2:
            suggestions[sa.name] = (
                f"Add {sa.name.replace('_', ' ')} — "
                f"posts with this get {sa.lift:.1f}x more engagement"
            )
    for sa in anti:
        suggestions[sa.name] = (
            f"Remove {sa.name.replace('_', ' ')} — "
            f"posts with this get {(1 / sa.lift) if sa.lift > 0 else 0:.1f}x less engagement"
        )

    return MidasConfig(
        name="analyzed",
        description="Auto-generated config from post analysis",
        signals=signal_defs,
        penalties=penalty_defs,
        tiers=tiers,
        hook_max_chars=hook_max_chars,
        close_lines=close_lines,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# File I/O convenience functions
# ---------------------------------------------------------------------------

def analyze_file(
    path: str,
    hook_max_chars: int = 100,
    close_lines: int = 3,
    min_frequency: float = 0.02,
) -> AnalysisResult:
    """Load a JSONL file and run analysis.

    Each line of the file should be a JSON object with keys:
        text, reactions, comments, reposts, date, has_image

    Args:
        path: Path to the JSONL file.
        hook_max_chars: Characters defining the hook region.
        close_lines: Lines from the end defining the close region.
        min_frequency: Minimum signal frequency to include.

    Returns:
        AnalysisResult with signal analyses and generated config.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    posts: list[dict] = []
    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e

    if not posts:
        raise ValueError(f"No posts found in {path}")

    return analyze_posts(
        posts,
        hook_max_chars=hook_max_chars,
        close_lines=close_lines,
        min_frequency=min_frequency,
    )


def export_config(result: AnalysisResult, path: str) -> None:
    """Save the derived MidasConfig from an analysis to a YAML file.

    Args:
        result: The AnalysisResult from analyze_posts or analyze_file.
        path: Output path for the YAML config file.
    """
    save_config(result.config, path)

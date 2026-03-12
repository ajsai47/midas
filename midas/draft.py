"""LLM-powered drafting and optimization for LinkedIn posts.

Generates and rewrites posts using LLM APIs, guided by the user's
MidasConfig scoring formula.  Supports Anthropic, OpenAI, and local
providers.  Install the optional dependency group for LLM support:

    pip install midas-linkedin[llm]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .config import MidasConfig, SignalDef, PenaltyDef
from .scorer import ScoreResult, score


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DraftResult:
    """A generated or rewritten draft with its scoring breakdown."""

    text: str
    score_result: ScoreResult
    provider: str
    model: str

    def __str__(self) -> str:
        header = f"[{self.provider}/{self.model}] Score: {self.score_result.score:.0f} ({self.score_result.tier})"
        return f"{header}\n{'─' * len(header)}\n{self.text}"


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------

def generate_system_prompt(config: MidasConfig) -> str:
    """Convert a MidasConfig into a system prompt that steers an LLM.

    The prompt encodes the scoring formula as explicit writing rules so the
    model naturally produces high-scoring drafts without needing access to
    the scorer at inference time.

    Parameters
    ----------
    config : MidasConfig
        The scoring configuration to translate into writing instructions.

    Returns
    -------
    str
        A system prompt string ready to pass to an LLM.
    """
    sections: list[str] = []

    # Header
    sections.append(
        "You are an expert LinkedIn ghostwriter. Your goal is to write "
        "posts that maximize engagement according to a precise scoring "
        "formula. Follow every rule below.\n"
    )

    # --- Positive signals as rules (sorted by weight, descending) ---------
    sorted_signals = sorted(config.signals, key=lambda s: s.weight, reverse=True)
    if sorted_signals:
        rules: list[str] = []
        for sig in sorted_signals:
            rule = f"  (+{sig.weight:.0f}) {sig.name}"
            if sig.description:
                rule += f" — {sig.description}"
            # Add concrete guidance based on detection method
            if sig.regex:
                rule += f"  [pattern: {sig.regex}]"
            if sig.keywords:
                rule += f"  [include one of: {', '.join(sig.keywords[:5])}]"
            if sig.scope != "full":
                rule += f"  [applies to: {sig.scope}]"
            if sig.field and sig.min_value is not None:
                rule += f"  [requires {sig.field} >= {sig.min_value}]"
            rules.append(rule)
        sections.append("WRITING RULES (sorted by importance):\n" + "\n".join(rules))

    # --- Penalties as anti-rules ------------------------------------------
    sorted_penalties = sorted(config.penalties, key=lambda p: p.weight)
    if sorted_penalties:
        nevers: list[str] = []
        for pen in sorted_penalties:
            never = f"  ({pen.weight:.0f}) {pen.name}"
            if pen.description:
                never += f" — {pen.description}"
            if pen.regex:
                never += f"  [avoid pattern: {pen.regex}]"
            if pen.keywords:
                never += f"  [avoid: {', '.join(pen.keywords[:5])}]"
            nevers.append(never)
        sections.append("NEVER DO (penalties):\n" + "\n".join(nevers))

    # --- Structural guidance ----------------------------------------------
    structure: list[str] = []

    # Infer target length from signals
    length_signals = [s for s in config.signals if s.field == "char_count" and s.min_value]
    if length_signals:
        best = max(length_signals, key=lambda s: s.weight)
        structure.append(
            f"- Aim for at least {int(best.min_value)} characters "
            f"(the '{best.name}' signal is worth +{best.weight:.0f})."
        )

    hook_signals = [s for s in config.signals if s.scope == "hook"]
    if hook_signals:
        structure.append(
            f"- The hook is the first {config.hook_max_chars} characters. "
            "Make it punchy and attention-grabbing."
        )

    close_signals = [s for s in config.signals if s.scope == "close"]
    if close_signals:
        structure.append(
            f"- The close is the last {config.close_lines} lines. "
            "End with a clear call to action."
        )

    structure.append("- Use line breaks for readability. One idea per line.")
    structure.append("- Write in a natural, conversational tone.")

    if structure:
        sections.append("STRUCTURE:\n" + "\n".join(structure))

    # --- Output format ----------------------------------------------------
    sections.append(
        "OUTPUT FORMAT:\n"
        "  Return ONLY the post text. No preamble, no explanation, no "
        "markdown formatting. Just the raw LinkedIn post ready to publish."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------

def _get_anthropic_client(api_key: str | None = None) -> Any:
    """Return an Anthropic client, raising a helpful error if not installed."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic provider.\n"
            "Install it with:  pip install midas-linkedin[llm]"
        ) from None

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "No Anthropic API key provided. Pass api_key= or set "
            "the ANTHROPIC_API_KEY environment variable."
        )
    return anthropic.Anthropic(api_key=key)


def _get_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Return an OpenAI client, raising a helpful error if not installed."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for the OpenAI/local provider.\n"
            "Install it with:  pip install midas-linkedin[llm]"
        ) from None

    kwargs: dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = api_key or "not-needed"
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key provided. Pass api_key= or set "
                "the OPENAI_API_KEY environment variable."
            )
        kwargs["api_key"] = key

    return openai.OpenAI(**kwargs)


_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "local": "default",
}


def _call_llm(
    system: str,
    user_message: str,
    provider: str,
    api_key: str | None,
    model: str | None,
    temperature: float,
) -> tuple[str, str]:
    """Call an LLM and return (response_text, model_used).

    Parameters
    ----------
    system : str
        The system prompt.
    user_message : str
        The user prompt.
    provider : str
        One of "anthropic", "openai", "local".
    api_key : str | None
        API key override.
    model : str | None
        Model override.  Falls back to provider defaults.
    temperature : float
        Sampling temperature.

    Returns
    -------
    tuple[str, str]
        (generated_text, model_name)
    """
    resolved_model = model or _DEFAULT_MODELS.get(provider, "default")

    if provider == "anthropic":
        client = _get_anthropic_client(api_key)
        response = client.messages.create(
            model=resolved_model,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text
        return text.strip(), resolved_model

    elif provider in ("openai", "local"):
        base_url = "http://localhost:8000/v1" if provider == "local" else None
        client = _get_openai_client(api_key, base_url)
        response = client.chat.completions.create(
            model=resolved_model,
            temperature=temperature,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        text = response.choices[0].message.content
        return text.strip(), resolved_model

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Supported: 'anthropic', 'openai', 'local'."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draft(
    topic: str,
    config: MidasConfig,
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    num_samples: int = 3,
) -> list[DraftResult]:
    """Generate multiple LinkedIn post drafts for a topic.

    Produces ``num_samples`` drafts via the chosen LLM provider, scores
    each one against ``config``, and returns them sorted best-first.

    Parameters
    ----------
    topic : str
        The subject, angle, or raw idea for the post.
    config : MidasConfig
        The scoring configuration that guides both generation and evaluation.
    provider : str
        LLM provider: ``"anthropic"``, ``"openai"``, or ``"local"``.
    api_key : str | None
        API key override.  Falls back to ``ANTHROPIC_API_KEY`` or
        ``OPENAI_API_KEY`` environment variables.
    model : str | None
        Model override.  Defaults to ``claude-sonnet-4-20250514`` (Anthropic)
        or ``gpt-4o`` (OpenAI).
    temperature : float
        Sampling temperature.  Higher values produce more varied drafts.
    num_samples : int
        Number of drafts to generate.

    Returns
    -------
    list[DraftResult]
        Drafts sorted by score descending (best first).
    """
    system = generate_system_prompt(config)
    user_message = (
        f"Write a LinkedIn post about the following topic:\n\n{topic}\n\n"
        f"Make it engaging, authentic, and optimized for maximum engagement."
    )

    results: list[DraftResult] = []
    for _ in range(num_samples):
        text, model_used = _call_llm(
            system=system,
            user_message=user_message,
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        score_result = score(text, config)
        results.append(DraftResult(
            text=text,
            score_result=score_result,
            provider=provider,
            model=model_used,
        ))

    results.sort(key=lambda r: r.score_result.score, reverse=True)
    return results


def rewrite(
    draft_text: str,
    config: MidasConfig,
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.5,
) -> DraftResult:
    """Score an existing draft, then rewrite it for a higher score.

    Analyses the draft against the config, builds targeted improvement
    suggestions, and sends both the original text and suggestions to the
    LLM for a single optimized rewrite.

    Parameters
    ----------
    draft_text : str
        The original post text to improve.
    config : MidasConfig
        The scoring configuration used for evaluation and rewriting.
    provider : str
        LLM provider: ``"anthropic"``, ``"openai"``, or ``"local"``.
    api_key : str | None
        API key override.
    model : str | None
        Model override.
    temperature : float
        Sampling temperature.  Defaults to 0.5 (slightly more focused than
        ``draft()`` since we want a faithful improvement, not wild variation).

    Returns
    -------
    DraftResult
        The rewritten post with its new score.
    """
    original_result = score(draft_text, config)

    # Build improvement guidance from the scoring breakdown
    improvement_lines: list[str] = []

    # Missing high-value signals
    all_signal_names = {s.name for s in config.signals}
    matched_signal_names = set(original_result.signals.keys())
    missing_signals = [
        s for s in config.signals if s.name not in matched_signal_names
    ]
    missing_signals.sort(key=lambda s: s.weight, reverse=True)

    if missing_signals:
        improvement_lines.append("MISSING SIGNALS (add these):")
        for sig in missing_signals[:8]:
            line = f"  - {sig.name} (+{sig.weight:.0f})"
            if sig.description:
                line += f": {sig.description}"
            improvement_lines.append(line)

    # Active penalties
    if original_result.penalties:
        improvement_lines.append("ACTIVE PENALTIES (remove these):")
        for name, weight in sorted(
            original_result.penalties.items(), key=lambda x: x[1]
        ):
            pen_def = next((p for p in config.penalties if p.name == name), None)
            desc = f": {pen_def.description}" if pen_def and pen_def.description else ""
            improvement_lines.append(f"  - {name} ({weight:.0f}){desc}")

    # Built-in suggestions from config
    if original_result.suggestions:
        improvement_lines.append("QUICK WINS:")
        for suggestion in original_result.suggestions:
            improvement_lines.append(f"  - {suggestion}")

    improvement_text = "\n".join(improvement_lines) if improvement_lines else "No specific improvements identified."

    system = generate_system_prompt(config)
    user_message = (
        f"Rewrite and improve the following LinkedIn post.\n\n"
        f"ORIGINAL POST (current score: {original_result.score:.0f}/{original_result.tier}):\n"
        f"---\n{draft_text}\n---\n\n"
        f"IMPROVEMENT GUIDANCE:\n{improvement_text}\n\n"
        f"Rewrite the post to address as many improvements as possible while "
        f"preserving the original message and voice. Return ONLY the improved "
        f"post text."
    )

    text, model_used = _call_llm(
        system=system,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
    )

    new_score = score(text, config)
    return DraftResult(
        text=text,
        score_result=new_score,
        provider=provider,
        model=model_used,
    )

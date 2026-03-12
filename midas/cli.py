"""MIDAS CLI — Score, analyze, draft, and optimize LinkedIn posts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import load_config, MidasConfig
from .scorer import score, ScoreResult

console = Console()

DEFAULT_CONFIG = "midas_config.yaml"


def _find_config(config_path: str | None) -> MidasConfig:
    """Find and load config, checking common paths."""
    candidates = [config_path] if config_path else [
        DEFAULT_CONFIG,
        "config.yaml",
        "examples/sample_config.yaml",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return load_config(path)
    console.print("[red]No config found.[/red] Run `midas analyze` to generate one, or specify --config.")
    console.print("  Quick start: copy examples/sample_config.yaml to midas_config.yaml")
    sys.exit(1)


def _render_score(result: ScoreResult) -> None:
    """Pretty-print a score result using rich."""
    # Tier color
    color = {
        "VIRAL CANDIDATE": "bold magenta",
        "HIGH PERFORMER": "bold green",
        "ABOVE AVERAGE": "cyan",
        "AVERAGE": "yellow",
        "BELOW AVERAGE": "red",
    }.get(result.tier, "white")

    console.print()
    console.print(f"  Score: [bold]{result.score:.0f}[/bold]  [{color}]{result.tier}[/{color}]")
    if result.tier_description:
        console.print(f"  {result.tier_description}")
    console.print()

    if result.signals:
        table = Table(title="Signals", show_header=True, header_style="green")
        table.add_column("Signal", style="dim")
        table.add_column("Weight", justify="right")
        for name, w in sorted(result.signals.items(), key=lambda x: -x[1]):
            table.add_row(name, f"+{w:.0f}")
        table.add_row("[bold]Total[/bold]", f"[bold]+{result.signal_total:.0f}[/bold]")
        console.print(table)

    if result.penalties:
        table = Table(title="Penalties", show_header=True, header_style="red")
        table.add_column("Penalty", style="dim")
        table.add_column("Weight", justify="right")
        for name, w in sorted(result.penalties.items(), key=lambda x: x[1]):
            table.add_row(name, f"{w:.0f}")
        table.add_row("[bold]Total[/bold]", f"[bold]{result.penalty_total:.0f}[/bold]")
        console.print(table)

    if result.suggestions:
        console.print()
        console.print("[bold]Quick wins:[/bold]")
        for s in result.suggestions:
            console.print(f"  → {s}")
    console.print()


@click.group()
@click.version_option(package_name="midas-linkedin")
def main():
    """MIDAS — Reverse-engineer your LinkedIn into a personalized scoring formula."""


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read post from file")
@click.option("--config", "-c", type=click.Path(), help="Path to config YAML")
def score_cmd(text: str | None, file: str | None, config: str | None):
    """Score a LinkedIn post against your formula."""
    if file:
        text = Path(file).read_text()
    elif text is None:
        # Read from stdin
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            console.print("Enter your post (Ctrl+D when done):")
            text = sys.stdin.read()

    if not text or not text.strip():
        console.print("[red]No text provided.[/red]")
        sys.exit(1)

    cfg = _find_config(config)
    result = score(text.strip(), cfg)
    _render_score(result)


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="midas_config.yaml", help="Output config path")
@click.option("--hook-max-chars", default=100, help="Max chars for hook detection")
@click.option("--min-posts", default=10, help="Min posts with signal to include it")
def analyze(data_path: str, output: str, hook_max_chars: int, min_posts: int):
    """Analyze your posts and generate a scoring config.

    DATA_PATH should be a JSONL file with your posts and engagement data.
    """
    from .analyze import analyze_file, export_config

    console.print(f"[bold]Analyzing[/bold] {data_path}...")

    result = analyze_file(data_path, hook_max_chars=hook_max_chars, min_posts=min_posts)

    console.print(f"\n  Posts analyzed: [bold]{result.total_posts}[/bold]")
    console.print(f"  Signals found: [bold]{len(result.signals)}[/bold]")
    console.print(f"  Penalties found: [bold]{len(result.penalties)}[/bold]")

    if result.signals:
        console.print("\n[bold]Top signals by lift:[/bold]")
        for s in result.signals[:10]:
            console.print(f"  +{s.weight:.0f}  {s.name}  (lift: {s.lift:.2f}, freq: {s.frequency:.0%})")

    if result.anti_patterns:
        console.print("\n[bold]Anti-patterns (negative lift):[/bold]")
        for p in result.anti_patterns[:5]:
            console.print(f"  {p.weight:.0f}  {p.name}  (lift: {p.lift:.2f})")

    export_config(result, output)
    console.print(f"\n[green]Config saved to {output}[/green]")
    console.print(f"  Edit the config to tune weights, then use `midas score` to test.")


@main.command()
@click.argument("topic")
@click.option("--config", "-c", type=click.Path(), help="Path to config YAML")
@click.option("--provider", "-p", default="anthropic", type=click.Choice(["anthropic", "openai", "local"]))
@click.option("--api-key", envvar=["ANTHROPIC_API_KEY", "OPENAI_API_KEY"], help="API key")
@click.option("--model", "-m", help="Model name override")
@click.option("--samples", "-n", default=3, help="Number of samples to generate")
@click.option("--temperature", "-t", default=0.7, help="Sampling temperature")
def draft(topic: str, config: str | None, provider: str, api_key: str | None, model: str | None, samples: int, temperature: float):
    """Generate a LinkedIn post about a topic."""
    from .draft import draft as draft_fn

    cfg = _find_config(config)
    console.print(f"[bold]Drafting[/bold] {samples} samples about: {topic}")
    console.print(f"  Provider: {provider}")

    results = draft_fn(
        topic=topic,
        config=cfg,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        num_samples=samples,
    )

    for i, r in enumerate(results, 1):
        console.print(Panel(
            r.text,
            title=f"Draft {i} — Score: {r.score_result.score:.0f} ({r.score_result.tier})",
            border_style="green" if i == 1 else "dim",
        ))
        if i == 1 and r.score_result.suggestions:
            console.print("[bold]Quick wins for the best draft:[/bold]")
            for s in r.score_result.suggestions:
                console.print(f"  → {s}")


@main.command()
@click.argument("draft_path", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(), help="Path to config YAML")
@click.option("--provider", "-p", default="anthropic", type=click.Choice(["anthropic", "openai", "local"]))
@click.option("--api-key", help="API key")
@click.option("--model", "-m", help="Model name override")
def rewrite(draft_path: str, config: str | None, provider: str, api_key: str | None, model: str | None):
    """Rewrite and optimize an existing draft."""
    from .draft import rewrite as rewrite_fn

    cfg = _find_config(config)
    original = Path(draft_path).read_text().strip()

    console.print("[bold]Original:[/bold]")
    original_result = score(original, cfg)
    _render_score(original_result)

    console.print("[bold]Rewriting...[/bold]")
    result = rewrite_fn(
        draft_text=original,
        config=cfg,
        provider=provider,
        api_key=api_key,
        model=model,
    )

    console.print(Panel(result.text, title="Rewritten", border_style="green"))
    _render_score(result.score_result)

    delta = result.score_result.score - original_result.score
    color = "green" if delta > 0 else "red"
    console.print(f"  Score change: [{color}]{delta:+.0f}[/{color}]")


@main.command()
@click.option("--original", "-o", type=click.Path(exists=True), help="Original draft file")
@click.option("--edited", "-e", type=click.Path(exists=True), help="Edited draft file")
@click.option("--log", "-l", default="midas_feedback.jsonl", help="Feedback log path")
@click.option("--config", "-c", type=click.Path(), help="Path to config YAML")
@click.option("--stats", is_flag=True, help="Show editing pattern stats")
@click.option("--export-dpo", type=click.Path(), help="Export edits as DPO data")
def feedback(original: str | None, edited: str | None, log: str, config: str | None, stats: bool, export_dpo: str | None):
    """Log edits and track your editing patterns."""
    from .feedback import log_edit, get_stats, export_dpo as export_dpo_fn

    if stats:
        s = get_stats(log)
        console.print(f"\n[bold]Feedback Stats[/bold] ({s.total_edits} edits)")
        console.print(f"  Avg score improvement: [green]+{s.avg_score_improvement:.0f}[/green]")
        if s.most_commonly_added:
            console.print("  Most added signals:")
            for name, count in s.most_commonly_added[:5]:
                console.print(f"    +{count}x  {name}")
        if s.most_commonly_removed:
            console.print("  Most removed signals:")
            for name, count in s.most_commonly_removed[:5]:
                console.print(f"    -{count}x  {name}")
        return

    if export_dpo:
        n = export_dpo_fn(log_path=log, output_path=export_dpo)
        console.print(f"[green]Exported {n} DPO pairs to {export_dpo}[/green]")
        return

    if not original or not edited:
        console.print("[red]Provide --original and --edited files to log an edit.[/red]")
        sys.exit(1)

    cfg = _find_config(config)
    original_text = Path(original).read_text().strip()
    edited_text = Path(edited).read_text().strip()

    entry = log_edit(original_text, edited_text, cfg, log_path=log)
    delta = entry.edited_score - entry.original_score
    color = "green" if delta > 0 else "red"
    console.print(f"  Original score: {entry.original_score:.0f}")
    console.print(f"  Edited score:   {entry.edited_score:.0f}")
    console.print(f"  Delta:          [{color}]{delta:+.0f}[/{color}]")
    if entry.signals_added:
        console.print(f"  Signals added:  {', '.join(entry.signals_added)}")
    if entry.signals_removed:
        console.print(f"  Signals removed: {', '.join(entry.signals_removed)}")
    console.print(f"  Logged to {log}")


# Register score command with the right name
main.add_command(score_cmd, "score")

if __name__ == "__main__":
    main()

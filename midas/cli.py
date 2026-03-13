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


def _find_sample(name: str) -> Path | None:
    """Locate a bundled sample file (package data or dev examples/)."""
    pkg_data = Path(__file__).parent / "data" / name
    if pkg_data.exists():
        return pkg_data
    dev_examples = Path(__file__).parent.parent / "examples" / name
    if dev_examples.exists():
        return dev_examples
    return None


def _copy_sample_files(target: Path) -> tuple[Path | None, Path | None]:
    """Copy sample config and data into target dir. Returns (config_path, data_path) or None if skipped."""
    import shutil

    config_path = target / "midas_config.yaml"
    data_path = target / "posts.jsonl"
    sample_config = _find_sample("sample_config.yaml")
    sample_data = _find_sample("sample_data.jsonl")

    created_config = None
    created_data = None

    if config_path.exists():
        console.print(f"  [dim]Config already exists:[/dim] {config_path}")
        created_config = config_path
    elif sample_config:
        shutil.copy(sample_config, config_path)
        console.print(f"  [green]Created[/green] {config_path} (sample config)")
        created_config = config_path

    if data_path.exists():
        console.print(f"  [dim]Data already exists:[/dim] {data_path}")
        created_data = data_path
    elif sample_data:
        shutil.copy(sample_data, data_path)
        console.print(f"  [green]Created[/green] {data_path} (10 sample posts)")
        created_data = data_path

    return created_config, created_data


def _detect_data_format(path: str) -> str:
    """Auto-detect data format: 'apify' (JSON array), 'csv' (LinkedIn CSV), or 'jsonl'."""
    filepath = Path(path)
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        return "csv"

    with open(filepath, encoding="utf-8") as f:
        first_char = f.read(1).strip()

    if first_char == "[":
        return "apify"
    return "jsonl"


def _parse_data_file(path: str, fmt: str) -> list[dict]:
    """Parse a data file into MIDAS posts based on format."""
    from .export import parse_apify_posts, parse_linkedin_export, load_jsonl

    if fmt == "apify":
        return parse_apify_posts(path)
    elif fmt == "csv":
        return parse_linkedin_export(path)
    else:
        return load_jsonl(path)


def _interactive_score_demo(target: Path) -> None:
    """Prompt user to score a post interactively."""
    config_path = target / "midas_config.yaml"
    if not config_path.exists():
        return

    console.print()
    console.print("[bold]Let's try scoring a post![/bold]")
    console.print("  Paste a LinkedIn post below (or press Enter to use a sample):")
    console.print()

    try:
        text = click.prompt("", default="", prompt_suffix="  > ", show_default=False)
    except (click.Abort, EOFError):
        return

    if not text.strip():
        # Use a built-in sample
        text = (
            "I just spent 3 months building an AI agent from scratch.\n\n"
            "Everyone said to use a framework.\n\n"
            "But here's the thing → frameworks hide the complexity.\n\n"
            "They don't remove it.\n\n"
            "I learned more in those 3 months than in 2 years of using LangChain.\n\n"
            "Here's what actually matters:\n\n"
            "→ Prompt engineering is 80% of the work\n"
            "→ Memory management is harder than generation\n"
            "→ Error handling is where agents actually break\n"
            "→ Evaluation is still an unsolved problem\n\n"
            "The frameworks will catch up.\n\n"
            "But understanding the fundamentals won't go out of style.\n\n"
            "Comment AGENT if you've built from scratch too."
        )
        console.print("  [dim](Using sample post)[/dim]")

    cfg = load_config(str(config_path))
    result = score(text.strip(), cfg)
    _render_score(result)


def _print_static_next_steps(has_config: bool) -> None:
    """Print the static next-steps text (non-interactive fallback)."""
    console.print("[bold]Next steps:[/bold]")
    console.print()
    console.print("  [bold cyan]1.[/bold cyan] Get your LinkedIn data (you need posts + engagement numbers):")
    console.print("     Use the [bold]Apify LinkedIn Post Scraper[/bold] (free tier available):")
    console.print("     [dim]https://console.apify.com/actors/RE0MriXnFhR3IgVnJ/input[/dim]")
    console.print()
    console.print("     Then convert to MIDAS format:")
    console.print("     [dim]midas init --data apify_dataset.json[/dim]")
    console.print()
    console.print("  [bold cyan]2.[/bold cyan] Analyze your posts to build your formula:")
    console.print("     [dim]midas analyze posts.jsonl -o midas_config.yaml[/dim]")
    console.print()
    console.print("  [bold cyan]3.[/bold cyan] Score a draft before publishing:")
    console.print('     [dim]midas score "Your draft here..." --config midas_config.yaml[/dim]')
    console.print()
    console.print("  [bold cyan]4.[/bold cyan] Validate that your formula predicts engagement:")
    console.print("     [dim]midas validate posts.jsonl --config midas_config.yaml[/dim]")
    console.print()
    if has_config:
        console.print("  [dim]Tip: A sample config and data were created above — try steps 3-4 now to see it in action.[/dim]")
        console.print()


@main.command()
@click.option("--dir", "-d", default=".", help="Directory to initialize in")
@click.option("--data", type=click.Path(exists=True), help="Path to your LinkedIn data file (auto-detects format)")
def init(dir: str, data: str | None):
    """Set up MIDAS in your project — guided onboarding."""
    from .export import save_jsonl
    from .analyze import analyze_file, export_config

    target = Path(dir)
    target.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        "[bold]MIDAS[/bold] — Reverse-engineer your LinkedIn into a\npersonalized scoring formula.",
        border_style="yellow",
    ))
    console.print()

    is_interactive = sys.stdin.isatty() and data is None

    # ── Path A: User provided --data flag ──────────────────────────────
    if data:
        fmt = _detect_data_format(data)
        console.print(f"  Detected format: [bold]{fmt}[/bold]")

        posts = _parse_data_file(data, fmt)
        if not posts:
            console.print("[red]No posts found in the file.[/red]")
            sys.exit(1)

        data_path = target / "posts.jsonl"
        save_jsonl(posts, str(data_path))
        console.print(f"  [green]Parsed {len(posts)} posts[/green] → {data_path}")

        if fmt == "csv":
            console.print()
            console.print("  [yellow]Note:[/yellow] LinkedIn CSV exports don't include engagement metrics.")
            console.print("  You'll need to add reactions/comments/reposts manually or via the LinkedIn API.")
            console.print()

        # Analyze
        console.print()
        console.print("[bold]Analyzing your posts...[/bold]")
        result = analyze_file(str(data_path))
        sig_count = sum(1 for s in result.signals if s.significant)

        config_path = target / "midas_config.yaml"
        export_config(result, str(config_path))

        console.print(f"  Posts analyzed: [bold]{result.total_posts}[/bold]")
        console.print(f"  Signals found: [bold]{len(result.signals)}[/bold] ({sig_count} statistically significant)")
        console.print(f"  [green]Config saved to {config_path}[/green]")

        # Validate
        console.print()
        console.print("[bold]Validating your formula...[/bold]")
        from .validate import validate as validate_fn
        import json as _json

        loaded_posts = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    loaded_posts.append(_json.loads(line))

        if len(loaded_posts) >= 5:
            cfg = load_config(str(config_path))
            val_result = validate_fn(loaded_posts, cfg)

            color = "green" if val_result.spearman_rho > 0.3 else "yellow" if val_result.spearman_rho > 0 else "red"
            strength = val_result.correlation_strength.upper()
            sig_str = ", SIGNIFICANT" if val_result.is_significant else ""
            console.print(f"  Spearman rho: [{color}]{val_result.spearman_rho:+.2f}[/{color}] ({strength}{sig_str})")

            if val_result.spearman_rho > 0 and val_result.is_significant:
                console.print("  [green]Your formula predicts engagement![/green]")
        else:
            console.print("  [dim]Not enough posts to validate (need at least 5).[/dim]")

        console.print()
        console.print("[bold]You're all set.[/bold] Try scoring a draft:")
        console.print('  [dim]midas score "Your draft here..."[/dim]')
        console.print()
        return

    # ── Path B: Non-interactive (piped stdin / CI) ─────────────────────
    if not is_interactive:
        created_config, _ = _copy_sample_files(target)
        console.print()
        _print_static_next_steps(created_config is not None)
        return

    # ── Path C: Interactive onboarding ─────────────────────────────────
    has_data = click.confirm("Do you have your LinkedIn post data ready?", default=False)

    if not has_data:
        # No data — set up with samples and demo scoring
        console.print()
        console.print("  No worries! Here's how to get it:")
        console.print()
        console.print("  [bold cyan]1.[/bold cyan] Go to the Apify LinkedIn Post Scraper (free tier available):")
        console.print("     [dim]https://console.apify.com/actors/RE0MriXnFhR3IgVnJ/input[/dim]")
        console.print()
        console.print("  [bold cyan]2.[/bold cyan] Run the scraper on your profile")
        console.print()
        console.print("  [bold cyan]3.[/bold cyan] Download the JSON dataset and save it here, then run:")
        console.print("     [dim]midas init --data apify_dataset.json[/dim]")
        console.print()
        console.print("  In the meantime, let's set up with sample data so you can see how MIDAS works.")
        console.print()

        _copy_sample_files(target)
        _interactive_score_demo(target)

        console.print("  Your formula is working. Once you have your real data:")
        console.print("    [dim]midas init --data your_posts.json[/dim]")
        console.print()
    else:
        # User has data — walk them through import
        console.print()
        fmt_choice = click.prompt(
            "What format is your data in?\n"
            "  [1] Apify JSON export\n"
            "  [2] LinkedIn CSV export (Settings → Data privacy)\n"
            "  [3] JSONL (already in MIDAS format)\n"
            "  Choose",
            type=click.Choice(["1", "2", "3"]),
            show_choices=False,
        )

        fmt_map = {"1": "apify", "2": "csv", "3": "jsonl"}
        fmt = fmt_map[fmt_choice]

        file_path = click.prompt("\nPath to your data file", type=click.Path(exists=True))

        posts = _parse_data_file(file_path, fmt)
        if not posts:
            console.print("[red]No posts found in the file.[/red]")
            sys.exit(1)

        data_path = target / "posts.jsonl"
        save_jsonl(posts, str(data_path))
        console.print(f"  [green]Parsed {len(posts)} posts[/green] → {data_path}")

        if fmt == "csv":
            console.print()
            console.print("  [yellow]Note:[/yellow] LinkedIn CSV exports don't include engagement metrics.")
            console.print("  You'll need to add reactions/comments/reposts manually or via the LinkedIn API.")
            console.print()

        # Analyze
        console.print()
        console.print("[bold]Analyzing your posts...[/bold]")
        result = analyze_file(str(data_path))
        sig_count = sum(1 for s in result.signals if s.significant)

        config_path = target / "midas_config.yaml"
        export_config(result, str(config_path))

        console.print(f"  Posts analyzed: [bold]{result.total_posts}[/bold]")
        console.print(f"  Signals found: [bold]{len(result.signals)}[/bold] ({sig_count} statistically significant)")
        console.print(f"  [green]Config saved to {config_path}[/green]")

        # Validate
        console.print()
        console.print("[bold]Validating your formula...[/bold]")
        from .validate import validate as validate_fn
        import json as _json

        loaded_posts = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    loaded_posts.append(_json.loads(line))

        if len(loaded_posts) >= 5:
            cfg = load_config(str(config_path))
            val_result = validate_fn(loaded_posts, cfg)

            color = "green" if val_result.spearman_rho > 0.3 else "yellow" if val_result.spearman_rho > 0 else "red"
            strength = val_result.correlation_strength.upper()
            sig_str = ", SIGNIFICANT" if val_result.is_significant else ""
            console.print(f"  Spearman rho: [{color}]{val_result.spearman_rho:+.2f}[/{color}] ({strength}{sig_str})")

            if val_result.spearman_rho > 0 and val_result.is_significant:
                console.print("  [green]Your formula predicts engagement![/green]")
        else:
            console.print("  [dim]Not enough posts to validate (need at least 5).[/dim]")

        console.print()
        console.print("[bold]You're all set.[/bold] Try scoring a draft:")
        console.print('  [dim]midas score "Your draft here..."[/dim]')
        console.print()


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="midas_config.yaml", help="Output config path")
@click.option("--hook-max-chars", default=100, help="Max chars for hook detection")
@click.option("--min-frequency", default=0.02, help="Min signal frequency to include (e.g. 0.05 = 5%)")
def analyze(data_path: str, output: str, hook_max_chars: int, min_frequency: float):
    """Analyze your posts and generate a scoring config.

    DATA_PATH should be a JSONL file with your posts and engagement data.
    """
    from .analyze import analyze_file, export_config

    console.print(f"[bold]Analyzing[/bold] {data_path}...")

    result = analyze_file(data_path, hook_max_chars=hook_max_chars, min_frequency=min_frequency)

    sig_count = sum(1 for s in result.signals if s.significant)
    anti_sig = sum(1 for a in result.anti_patterns if a.significant)

    console.print(f"\n  Posts analyzed: [bold]{result.total_posts}[/bold]")
    console.print(f"  Signals found: [bold]{len(result.signals)}[/bold] ({sig_count} statistically significant)")
    console.print(f"  Anti-patterns found: [bold]{len(result.anti_patterns)}[/bold] ({anti_sig} statistically significant)")

    if result.signals:
        console.print("\n[bold]Top signals by lift:[/bold]")
        for s in sorted(result.signals, key=lambda x: -x.median_lift)[:10]:
            sig_marker = " [green]*[/green]" if s.significant else ""
            p_str = f"p={s.p_value:.3f}" if s.p_value < 1.0 else ""
            console.print(
                f"  +{s.weight:.0f}  {s.name}  "
                f"(lift: {s.median_lift:.2f}x, freq: {s.frequency:.0%}, "
                f"CI: [{s.ci_lower:.2f}-{s.ci_upper:.2f}], {p_str}){sig_marker}"
            )

    if result.anti_patterns:
        console.print("\n[bold]Anti-patterns (negative lift):[/bold]")
        for p in sorted(result.anti_patterns, key=lambda x: x.median_lift)[:5]:
            sig_marker = " [green]*[/green]" if p.significant else ""
            console.print(
                f"  {p.weight:.0f}  {p.name}  "
                f"(lift: {p.median_lift:.2f}x, p={p.p_value:.3f}){sig_marker}"
            )

    export_config(result, output)
    console.print(f"\n[green]Config saved to {output}[/green]")
    console.print(f"  Edit the config to tune weights, then use `midas score` to test.")


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(), help="Path to config YAML")
@click.option("--holdout", "-k", type=int, default=0, help="Run k-fold holdout validation (e.g. --holdout 5)")
@click.option("--min-frequency", default=0.02, help="Min signal frequency for holdout re-analysis")
def validate(data_path: str, config: str | None, holdout: int, min_frequency: float):
    """Validate your formula against actual engagement data.

    Scores every post in DATA_PATH and measures how well MIDAS scores
    predict actual engagement using Spearman rank correlation.

    Use --holdout N for k-fold cross-validation (proves generalization).
    """
    from .validate import validate as validate_fn, holdout_validate
    from .analyze import analyze_file

    import json

    # Load posts
    posts: list[dict] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                posts.append(json.loads(line))

    if not posts:
        console.print("[red]No posts found in data file.[/red]")
        sys.exit(1)

    console.print(f"[bold]Validating[/bold] against {len(posts)} posts...")

    if holdout > 0:
        # K-fold cross-validation
        min_posts = holdout * 5
        if len(posts) < min_posts:
            console.print(
                f"[red]Need at least {min_posts} posts for {holdout}-fold CV. "
                f"Got {len(posts)}.[/red]"
            )
            sys.exit(1)
        console.print(f"  Running {holdout}-fold holdout validation...\n")
        cv_result = holdout_validate(posts, n_splits=holdout, min_frequency=min_frequency)

        for i, fold in enumerate(cv_result.fold_results, 1):
            sig = "[green]*[/green]" if fold.is_significant else ""
            color = "green" if fold.spearman_rho > 0.3 else "yellow" if fold.spearman_rho > 0 else "red"
            console.print(
                f"  Fold {i}: rho=[{color}]{fold.spearman_rho:+.4f}[/{color}]  "
                f"p={fold.spearman_p:.4f} {sig}  (n={fold.total_posts})"
            )

        console.print()
        color = "green" if cv_result.mean_rho > 0.3 else "yellow" if cv_result.mean_rho > 0 else "red"
        console.print(f"  Mean rho:  [{color}]{cv_result.mean_rho:+.4f}[/{color}] +/- {cv_result.std_rho:.4f}")

        if cv_result.all_significant and cv_result.mean_rho > 0.3:
            console.print("\n  [bold green]Your formula generalizes to unseen data.[/bold green]")
        elif cv_result.mean_rho > 0:
            console.print("\n  [yellow]Weak positive signal. Collect more data.[/yellow]")
        else:
            console.print("\n  [red]Formula does not generalize. Re-analyze with more data.[/red]")
    else:
        # Standard validation
        cfg = _find_config(config)
        result = validate_fn(posts, cfg)

        color = "green" if result.spearman_rho > 0.3 else "yellow" if result.spearman_rho > 0 else "red"
        console.print(f"\n  Spearman rho:  [{color}]{result.spearman_rho:+.4f}[/{color}]  ({result.correlation_strength})")
        console.print(f"  p-value:       {result.spearman_p:.6f}  ({'[green]SIGNIFICANT[/green]' if result.is_significant else '[red]NOT SIGNIFICANT[/red]'})")

        if result.tier_calibration:
            console.print()
            table = Table(title="Tier Calibration", show_header=True)
            table.add_column("Tier")
            table.add_column("Posts", justify="right")
            table.add_column("Med. Engagement", justify="right")
            table.add_column("Range", justify="right")
            for tc in result.tier_calibration:
                table.add_row(
                    tc.tier_name,
                    str(tc.count),
                    f"{tc.actual_median_engagement:.0f}",
                    f"{tc.engagement_range[0]:.0f}-{tc.engagement_range[1]:.0f}",
                )
            console.print(table)

        if result.spearman_rho > 0 and result.is_significant:
            console.print("\n  [green]Your formula predicts engagement.[/green]")
        else:
            console.print("\n  [yellow]Consider re-analyzing with more data for better predictions.[/yellow]")


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
        console.print(f"  Avg score improvement: [green]{s.avg_score_improvement:+.0f}[/green]")
        console.print(f"  Improvement rate: {s.improvement_rate:.0%}")

        if s.streak > 0:
            console.print(f"  Current streak: [green]{s.streak}[/green] consecutive improvements")
        if s.best_streak > 1:
            console.print(f"  Best streak: {s.best_streak}")
        if s.skill_trend:
            color = "green" if "improving" in s.skill_trend else "yellow" if "stable" in s.skill_trend else "dim"
            console.print(f"  Skill trend: [{color}]{s.skill_trend}[/{color}]")

        if s.most_commonly_added:
            console.print("\n  [bold]Most added signals:[/bold]")
            for name, count in s.most_commonly_added[:5]:
                console.print(f"    +{count}x  {name}")
        if s.most_commonly_removed:
            console.print("  [bold]Most removed signals:[/bold]")
            for name, count in s.most_commonly_removed[:5]:
                console.print(f"    -{count}x  {name}")

        if s.signal_win_rates:
            console.print("\n  [bold]Signal win rates:[/bold]")
            for swr in s.signal_win_rates[:8]:
                if swr.times_added > 0:
                    color = "green" if swr.add_win_rate > 0.6 else "yellow" if swr.add_win_rate > 0.4 else "red"
                    console.print(
                        f"    + {swr.signal_name}: [{color}]{swr.add_win_rate:.0%}[/{color}] win rate "
                        f"({swr.times_adding_improved}/{swr.times_added}), avg delta {swr.avg_delta_when_added:+.0f}"
                    )
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

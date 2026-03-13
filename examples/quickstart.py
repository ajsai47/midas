"""MIDAS Quickstart — End-to-end example of the scoring and analysis workflow."""

from pathlib import Path

from midas.config import load_config
from midas.scorer import score
from midas.export import load_jsonl, get_engagement
from midas.analyze import analyze_posts, export_config


def main():
    # ── Step 1: Load sample data ──
    data_path = Path(__file__).parent / "sample_data.jsonl"
    posts = load_jsonl(str(data_path))
    print(f"Loaded {len(posts)} posts\n")

    # Show engagement distribution
    for post in sorted(posts, key=get_engagement, reverse=True)[:3]:
        eng = get_engagement(post)
        hook = post["text"].split("\n")[0][:60]
        print(f"  {eng:>6.0f}  {hook}...")
    print()

    # ── Step 2: Analyze posts to derive a formula ──
    result = analyze_posts(posts)
    print(f"Analysis complete: {len(result.signals)} signals, {len(result.anti_patterns)} anti-patterns\n")

    # Show top signals
    print("Top signals by engagement lift:")
    for s in result.signals[:5]:
        print(f"  +{s.weight:>4.0f}  {s.name}  (lift: {s.lift:.2f})")
    print()

    # Save as config
    config_path = Path("/tmp/midas_quickstart_config.yaml")
    export_config(result, str(config_path))
    print(f"Config saved to {config_path}\n")

    # ── Step 3: Score posts with the derived config ──
    config = load_config(config_path)

    print("Scoring all posts:")
    scored = []
    for post in posts:
        result = score(post["text"], config)
        hook = post["text"].split("\n")[0][:50]
        scored.append((result.score, result.tier, hook, post["reactions"]))

    for s, tier, hook, reactions in sorted(scored, reverse=True):
        print(f"  {s:>6.0f} [{tier:<17s}] ({reactions:>3d}r) {hook}...")
    print()

    # ── Step 4: Score a new post ──
    new_post = """I spent 6 months reverse-engineering my LinkedIn data.

Here's what I found → engagement isn't random.

It follows patterns. Predictable ones.

Every viral post I've written shares 5 signals:

→ Short punchy hook (under 50 chars)
→ Personal story in the first 3 lines
→ Heavy whitespace between paragraphs
→ Specific numbers and data points
→ Clear CTA in the closing

But here's the thing — YOUR signals are different from mine.

What works for a founder won't work for an engineer.

So I built a tool that finds YOUR formula.

It analyzes your past posts, extracts what works for YOUR audience,
and turns it into a scoring engine you can use before you hit publish.

Comment MIDAS if you want to try it."""

    result = score(new_post, config)
    print(f"New post score: {result.score:.0f} — {result.tier}")
    print(f"  Signals: {', '.join(result.signals.keys())}")
    if result.penalties:
        print(f"  Penalties: {', '.join(result.penalties.keys())}")
    if result.suggestions:
        print(f"  Quick wins:")
        for s in result.suggestions:
            print(f"    → {s}")


if __name__ == "__main__":
    main()

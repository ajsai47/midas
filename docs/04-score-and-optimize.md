# Step 4: Use the Scorer

Your YAML config encodes what works for your audience. The scorer applies it to
any draft and tells you exactly what is working, what is missing, and how to fix it.

## Scoring a Post

**From the command line:**

```bash
# Score inline text
midas score "I spent 6 months reverse-engineering LinkedIn.\n\nHere's what I found →" \
  --config my_config.yaml

# Score from a file
midas score draft.txt --config my_config.yaml
```

**Output:**

```
Score: 620 — HIGH PERFORMER
  Signals:
    +200  topic_primary
    +140  personal_anecdote
    +120  uses_arrows
    +90   hook_personal_i
    +70   word_count_long
  Penalties:
    -55   has_hashtag
  Quick wins:
    → End with 'Comment [WORD]' to drive engagement
    → Remove hashtags — they reduce engagement on LinkedIn
    → Add a narrative pivot ('but here's the thing...') for tension
```

## Reading the Breakdown

The output has four parts:

**Score and Tier.** The raw numeric score and which tier it falls into. This is
the sum of all triggered signals minus all triggered penalties.

**Signals.** Every positive signal that matched, sorted by weight descending.
These are the things you are doing right.

**Penalties.** Every penalty that triggered. These are dragging your score down.
Fix them for easy wins.

**Quick wins.** Suggestions for missing high-value signals. These are signals
in your config that did NOT match the current draft. Adding them would increase
your score.

## The Optimization Workflow

The real power of MIDAS is iterative scoring. The workflow:

```
1. Write your draft
2. Score it
3. Read the "Quick wins" section
4. Add 1-2 missing signals that fit naturally
5. Rescore
6. Repeat until you hit your target tier
```

Example iteration:

```
Draft v1:  Score 280 (ABOVE AVERAGE)
  Missing: cta_comment, uses_arrows, personal_anecdote

  → Add arrows for structure
Draft v2:  Score 400 (ABOVE AVERAGE)
  Missing: cta_comment, personal_anecdote

  → Add closing CTA: "Comment 'build' if you want the template"
Draft v3:  Score 700 (HIGH PERFORMER)
  Missing: personal_anecdote

  → Good enough. The personal_anecdote signal does not fit this topic.
  → Ship it.
```

Do not chase every signal. A post that awkwardly stuffs in every pattern will
read like spam. Use the scorer as a checklist, not a straitjacket.

## Backtesting Against Historical Posts

Score your entire dataset to validate the config:

```bash
midas backtest posts.jsonl --config my_config.yaml
```

Output:

```
Backtest Results — 187 posts
============================
Correlation (score vs reactions): 0.72
Mean score for top-25% posts:     680
Mean score for bottom-25% posts:  140

Tier Accuracy:
  VIRAL CANDIDATE:  avg 234 reactions (12 posts)
  HIGH PERFORMER:   avg 87 reactions  (28 posts)
  ABOVE AVERAGE:    avg 41 reactions  (52 posts)
  AVERAGE:          avg 19 reactions  (61 posts)
  BELOW AVERAGE:    avg 8 reactions   (34 posts)
```

If the correlation is below 0.5, your config needs tuning. Go back to Step 3
and adjust weights or add custom signals.

## JSON Output

For programmatic use, add `--json`:

```bash
midas score draft.txt --config my_config.yaml --json
```

```json
{
  "score": 620,
  "tier": "HIGH PERFORMER",
  "tier_description": "Top 15% — expect 30-50 reactions",
  "signals": {"topic_primary": 200, "personal_anecdote": 140, "uses_arrows": 120},
  "penalties": {"has_hashtag": -55},
  "signal_total": 620,
  "penalty_total": -55,
  "suggestions": ["End with 'Comment [WORD]' to drive engagement"],
  "stats": {"char_count": 1247, "line_count": 32, "word_count": 198}
}
```

## Python API

```python
from midas.config import load_config
from midas.scorer import score, score_text

config = load_config("my_config.yaml")

# Full breakdown
result = score("Your post text here...", config)
print(result)              # Formatted output
print(result.score)        # 620.0
print(result.tier)         # "HIGH PERFORMER"
print(result.signals)      # {"topic_primary": 200, ...}
print(result.suggestions)  # ["End with 'Comment [WORD]'..."]

# Quick numeric score only
numeric = score_text("Your post text here...", config)
print(numeric)  # 620.0
```

The `ScoreResult` object gives you structured access to every field:

```python
from midas.scorer import ScoreResult

result: ScoreResult = score(text, config)

# Check tier programmatically
if result.tier in ("VIRAL CANDIDATE", "HIGH PERFORMER"):
    print("Ready to post")
else:
    print(f"Score {result.score} — review suggestions:")
    for suggestion in result.suggestions:
        print(f"  - {suggestion}")
```

## Integration Tips

**Pre-commit hook.** Score drafts before they leave your editor:

```bash
#!/bin/bash
# .git/hooks/pre-commit (for a content repo)
SCORE=$(midas score "$1" --config my_config.yaml --json | python3 -c "import sys,json; print(json.load(sys.stdin)['score'])")
if [ "$SCORE" -lt 250 ]; then
  echo "MIDAS score $SCORE is below threshold (250). Revise before posting."
  exit 1
fi
```

**CI/CD.** If you store drafts in a git repo, run MIDAS in CI to flag low-scoring
posts before they are scheduled.

**Batch scoring.** Score many drafts at once:

```python
from pathlib import Path
from midas.config import load_config
from midas.scorer import score

config = load_config("my_config.yaml")
for draft in Path("drafts/").glob("*.txt"):
    result = score(draft.read_text(), config)
    print(f"{draft.name}: {result.score:.0f} ({result.tier})")
```

## Next Step

To generate drafts with AI assistance, continue to
[Step 5a: Draft with Claude/GPT](05-llm-integration.md).

To score manually and skip AI, jump to
[Step 6: Close the Loop](07-feedback-loop.md).

# Step 6: Close the Loop

Every edit you make to an AI-generated draft is a training signal. MIDAS captures
these edits and feeds them back into the system -- improving your scoring formula
and fine-tuned model over time.

## Why Feedback Matters

When you rewrite an AI draft before posting, you are expressing a preference:
"this version is better than that version." That preference data is exactly what
DPO training needs.

Without feedback, the system is static. With it:

- Your scoring config gets validated (did high-scoring posts actually perform?)
- Your fine-tuned model learns to avoid the mistakes you keep correcting
- You build a dataset of before/after pairs for continuous improvement

## Logging Edits

After MIDAS generates or rewrites a draft, log your changes:

```bash
# Interactive: shows the generated draft, opens your editor, saves the diff
midas feedback

# Direct: provide both versions
midas feedback --before generated.txt --after final.txt

# With metadata
midas feedback --before generated.txt --after final.txt \
  --topic "open source lessons" \
  --notes "Removed the numbered list, added personal story about GitHub stars"
```

The feedback is stored in `~/.midas/feedback.jsonl`:

```json
{
  "timestamp": "2026-03-12T14:30:00Z",
  "before": "The generated draft text...",
  "after": "Your edited final version...",
  "before_score": 680,
  "after_score": 740,
  "topic": "open source lessons",
  "notes": "Removed the numbered list, added personal story",
  "config": "my_config.yaml"
}
```

## Viewing Feedback Stats

```bash
midas feedback --stats
```

Output:

```
Feedback Summary — 47 entries
==============================
Average score change:  +85 (before: 590 → after: 675)
Score improved:        38/47 (81%)
Score decreased:       6/47 (13%)
Score unchanged:       3/47 (6%)

Most common edits:
  Removed hashtags             18 times
  Shortened hook               14 times
  Added personal anecdote      12 times
  Changed CTA                  11 times
  Removed numbered list         8 times

Signals you keep adding:
  personal_anecdote            12 times (+140 each)
  has_pivot                     9 times (+50 each)
  hook_personal_i               7 times (+90 each)

Signals you keep removing:
  hook_question                 6 times
  has_hashtag                   5 times
```

This tells you two things:
1. Which signals the AI keeps missing (upweight them or improve detection).
2. Which patterns you consistently reject (the model needs DPO training on these).

## Logging Actual Performance

After posting, record the real engagement to validate your formula:

```bash
midas feedback --post-id 2026-03-12-open-source \
  --reactions 156 --comments 23 --reposts 8
```

This links the feedback entry to actual performance data, enabling backtesting.

## Exporting to DPO Format

Once you have 30+ feedback entries, export them for DPO training:

```bash
midas prepare-dpo ~/.midas/feedback.jsonl --output dpo_data.jsonl
```

Each entry becomes a preference pair:

```json
{
  "prompt": "Write a LinkedIn post about open source lessons",
  "chosen": "Your edited version (after)",
  "rejected": "The AI's original draft (before)"
}
```

Entries where the after-score is lower than the before-score are flipped (the
original becomes the chosen output). MIDAS assumes score improvements indicate
genuine preferences, but score decreases might mean you prioritized voice over
formula -- so those pairs are excluded by default. Override with `--include-all`.

## Updating Your Config

Your feedback data can also improve the scoring config itself:

```bash
midas recalibrate --feedback ~/.midas/feedback.jsonl \
  --posts posts.jsonl \
  --config my_config.yaml \
  --output my_config_v2.yaml
```

This re-runs signal analysis on the combined dataset (original posts plus actual
performance of new posts) and adjusts weights. Signals you consistently add
manually get upweighted. Signals the AI over-relies on get downweighted.

## The Virtuous Cycle

The complete MIDAS loop:

```
  Analyze your data
       |
       v
  Build scoring formula ←--------+
       |                          |
       v                          |
  Score & optimize drafts         |
       |                          |
       v                          |
  Post to LinkedIn                |
       |                          |
       v                          |
  Measure real engagement         |
       |                          |
       v                          |
  Log edits + performance --------+
       |
       v
  Retrain model (DPO)
       |
       v
  Better drafts next time
```

Each pass through the loop:
1. Validates or updates your signal weights with real data.
2. Adds preference pairs for model improvement.
3. Reveals new patterns as your content strategy evolves.

## Automation

Set up a weekly routine:

```bash
# Weekly: update your posts file with new performance data
# (manual step or API pull)

# Re-analyze with fresh data
midas analyze posts_updated.jsonl --output my_config_v3.yaml

# If you have 50+ new feedback entries, retrain
midas train-dpo dpo_data.jsonl \
  --sft-checkpoint ./checkpoints/sft-v1 \
  --output ./checkpoints/dpo-v2

# Backtest the new config against historical posts
midas backtest posts_updated.jsonl --config my_config_v3.yaml
```

## Python API

```python
from midas.feedback import log_feedback, get_stats, export_dpo

# Log an edit
log_feedback(
    before="The AI generated this...",
    after="You rewrote it to this...",
    topic="open source lessons",
)

# Get aggregate stats
stats = get_stats()
print(f"Average score improvement: {stats.avg_score_change:+.0f}")
print(f"Most added signal: {stats.most_added_signals[0]}")

# Export for DPO
export_dpo(
    feedback_path="~/.midas/feedback.jsonl",
    output_path="dpo_data.jsonl",
    min_score_improvement=20,
)
```

## Tips

1. **Log every edit.** Even small changes are valuable training signals. The
   more pairs you collect, the better DPO works.
2. **Review feedback stats monthly.** If you keep adding the same signal
   manually, the AI is not learning it -- time for DPO training.
3. **Do not over-optimize.** A post scoring 900 that sounds robotic will
   underperform a post scoring 600 that sounds human. Trust the feedback
   loop to find the balance.
4. **Version your configs.** Name them `my_config_v1.yaml`, `v2`, etc. so you
   can compare performance across config versions.
5. **Share your config.** MIDAS configs are portable. Share yours with your
   team or community to help others calibrate faster (though their weights
   will differ based on their audience).

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
# Provide both versions
midas feedback --original generated.txt --edited final.txt

# With a custom log path and config
midas feedback --original generated.txt --edited final.txt \
  --log my_feedback.jsonl --config my_config.yaml
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--original` / `-o` | (required) | Path to the original/generated draft |
| `--edited` / `-e` | (required) | Path to your edited version |
| `--log` | `midas_feedback.jsonl` | Path to the feedback log file |
| `--config` | `midas_config.yaml` | Your scoring config |

The feedback is stored in `midas_feedback.jsonl` (default) or the path you
specify with `--log`:

```json
{
  "timestamp": "2026-03-12T14:30:00Z",
  "original_text": "The generated draft text...",
  "edited_text": "Your edited final version...",
  "original_score": 680,
  "edited_score": 740,
  "signals_added": ["personal_anecdote", "has_pivot"],
  "signals_removed": ["hook_question", "has_hashtag"]
}
```

## Viewing Feedback Stats

```bash
midas feedback --stats

# Or with a custom log path
midas feedback --stats --log my_feedback.jsonl
```

Output:

```
Feedback Summary — 47 entries
==============================
Average score improvement:  +85

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

## Exporting to DPO Format

Once you have 30+ feedback entries, export them for DPO training:

```bash
midas feedback --export-dpo dpo_data.jsonl

# Or with a custom log path
midas feedback --export-dpo dpo_data.jsonl --log my_feedback.jsonl
```

Each entry becomes a preference pair:

```json
{
  "prompt": "Write a LinkedIn post about open source lessons",
  "chosen": "Your edited version (after)",
  "rejected": "The AI's original draft (before)"
}
```

Entries where the edited score is lower than the original score are flipped (the
original becomes the chosen output). MIDAS assumes score improvements indicate
genuine preferences, but score decreases might mean you prioritized voice over
formula -- so those pairs are excluded by default.

The command returns the number of pairs exported.

## Updating Your Config

Your feedback stats reveal which signals to adjust. If you keep adding a signal
manually, consider upweighting it. If the model over-relies on a signal you
keep removing, downweight it. Re-run `midas analyze` with updated posts to
regenerate your config with fresh engagement data.

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

# If you have 50+ new feedback entries, export and retrain
midas feedback --export-dpo dpo_data.jsonl
python training/prepare_dpo.py dpo_data.jsonl --output dpo_training_data.jsonl
python training/train_dpo.py dpo_training_data.jsonl \
  --sft-checkpoint ./checkpoints/sft-v1 \
  --output ./checkpoints/dpo-v2

# Validate the new config against historical posts
midas validate posts_updated.jsonl --config my_config_v3.yaml
```

## Python API

```python
from midas.config import load_config
from midas.feedback import log_edit, get_stats, export_dpo

config = load_config("my_config.yaml")

# Log an edit
# Returns a FeedbackEntry with: original_text, edited_text, original_score,
# edited_score, signals_added, signals_removed, timestamp
entry = log_edit(
    original="The AI generated this...",
    edited="You rewrote it to this...",
    config=config,
    log_path="midas_feedback.jsonl",
)
print(f"Score: {entry.original_score} → {entry.edited_score}")

# Get aggregate stats
# Returns FeedbackStats with: total_edits, avg_score_improvement,
# most_commonly_added (list of tuples), most_commonly_removed (list of tuples)
stats = get_stats("midas_feedback.jsonl")
print(f"Total edits: {stats.total_edits}")
print(f"Average score improvement: {stats.avg_score_improvement:+.0f}")
print(f"Most added signal: {stats.most_commonly_added[0]}")
print(f"Most removed signal: {stats.most_commonly_removed[0]}")

# Export for DPO training
# Returns int (number of pairs exported)
n_pairs = export_dpo(
    log_path="midas_feedback.jsonl",
    output_path="dpo_data.jsonl",
)
print(f"Exported {n_pairs} DPO pairs")
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

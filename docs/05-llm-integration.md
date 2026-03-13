# Step 5a: Draft with Claude/GPT

MIDAS turns your scoring config into a system prompt, then uses an LLM to generate
drafts that are pre-optimized for your formula. Instead of generic LinkedIn tips,
the model gets your exact signals, weights, and anti-patterns.

## How It Works

1. MIDAS reads your YAML config and builds a system prompt listing every signal,
   penalty, and suggestion with their weights.
2. The LLM generates N candidate drafts for your topic.
3. MIDAS scores each candidate with the scorer.
4. The best-scoring draft is presented, along with its breakdown.

The model does not guess what performs well. It has your formula.

## Setup

Install the LLM extras:

```bash
pip install "midas-linkedin[llm]"
```

Set your API key:

```bash
# Claude (recommended)
export ANTHROPIC_API_KEY="sk-ant-..."

# or GPT
export OPENAI_API_KEY="sk-..."
```

## Generating Drafts

```bash
midas draft "lessons from building an open-source project" --config my_config.yaml
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--config` / `-c` | `midas_config.yaml` | Your scoring config |
| `--provider` / `-p` | `anthropic` | `anthropic`, `openai`, or `local` |
| `--api-key` | env var | API key (overrides env var) |
| `--model` / `-m` | Provider default | Model ID (e.g., `claude-sonnet-4-20250514`) |
| `--samples` / `-n` | `3` | Number of candidates to generate |
| `--temperature` / `-t` | `0.7` | Sampling temperature |

Output:

```
Generating 3 candidates...

--- Candidate 1 — Score: 780 (HIGH PERFORMER) ---
I almost quit open source 4 months in.

Not because the code was hard.
Because nobody cared.

Here's what changed everything →
[...]

--- Candidate 2 — Score: 640 (HIGH PERFORMER) ---
[...]

--- Candidate 3 — Score: 520 (ABOVE AVERAGE) ---
[...]

Best: Candidate 1 (780)
```

## Rewriting an Existing Draft

Already have a draft? Let MIDAS optimize it:

```bash
midas rewrite draft.txt --config my_config.yaml
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--config` / `-c` | `midas_config.yaml` | Your scoring config |
| `--provider` / `-p` | `anthropic` | `anthropic`, `openai`, or `local` |
| `--api-key` | env var | API key (overrides env var) |
| `--model` / `-m` | Provider default | Model ID |

This sends your draft to the LLM along with the scoring breakdown and asks it to
improve the score by adding missing signals and removing penalties. The original
text and intent are preserved.

```
Original:  Score 340 (ABOVE AVERAGE)
Rewrite:   Score 720 (HIGH PERFORMER)

Changes made:
  + Added personal hook ("I" opener)
  + Restructured with → arrows
  + Added closing CTA
  - Removed hashtags
```

## The System Prompt

MIDAS auto-generates a system prompt like this from your config:

```
You are a LinkedIn post writer. Your goal is to write posts that score
highly on the following formula:

POSITIVE SIGNALS (include as many as natural):
  +300  cta_comment — Asking for comments in closing drives engagement
        Detection: closing text contains "comment"
  +200  topic_primary — Posts about [TOPIC] outperform others
        Detection: keywords ["ai agent", "autonomous agent", "agentic"]
  +140  personal_anecdote — Personal stories outperform information-only
        Detection: keywords ["i remember", "years ago", "back in"]
  [...]

PENALTIES (avoid these):
  -55   has_hashtag — Hashtags reduce engagement on LinkedIn
  -30   hook_question — Question hooks underperform declarative hooks
  [...]

STRUCTURE:
  - Hook should be under 100 characters
  - Use heavy line breaks (25+ newlines)
  - Close with a CTA in the last 3 lines

Write naturally. Do not force signals that do not fit the topic.
```

You can generate this prompt programmatically with `generate_system_prompt(config)`
from `midas.draft` (see the Python API section below).

## Provider Options

### Claude (Anthropic)

Default and recommended. Claude tends to produce more natural-sounding posts
with fewer "AI tells" (overused phrases, excessive structure).

```bash
midas draft "topic" --provider anthropic --model claude-sonnet-4-20250514
```

### GPT (OpenAI)

```bash
midas draft "topic" --provider openai --model gpt-4o
```

### Local Models

Use any OpenAI-compatible local server (vLLM, Ollama, llama.cpp):

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="not-needed"
midas draft "topic" --provider local --model my-model
```

This is especially useful if you fine-tune a model on your posts (see
[Step 5b: Fine-Tuning](06-fine-tuning.md)).

## Best-of-N Sampling

The `--samples` flag controls how many candidates MIDAS generates and scores.
Higher N means better best-of-N results but more API cost.

| Samples | Typical Best Score Improvement | API Cost (Claude Sonnet) |
|---------|-------------------------------|-------------------------|
| 1 | Baseline | ~$0.01 |
| 3 | +15-25% over single sample | ~$0.03 |
| 5 | +20-35% over single sample | ~$0.05 |
| 10 | +25-40% over single sample | ~$0.10 |

Three samples is the sweet spot for most use cases.

## Python API

```python
from midas.config import load_config
from midas.draft import draft, rewrite, generate_system_prompt

config = load_config("my_config.yaml")

# Inspect the system prompt MIDAS builds from your config
prompt = generate_system_prompt(config)
print(prompt)

# Generate new drafts
results = draft(
    topic="lessons from building an open-source project",
    config=config,
    provider="anthropic",
    num_samples=3,
    temperature=0.7,
)

# results is a list[DraftResult], sorted by score descending
# DraftResult has: text, score_result, provider, model
best = results[0]
print(best.text)
print(best.score_result)

# Rewrite an existing draft
original = open("draft.txt").read()
result = rewrite(
    draft_text=original,
    config=config,
    provider="anthropic",
    temperature=0.5,
)
# result is a DraftResult
print(result.text)
print(result.score_result)
```

## Tips

1. **Do not accept AI output verbatim.** Use it as a starting point and add your
   voice, specific details, and real anecdotes.
2. **Score after editing.** Run `midas score` on your final version to make sure
   your edits did not drop key signals.
3. **Iterate on the prompt.** If the LLM keeps producing a style you dislike,
   adjust your config's signals and suggestions to steer the output.
4. **Combine with rewrite.** Write a rough draft, score it, then use
   `midas rewrite` to optimize structure while keeping your voice.

## Next Step

For even better results, fine-tune a model on your own posts:
[Step 5b: Fine-Tune Your Own Model](06-fine-tuning.md).

Or skip to [Step 6: Close the Loop](07-feedback-loop.md).

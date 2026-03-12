# MIDAS

**Everything you touch turns to gold.**

MIDAS is an open-source framework for reverse-engineering your LinkedIn performance data into a personalized scoring formula and AI-powered post optimization.

Built from 1,046 posts over 2 years. Battle-tested. Now open-source.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Why MIDAS?

Generic "LinkedIn tips" don't work because **your audience is unique**. What drives engagement for a founder in Portland is different from an engineer in SF or a recruiter in NYC.

MIDAS takes YOUR data and extracts YOUR formula:

```
Your posts + engagement data
        ↓
  Signal analysis (what patterns correlate with high engagement?)
        ↓
  Personalized scoring config (YAML)
        ↓
  Score drafts before publishing / Generate optimized posts with AI
        ↓
  Track edits → close the feedback loop → improve over time
```

## Quickstart

```bash
pip install midas-linkedin
```

### 1. Analyze your data

```bash
# Your data: JSONL with {text, reactions, comments, reposts, date}
midas analyze my_posts.jsonl --output my_config.yaml
```

This analyzes every post, computes signal lifts, and generates a scoring config with weights calibrated to YOUR audience.

### 2. Score a post

```bash
midas score "I just built something incredible..."
```

```
  Score: 485 — HIGH PERFORMER

  Signals:
    +300  cta_comment
    +140  personal_anecdote
    +100  uses_arrows

  Penalties:
    -55   has_hashtag

  Quick wins:
    → Shorten your hook to under 50 characters
    → Add specific numbers for credibility
```

### 3. Generate with AI

```bash
export ANTHROPIC_API_KEY=sk-...
midas draft "my experience building AI agents from scratch"
```

MIDAS generates a system prompt from your config, creates multiple drafts, scores each one, and shows you the best.

### 4. Close the loop

```bash
midas feedback --original draft.txt --edited final.txt
midas feedback --stats
```

Track what you change, learn your editing patterns, export to DPO format for fine-tuning.

## How It Works

### The Scoring Engine

Every post is evaluated against **signals** (positive patterns) and **penalties** (anti-patterns). Each has a weight derived from your data:

```yaml
signals:
  - name: hook_personal_i
    weight: 90                    # Engagement lift when present
    scope: hook                   # Only checks the first line
    regex: "^I[' ]"              # Starts with "I " or "I'"

  - name: cta_comment
    weight: 300
    scope: close                  # Only checks last 3 lines
    keywords: ["comment"]

penalties:
  - name: has_hashtag
    weight: 55                    # Gets subtracted
    regex: "#\\w+"
```

Weights come from **statistical lift**: `mean_engagement_with_signal / mean_engagement_without`. A signal with 2x lift and 50 average reactions might get a weight of 100.

### Two Paths to AI-Powered Drafting

| | Path A: LLM API | Path B: Fine-tuning |
|---|---|---|
| **Setup** | API key only | GPU + training pipeline |
| **How** | System prompt from your config | SFT → DPO on your posts |
| **Data needed** | 50+ posts | 200+ posts |
| **Quality** | Good | Better |
| **Cost** | Per-request API cost | One-time training cost |
| **Best for** | Getting started | Power users |

## Architecture

```
midas/
├── config.py       # YAML config loader (signals, penalties, tiers)
├── scorer.py       # Scoring engine — config-driven, no hardcoded weights
├── analyze.py      # Signal extraction from raw post data
├── export.py       # LinkedIn data export helpers
├── draft.py        # LLM-powered drafting (Claude/GPT/local)
├── feedback.py     # Edit logging + DPO data generation
└── cli.py          # CLI interface

training/           # Optional fine-tuning pipeline
├── prepare_sft.py  # Convert posts → SFT data
├── prepare_dpo.py  # Create preference pairs
├── train_sft.py    # SFT with HuggingFace + TRL
└── train_dpo.py    # DPO training
```

## Documentation

| Guide | What you'll learn |
|-------|-------------------|
| [01 — Export Your Data](docs/01-export-your-data.md) | Get your LinkedIn posts into MIDAS format |
| [02 — Analyze Signals](docs/02-analyze-signals.md) | Extract what drives YOUR engagement |
| [03 — Build Your Formula](docs/03-build-your-formula.md) | Customize your scoring config |
| [04 — Score & Optimize](docs/04-score-and-optimize.md) | Score posts before publishing |
| [05 — LLM Integration](docs/05-llm-integration.md) | Generate posts with Claude/GPT |
| [06 — Fine-Tuning](docs/06-fine-tuning.md) | Train your own model (optional) |
| [07 — Feedback Loop](docs/07-feedback-loop.md) | Close the loop with edit tracking |

## Installation

```bash
# Core (scoring + analysis + CLI)
pip install midas-linkedin

# With LLM support (Claude/GPT drafting)
pip install "midas-linkedin[llm]"

# With training support (fine-tuning pipeline)
pip install "midas-linkedin[training]"

# Everything
pip install "midas-linkedin[all]"
```

## Python API

```python
from midas.config import load_config
from midas.scorer import score
from midas.analyze import analyze_file, export_config
from midas.draft import draft
from midas.feedback import log_edit, get_stats

# Analyze your data
result = analyze_file("my_posts.jsonl")
export_config(result, "my_config.yaml")

# Score a post
config = load_config("my_config.yaml")
result = score("Your post text here...", config)
print(f"{result.score:.0f} — {result.tier}")

# Generate with AI
drafts = draft("topic", config, provider="anthropic")
print(drafts[0].text)  # Best-scoring draft
```

## Data Format

MIDAS uses a simple JSONL format. One post per line:

```json
{
  "text": "Your post text here...",
  "reactions": 47,
  "comments": 23,
  "reposts": 8,
  "date": "2026-02-15",
  "has_image": false
}
```

See [01 — Export Your Data](docs/01-export-your-data.md) for how to get your data into this format.

## Requirements

- Python 3.10+
- For LLM drafting: `anthropic` or `openai` SDK + API key
- For fine-tuning: GPU with 24GB+ VRAM (or cloud GPU)

## Contributing

PRs welcome. The most impactful contributions:

1. **New signal detectors** — found a pattern that predicts engagement? Add it
2. **Export helpers** — parsers for different data sources
3. **Provider integrations** — support for more LLM providers
4. **Documentation** — tutorials, case studies, guides

## License

MIT


```
 ███╗   ███╗ ██╗ ██████╗   █████╗  ███████╗
 ████╗ ████║ ██║ ██╔══██╗ ██╔══██╗ ██╔════╝
 ██╔████╔██║ ██║ ██║  ██║ ███████║ ███████╗
 ██║╚██╔╝██║ ██║ ██║  ██║ ██╔══██║ ╚════██║
 ██║ ╚═╝ ██║ ██║ ██████╔╝ ██║  ██║ ███████║
 ╚═╝     ╚═╝ ╚═╝ ╚═════╝  ╚═╝  ╚═╝ ╚══════╝
```

**Everything you touch turns to gold.**

Reverse-engineer your LinkedIn into a personalized scoring formula. 1,046 posts. 2 years of data. Now open-source.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

Generic LinkedIn tips don't work. Your audience is unique. What goes viral for a founder in Portland is invisible to an engineer in SF.

## The Fix

```
Your posts + engagement data
        ↓
  Statistical signal analysis
        ↓
  Personalized scoring formula (YAML)
        ↓
  Score before you post / Generate with AI
        ↓
  Track edits → feedback loop → compound
```

---

## Quickstart

```bash
pip install midas-linkedin
```

**Step 1 — Analyze your data**
```bash
midas analyze posts.jsonl -o my_config.yaml
```

**Step 2 — Score a draft**
```bash
midas score "I spent 6 months building something nobody asked for..."
```

```
  Score: 485  VIRAL CANDIDATE

  Signals
  ────────────────────────────
  +300  cta_comment
  +140  personal_anecdote
  +100  uses_arrows

  Penalties
  ────────────────────────────
  -55   has_hashtag

  Quick wins
  → Shorten your hook to under 50 characters
  → Add specific numbers for credibility
```

**Step 3 — Generate with AI**
```bash
export ANTHROPIC_API_KEY=sk-...
midas draft "lessons from building AI agents for 2 years"
```

**Step 4 — Close the loop**
```bash
midas feedback --original draft.txt --edited final.txt
midas feedback --stats
```

---

## How It Works

### Signals + Penalties = Your Formula

Every post is scored against **signals** (patterns that predict engagement) and **penalties** (patterns that kill it). Weights come from your data — not opinions.

```yaml
signals:
  - name: hook_personal_i
    weight: 90
    scope: hook
    regex: "^I[' ]"

  - name: cta_comment
    weight: 300
    scope: close
    keywords: ["comment"]

penalties:
  - name: has_hashtag
    weight: 55
    regex: "#\\w+"
```

Weights = **statistical lift** × baseline engagement. A signal with 2x lift means posts with that pattern get 2x the engagement.

### Two Paths

| | LLM API | Fine-tuning |
|---|---|---|
| **Setup** | API key | GPU + training |
| **How** | System prompt from config | SFT → DPO on your posts |
| **Data** | 50+ posts | 200+ posts |
| **Quality** | Good | Better |
| **Best for** | Getting started | Power users |

---

## Architecture

```
midas/
├── config.py       Config loader — signals, penalties, tiers
├── scorer.py       Scoring engine — config-driven, zero hardcoded weights
├── analyze.py      Signal extraction — lift computation from raw data
├── export.py       Data helpers — LinkedIn CSV, Apify JSON, JSONL
├── draft.py        LLM drafting — Claude, GPT, or local models
├── feedback.py     Edit tracking — DPO export for fine-tuning
└── cli.py          CLI — analyze, score, draft, rewrite, feedback

training/
├── prepare_sft.py  Posts → supervised fine-tuning data
├── prepare_dpo.py  Engagement pairs → preference data
├── train_sft.py    SFT with HuggingFace + TRL + LoRA
└── train_dpo.py    DPO training
```

---

## Docs

| Step | Guide | What you'll learn |
|------|-------|-------------------|
| 1 | [Export Your Data](docs/01-export-your-data.md) | Get posts into MIDAS format |
| 2 | [Analyze Signals](docs/02-analyze-signals.md) | Extract what drives YOUR engagement |
| 3 | [Build Your Formula](docs/03-build-your-formula.md) | Customize scoring config |
| 4 | [Score & Optimize](docs/04-score-and-optimize.md) | Score before publishing |
| 5a | [LLM Integration](docs/05-llm-integration.md) | Generate with Claude/GPT |
| 5b | [Fine-Tuning](docs/06-fine-tuning.md) | Train your own model |
| 6 | [Feedback Loop](docs/07-feedback-loop.md) | Close the loop |

---

## Install

```bash
pip install midas-linkedin              # Core — scoring + analysis + CLI
pip install "midas-linkedin[llm]"       # + Claude/GPT drafting
pip install "midas-linkedin[training]"  # + fine-tuning pipeline
pip install "midas-linkedin[all]"       # Everything
```

## Python API

```python
from midas.analyze import analyze_file, export_config
from midas.config import load_config
from midas.scorer import score
from midas.draft import draft

# Analyze → config → score → draft
result = analyze_file("posts.jsonl")
export_config(result, "config.yaml")

config = load_config("config.yaml")
print(score("Your post here...", config).tier)

drafts = draft("topic", config, provider="anthropic")
```

## Data Format

One post per line. JSONL.

```json
{"text": "Your post...", "reactions": 47, "comments": 23, "reposts": 8, "date": "2026-02-15", "has_image": false}
```

Get your data via [Apify scraper](docs/01-export-your-data.md) (recommended), LinkedIn export, or manual tracking.

---

## How Much Data?

| Posts | Quality |
|-------|---------|
| 20-49 | Rough signals. Better than nothing. |
| 50-99 | Usable. Main signals clear. |
| 100-199 | Solid. Statistically meaningful lifts. |
| 200+ | Best. Enough for fine-tuning. |

---

## Contributing

PRs welcome.

1. **Signal detectors** — found a pattern that predicts engagement? Add it
2. **Export helpers** — parsers for new data sources
3. **LLM providers** — support for more models
4. **Case studies** — share your results

## License

MIT

# MIDAS

**MIDAS turns your LinkedIn engagement data into a personalized scoring formula, so you know if a post will perform before you hit publish.**

Analyze your past posts, extract the patterns that drive YOUR engagement, score drafts against your formula, generate new posts with AI, and close the feedback loop. Your audience is unique. Your formula should be too.

### Without MIDAS

- You follow generic LinkedIn advice that was written for someone else's audience
- You have no idea if a post will perform until you publish it and wait
- You copy "viral post" templates that worked for a founder in SF but die for an engineer in Portland
- Hashtags? No hashtags? Long hooks? Short hooks? You are guessing every time
- Your best posts felt lucky — you cannot explain why they worked
- You edit drafts by gut, never tracking whether your edits actually improved anything

### With MIDAS

| Command | What it does |
|---------|-------------|
| `midas analyze posts.jsonl` | Extract signals from your post history. Compute statistical lift. Generate your personal scoring formula. |
| `midas score "your draft..."` | Score a draft against your formula before you publish. See exactly which signals hit and which are missing. |
| `midas draft "topic"` | Generate 3 LinkedIn posts with Claude or GPT, guided by your formula. Scored and ranked automatically. |
| `midas rewrite draft.txt` | Take an existing draft and rewrite it to score higher. See the before/after delta. |
| `midas feedback -o draft.txt -e final.txt` | Log your edits. Track which signals you add and remove. Export as DPO data for fine-tuning. |
| `midas init` | Set up MIDAS in your project with sample config and guided onboarding. |

## Demo: from raw data to scored draft

I start by analyzing my post history. MIDAS finds the patterns — what I write when engagement is high vs low. Then I use the formula to score drafts before publishing.

```
$ midas analyze posts.jsonl -o my_config.yaml

  Analyzing posts.jsonl...

  Posts analyzed: 187
  Signals found: 12
  Anti-patterns found: 3

  Top signals by lift:
  +300  cta_comment       (lift: 3.20, freq: 24%)
  +200  personal_anecdote (lift: 2.10, freq: 31%)
  +160  hook_exclamation  (lift: 1.60, freq: 14%)
  +120  uses_arrows       (lift: 1.25, freq: 42%)

  Anti-patterns (negative lift):
  -110  has_hashtag       (lift: 0.55, freq: 38%)
  -60   has_link          (lift: 0.72, freq: 19%)

  Config saved to my_config.yaml
```

The weights are not opinions. They are computed from my data. `cta_comment` gets +300 because posts where I ask for comments get 3.2x more engagement than posts where I don't. `has_hashtag` gets -110 because my posts with hashtags get roughly half the engagement.

Now I score a draft:

```
$ midas score "I spent 6 months reverse-engineering my LinkedIn data.

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

Comment MIDAS if you want to try it."

  Score: 710  HIGH PERFORMER

  Signals
  ──────────────────────────────
  +300  cta_comment
  +200  personal_anecdote
  +120  uses_arrows
  +90   hook_personal_i
  +50   has_pivot

  Penalties
  ──────────────────────────────
  -50   (none significant)

  Quick wins:
  → Add specific numbers for credibility
  → Add heavy linebreaks (25+ newlines) for scannability
```

710 is HIGH PERFORMER. The formula tells me exactly which signals are firing and what I could add to push it higher.

Now I generate drafts with AI:

```
$ midas draft "lessons from building AI agents for 2 years"

  Drafting 3 samples about: lessons from building AI agents for 2 years
  Provider: anthropic

  ╭─ Draft 1 — Score: 820 (VIRAL CANDIDATE) ───────────╮
  │ I've built 14 AI agents in the last 2 years.       │
  │                                                     │
  │ 11 of them failed.                                  │
  │                                                     │
  │ But the 3 that worked changed everything...         │
  │ ...                                                 │
  │ Comment AGENT if you want the full breakdown.       │
  ╰─────────────────────────────────────────────────────╯

  Quick wins for the best draft:
  → Add heavy linebreaks for scannability
```

The system prompt is generated directly from my scoring config. The LLM is not writing generic LinkedIn content — it is writing to MY formula. Then each draft is scored and ranked.

After I edit the draft and publish, I log the edit:

```
$ midas feedback --original draft.txt --edited final.txt

  Original score: 820
  Edited score:   890
  Delta:          +70
  Signals added:  has_data, heavy_linebreaks
  Signals removed: (none)
  Logged to midas_feedback.jsonl
```

Over time, my editing patterns become training data. I can export them as DPO pairs for fine-tuning a model that writes like me.

## Who this is for

You post on LinkedIn regularly and want to stop guessing what works. You have at least 20-50 posts of history. You want a data-driven formula, not vibes.

This is not a "10 tips for LinkedIn success" blog post. It is a scoring engine calibrated to your specific audience, your specific voice, your specific niche.

If you have never posted on LinkedIn, start posting first. Come back when you have data.

## Install

**Requirements:** Python 3.10+

```bash
pip install git+https://github.com/ajsai47/midas.git
```

That gives you the CLI. For AI drafting, add the LLM extra:

```bash
pip install "midas-linkedin[llm] @ git+https://github.com/ajsai47/midas.git"
```

From source:

```bash
git clone https://github.com/ajsai47/midas.git && cd midas
pip install -e ".[all]"
```

### Getting started

```bash
midas init
```

This copies a sample config and sample data into your directory, then shows you what to do next.

---

## `midas analyze`

This is where your formula comes from.

Give it a JSONL file of your posts with engagement data. It tests every candidate signal against your data and computes the statistical lift — how much more engagement you get when that pattern is present vs absent.

Signals with lift > 1.0 become positive weights. Signals with lift < 1.0 become penalties.

The output is a YAML config file. Every weight in it is justified by your data.

### What it detects

MIDAS tests 16 candidate signals across four categories:

**Hook patterns** — what makes people click "see more":
- Starting with "I" (personal story)
- Short punchy hooks under 50 chars
- Numbers in the hook
- Emotional openers (Wow, Whoa, Wait)
- Question hooks
- Superlatives

**Structure patterns** — formatting that keeps people reading:
- Arrow (→) formatting
- Heavy linebreaks (25+ newlines)
- Narrative pivots ("but here's the thing...")
- Long-form posts (1000+ chars)

**Content patterns** — what topics drive engagement:
- Personal anecdotes
- Specific numbers and data
- Images

**CTA patterns** — what drives comments:
- "Comment" in the closing
- Newsletter/subscribe CTAs (often a penalty)

Every signal that appears in at least 2% of your posts gets tested. The lift is computed, the weight is derived, and it lands in your config.

### Example

```bash
midas analyze my_posts.jsonl -o my_config.yaml --min-frequency 0.05
```

You can also add your own signals to the config after generation. The format is simple YAML:

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

The config is yours. Edit it. Add topic-specific signals. Tune the weights. The tool works for you, not the other way around.

---

## `midas score`

This is the core loop.

Write a draft. Score it. See what is working and what is missing. Edit. Score again. Repeat until the number is where you want it.

```bash
# Inline
midas score "Your draft here..."

# From file
midas score -f draft.txt

# From stdin
cat draft.txt | midas score
```

The output shows the total score, the tier, every signal that matched, every penalty that fired, and specific suggestions for what you could add.

### Tiers

Tiers map raw scores to human-readable performance predictions. They are calibrated from your data during analysis:

| Tier | Score | What it means |
|------|-------|--------------|
| VIRAL CANDIDATE | 800+ | Top 5% of your posts |
| HIGH PERFORMER | 500+ | Top 15% |
| ABOVE AVERAGE | 250+ | Top 35% |
| AVERAGE | 100+ | Median range |
| BELOW AVERAGE | <100 | Consider revising |

---

## `midas draft` and `midas rewrite`

These are the AI modes.

`draft` generates new posts from a topic. `rewrite` takes an existing draft and improves it.

Both work by converting your scoring config into a system prompt. The LLM is not writing generic LinkedIn content. It is writing to your specific formula — your signals, your penalties, your structure rules. Then each output is scored against the same formula.

```bash
# Generate 3 drafts
midas draft "your topic" --provider anthropic --samples 3

# Rewrite an existing draft
midas rewrite draft.txt --provider openai
```

Supports Anthropic (Claude), OpenAI (GPT), and local models via OpenAI-compatible APIs.

| Provider | Default model | Setup |
|----------|--------------|-------|
| `anthropic` | claude-sonnet-4-20250514 | `export ANTHROPIC_API_KEY=sk-...` |
| `openai` | gpt-4o | `export OPENAI_API_KEY=sk-...` |
| `local` | default | Run an OpenAI-compatible server on port 8000 |

---

## `midas feedback`

This is the feedback loop.

Every time you edit a draft before publishing, log the edit. MIDAS scores both versions, computes the delta, and tracks which signals you added or removed.

```bash
# Log an edit
midas feedback --original draft.txt --edited final.txt

# See your editing patterns
midas feedback --stats

# Export as DPO data for fine-tuning
midas feedback --export-dpo training_pairs.jsonl
```

Over time, this builds a dataset of your editing preferences — what you keep, what you cut, what you add. That dataset can train a model that writes like you from the start.

---

## Fine-tuning

For power users with 200+ posts. The `training/` directory contains standalone scripts for:

```bash
python training/prepare_sft.py -i posts.jsonl -o ./data/     # Posts → SFT data
python training/train_sft.py --data ./data/ --output ./model/ # Train with LoRA
python training/prepare_dpo.py -i posts.jsonl -o ./data/      # Engagement pairs → DPO data
python training/train_dpo.py --data ./data/ --output ./model/  # DPO training
```

Uses HuggingFace Transformers + TRL + PEFT. Works with any causal LM.

---

## Data format

One post per line. JSONL.

```json
{"text": "Your post...", "reactions": 47, "comments": 23, "reposts": 8, "date": "2026-02-15", "has_image": false}
```

Get your data via [Apify scraper](docs/01-export-your-data.md) (recommended), LinkedIn CSV export, or manual tracking.

| Posts | Quality |
|-------|---------|
| 20-49 | Rough signals. Better than nothing. |
| 50-99 | Usable. Main signals clear. |
| 100-199 | Solid. Statistically meaningful lifts. |
| 200+ | Best. Enough for fine-tuning. |

---

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
print(drafts[0].text)           # Best draft
print(drafts[0].score_result)   # Its score breakdown
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

## Contributing

PRs welcome.

1. **Signal detectors** — found a pattern that predicts engagement? Add it
2. **Export helpers** — parsers for new data sources
3. **LLM providers** — support for more models
4. **Case studies** — share your results

## License

MIT

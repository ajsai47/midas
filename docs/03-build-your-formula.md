# Step 3: Derive Your Weights

The analysis from Step 2 generates a YAML config with auto-suggested weights.
This step is about understanding that config, refining it, and making it yours.

## The YAML Config Structure

A MIDAS config has five sections:

```yaml
name: my_formula
description: Personalized scoring formula for @yourhandle
hook_max_chars: 100
close_lines: 3

signals:       # Patterns that boost engagement (positive weights)
penalties:     # Patterns that hurt engagement (negative weights)
tiers:         # Score thresholds mapped to performance labels
suggestions:   # Actionable tips when signals are missing
```

### Signals

Each signal has a detection method and a weight:

```yaml
signals:
  - name: hook_personal_i
    weight: 90
    description: Starting with "I" signals personal story
    scope: hook
    regex: "^I[' ]"

  - name: personal_anecdote
    weight: 140
    description: Personal stories outperform information-only posts
    keywords: ["i remember", "years ago", "back in", "my first", "when i"]

  - name: word_count_long
    weight: 70
    description: Longer posts tend to perform better
    field: char_count
    min_value: 1000
```

**Detection methods** (choose one per signal):

| Method | Fields | How It Works |
|--------|--------|-------------|
| Regex | `regex` | Python regex match against text (case-insensitive) |
| Keywords | `keywords` | Any keyword found in text (case-insensitive) |
| Threshold | `field` + `min_value` | Numeric stat >= threshold |

**Scope** controls which part of the post is checked:

| Scope | Text Region |
|-------|------------|
| `full` (default) | Entire post |
| `hook` | First line only |
| `close` | Last N lines (set by `close_lines`) |

### Penalties

Same structure as signals but weights get negated automatically:

```yaml
penalties:
  - name: has_hashtag
    weight: 55          # Stored as -55 internally
    description: Hashtags reduce engagement on LinkedIn
    regex: "#\\w+"
```

### Tiers

Map raw scores to human-readable labels. Sorted highest-first:

```yaml
tiers:
  - name: "VIRAL CANDIDATE"
    min_score: 800
    description: "Top 5% — expect 50+ reactions"
  - name: "HIGH PERFORMER"
    min_score: 500
    description: "Top 15% — expect 30-50 reactions"
  - name: "ABOVE AVERAGE"
    min_score: 250
  - name: "AVERAGE"
    min_score: 100
  - name: "BELOW AVERAGE"
    min_score: 0
```

### Suggestions

Shown when a signal is absent or a penalty is triggered:

```yaml
suggestions:
  - signal: uses_arrows
    message: "Use → arrows instead of bullets for better scannability"
  - signal: has_hashtag
    message: "Remove hashtags — they reduce engagement on LinkedIn"
```

## Interpreting Auto-Generated Weights

The analyzer sets weights proportional to engagement lift. A signal with 3.0x lift
gets a higher weight than one with 1.2x lift. But raw lift is not the whole story.

Consider two signals:

| Signal | Lift | Frequency | Auto Weight |
|--------|------|-----------|-------------|
| cta_comment | 3.2x | 18% | +300 |
| word_count_long | 1.2x | 52% | +70 |

`cta_comment` has the higher weight, but it appears in only 18% of posts. If you
add it to every post, it will stop being a differentiator and the lift may drop.
`word_count_long` appears in 52% of posts and consistently performs above average.

Rule of thumb: **trust high-frequency, moderate-lift signals more than low-frequency,
high-lift signals.**

## Adjusting Weights for Your Goals

Not all engagement is equal. You might want to optimize for:

- **Comments over reactions.** Upweight `cta_comment`, `has_pivot`, `hook_question`
  (yes, even if questions have lower reaction lift, they may drive replies).
- **Reach/impressions.** Upweight signals that drive reposts: controversial takes,
  data-heavy posts, frameworks.
- **Authority.** Downweight engagement-bait patterns; upweight `has_data`,
  `topic_primary`, `personal_anecdote`.

MIDAS does not prescribe a goal. It gives you the data; you set the priorities.

## Adding Custom Signals

Add signals specific to your niche. Examples:

```yaml
# Detect mentions of a specific technology
- name: topic_ai_agents
  weight: 200
  keywords: ["ai agent", "autonomous agent", "agentic", "tool use"]

# Detect a writing pattern (numbered lists)
- name: numbered_list
  weight: 80
  regex: "\\n\\d+[.)\\s]"

# Detect image presence (requires has_image field in data)
- name: includes_image
  weight: 60
  field: has_image
  min_value: 1
```

For regex signals, test your patterns before adding them:

```python
import re
pattern = r"\n\d+[.)\s]"
text = "Here are 3 lessons:\n1. Start small\n2. Ship fast\n3. Iterate"
print(bool(re.search(pattern, text)))  # True
```

## Setting Tier Thresholds

Tiers should reflect your actual engagement distribution. Score your historical
posts and look at the score distribution:

```python
from midas.export import load_jsonl
from midas.config import load_config
from midas.scorer import score
import statistics

config = load_config("my_config.yaml")
posts = load_jsonl("posts.jsonl")
scores = [score(p["text"], config).score for p in posts]
scores.sort()

n = len(scores)
print(f"Score Distribution ({n} posts):")
print(f"  p95:  {scores[int(n * 0.95)]:.0f}   → VIRAL CANDIDATE threshold")
print(f"  p85:  {scores[int(n * 0.85)]:.0f}   → HIGH PERFORMER threshold")
print(f"  p65:  {scores[int(n * 0.65)]:.0f}   → ABOVE AVERAGE threshold")
print(f"  p50:  {scores[int(n * 0.50)]:.0f}   → AVERAGE threshold")
print(f"  below p50   → BELOW AVERAGE")
```

Set your tier thresholds to roughly match these percentiles so that tiers
correspond to real-world outcomes.

## Walkthrough: From Analysis to Config

Say the analysis of 150 posts yields:

```
hook_personal_i    1.6x lift   40% freq
uses_arrows        2.0x lift   20% freq
topic_saas         2.5x lift   35% freq
cta_question       1.8x lift   15% freq
has_hashtag        0.65x lift  50% freq
```

Step through the reasoning:

1. `topic_saas` has the highest lift and solid frequency. Top signal. Weight: 200.
2. `uses_arrows` has 2.0x lift but only 20% frequency. Moderate signal. Weight: 120.
3. `cta_question` has 1.8x lift at 15% frequency. Borderline. Weight: 80.
4. `hook_personal_i` has moderate lift but highest frequency -- reliable. Weight: 90.
5. `has_hashtag` is below 1.0 -- it is a penalty. Weight: -55.

Tier thresholds: the max possible score if all signals fire is
200 + 120 + 80 + 90 = 490. Set VIRAL at 400, HIGH at 280, ABOVE AVG at 150,
AVG at 60.

The final config is ready to score new posts.

## Next Step

With your config built, move on to
[Step 4: Use the Scorer](04-score-and-optimize.md).

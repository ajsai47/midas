# Step 2: Run the Analysis

This step takes your raw post data and extracts the patterns that actually drive
engagement for *your* audience. No generic LinkedIn tips -- just math.

## What Signal Analysis Does

MIDAS scans every post in your dataset for detectable patterns (signals) and
computes how each one correlates with engagement. The core metric is **lift**:
how much better does a post perform when a signal is present versus absent?

For example, if your posts that start with "I" average 180 reactions, and posts
that do not average 95 reactions, the lift for `hook_personal_i` is 1.89x.

## Running the Analysis

```bash
midas analyze posts.jsonl --output my_config.yaml
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--output` / `-o` | `midas_config.yaml` | Output config path |
| `--min-frequency` | `0.02` | Minimum signal frequency to include (2%) |
| `--hook-max-chars` | `100` | Characters to consider as the "hook" |

Engagement is always computed as a weighted sum: `reactions + comments*2 + reposts*3`.

## Understanding the Output

The analyzer prints a table like this:

```
MIDAS Signal Analysis
==================================================
Signal                   Lift   Freq   Suggested Weight
---------------------------------------------------------
cta_comment              3.2x   18%    +300
topic_ai_agents          2.8x   22%    +200
hook_exclamation         2.1x   12%    +160
personal_anecdote        1.9x   31%    +140
uses_arrows              1.8x   24%    +120
heavy_linebreaks         1.6x   45%    +95
hook_personal_i          1.5x   38%    +90
topic_developer_tools    1.3x   19%    +75
word_count_long          1.2x   52%    +70
hook_number              1.2x    8%    +60
has_pivot                1.2x   27%    +50
hook_short_teaser        1.1x   15%    +40
has_data                 1.0x   33%     +20

Anti-patterns (lift < 1.0):
---------------------------------------------------------
has_hashtag              0.7x   41%    -55
hook_question            0.8x   14%    -30
cta_newsletter           0.9x    9%    -20
```

### Reading the Columns

**Lift** is the ratio of mean engagement with the signal to mean engagement without
it. A lift of 2.0x means posts with this signal get twice the engagement on average.

```
lift = mean_engagement_with_signal / mean_engagement_without_signal
```

Lifts above 1.0 become positive signals. Lifts below 1.0 become penalties.

**Frequency** is how often the signal appears in your dataset. A signal with 3.0x
lift but 2% frequency might be noise. MIDAS filters out signals below
`--min-frequency` (default 2%) to avoid overfitting to rare patterns.

**Suggested Weight** is a scaled version of lift, normalized so that your highest-
impact signal gets the largest weight. These are starting points -- you can and
should tune them.

## How Lift Is Calculated

For each candidate signal, MIDAS splits your posts into two groups:

```
Group A: posts where the signal is present
Group B: posts where the signal is absent

lift = mean(Group A engagement) / mean(Group B engagement)
```

When `mean(Group B)` is zero (rare), MIDAS falls back to comparing against the
global mean. When Group A has fewer than 3 posts, the signal is marked unreliable
and excluded.

## The Hook Taxonomy

MIDAS categorizes hooks (the first line of your post) into patterns:

| Hook Type | Detection | Example |
|-----------|-----------|---------|
| Personal "I" | Starts with `I` + space/apostrophe | "I spent 3 months..." |
| Exclamation | Starts with Wow/Well/Wait/Whoa/Holy | "Wow. This changed everything." |
| Number lead | Starts with digit or $ | "7 lessons from..." |
| Superlative | Contains most/biggest/first/best/worst | "The biggest mistake I see..." |
| Question | Ends with `?` | "Why do most startups fail?" |
| Short teaser | Under 50 characters | "Nobody talks about this." |

The analysis tells you which hook types your audience responds to. Most people are
surprised: question hooks, which feel engaging to write, often underperform
declarative hooks.

## Identifying Your Anti-Patterns

Signals with lift below 1.0 are anti-patterns -- things you do that actively hurt
engagement. Common ones:

- **Hashtags**: Almost universally negative on LinkedIn. The algorithm deprioritizes
  posts with hashtags in the feed.
- **Question hooks**: Feel interactive but get scrolled past.
- **Newsletter CTAs**: "Subscribe to my newsletter" triggers the audience's
  promotional-content filter.

The analyzer flags these as penalties in the generated config.

## Manual Weight Tuning

The auto-generated weights are a starting point. Reasons to adjust:

1. **Strategic priorities.** You might want to emphasize thought leadership signals
   over engagement-bait signals, even if the latter has higher lift.
2. **Sample size concerns.** A signal with 3.0x lift from 5 posts is less reliable
   than 1.5x lift from 80 posts.
3. **Goal alignment.** Engagement is computed as `reactions + comments*2 + reposts*3`,
   which already weights deeper interactions higher. If you want to further emphasize
   certain interactions, adjust the signal weights in the YAML directly.

Edit the output YAML directly, or re-run with different parameters.

## Programmatic Usage

```python
from midas.export import load_jsonl
from midas.analyze import analyze_posts, export_config

posts = load_jsonl("posts.jsonl")
result = analyze_posts(posts, hook_max_chars=100, close_lines=3, min_frequency=0.02)

for signal in result.signals:
    print(f"{signal.name}: {signal.lift:.2f}x lift, {signal.frequency:.0%} freq")

# Save the derived config to YAML
export_config(result, "my_config.yaml")
```

## Next Step

Take the generated YAML and move on to
[Step 3: Derive Your Weights](03-build-your-formula.md) to refine your
scoring formula.

# Step 1: Get Your LinkedIn Data

MIDAS needs your historical post data to figure out what works for *your* audience.
LinkedIn does not make this easy. This guide covers every path, from the official
export to manual tracking.

## Option A: Apify LinkedIn Post Scraper (Recommended)

The fastest way to get all your posts **with engagement data** is the
[LinkedIn Post Search Scraper](https://console.apify.com/actors/RE0MriXnFhR3IgVnJ/input)
on Apify. It scrapes your public posts and returns text, reactions, comments,
reposts, images, and timestamps — everything MIDAS needs in one shot.

**Setup:**

1. Create a free [Apify account](https://apify.com) (comes with free monthly credits).
2. Open the [LinkedIn Post Search Scraper](https://console.apify.com/actors/RE0MriXnFhR3IgVnJ/input).
3. Enter your LinkedIn profile URL or search keywords for your posts.
4. Run the actor and export results as JSON.

**Convert to MIDAS format:**

```python
from midas.export import parse_apify_posts, save_jsonl

# Download the Apify dataset as JSON and pass the file path
posts = parse_apify_posts("apify_dataset.json")
save_jsonl(posts, "posts.jsonl")
print(f"Converted {len(posts)} posts")
```

The parser handles all common Apify output field names automatically. This is the
recommended path because you get text + engagement in a single export, no manual
enrichment needed.

## Option B: LinkedIn Native Export (Text Only — No Engagement)

1. Go to **Settings & Privacy** on LinkedIn.
2. Navigate to **Data Privacy > Get a copy of your data**.
3. Select **Posts** (deselect everything else to speed up the export).
4. Click **Request archive**. LinkedIn emails you a download link within 24 hours.

The archive gives you a CSV with your post text and timestamps. That is it.

**What you get:**

| Field | Included |
|-------|----------|
| Post text | Yes |
| Date posted | Yes |
| Reactions | No |
| Comments | No |
| Reposts | No |
| Impressions | No |
| Images/media | No |

The CSV alone is not enough for MIDAS -- you need engagement numbers to compute
signal lifts. You must manually enrich this data or use Option A instead.

## Option C: Manual Enrichment

Open the LinkedIn CSV alongside your LinkedIn activity page. For each post, record
the reaction count, comment count, and repost count. Tedious, but it works for
a backlog of 50-100 posts.

Save the result as JSONL (one JSON object per line):

```jsonl
{"text": "I spent 6 months building...", "reactions": 342, "comments": 47, "reposts": 12, "date": "2025-11-14", "has_image": false}
{"text": "Hot take: most founders...", "reactions": 89, "comments": 31, "reposts": 3, "date": "2025-11-10", "has_image": true}
```

## Option D: LinkedIn API

If you have a LinkedIn app with the `r_organization_social` or `r_member_social`
scope approved, you can pull engagement data programmatically. This requires:

1. A LinkedIn Developer App at https://www.linkedin.com/developers/
2. OAuth 2.0 authentication with the appropriate scopes
3. Calling the UGC Posts or Shares API to retrieve post stats

Most individual creators will not have API access. LinkedIn restricts these scopes
to approved marketing platforms. If you do have access, format the output into the
MIDAS JSONL schema described below.

## Option E: Manual Tracking Spreadsheet

Start tracking today. Each time you post, log:

| Column | Example |
|--------|---------|
| date | 2026-03-12 |
| text | Full post text |
| reactions | 156 |
| comments | 23 |
| reposts | 8 |
| has_image | true |

After 50+ posts, export to JSONL and run the analysis.

## The MIDAS JSONL Schema

Every path converges on this format. One JSON object per line, UTF-8 encoded:

```json
{
  "text": "Full post text including line breaks",
  "reactions": 156,
  "comments": 23,
  "reposts": 8,
  "date": "2026-03-12",
  "has_image": true
}
```

**Required fields:**
- `text` (string) -- the full post body
- `reactions` (integer) -- total reaction count (likes, celebrates, etc.)
- `comments` (integer) -- comment count
- `reposts` (integer) -- repost/share count

**Optional fields:**
- `date` (string, YYYY-MM-DD) -- used for time-based analysis
- `has_image` (boolean) -- whether the post included an image or carousel

Save the file as `posts.jsonl`. Verify it with:

```bash
python3 -c "
import json
with open('posts.jsonl') as f:
    posts = [json.loads(line) for line in f if line.strip()]
print(f'{len(posts)} posts loaded')
print(f'Avg reactions: {sum(p[\"reactions\"] for p in posts) / len(posts):.1f}')
"
```

## Format Conversion Helpers

MIDAS includes helpers for common input formats.

**From LinkedIn's native CSV export:**

```python
from midas.export import parse_linkedin_export, save_jsonl

# Parse the LinkedIn Shares CSV (text + dates only, no engagement)
posts = parse_linkedin_export("Shares.csv")
save_jsonl(posts, "posts.jsonl")
# NOTE: engagement fields will be 0 — you must enrich manually
```

**From a list of dicts:**

```python
from midas.export import save_jsonl

posts = [
    {"text": "My post...", "reactions": 50, "comments": 10, "reposts": 2},
    # ...
]
save_jsonl(posts, "posts.jsonl")
```

## How Much Data Do You Need?

| Posts | What You Get |
|-------|-------------|
| 20-49 | Rough signals, unreliable weights. Better than nothing. |
| 50-99 | Usable analysis. Main signals will be clear. |
| 100-199 | Solid analysis. Signal lifts become statistically meaningful. |
| 200+ | Best results. Enough data for fine-tuning later. |

MIDAS will warn you if your dataset is too small for reliable analysis. Aim for
50 posts minimum, 200+ if you plan to fine-tune a model (Step 5b).

## Next Step

Once you have `posts.jsonl`, move on to
[Step 2: Run the Analysis](02-analyze-signals.md).

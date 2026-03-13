"""LinkedIn data export helpers — get your post data into MIDAS format.

This module provides utilities for converting LinkedIn data exports and other
post data into the standard JSONL format that MIDAS expects for analysis.

Standard post schema:
    {
        "text": "Post content...",
        "reactions": 42,
        "comments": 7,
        "reposts": 3,
        "date": "2026-01-15",
        "has_image": true
    }
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Engagement calculation
# ---------------------------------------------------------------------------

def get_engagement(post: dict) -> float:
    """Compute total weighted engagement for a post.

    Formula: reactions + comments*2 + reposts*3

    Comments are weighted 2x because they require more effort than a reaction.
    Reposts are weighted 3x because they put the author's reputation on the line.

    Args:
        post: A post dict with keys: reactions, comments, reposts.

    Returns:
        Weighted engagement score as a float.
    """
    return (
        post.get("reactions", 0)
        + post.get("comments", 0) * 2
        + post.get("reposts", 0) * 3
    )


# ---------------------------------------------------------------------------
# Post normalization
# ---------------------------------------------------------------------------

_STANDARD_KEYS = {"text", "reactions", "comments", "reposts", "date", "has_image"}


def normalize_post(raw: dict) -> dict:
    """Normalize any post dict to the standard MIDAS schema.

    Handles common variations in key names and types:
    - "body" / "content" / "ShareCommentary" -> "text"
    - "likes" / "reaction_count" -> "reactions"
    - "comment_count" -> "comments"
    - "repost_count" / "shares" / "share_count" -> "reposts"
    - "created_at" / "published" / "Date" -> "date"
    - "image" / "media" / "MediaUrl" -> "has_image"

    Missing engagement fields default to 0. Missing date defaults to empty
    string. Missing has_image defaults to False.

    Args:
        raw: A dict with post data in any common format.

    Returns:
        A dict with exactly the standard keys:
        text, reactions, comments, reposts, date, has_image.
    """
    # --- Text ---
    text = (
        raw.get("text")
        or raw.get("body")
        or raw.get("content")
        or raw.get("ShareCommentary")
        or raw.get("share_commentary")
        or ""
    )
    text = str(text).strip()

    # --- Reactions (use explicit None checks so 0 is preserved) ---
    reactions = _first_int(raw, [
        "reactions", "likes", "reaction_count", "like_count",
    ])

    # --- Comments ---
    comments = _first_int(raw, [
        "comments", "comment_count", "num_comments",
    ])

    # --- Reposts ---
    reposts = _first_int(raw, [
        "reposts", "shares", "share_count", "repost_count",
    ])

    # --- Date ---
    date_val = (
        raw.get("date")
        or raw.get("Date")
        or raw.get("created_at")
        or raw.get("published")
        or raw.get("published_at")
        or ""
    )
    date_str = _normalize_date(date_val)

    # --- Has Image ---
    has_image_val = (
        raw.get("has_image")
        if raw.get("has_image") is not None
        else raw.get("image")
        if raw.get("image") is not None
        else raw.get("media")
        if raw.get("media") is not None
        else raw.get("MediaUrl")
        if raw.get("MediaUrl") is not None
        else raw.get("media_url")
        if raw.get("media_url") is not None
        else False
    )
    has_image = _to_bool(has_image_val)

    return {
        "text": text,
        "reactions": int(reactions),
        "comments": int(comments),
        "reposts": int(reposts),
        "date": date_str,
        "has_image": has_image,
    }


def _normalize_date(val: Any) -> str:
    """Best-effort date normalization to YYYY-MM-DD string.

    Handles datetime objects, ISO format strings, and common date formats.
    Returns empty string if parsing fails.
    """
    if not val:
        return ""

    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")

    val_str = str(val).strip()

    # Already in YYYY-MM-DD format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):
        return val_str

    # ISO format with time component (2026-01-15T10:30:00Z)
    if re.match(r"^\d{4}-\d{2}-\d{2}[T ]", val_str):
        return val_str[:10]

    # LinkedIn export format: "2026/01/15 10:30:00"
    if re.match(r"^\d{4}/\d{2}/\d{2}", val_str):
        return val_str[:10].replace("/", "-")

    # US format: MM/DD/YYYY
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", val_str)
    if match:
        m, d, y = match.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"

    return val_str


def _to_bool(val: Any) -> bool:
    """Convert various truthy representations to a bool.

    Handles: True/False, 1/0, "true"/"false", non-empty strings/URLs.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val > 0
    if isinstance(val, str):
        if val.lower() in ("true", "1", "yes"):
            return True
        if val.lower() in ("false", "0", "no", ""):
            return False
        # Non-empty string (e.g., a URL) means media is present
        return bool(val.strip())
    return bool(val)


# ---------------------------------------------------------------------------
# Apify LinkedIn Post Scraper parser
# ---------------------------------------------------------------------------

def parse_apify_posts(path: str) -> list[dict]:
    """Parse output from the Apify LinkedIn Post Search Scraper.

    Supports the JSON dataset export from:
    https://console.apify.com/actors/RE0MriXnFhR3IgVnJ/input

    Handles the common Apify output field names:
        text/postText/commentary -> text
        numLikes/reactionCount/likesCount/totalReactionCount -> reactions
        numComments/commentsCount -> comments
        numShares/repostsCount/sharesCount -> reposts
        postedAt/postedDate/publishedAt -> date
        images/media/imageUrl -> has_image

    Args:
        path: Path to the Apify JSON export file (array of objects).

    Returns:
        List of post dicts in the standard MIDAS schema.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Apify export file not found: {path}")

    with open(filepath, encoding="utf-8") as f:
        content = f.read().strip()

    # Support both JSON array and JSONL formats
    if content.startswith("["):
        raw_posts = json.loads(content)
    else:
        raw_posts = [json.loads(line) for line in content.split("\n") if line.strip()]

    posts: list[dict] = []
    for raw in raw_posts:
        text = (
            raw.get("text")
            or raw.get("postText")
            or raw.get("commentary")
            or raw.get("postContent")
            or raw.get("content")
            or ""
        )
        text = str(text).strip()
        if not text:
            continue

        reactions = _first_int(raw, [
            "numLikes", "reactionCount", "totalReactionCount",
            "likesCount", "reactions", "likes",
        ])

        comments = _first_int(raw, [
            "numComments", "commentsCount", "commentCount",
            "comments", "comment_count",
        ])

        reposts = _first_int(raw, [
            "numShares", "repostsCount", "sharesCount",
            "shareCount", "reposts", "shares",
        ])

        date_val = (
            raw.get("postedAt")
            or raw.get("postedDate")
            or raw.get("publishedAt")
            or raw.get("date")
            or raw.get("createdAt")
            or ""
        )
        date_str = _normalize_date(date_val)

        has_image = bool(
            raw.get("images")
            or raw.get("image")
            or raw.get("imageUrl")
            or raw.get("media")
            or raw.get("mediaUrl")
        )

        posts.append({
            "text": text,
            "reactions": reactions,
            "comments": comments,
            "reposts": reposts,
            "date": date_str,
            "has_image": has_image,
        })

    return posts


def _first_int(d: dict, keys: list[str]) -> int:
    """Return the first non-None integer value found among the given keys."""
    for key in keys:
        val = d.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    return 0


# ---------------------------------------------------------------------------
# LinkedIn CSV export parser
# ---------------------------------------------------------------------------

def parse_linkedin_export(shares_path: str) -> list[dict]:
    """Parse LinkedIn's native data export CSV into the standard post schema.

    LinkedIn allows you to download your data via:
        Settings -> Data privacy -> Get a copy of your data

    Select "Shares" to get a CSV of your posts. The file typically has columns:
        Date, ShareLink, ShareCommentary, SharedUrl, MediaUrl, Visibility

    IMPORTANT: LinkedIn's data export does NOT include engagement metrics
    (reactions, comments, reposts). The returned posts will have all engagement
    fields set to 0. You will need to enrich this data with engagement metrics
    from one of these sources:

    1. LinkedIn Analytics (manual): View each post's analytics on LinkedIn
       and add the numbers to the JSONL file.
    2. LinkedIn API: If you have API access, use the ugcPosts or shares
       endpoints to fetch engagement counts.
    3. Third-party tools: Tools like Shield, AuthoredUp, or Taplio can
       export your posts with engagement data.

    Args:
        shares_path: Path to the LinkedIn Shares CSV file.

    Returns:
        List of post dicts in the standard schema. Engagement fields will
        be 0 — you must add engagement data separately.
    """
    filepath = Path(shares_path)
    if not filepath.exists():
        raise FileNotFoundError(f"LinkedIn export file not found: {shares_path}")

    posts: list[dict] = []

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows or rows without text
            text = (
                row.get("ShareCommentary")
                or row.get("Commentary")
                or row.get("share_commentary")
                or ""
            ).strip()
            if not text:
                continue

            # Parse date
            date_raw = row.get("Date") or row.get("date") or ""
            date_str = _normalize_date(date_raw)

            # Detect image/media presence
            media_url = (
                row.get("MediaUrl")
                or row.get("media_url")
                or row.get("SharedUrl")
                or ""
            ).strip()
            has_image = bool(media_url)

            posts.append({
                "text": text,
                "reactions": 0,
                "comments": 0,
                "reposts": 0,
                "date": date_str,
                "has_image": has_image,
            })

    return posts


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file of posts.

    Each line should be a JSON object. Blank lines are skipped.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of post dicts as parsed from JSON.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line contains invalid JSON.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    posts: list[dict] = []
    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e

    return posts


def save_jsonl(posts: list[dict], path: str) -> None:
    """Save posts as a JSONL file (one JSON object per line).

    Posts are normalized to the standard schema before writing.

    Args:
        posts: List of post dicts. Each will be normalized via normalize_post().
        path: Output file path.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for post in posts:
            normalized = normalize_post(post)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

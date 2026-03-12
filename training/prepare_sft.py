#!/usr/bin/env python3
"""MIDAS SFT Data Preparation — Convert high-performing posts into supervised fine-tuning data.

Reads a JSONL file of posts (standard schema: text, reactions, comments, reposts,
date, has_image) and produces chat-format JSONL suitable for SFT training with
HuggingFace TRL.

Usage:
    python training/prepare_sft.py \
        --input data/posts.jsonl \
        --output-dir ./data/ \
        --system-prompt "You are a LinkedIn thought leader..." \
        --min-reactions 10 \
        --eval-size 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Topic taxonomy — keyword-based classification
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "community_events": [
        "community", "event", "meetup", "chapter", "members", "joined",
        "attendees", "conference", "summit", "workshop", "gathering",
        "hosted", "spoke at", "panel", "fireside",
    ],
    "ai_tech": [
        "ai", "machine learning", "llm", "gpt", "model", "training",
        "fine-tuning", "neural", "deep learning", "transformer", "agent",
        "inference", "gpu", "dataset", "benchmark", "open source",
        "engineering", "developer", "code", "api", "deploy", "startup",
        "tech", "software", "product", "saas", "platform",
    ],
    "personal_milestone": [
        "proud", "milestone", "anniversary", "launched", "shipped",
        "first time", "just hit", "reached", "accomplished", "grew to",
        "sold", "raised", "hired", "quit", "started",
    ],
    "industry_insight": [
        "trend", "prediction", "market", "industry", "report",
        "research shows", "data says", "study", "survey", "analysis",
        "future of", "landscape", "shift", "disruption",
    ],
    "career_advice": [
        "career", "hiring", "interview", "resume", "linkedin",
        "job search", "salary", "promotion", "leadership", "management",
        "mentor", "advice", "lesson", "mistake", "learned",
    ],
    "storytelling": [
        "story", "years ago", "i remember", "back in", "when i was",
        "let me tell you", "here's what happened", "true story",
    ],
}

# Personal-angle markers — first-person ownership / experience
PERSONAL_MARKERS: list[str] = [
    "i built", "i created", "i launched", "i shipped", "i spent",
    "my team", "our team", "we built", "we launched", "we grew",
    "my experience", "i learned", "i realized", "i discovered",
    "as a founder", "as a ceo", "as an engineer", "running a",
]


# ---------------------------------------------------------------------------
# Post data model
# ---------------------------------------------------------------------------

@dataclass
class Post:
    """A single post loaded from the JSONL input."""

    text: str
    reactions: int = 0
    comments: int = 0
    reposts: int = 0
    date: str = ""
    has_image: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def engagement(self) -> int:
        """Total engagement count: reactions + comments + reposts."""
        return self.reactions + self.comments + self.reposts


def load_posts(path: Path) -> list[Post]:
    """Load posts from a JSONL file.

    Expected fields per line: text, reactions, comments, reposts, date, has_image.
    All fields except ``text`` are optional.
    """
    posts: list[Post] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed JSON on line {lineno}: {exc}", file=sys.stderr)
                continue

            text = obj.get("text", "").strip()
            if not text:
                continue

            posts.append(Post(
                text=text,
                reactions=int(obj.get("reactions", 0)),
                comments=int(obj.get("comments", 0)),
                reposts=int(obj.get("reposts", 0)),
                date=str(obj.get("date", "")),
                has_image=bool(obj.get("has_image", False)),
                raw=obj,
            ))
    return posts


# ---------------------------------------------------------------------------
# Topic & angle extraction
# ---------------------------------------------------------------------------

def classify_topic(text: str) -> str:
    """Return the best-matching topic for *text*, or ``"general"`` if no match."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return "general"
    return best


def detect_personal_angle(text: str) -> bool:
    """Return ``True`` if the post contains first-person experience markers."""
    lower = text.lower()
    return any(marker in lower for marker in PERSONAL_MARKERS)


def extract_hook_hint(text: str) -> str:
    """Return a short description of the hook style used in *text*."""
    first_line = text.split("\n")[0].strip()
    if first_line.endswith("?"):
        return "question hook"
    if re.match(r"^[0-9$]", first_line):
        return "number-led hook"
    if re.match(r"^I[ ']", first_line):
        return "personal-I hook"
    if re.match(r"^(Wow|Well|Wait|Whoa|Holy)", first_line, re.IGNORECASE):
        return "exclamation hook"
    if len(first_line) < 50:
        return "short teaser hook"
    return "declarative hook"


# ---------------------------------------------------------------------------
# Instruction template generation
# ---------------------------------------------------------------------------

# Templates are grouped by style. Each template is a format string that may
# reference: {topic_label}, {hook_hint}, {angle_clause}, {word_count}.

TEMPLATES_TOPIC: list[str] = [
    "Write a LinkedIn post about {topic_label}.",
    "Draft a {topic_label} post for LinkedIn.",
    "Create an engaging LinkedIn post on the topic of {topic_label}.",
]

TEMPLATES_DETAILED: list[str] = [
    "Write a LinkedIn post about {topic_label}. Use a {hook_hint} to open. {angle_clause}",
    "Draft a LinkedIn post on {topic_label} ({hook_hint}). Aim for roughly {word_count} words. {angle_clause}",
]

TEMPLATES_PERSONA: list[str] = [
    "Write a LinkedIn post about {topic_label} from a first-person perspective. {angle_clause}",
    "Create a LinkedIn post sharing a personal take on {topic_label}. Use a {hook_hint}.",
]

TEMPLATES_BRIEF: list[str] = [
    "LinkedIn post: {topic_label}.",
    "Post about {topic_label} for LinkedIn.",
]

TEMPLATES_STRUCTURAL: list[str] = [
    "Write a LinkedIn post about {topic_label}. Open with a {hook_hint}, include a narrative pivot, and end with a call to action.",
    "Draft a LinkedIn post on {topic_label} (~{word_count} words). Start strong, add a personal angle, close with engagement CTA.",
]

ALL_TEMPLATE_GROUPS: list[list[str]] = [
    TEMPLATES_TOPIC,
    TEMPLATES_DETAILED,
    TEMPLATES_PERSONA,
    TEMPLATES_BRIEF,
    TEMPLATES_STRUCTURAL,
]

# Human-readable labels for topics
TOPIC_LABELS: dict[str, str] = {
    "community_events": "community building and events",
    "ai_tech": "AI and technology",
    "personal_milestone": "a personal or professional milestone",
    "industry_insight": "an industry trend or insight",
    "career_advice": "career development and professional growth",
    "storytelling": "a personal story or experience",
    "general": "a professional topic",
}


def generate_instruction(topic: str, hook_hint: str, is_personal: bool, word_count: int) -> str:
    """Generate a varied user instruction from the template pool."""
    topic_label = TOPIC_LABELS.get(topic, topic.replace("_", " "))
    angle_clause = "Include a first-person experience or personal angle." if is_personal else ""

    group = random.choice(ALL_TEMPLATE_GROUPS)
    template = random.choice(group)

    return template.format(
        topic_label=topic_label,
        hook_hint=hook_hint,
        angle_clause=angle_clause,
        word_count=word_count,
    ).strip()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def filter_posts(
    posts: list[Post],
    *,
    top_percentile: float = 0.30,
    min_reactions: int = 0,
    min_text_length: int = 200,
) -> list[Post]:
    """Filter to the top-performing slice of posts.

    Parameters
    ----------
    posts:
        All loaded posts.
    top_percentile:
        Keep the top N% by engagement (e.g. 0.30 = top 30%).
    min_reactions:
        Hard floor: post must have at least this many reactions.
    min_text_length:
        Minimum character count for the post body.
    """
    # Apply hard filters first
    candidates = [
        p for p in posts
        if p.reactions >= min_reactions and len(p.text) >= min_text_length
    ]

    if not candidates:
        return []

    # Sort by engagement descending and take the top slice
    candidates.sort(key=lambda p: p.engagement, reverse=True)
    cutoff = max(1, math.ceil(len(candidates) * top_percentile))
    return candidates[:cutoff]


def build_sft_example(post: Post, system_prompt: str) -> dict[str, Any]:
    """Convert a single post into a chat-format SFT training example.

    Returns
    -------
    dict with ``messages`` key containing system / user / assistant turns.
    """
    topic = classify_topic(post.text)
    is_personal = detect_personal_angle(post.text)
    hook_hint = extract_hook_hint(post.text)
    word_count = len(post.text.split())

    instruction = generate_instruction(topic, hook_hint, is_personal, word_count)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": post.text},
        ],
    }


def prepare_sft_data(
    posts: list[Post],
    system_prompt: str,
    *,
    top_percentile: float = 0.30,
    min_reactions: int = 0,
    min_text_length: int = 200,
    eval_size: int = 30,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Full SFT data preparation pipeline.

    Returns
    -------
    (train_examples, eval_examples)
    """
    random.seed(seed)

    filtered = filter_posts(
        posts,
        top_percentile=top_percentile,
        min_reactions=min_reactions,
        min_text_length=min_text_length,
    )

    if not filtered:
        print("WARNING: no posts passed filtering — check thresholds", file=sys.stderr)
        return [], []

    examples = [build_sft_example(p, system_prompt) for p in filtered]
    random.shuffle(examples)

    eval_size = min(eval_size, len(examples) // 5)  # Cap eval at 20% of data
    eval_set = examples[:eval_size]
    train_set = examples[eval_size:]

    return train_set, eval_set


def write_jsonl(data: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a LinkedIn content strategist and ghostwriter. "
    "Write engaging, authentic posts that feel personal and drive meaningful engagement. "
    "Use short punchy hooks, narrative pivots, and clear calls to action. "
    "Match the voice and style of the user's best-performing content."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data from high-performing posts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to input JSONL file (one post per line).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./data"),
        help="Directory to write sft_train.jsonl and sft_eval.jsonl (default: ./data/).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to embed in every training example.",
    )
    parser.add_argument(
        "--top-percentile",
        type=float,
        default=0.30,
        help="Keep the top N%% of posts by engagement (default: 0.30 = top 30%%).",
    )
    parser.add_argument(
        "--min-reactions",
        type=int,
        default=10,
        help="Minimum reactions for a post to be included (default: 10).",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=200,
        help="Minimum character count for post text (default: 200).",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=30,
        help="Number of examples to hold out for evaluation (default: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load
    print(f"Loading posts from {args.input} ...")
    posts = load_posts(args.input)
    print(f"  Loaded {len(posts)} posts.")

    if not posts:
        print("ERROR: no posts found in input file.", file=sys.stderr)
        sys.exit(1)

    # Prepare
    train, eval_ = prepare_sft_data(
        posts,
        system_prompt=args.system_prompt,
        top_percentile=args.top_percentile,
        min_reactions=args.min_reactions,
        min_text_length=args.min_text_length,
        eval_size=args.eval_size,
        seed=args.seed,
    )

    if not train:
        print("ERROR: no training examples produced — try lowering thresholds.", file=sys.stderr)
        sys.exit(1)

    # Write
    train_path = args.output_dir / "sft_train.jsonl"
    eval_path = args.output_dir / "sft_eval.jsonl"

    write_jsonl(train, train_path)
    write_jsonl(eval_, eval_path)

    # Summary
    topics = {}
    for ex in train + eval_:
        user_msg = ex["messages"][1]["content"]
        for topic, label in TOPIC_LABELS.items():
            if label in user_msg:
                topics[topic] = topics.get(topic, 0) + 1
                break

    print(f"\nSFT data prepared:")
    print(f"  Train: {len(train)} examples -> {train_path}")
    print(f"  Eval:  {len(eval_)} examples -> {eval_path}")
    print(f"\nTopic distribution:")
    for topic, count in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()

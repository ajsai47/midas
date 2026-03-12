#!/usr/bin/env python3
"""MIDAS DPO Data Preparation — Create preference pairs from posts with different engagement.

Compares posts within the same topic category to build (chosen, rejected) pairs
where the chosen post has significantly higher engagement. Output is compatible
with HuggingFace TRL's DPOTrainer.

Usage:
    python training/prepare_dpo.py \
        --input data/posts.jsonl \
        --output-dir ./data/ \
        --max-pairs 500 \
        --eval-size 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from training.prepare_sft import (
        DEFAULT_SYSTEM_PROMPT,
        TOPIC_LABELS,
        Post,
        classify_topic,
        detect_personal_angle,
        extract_hook_hint,
        generate_instruction,
        load_posts,
        write_jsonl,
    )
except ImportError:
    from prepare_sft import (
        DEFAULT_SYSTEM_PROMPT,
        TOPIC_LABELS,
        Post,
        classify_topic,
        detect_personal_angle,
        extract_hook_hint,
        generate_instruction,
        load_posts,
        write_jsonl,
    )


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def build_topic_buckets(
    posts: list[Post],
    *,
    min_text_length: int = 200,
) -> dict[str, list[Post]]:
    """Group posts by topic, filtering out short posts.

    Returns a dict mapping topic -> list of posts sorted by engagement descending.
    """
    buckets: dict[str, list[Post]] = defaultdict(list)
    for post in posts:
        if len(post.text) < min_text_length:
            continue
        topic = classify_topic(post.text)
        buckets[topic].append(post)

    # Sort each bucket by engagement descending
    for topic in buckets:
        buckets[topic].sort(key=lambda p: p.engagement, reverse=True)

    return dict(buckets)


def generate_pairs(
    buckets: dict[str, list[Post]],
    system_prompt: str,
    *,
    min_engagement_ratio: float = 2.0,
    min_chosen_reactions: int = 10,
    max_appearances: int = 5,
    max_pairs: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate DPO preference pairs from topic-grouped posts.

    Within each topic, the top half of posts (by engagement) are candidates
    for ``chosen`` and the bottom half for ``rejected``. A pair is valid when:

    - chosen engagement >= ``min_engagement_ratio`` * rejected engagement
    - chosen reactions >= ``min_chosen_reactions``
    - neither post has appeared in more than ``max_appearances`` pairs already

    Parameters
    ----------
    buckets:
        Topic -> sorted posts (descending by engagement).
    system_prompt:
        System message to include in the prompt turns.
    min_engagement_ratio:
        Chosen must have at least this multiple of rejected engagement.
    min_chosen_reactions:
        Hard floor on reactions for the chosen post.
    max_appearances:
        Each post can appear in at most this many pairs (prevents overfitting).
    max_pairs:
        Total cap on generated pairs.
    seed:
        Random seed for shuffling.
    """
    random.seed(seed)

    # Track how many times each post (by text hash) has been used
    appearance_count: dict[int, int] = defaultdict(int)
    pairs: list[dict[str, Any]] = []

    for topic, posts in buckets.items():
        if len(posts) < 4:
            # Need at least 4 posts to form meaningful top/bottom halves
            continue

        midpoint = len(posts) // 2
        top_half = posts[:midpoint]      # Higher engagement (chosen candidates)
        bottom_half = posts[midpoint:]   # Lower engagement (rejected candidates)

        for chosen in top_half:
            if chosen.reactions < min_chosen_reactions:
                continue

            chosen_id = hash(chosen.text)
            if appearance_count[chosen_id] >= max_appearances:
                continue

            for rejected in bottom_half:
                if len(pairs) >= max_pairs:
                    break

                rejected_id = hash(rejected.text)
                if appearance_count[rejected_id] >= max_appearances:
                    continue

                # Engagement ratio check (avoid division by zero)
                rejected_eng = max(rejected.engagement, 1)
                if chosen.engagement < min_engagement_ratio * rejected_eng:
                    continue

                # Build the pair — use a shared instruction for both
                hook_hint = extract_hook_hint(chosen.text)
                is_personal = detect_personal_angle(chosen.text)
                word_count = len(chosen.text.split())
                instruction = generate_instruction(topic, hook_hint, is_personal, word_count)

                pair = {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": instruction},
                    ],
                    "chosen": [
                        {"role": "assistant", "content": chosen.text},
                    ],
                    "rejected": [
                        {"role": "assistant", "content": rejected.text},
                    ],
                }

                pairs.append(pair)
                appearance_count[chosen_id] += 1
                appearance_count[rejected_id] += 1

            if len(pairs) >= max_pairs:
                break

    random.shuffle(pairs)
    return pairs[:max_pairs]


def prepare_dpo_data(
    posts: list[Post],
    system_prompt: str,
    *,
    min_text_length: int = 200,
    min_engagement_ratio: float = 2.0,
    min_chosen_reactions: int = 10,
    max_appearances: int = 5,
    max_pairs: int = 500,
    eval_size: int = 20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Full DPO data preparation pipeline.

    Returns
    -------
    (train_pairs, eval_pairs)
    """
    buckets = build_topic_buckets(posts, min_text_length=min_text_length)

    if not buckets:
        print("WARNING: no topic buckets formed — check min_text_length", file=sys.stderr)
        return [], []

    pairs = generate_pairs(
        buckets,
        system_prompt,
        min_engagement_ratio=min_engagement_ratio,
        min_chosen_reactions=min_chosen_reactions,
        max_appearances=max_appearances,
        max_pairs=max_pairs,
        seed=seed,
    )

    if not pairs:
        print("WARNING: no valid pairs generated — try lowering thresholds", file=sys.stderr)
        return [], []

    eval_size = min(eval_size, len(pairs) // 5)  # Cap eval at 20%
    eval_set = pairs[:eval_size]
    train_set = pairs[eval_size:]

    return train_set, eval_set


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DPO preference pairs from posts with varying engagement.",
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
        help="Directory to write dpo_train.jsonl and dpo_eval.jsonl (default: ./data/).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the prompt turns.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=200,
        help="Minimum character count for post text (default: 200).",
    )
    parser.add_argument(
        "--min-engagement-ratio",
        type=float,
        default=2.0,
        help="Chosen engagement must be >= this multiple of rejected (default: 2.0).",
    )
    parser.add_argument(
        "--min-chosen-reactions",
        type=int,
        default=10,
        help="Minimum reactions for the chosen post (default: 10).",
    )
    parser.add_argument(
        "--max-appearances",
        type=int,
        default=5,
        help="Max times a single post can appear across all pairs (default: 5).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=500,
        help="Maximum total pairs to generate (default: 500).",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=20,
        help="Number of pairs to hold out for evaluation (default: 20).",
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

    # Build topic buckets for summary
    buckets = build_topic_buckets(posts, min_text_length=args.min_text_length)
    print(f"\nTopic buckets:")
    for topic, topic_posts in sorted(buckets.items(), key=lambda x: -len(x[1])):
        top_eng = topic_posts[0].engagement if topic_posts else 0
        bot_eng = topic_posts[-1].engagement if topic_posts else 0
        print(f"  {topic}: {len(topic_posts)} posts (engagement range: {bot_eng}-{top_eng})")

    # Prepare
    train, eval_ = prepare_dpo_data(
        posts,
        system_prompt=args.system_prompt,
        min_text_length=args.min_text_length,
        min_engagement_ratio=args.min_engagement_ratio,
        min_chosen_reactions=args.min_chosen_reactions,
        max_appearances=args.max_appearances,
        max_pairs=args.max_pairs,
        eval_size=args.eval_size,
        seed=args.seed,
    )

    if not train:
        print("ERROR: no training pairs produced — try lowering thresholds.", file=sys.stderr)
        sys.exit(1)

    # Write
    train_path = args.output_dir / "dpo_train.jsonl"
    eval_path = args.output_dir / "dpo_eval.jsonl"

    write_jsonl(train, train_path)
    write_jsonl(eval_, eval_path)

    # Summary
    print(f"\nDPO data prepared:")
    print(f"  Train: {len(train)} pairs -> {train_path}")
    print(f"  Eval:  {len(eval_)} pairs -> {eval_path}")

    # Compute avg engagement gap
    chosen_engs = []
    rejected_engs = []
    for pair in train + eval_:
        # We can't directly read engagement from the pair (it's text only),
        # but we can count the ratio of pairs per topic from the instruction.
        pass

    print(f"\nConfig used:")
    print(f"  min_engagement_ratio: {args.min_engagement_ratio}x")
    print(f"  min_chosen_reactions: {args.min_chosen_reactions}")
    print(f"  max_appearances: {args.max_appearances}")
    print(f"  max_pairs: {args.max_pairs}")


if __name__ == "__main__":
    main()

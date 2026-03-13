"""Tests for midas.export — normalization, parsing, JSONL I/O."""

import json
import textwrap
from pathlib import Path

import pytest

from midas.export import (
    get_engagement,
    normalize_post,
    parse_apify_posts,
    parse_linkedin_export,
    load_jsonl,
    save_jsonl,
    _first_int,
    _normalize_date,
    _to_bool,
)


# ---------------------------------------------------------------------------
# get_engagement
# ---------------------------------------------------------------------------

class TestGetEngagement:
    def test_basic(self):
        post = {"reactions": 10, "comments": 5, "reposts": 2}
        assert get_engagement(post) == 10 + 5 * 2 + 2 * 3  # 26

    def test_missing_fields(self):
        assert get_engagement({}) == 0

    def test_partial_fields(self):
        assert get_engagement({"reactions": 7}) == 7

    def test_zero_values(self):
        post = {"reactions": 0, "comments": 0, "reposts": 0}
        assert get_engagement(post) == 0


# ---------------------------------------------------------------------------
# _first_int
# ---------------------------------------------------------------------------

class TestFirstInt:
    def test_returns_first_match(self):
        d = {"a": 10, "b": 20}
        assert _first_int(d, ["a", "b"]) == 10

    def test_skips_none(self):
        d = {"a": None, "b": 5}
        assert _first_int(d, ["a", "b"]) == 5

    def test_preserves_zero(self):
        """Zero is a valid value, not the same as missing."""
        d = {"reactions": 0, "likes": 42}
        assert _first_int(d, ["reactions", "likes"]) == 0

    def test_returns_zero_when_missing(self):
        assert _first_int({}, ["x", "y"]) == 0

    def test_skips_non_numeric(self):
        d = {"a": "not a number", "b": 7}
        assert _first_int(d, ["a", "b"]) == 7


# ---------------------------------------------------------------------------
# _normalize_date
# ---------------------------------------------------------------------------

class TestNormalizeDate:
    def test_empty(self):
        assert _normalize_date("") == ""
        assert _normalize_date(None) == ""

    def test_already_normalized(self):
        assert _normalize_date("2026-01-15") == "2026-01-15"

    def test_iso_with_time(self):
        assert _normalize_date("2026-01-15T10:30:00Z") == "2026-01-15"

    def test_linkedin_format(self):
        assert _normalize_date("2026/01/15 10:30:00") == "2026-01-15"

    def test_us_format(self):
        assert _normalize_date("1/5/2026") == "2026-01-05"
        assert _normalize_date("12/25/2026") == "2026-12-25"

    def test_datetime_object(self):
        from datetime import datetime
        dt = datetime(2026, 3, 12)
        assert _normalize_date(dt) == "2026-03-12"


# ---------------------------------------------------------------------------
# _to_bool
# ---------------------------------------------------------------------------

class TestToBool:
    def test_booleans(self):
        assert _to_bool(True) is True
        assert _to_bool(False) is False

    def test_numbers(self):
        assert _to_bool(1) is True
        assert _to_bool(0) is False

    def test_strings(self):
        assert _to_bool("true") is True
        assert _to_bool("false") is False
        assert _to_bool("") is False
        assert _to_bool("https://example.com/img.png") is True

    def test_none(self):
        assert _to_bool(None) is False


# ---------------------------------------------------------------------------
# normalize_post
# ---------------------------------------------------------------------------

class TestNormalizePost:
    def test_standard_keys(self):
        raw = {
            "text": "hello",
            "reactions": 10,
            "comments": 5,
            "reposts": 2,
            "date": "2026-01-15",
            "has_image": True,
        }
        result = normalize_post(raw)
        assert result == raw

    def test_alternate_key_names(self):
        raw = {
            "body": "hello",
            "likes": 10,
            "comment_count": 5,
            "shares": 2,
            "created_at": "2026-01-15T10:00:00Z",
            "media": "https://example.com/img.png",
        }
        result = normalize_post(raw)
        assert result["text"] == "hello"
        assert result["reactions"] == 10
        assert result["comments"] == 5
        assert result["reposts"] == 2
        assert result["date"] == "2026-01-15"
        assert result["has_image"] is True

    def test_zero_engagement_preserved(self):
        """Zero engagement should not fall through to other keys."""
        raw = {
            "text": "test",
            "reactions": 0,
            "likes": 99,
        }
        result = normalize_post(raw)
        assert result["reactions"] == 0  # Not 99

    def test_missing_fields_default(self):
        result = normalize_post({"text": "hello"})
        assert result["reactions"] == 0
        assert result["comments"] == 0
        assert result["reposts"] == 0
        assert result["date"] == ""
        assert result["has_image"] is False


# ---------------------------------------------------------------------------
# parse_apify_posts
# ---------------------------------------------------------------------------

class TestParseApifyPosts:
    def test_json_array(self, tmp_path):
        data = [
            {
                "text": "Post one",
                "numLikes": 10,
                "numComments": 3,
                "numShares": 1,
                "postedAt": "2026-01-15T10:00:00Z",
                "images": ["img.png"],
            },
            {
                "postText": "Post two",
                "reactionCount": 20,
            },
        ]
        path = tmp_path / "apify.json"
        path.write_text(json.dumps(data))

        posts = parse_apify_posts(str(path))
        assert len(posts) == 2
        assert posts[0]["text"] == "Post one"
        assert posts[0]["reactions"] == 10
        assert posts[0]["has_image"] is True
        assert posts[1]["text"] == "Post two"
        assert posts[1]["reactions"] == 20

    def test_jsonl_format(self, tmp_path):
        lines = [
            json.dumps({"text": "Post A", "numLikes": 5}),
            json.dumps({"text": "Post B", "numLikes": 8}),
        ]
        path = tmp_path / "apify.jsonl"
        path.write_text("\n".join(lines))

        posts = parse_apify_posts(str(path))
        assert len(posts) == 2

    def test_skips_empty_text(self, tmp_path):
        data = [{"text": "", "numLikes": 5}, {"text": "Real post"}]
        path = tmp_path / "apify.json"
        path.write_text(json.dumps(data))

        posts = parse_apify_posts(str(path))
        assert len(posts) == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_apify_posts("/nonexistent/file.json")


# ---------------------------------------------------------------------------
# parse_linkedin_export
# ---------------------------------------------------------------------------

class TestParseLinkedinExport:
    def test_basic_csv(self, tmp_path):
        csv_content = textwrap.dedent("""\
            Date,ShareLink,ShareCommentary,SharedUrl,MediaUrl,Visibility
            2026/01/15 10:00:00,https://linkedin.com/post/1,My great post,,https://media.com/img.png,PUBLIC
            2026/01/16 12:00:00,https://linkedin.com/post/2,Another post,,,PUBLIC
        """)
        path = tmp_path / "shares.csv"
        path.write_text(csv_content)

        posts = parse_linkedin_export(str(path))
        assert len(posts) == 2
        assert posts[0]["text"] == "My great post"
        assert posts[0]["reactions"] == 0  # No engagement in CSV
        assert posts[0]["has_image"] is True
        assert posts[0]["date"] == "2026-01-15"
        assert posts[1]["has_image"] is False

    def test_skips_empty_rows(self, tmp_path):
        csv_content = textwrap.dedent("""\
            Date,ShareCommentary
            2026/01/15,
            2026/01/16,Real post
        """)
        path = tmp_path / "shares.csv"
        path.write_text(csv_content)

        posts = parse_linkedin_export(str(path))
        assert len(posts) == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_linkedin_export("/nonexistent/shares.csv")


# ---------------------------------------------------------------------------
# load_jsonl / save_jsonl
# ---------------------------------------------------------------------------

class TestJsonlIO:
    def test_round_trip(self, tmp_path):
        posts = [
            {"text": "hello", "reactions": 10, "comments": 5, "reposts": 2,
             "date": "2026-01-15", "has_image": True},
            {"text": "world", "reactions": 0, "comments": 0, "reposts": 0,
             "date": "", "has_image": False},
        ]
        path = tmp_path / "posts.jsonl"
        save_jsonl(posts, str(path))
        loaded = load_jsonl(str(path))
        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"
        assert loaded[1]["reactions"] == 0

    def test_load_skips_blank_lines(self, tmp_path):
        path = tmp_path / "posts.jsonl"
        path.write_text('{"text": "a"}\n\n{"text": "b"}\n')
        loaded = load_jsonl(str(path))
        assert len(loaded) == 2

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"text": "ok"}\nnot json\n')
        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            load_jsonl(str(path))

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/file.jsonl")

    def test_save_creates_directories(self, tmp_path):
        posts = [{"text": "test"}]
        path = tmp_path / "sub" / "dir" / "posts.jsonl"
        save_jsonl(posts, str(path))
        assert path.exists()

"""
Tests for the RSS watcher trigger module.

Validates episode detection, GUID deduplication, error handling,
and output formatting using mock feedparser responses.

Covers:
- Detecting new episodes from RSS feed
- Deduplication via known GUID filtering
- Empty feed handling
- Malformed feed handling
- WatchResult output format and JSON serialization
- Edge cases: missing audio, missing title, mixed enclosures
"""

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from podcast_intel.triggers.rss_watcher import (
    EpisodeMetadata,
    WatchResult,
    check_new_episodes,
    run_watch,
    load_known_guids,
    _extract_audio_url,
    _extract_file_size,
    _load_known_guids_from_json,
)


# ---------------------------------------------------------------------------
#  Helpers -- build realistic feedparser-like objects
# ---------------------------------------------------------------------------

def _make_entry(
    guid: str = "guid-001",
    title: str = "Episode 1 - Pilot",
    audio_url: str = "https://cdn.example.com/ep1.mp3",
    audio_type: str = "audio/mpeg",
    published: str = "Mon, 01 Jan 2024 12:00:00 GMT",
    duration: str = "01:05:30",
    summary: str = "First episode of the podcast.",
    file_size: str = "52428800",
    episode_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a feedparser-style entry dict with enclosures.

    Returns a namespace-like dict that mimics feedparser's entry objects.
    """
    entry = SimpleNamespace()
    entry.id = guid
    entry.title = title
    entry.published = published
    entry.summary = summary
    entry.itunes_duration = duration
    entry.enclosures = [
        {"type": audio_type, "href": audio_url, "length": file_size}
    ]
    entry.links = []

    if episode_type is not None:
        entry.itunes_episodetype = episode_type

    # Make dict-style access work via __getitem__
    def _get(key, default=None):
        return getattr(entry, key, default)

    entry.get = _get
    return entry


def _make_feed(
    entries: Optional[List] = None,
    title: str = "My Podcast",
    bozo: bool = False,
    bozo_exception: Optional[Exception] = None,
) -> SimpleNamespace:
    """Build a feedparser-style feed result."""
    feed = SimpleNamespace()
    feed.entries = entries or []
    feed.bozo = 1 if bozo else 0
    feed.bozo_exception = bozo_exception
    feed.feed = {"title": title}
    return feed


# ===================================================================
# Test 1: Detecting new episodes
# ===================================================================

class TestCheckNewEpisodes:
    """Tests for check_new_episodes()."""

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_finds_new_episode(self, mock_parse):
        """A single new episode is detected when GUID is not in known set."""
        entry = _make_entry(guid="new-guid-001", title="Episode 10")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 1
        assert result[0].guid == "new-guid-001"
        assert result[0].title == "Episode 10"
        assert result[0].audio_url == "https://cdn.example.com/ep1.mp3"

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_finds_multiple_new_episodes(self, mock_parse):
        """Multiple new episodes are all returned."""
        entries = [
            _make_entry(guid=f"guid-{i}", title=f"Episode {i}")
            for i in range(1, 4)
        ]
        mock_parse.return_value = _make_feed(entries=entries)

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 3
        guids = {ep.guid for ep in result}
        assert guids == {"guid-1", "guid-2", "guid-3"}

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_extracts_pub_date(self, mock_parse):
        """Publication date is parsed into ISO-8601 format."""
        entry = _make_entry(
            guid="date-test",
            published="Wed, 15 Jan 2025 08:30:00 GMT",
        )
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 1
        assert "2025-01-15" in result[0].pub_date

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_extracts_duration(self, mock_parse):
        """Duration string is captured from iTunes tag."""
        entry = _make_entry(guid="dur-test", duration="01:23:45")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert result[0].duration == "01:23:45"

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_extracts_file_size(self, mock_parse):
        """File size is parsed as integer from enclosure length."""
        entry = _make_entry(guid="size-test", file_size="104857600")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert result[0].file_size_bytes == 104857600

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_extracts_episode_type(self, mock_parse):
        """Episode type is extracted from iTunes episode type tag."""
        entry = _make_entry(guid="type-test", episode_type="bonus")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert result[0].episode_type == "bonus"

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_defaults_to_full_episode_type(self, mock_parse):
        """Episode type defaults to 'full' when not specified."""
        entry = _make_entry(guid="default-type")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert result[0].episode_type == "full"


# ===================================================================
# Test 2: Deduplication (known GUIDs filtered)
# ===================================================================

class TestDeduplication:
    """Tests for GUID-based deduplication."""

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_known_guid_is_filtered(self, mock_parse):
        """An episode whose GUID is already known is excluded."""
        entry = _make_entry(guid="known-001", title="Old Episode")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes(
            "https://example.com/feed.rss",
            known_guids={"known-001"},
        )

        assert len(result) == 0

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_mixed_known_and_new(self, mock_parse):
        """Only new GUIDs are returned when feed has both known and new."""
        entries = [
            _make_entry(guid="old-001", title="Old One"),
            _make_entry(guid="new-001", title="New One"),
            _make_entry(guid="old-002", title="Old Two"),
            _make_entry(guid="new-002", title="New Two"),
        ]
        mock_parse.return_value = _make_feed(entries=entries)

        result = check_new_episodes(
            "https://example.com/feed.rss",
            known_guids={"old-001", "old-002"},
        )

        assert len(result) == 2
        guids = {ep.guid for ep in result}
        assert guids == {"new-001", "new-002"}

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_all_known_returns_empty(self, mock_parse):
        """When all feed GUIDs are known, result is empty."""
        entries = [
            _make_entry(guid=f"ep-{i}", title=f"Episode {i}")
            for i in range(5)
        ]
        mock_parse.return_value = _make_feed(entries=entries)

        known = {f"ep-{i}" for i in range(5)}
        result = check_new_episodes("https://example.com/feed.rss", known)

        assert len(result) == 0

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_empty_known_guids_returns_all(self, mock_parse):
        """With no known GUIDs, all feed episodes are new."""
        entries = [
            _make_entry(guid=f"ep-{i}", title=f"Episode {i}")
            for i in range(3)
        ]
        mock_parse.return_value = _make_feed(entries=entries)

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 3


# ===================================================================
# Test 3: Empty feed handling
# ===================================================================

class TestEmptyFeed:
    """Tests for empty or no-entry feeds."""

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_empty_feed_returns_empty_list(self, mock_parse):
        """An RSS feed with zero entries returns an empty list."""
        mock_parse.return_value = _make_feed(entries=[])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert result == []

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_entries_without_audio_are_skipped(self, mock_parse):
        """Entries that lack an audio enclosure are excluded."""
        entry = _make_entry(guid="no-audio")
        entry.enclosures = []  # Remove audio enclosures
        entry.links = []
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 0

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_entries_without_title_are_skipped(self, mock_parse):
        """Entries that lack a title are excluded."""
        entry = _make_entry(guid="no-title", title="")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 0

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_entries_without_guid_are_skipped(self, mock_parse):
        """Entries with empty GUID are excluded."""
        entry = _make_entry(guid="")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 0


# ===================================================================
# Test 4: Malformed feed handling
# ===================================================================

class TestMalformedFeed:
    """Tests for error handling with broken or unusual feeds."""

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_bozo_feed_with_entries_still_parses(self, mock_parse):
        """A feed marked as bozo but with valid entries still returns results."""
        entry = _make_entry(guid="bozo-ep", title="Bozo Episode")
        mock_parse.return_value = _make_feed(
            entries=[entry],
            bozo=True,
            bozo_exception=Exception("XML not well-formed"),
        )

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 1
        assert result[0].title == "Bozo Episode"

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_bozo_feed_without_entries_raises(self, mock_parse):
        """A bozo feed with zero entries raises ValueError."""
        mock_parse.return_value = _make_feed(
            entries=[],
            bozo=True,
            bozo_exception=Exception("Connection refused"),
        )

        with pytest.raises(ValueError, match="Failed to parse RSS feed"):
            check_new_episodes("https://example.com/feed.rss", set())

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_entry_with_non_audio_enclosure(self, mock_parse):
        """Entries with only non-audio enclosures (e.g., image) are skipped."""
        entry = _make_entry(guid="img-only")
        entry.enclosures = [
            {"type": "image/jpeg", "href": "https://cdn.example.com/cover.jpg"}
        ]
        entry.links = []
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 0

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_invalid_date_uses_raw_string(self, mock_parse):
        """An unparseable date falls back to the raw string."""
        entry = _make_entry(guid="bad-date", published="not-a-date")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 1
        assert result[0].pub_date == "not-a-date"

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_invalid_file_size_returns_none(self, mock_parse):
        """Non-numeric file size in enclosure produces None."""
        entry = _make_entry(guid="bad-size", file_size="unknown")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 1
        assert result[0].file_size_bytes is None

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    def test_audio_url_from_links_fallback(self, mock_parse):
        """When enclosures are absent, audio URL is extracted from links."""
        entry = _make_entry(guid="link-audio")
        entry.enclosures = []
        entry.links = [
            SimpleNamespace(
                **{"type": "audio/mpeg", "href": "https://cdn.example.com/via-link.mp3"}
            )
        ]
        # Make links iterable and support .get()
        link_obj = entry.links[0]
        link_obj.get = lambda key, default=None: getattr(link_obj, key, default)
        mock_parse.return_value = _make_feed(entries=[entry])

        result = check_new_episodes("https://example.com/feed.rss", set())

        assert len(result) == 1
        assert result[0].audio_url == "https://cdn.example.com/via-link.mp3"


# ===================================================================
# Test 5: WatchResult output format
# ===================================================================

class TestWatchResultFormat:
    """Tests for WatchResult dataclass and serialization."""

    def test_default_watch_result(self):
        """Default WatchResult has no episodes and no errors."""
        result = WatchResult()
        assert result.has_new_episodes is False
        assert result.episodes == []
        assert result.errors == []

    def test_watch_result_with_episodes(self):
        """WatchResult correctly reports has_new_episodes=True."""
        ep = EpisodeMetadata(
            title="Episode 42",
            guid="guid-42",
            audio_url="https://cdn.example.com/ep42.mp3",
            pub_date="2025-01-15T12:00:00+00:00",
            duration="00:45:00",
        )
        result = WatchResult(
            has_new_episodes=True,
            episodes=[ep],
            checked_at="2025-01-15T13:00:00+00:00",
        )
        assert result.has_new_episodes is True
        assert len(result.episodes) == 1
        assert result.episodes[0].title == "Episode 42"

    def test_to_dict_structure(self):
        """to_dict() returns a dictionary with all expected keys."""
        ep = EpisodeMetadata(
            title="Test",
            guid="g1",
            audio_url="https://example.com/test.mp3",
        )
        result = WatchResult(
            has_new_episodes=True,
            episodes=[ep],
            checked_at="2025-01-01T00:00:00+00:00",
            feed_title="My Podcast",
            total_feed_episodes=100,
            known_guid_count=99,
        )

        d = result.to_dict()

        assert d["has_new_episodes"] is True
        assert d["new_episode_count"] == 1
        assert d["feed_title"] == "My Podcast"
        assert d["total_feed_episodes"] == 100
        assert d["known_guid_count"] == 99
        assert d["checked_at"] == "2025-01-01T00:00:00+00:00"
        assert len(d["episodes"]) == 1
        assert d["episodes"][0]["guid"] == "g1"

    def test_to_json_is_valid_json(self):
        """to_json() produces valid JSON that can be parsed back."""
        result = WatchResult(
            has_new_episodes=False,
            checked_at="2025-01-01T00:00:00",
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["has_new_episodes"] is False

    def test_to_json_with_episodes(self):
        """to_json() includes all episode metadata."""
        eps = [
            EpisodeMetadata(
                title=f"Episode {i}",
                guid=f"guid-{i}",
                audio_url=f"https://example.com/ep{i}.mp3",
                pub_date="2025-06-01T00:00:00",
                duration="01:00:00",
                description=f"Description for episode {i}",
                file_size_bytes=50_000_000,
                episode_type="full",
            )
            for i in range(3)
        ]
        result = WatchResult(
            has_new_episodes=True,
            episodes=eps,
            checked_at="2025-06-01T12:00:00",
        )
        parsed = json.loads(result.to_json())
        assert parsed["new_episode_count"] == 3
        assert all("title" in ep for ep in parsed["episodes"])
        assert all("audio_url" in ep for ep in parsed["episodes"])

    def test_episode_metadata_to_dict(self):
        """EpisodeMetadata.to_dict() includes all fields."""
        ep = EpisodeMetadata(
            title="Roundtable Discussion",
            guid="rt-001",
            audio_url="https://cdn.example.com/rt.mp3",
            pub_date="2025-03-15T09:00:00+00:00",
            duration="00:55:12",
            description="A roundtable discussion.",
            file_size_bytes=45_000_000,
            episode_type="full",
        )
        d = ep.to_dict()
        assert d["title"] == "Roundtable Discussion"
        assert d["guid"] == "rt-001"
        assert d["audio_url"] == "https://cdn.example.com/rt.mp3"
        assert d["pub_date"] == "2025-03-15T09:00:00+00:00"
        assert d["duration"] == "00:55:12"
        assert d["description"] == "A roundtable discussion."
        assert d["file_size_bytes"] == 45_000_000
        assert d["episode_type"] == "full"

    def test_watch_result_errors_list(self):
        """WatchResult can carry multiple error messages."""
        result = WatchResult(
            errors=["Error 1: timeout", "Error 2: parse failure"],
        )
        d = result.to_dict()
        assert len(d["errors"]) == 2
        assert "timeout" in d["errors"][0]


# ===================================================================
# Test 6: run_watch integration (mocked)
# ===================================================================

class TestRunWatch:
    """Tests for the run_watch() orchestration function."""

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    @patch("podcast_intel.triggers.rss_watcher.load_known_guids")
    @patch("podcast_intel.triggers.rss_watcher.get_config")
    def test_run_watch_finds_new_episodes(self, mock_config, mock_guids, mock_parse):
        """run_watch returns WatchResult with new episodes."""
        cfg = MagicMock()
        cfg.rss_url = "https://example.com/feed.rss"
        cfg.db_path = "/tmp/test.db"
        mock_config.return_value = cfg

        mock_guids.return_value = {"old-001"}

        entry = _make_entry(guid="new-001", title="Brand New Episode")
        mock_parse.return_value = _make_feed(entries=[entry], title="Test Pod")

        result = run_watch(config=cfg, once=True)

        assert result.has_new_episodes is True
        assert len(result.episodes) == 1
        assert result.episodes[0].title == "Brand New Episode"

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    @patch("podcast_intel.triggers.rss_watcher.load_known_guids")
    @patch("podcast_intel.triggers.rss_watcher.get_config")
    def test_run_watch_no_new_episodes(self, mock_config, mock_guids, mock_parse):
        """run_watch reports no new episodes when all GUIDs are known."""
        cfg = MagicMock()
        cfg.rss_url = "https://example.com/feed.rss"
        cfg.db_path = "/tmp/test.db"
        mock_config.return_value = cfg

        mock_guids.return_value = {"ep-001"}

        entry = _make_entry(guid="ep-001", title="Known Episode")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = run_watch(config=cfg, once=True)

        assert result.has_new_episodes is False
        assert len(result.episodes) == 0

    @patch("podcast_intel.triggers.rss_watcher.load_podcast_yaml")
    @patch("podcast_intel.triggers.rss_watcher.get_config")
    def test_run_watch_no_rss_url(self, mock_config, mock_yaml):
        """run_watch returns error when no RSS URL is configured."""
        cfg = MagicMock()
        cfg.rss_url = ""
        mock_config.return_value = cfg
        mock_yaml.return_value = {}

        result = run_watch(config=cfg, once=True)

        assert result.has_new_episodes is False
        assert len(result.errors) == 1
        assert "No RSS URL" in result.errors[0]

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    @patch("podcast_intel.triggers.rss_watcher.load_known_guids")
    @patch("podcast_intel.triggers.rss_watcher.get_config")
    def test_run_watch_dry_run(self, mock_config, mock_guids, mock_parse):
        """run_watch in dry_run mode still detects episodes."""
        cfg = MagicMock()
        cfg.rss_url = "https://example.com/feed.rss"
        cfg.db_path = "/tmp/test.db"
        mock_config.return_value = cfg

        mock_guids.return_value = set()

        entry = _make_entry(guid="dry-001", title="Dry Run Episode")
        mock_parse.return_value = _make_feed(entries=[entry])

        result = run_watch(config=cfg, once=True, dry_run=True)

        assert result.has_new_episodes is True
        assert len(result.episodes) == 1

    @patch("podcast_intel.triggers.rss_watcher.feedparser.parse")
    @patch("podcast_intel.triggers.rss_watcher.load_known_guids")
    @patch("podcast_intel.triggers.rss_watcher.get_config")
    def test_run_watch_populates_feed_metadata(self, mock_config, mock_guids, mock_parse):
        """run_watch populates feed_title and total_feed_episodes."""
        cfg = MagicMock()
        cfg.rss_url = "https://example.com/feed.rss"
        cfg.db_path = "/tmp/test.db"
        mock_config.return_value = cfg
        mock_guids.return_value = set()

        entries = [_make_entry(guid=f"ep-{i}") for i in range(5)]
        mock_parse.return_value = _make_feed(entries=entries, title="Great Podcast")

        result = run_watch(config=cfg, once=True)

        assert result.feed_title == "Great Podcast"
        assert result.total_feed_episodes == 5


# ===================================================================
# Test 7: Known GUID loading from JSON
# ===================================================================

class TestLoadKnownGuidsJson:
    """Tests for _load_known_guids_from_json()."""

    def test_load_from_valid_json(self, tmp_path):
        """Loads GUIDs from a well-formed episodes.json."""
        data = [
            {"guid": "g1", "title": "Ep 1"},
            {"guid": "g2", "title": "Ep 2"},
            {"guid": "g3", "title": "Ep 3"},
        ]
        json_path = tmp_path / "episodes.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        guids = _load_known_guids_from_json(json_path)

        assert guids == {"g1", "g2", "g3"}

    def test_load_from_missing_file(self, tmp_path):
        """Returns empty set when file does not exist."""
        guids = _load_known_guids_from_json(tmp_path / "nonexistent.json")
        assert guids == set()

    def test_load_from_empty_json_list(self, tmp_path):
        """Returns empty set for an empty JSON array."""
        json_path = tmp_path / "episodes.json"
        json_path.write_text("[]", encoding="utf-8")

        guids = _load_known_guids_from_json(json_path)

        assert guids == set()

    def test_load_skips_entries_without_guid(self, tmp_path):
        """Entries missing a 'guid' key are excluded."""
        data = [
            {"guid": "g1", "title": "Ep 1"},
            {"title": "No GUID Episode"},
            {"guid": "g2", "title": "Ep 2"},
        ]
        json_path = tmp_path / "episodes.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        guids = _load_known_guids_from_json(json_path)

        assert guids == {"g1", "g2"}

    def test_load_handles_corrupt_json(self, tmp_path):
        """Returns empty set when JSON is invalid."""
        json_path = tmp_path / "episodes.json"
        json_path.write_text("{not valid json", encoding="utf-8")

        guids = _load_known_guids_from_json(json_path)

        assert guids == set()

"""
RSS feed watcher for detecting new podcast episodes.

Polls an RSS feed and compares episode GUIDs against a set of known
episodes (from the database or a JSON registry file). Returns metadata
for any new episodes that have not yet been processed.

This module is designed to be used in three ways:

1. **Programmatic** -- call ``run_watch()`` from Python.
2. **CLI** -- invoked via ``podcast-intel watch``.
3. **Scheduled** -- called periodically by an external scheduler (cron,
   systemd timer, GitHub Actions, etc.).

Example:
    >>> from podcast_intel.triggers.rss_watcher import run_watch
    >>> result = run_watch(config, once=True)
    >>> if result.has_new_episodes:
    ...     for ep in result.episodes:
    ...         print(f"New: {ep.title}")
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import feedparser
from dateutil import parser as date_parser

from podcast_intel.config import get_config, load_podcast_yaml, PROJECT_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data models
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetadata:
    """
    Metadata for a single episode discovered in the RSS feed.

    Attributes:
        title: Episode title from RSS
        guid: Globally unique identifier from RSS
        audio_url: URL to the audio enclosure
        pub_date: Publication date as ISO-8601 string
        duration: Duration string from iTunes tag (e.g., "01:23:45")
        description: Episode description or summary
        file_size_bytes: Audio file size in bytes, if available
        episode_type: Episode type (full, trailer, bonus)
    """

    title: str
    guid: str
    audio_url: str
    pub_date: str = ""
    duration: str = ""
    description: str = ""
    file_size_bytes: Optional[int] = None
    episode_type: str = "full"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class WatchResult:
    """
    Result of an RSS watch operation.

    Attributes:
        has_new_episodes: True if at least one new episode was found
        episodes: List of new episode metadata objects
        errors: List of error messages encountered during the check
        checked_at: ISO-8601 timestamp of when the check was performed
        feed_title: Title of the RSS feed, if available
        total_feed_episodes: Total number of episodes in the feed
        known_guid_count: Number of previously known GUIDs
    """

    has_new_episodes: bool = False
    episodes: List[EpisodeMetadata] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checked_at: str = ""
    feed_title: str = ""
    total_feed_episodes: int = 0
    known_guid_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_new_episodes": self.has_new_episodes,
            "episodes": [ep.to_dict() for ep in self.episodes],
            "errors": self.errors,
            "checked_at": self.checked_at,
            "feed_title": self.feed_title,
            "total_feed_episodes": self.total_feed_episodes,
            "known_guid_count": self.known_guid_count,
            "new_episode_count": len(self.episodes),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
#  Core functions
# ---------------------------------------------------------------------------

def check_new_episodes(
    rss_url: str,
    known_guids: Set[str],
) -> List[EpisodeMetadata]:
    """
    Fetch the RSS feed and return episodes whose GUIDs are not in known_guids.

    Parses the feed using feedparser, iterates over all entries, and
    filters out any episode whose GUID is already in the ``known_guids``
    set. Only episodes with a valid audio enclosure URL are returned.

    Args:
        rss_url: URL of the podcast RSS feed
        known_guids: Set of GUIDs for episodes already tracked

    Returns:
        List of EpisodeMetadata for episodes not in known_guids

    Example:
        >>> known = {"guid-001", "guid-002"}
        >>> new_eps = check_new_episodes("https://example.com/feed.rss", known)
        >>> print(f"Found {len(new_eps)} new episodes")
    """
    feed = feedparser.parse(rss_url)

    if feed.bozo and not feed.entries:
        raise ValueError(
            f"Failed to parse RSS feed: {feed.bozo_exception}"
        )

    new_episodes: List[EpisodeMetadata] = []

    for entry in feed.entries:
        guid = entry.get("id") or entry.get("guid", "")
        if not guid or guid in known_guids:
            continue

        title = entry.get("title", "")
        if not title:
            continue

        # Extract audio URL from enclosures
        audio_url = _extract_audio_url(entry)
        if not audio_url:
            continue

        # Parse publication date
        pub_date = ""
        raw_date = getattr(entry, "published", "") or getattr(entry, "pubDate", "")
        if raw_date:
            try:
                parsed = date_parser.parse(raw_date)
                pub_date = parsed.strftime("%Y-%m-%dT%H:%M:%S%z")
            except (ValueError, OverflowError):
                pub_date = raw_date

        # Duration from iTunes tag
        duration = getattr(entry, "itunes_duration", "") or ""

        # Description
        description = entry.get("summary") or entry.get("description") or ""

        # File size
        file_size = _extract_file_size(entry)

        # Episode type
        episode_type = "full"
        if hasattr(entry, "itunes_episodetype"):
            ep_type = entry.itunes_episodetype.lower()
            if ep_type in ("full", "trailer", "bonus"):
                episode_type = ep_type

        new_episodes.append(
            EpisodeMetadata(
                title=title,
                guid=guid,
                audio_url=audio_url,
                pub_date=pub_date,
                duration=duration,
                description=description,
                file_size_bytes=file_size,
                episode_type=episode_type,
            )
        )

    return new_episodes


def _extract_audio_url(entry: Any) -> Optional[str]:
    """
    Extract audio URL from a feedparser entry's enclosures or links.

    Args:
        entry: feedparser entry object

    Returns:
        Audio URL string or None if not found
    """
    # Check enclosures first
    if hasattr(entry, "enclosures") and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get("type", "").startswith("audio/"):
                url = enc.get("href") or enc.get("url")
                if url:
                    return url

    # Fallback to links
    if hasattr(entry, "links"):
        for link in entry.links:
            if link.get("type", "").startswith("audio/"):
                url = link.get("href")
                if url:
                    return url

    return None


def _extract_file_size(entry: Any) -> Optional[int]:
    """
    Extract audio file size from a feedparser entry's enclosures.

    Args:
        entry: feedparser entry object

    Returns:
        File size in bytes or None if not available
    """
    if hasattr(entry, "enclosures") and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get("type", "").startswith("audio/"):
                length = enc.get("length")
                if length:
                    try:
                        return int(length)
                    except (ValueError, TypeError):
                        pass
    return None


def _load_known_guids_from_db(db_path: Path) -> Set[str]:
    """
    Load known episode GUIDs from the SQLite database.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        Set of known GUIDs
    """
    if not db_path.exists():
        return set()

    from podcast_intel.models.database import Database

    db = Database(db_path)
    guids: Set[str] = set()
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT guid FROM episodes")
            guids = {row[0] for row in cursor.fetchall() if row[0]}
    except Exception as exc:
        logger.warning("Could not load GUIDs from database: %s", exc)

    return guids


def _load_known_guids_from_json(json_path: Path) -> Set[str]:
    """
    Load known episode GUIDs from a JSON registry file.

    The JSON file is expected to be a list of objects, each containing
    a ``guid`` key.

    Args:
        json_path: Path to the episodes JSON file

    Returns:
        Set of known GUIDs
    """
    if not json_path.exists():
        return set()

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {ep.get("guid", "") for ep in data if ep.get("guid")}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load GUIDs from JSON: %s", exc)

    return set()


def load_known_guids(config: Optional[Any] = None) -> Set[str]:
    """
    Load all known episode GUIDs from both DB and JSON sources.

    Merges GUIDs from the SQLite database and the episodes.json
    registry file to provide a comprehensive set of known episodes.

    Args:
        config: Application Config object (optional, uses default if None)

    Returns:
        Set of all known GUIDs

    Example:
        >>> guids = load_known_guids()
        >>> print(f"Tracking {len(guids)} known episodes")
    """
    if config is None:
        config = get_config()

    guids: Set[str] = set()

    # Load from database
    guids |= _load_known_guids_from_db(config.db_path)

    # Load from JSON registry
    json_path = PROJECT_ROOT / "data" / "episodes.json"
    guids |= _load_known_guids_from_json(json_path)

    # Discard empty strings
    guids.discard("")

    return guids


# ---------------------------------------------------------------------------
#  Main watch entry point
# ---------------------------------------------------------------------------

def run_watch(
    config: Optional[Any] = None,
    once: bool = True,
    dry_run: bool = False,
) -> WatchResult:
    """
    Run an RSS watch check and return the result.

    Fetches the podcast RSS feed, compares episode GUIDs against known
    episodes, and returns information about any new episodes found.

    When ``dry_run`` is True, the check is performed normally but no
    side effects (database writes, pipeline triggers) occur. This is
    useful for CI or preview purposes.

    Args:
        config: Application Config object (optional, uses default if None)
        once: If True, run a single check and return (default).
              If False, reserved for future continuous-watch mode.
        dry_run: If True, do not trigger any pipelines or write state.

    Returns:
        WatchResult with details about new episodes and any errors

    Example:
        >>> result = run_watch(once=True, dry_run=True)
        >>> print(result.to_json())
    """
    if config is None:
        config = get_config()

    result = WatchResult(
        checked_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z") or datetime.now().isoformat(),
    )

    # Validate RSS URL
    rss_url = config.rss_url
    if not rss_url:
        # Try loading from podcast.yaml
        yaml_config = load_podcast_yaml()
        rss_url = yaml_config.get("podcast", {}).get("rss_url", "") or yaml_config.get("rss_url", "")

    if not rss_url:
        result.errors.append(
            "No RSS URL configured. Set PODCAST_INTEL_RSS_URL or add rss_url to podcast.yaml."
        )
        return result

    # Load known GUIDs
    known_guids = load_known_guids(config)
    result.known_guid_count = len(known_guids)

    # Check for new episodes
    try:
        feed = feedparser.parse(rss_url)
        result.feed_title = feed.feed.get("title", "") if hasattr(feed, "feed") else ""
        result.total_feed_episodes = len(feed.entries)

        new_episodes = check_new_episodes(rss_url, known_guids)
        result.episodes = new_episodes
        result.has_new_episodes = len(new_episodes) > 0

    except ValueError as exc:
        result.errors.append(str(exc))
    except Exception as exc:
        result.errors.append(f"RSS fetch error: {exc}")

    # Log results
    if result.has_new_episodes:
        logger.info(
            "Found %d new episode(s) in feed '%s'",
            len(result.episodes),
            result.feed_title,
        )
        for ep in result.episodes:
            logger.info("  New: %s (guid=%s)", ep.title, ep.guid[:40])
    else:
        logger.debug("No new episodes found (checked %d known GUIDs)", result.known_guid_count)

    if result.errors:
        for err in result.errors:
            logger.error("Watch error: %s", err)

    return result

"""
RSS feed parsing and episode metadata extraction.

Handles parsing of podcast RSS feeds, extracting episode metadata including
title, description, publication date, audio URL, and duration.
Supports incremental sync via GUID tracking to avoid redundant downloads.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import feedparser
from dateutil import parser as date_parser


def parse_rss_feed(url: str) -> List[Dict[str, Any]]:
    """
    Parse RSS feed and extract episode metadata.

    Uses feedparser to parse the RSS feed and extracts all relevant episode
    information including title, description, publication date, audio URL,
    duration, and file size.

    Args:
        url: URL of the RSS feed

    Returns:
        List of episode metadata dictionaries

    Example:
        >>> episodes = parse_rss_feed("https://example.com/podcast/rss")
        >>> print(f"Found {len(episodes)} episodes")
        >>> print(episodes[0]["title"])
    """
    print(f"Fetching RSS feed from: {url}")
    feed = feedparser.parse(url)

    if feed.bozo:
        print(f"Warning: Feed parsing encountered errors: {feed.bozo_exception}")

    episodes = []

    for entry in feed.entries:
        episode = extract_episode_metadata(entry)
        if episode:
            episodes.append(episode)

    print(f"Successfully parsed {len(episodes)} episodes from feed")
    return episodes


def parse_duration(duration_str: str) -> int:
    """
    Parse iTunes duration string to total seconds.

    Supports multiple duration formats:
    - HH:MM:SS (e.g., "01:23:45" = 5025 seconds)
    - MM:SS (e.g., "45:30" = 2730 seconds)
    - SS (e.g., "90" = 90 seconds)

    Args:
        duration_str: Duration string from RSS feed

    Returns:
        Total duration in seconds

    Example:
        >>> parse_duration("01:23:45")
        5025
        >>> parse_duration("45:30")
        2730
        >>> parse_duration("90")
        90
    """
    if not duration_str:
        return 0

    duration_str = duration_str.strip()
    parts = duration_str.split(":")

    try:
        if len(parts) == 3:
            # HH:MM:SS format
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            # MM:SS format
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 1:
            # SS format (just seconds)
            return int(parts[0])
        else:
            print(f"Warning: Unexpected duration format: {duration_str}")
            return 0
    except (ValueError, IndexError) as e:
        print(f"Warning: Failed to parse duration '{duration_str}': {e}")
        return 0


def extract_episode_metadata(entry: Any) -> Optional[Dict[str, Any]]:
    """
    Extract structured metadata from a feed entry.

    Extracts all relevant information from a feedparser entry object,
    including title, description, GUID, publication date, audio URL,
    duration, and file size.

    Args:
        entry: feedparser entry object

    Returns:
        Dictionary with episode metadata, or None if required fields are missing

    Example:
        >>> feed = feedparser.parse(rss_url)
        >>> metadata = extract_episode_metadata(feed.entries[0])
        >>> print(metadata["title"])
    """
    try:
        # Required fields
        guid = entry.get("id") or entry.get("guid")
        title = entry.get("title")

        if not guid or not title:
            print(f"Warning: Episode missing required fields (guid={guid}, title={title})")
            return None

        # Description
        description = entry.get("summary") or entry.get("description") or ""

        # Publication date
        pub_date_str = None
        if hasattr(entry, "published"):
            pub_date_str = entry.published
        elif hasattr(entry, "pubDate"):
            pub_date_str = entry.pubDate

        # Parse and normalize date to ISO-8601 format
        if pub_date_str:
            try:
                pub_date = date_parser.parse(pub_date_str)
                pub_date_iso = pub_date.strftime("%Y-%m-%dT%H:%M:%S%z")
                # Ensure timezone format is correct
                if not pub_date_iso.endswith("+00:00") and not pub_date_iso.endswith("Z"):
                    if pub_date_iso[-5:][0] in ['+', '-']:
                        pass  # Already has timezone
                    else:
                        pub_date_iso += "+00:00"
            except Exception as e:
                print(f"Warning: Failed to parse date '{pub_date_str}': {e}")
                pub_date_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")
        else:
            pub_date_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")

        # Audio URL and file size
        audio_url = None
        file_size_bytes = None

        # Try to find audio enclosure
        if hasattr(entry, "enclosures") and entry.enclosures:
            for enclosure in entry.enclosures:
                if enclosure.get("type", "").startswith("audio/"):
                    audio_url = enclosure.get("href") or enclosure.get("url")
                    file_size_bytes = enclosure.get("length")
                    if file_size_bytes:
                        try:
                            file_size_bytes = int(file_size_bytes)
                        except (ValueError, TypeError):
                            file_size_bytes = None
                    break

        # Fallback: check for links with audio types
        if not audio_url and hasattr(entry, "links"):
            for link in entry.links:
                if link.get("type", "").startswith("audio/"):
                    audio_url = link.get("href")
                    break

        if not audio_url:
            print(f"Warning: No audio URL found for episode '{title}'")
            return None

        # Duration
        duration_seconds = None
        if hasattr(entry, "itunes_duration"):
            duration_seconds = parse_duration(entry.itunes_duration)

        # Episode type
        episode_type = "full"
        if hasattr(entry, "itunes_episodetype"):
            ep_type = entry.itunes_episodetype.lower()
            if ep_type in ["full", "trailer", "bonus"]:
                episode_type = ep_type

        return {
            "guid": guid,
            "title": title,
            "description": description,
            "pub_date": pub_date_iso,
            "audio_url": audio_url,
            "duration_seconds": duration_seconds,
            "file_size_bytes": file_size_bytes,
            "episode_type": episode_type,
        }

    except Exception as e:
        print(f"Error extracting metadata from entry: {e}")
        return None


def get_new_episodes(feed_entries: List[Any], existing_guids: set) -> List[Dict[str, Any]]:
    """
    Filter feed entries to only new episodes not in database.

    Compares feed entries against a set of existing GUIDs to identify
    which episodes need to be downloaded and processed.

    Args:
        feed_entries: List of feed entries from feedparser
        existing_guids: Set of GUIDs already in database

    Returns:
        List of new episode metadata dictionaries

    Example:
        >>> feed = feedparser.parse(rss_url)
        >>> existing = {"guid-1", "guid-2", "guid-3"}
        >>> new_eps = get_new_episodes(feed.entries, existing)
        >>> print(f"Found {len(new_eps)} new episodes")
    """
    new_episodes = []

    for entry in feed_entries:
        episode = extract_episode_metadata(entry)
        if episode and episode["guid"] not in existing_guids:
            new_episodes.append(episode)

    return new_episodes


def parse_feed(rss_url: str) -> List[Dict[str, Any]]:
    """
    Parse RSS feed and extract episode metadata.

    Convenience function that wraps parse_rss_feed for backwards compatibility.

    Args:
        rss_url: URL of the RSS feed

    Returns:
        List of episode metadata dictionaries

    Example:
        >>> episodes = parse_feed("https://example.com/podcast/rss")
        >>> print(f"Found {len(episodes)} episodes")
    """
    return parse_rss_feed(rss_url)


def main():
    """
    Main entry point for RSS parser testing.

    Can be run with: python -m podcast_intel.ingestion.rss_parser
    """
    from ..config import get_config

    print("=== RSS Feed Parser ===")
    print()

    config = get_config()
    print(f"RSS URL: {config.rss_url}")
    print()

    # Parse feed
    episodes = parse_rss_feed(config.rss_url)
    print()

    print(f"Successfully parsed {len(episodes)} episodes")
    print()

    # Show first few episodes
    print("First 5 episodes:")
    for i, episode in enumerate(episodes[:5], 1):
        duration_str = f"{episode['duration_seconds']//60} min" if episode.get('duration_seconds') else "unknown"
        size_str = f"{episode['file_size_bytes']//1024//1024} MB" if episode.get('file_size_bytes') else "unknown"
        print(f"{i}. {episode['title']}")
        print(f"   Duration: {duration_str}, Size: {size_str}")
        print(f"   Published: {episode['pub_date'][:10]}")
        print(f"   GUID: {episode['guid'][:50]}...")
        print()


if __name__ == "__main__":
    main()

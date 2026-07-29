"""
Ingestion module for RSS feed parsing and audio file downloading.

Provides functionality for incremental episode synchronization and
MP3 file management.
"""

from podcast_intel.ingestion.downloader import download_episode
from podcast_intel.ingestion.rss_parser import parse_feed

__all__ = ["parse_feed", "download_episode"]

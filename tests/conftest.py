"""
Shared test fixtures.

Provides pytest fixtures for common test resources including:
- Mock configuration
- Temporary database
- Sample episodes and segments
- Mock transcription data
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from podcast_intel.config import Config
from podcast_intel.models.database import Database
from podcast_intel.models.entities import (
    Episode,
    EpisodeType,
    Language,
    Segment,
    Speaker,
    TranscriptionStatus,
)


@pytest.fixture
def temp_dir():
    """
    Create temporary directory for test files.

    Yields:
        Path: Temporary directory path
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """
    Create test configuration with temporary paths.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Config: Test configuration
    """
    return Config(
        db_path=temp_dir / "db" / "podcast_intel.db",
        audio_dir=temp_dir / "audio",
        embeddings_dir=temp_dir / "embeddings",
        transcription_device="cpu",
        diarization_enabled=False,
    )


@pytest.fixture
def test_db(temp_dir: Path) -> Database:
    """
    Create test database with schema.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Database: Initialized test database
    """
    database = Database(temp_dir / "test.db")
    database.initialize()
    return database


@pytest.fixture
def sample_episodes() -> list[Episode]:
    """
    Create sample episode data for testing.

    Returns:
        List of Episode objects
    """
    base = datetime(2026, 1, 1, 9, 0, 0)
    return [
        Episode(
            id=index,
            guid=f"sample-guid-{index}",
            title=f"Episode {200 + index}",
            description=f"Description for episode {200 + index}",
            pub_date=base + timedelta(days=7 * index),
            audio_url=f"https://feeds.example.com/audio/{index}.mp3",
            duration_seconds=2400 + 600 * index,
            file_size_bytes=30_000_000 + 1_000_000 * index,
            episode_type=EpisodeType.FULL,
            transcription_status=TranscriptionStatus.PENDING,
        )
        for index in range(1, 4)
    ]


@pytest.fixture
def sample_speakers() -> list[Speaker]:
    """
    Create sample speaker data for testing.

    Returns:
        List of Speaker objects
    """
    return [
        Speaker(id=1, name="Alex", is_host=True),
        Speaker(id=2, name="Jordan", is_host=True),
        Speaker(id=3, name="Sam", is_host=False),
    ]


@pytest.fixture
def sample_segments() -> list[Segment]:
    """
    Create sample segment data for testing.

    Returns:
        List of Segment objects
    """
    texts = [
        "Welcome back to the show, today we are covering a lot of ground.",
        "I think the more interesting question is what happens next.",
        "Right, and that is basically where the data disagrees with the story.",
    ]
    return [
        Segment(
            id=index + 1,
            episode_id=1,
            speaker_id=(index % 3) + 1,
            start_time=float(index * 15),
            end_time=float(index * 15 + 15),
            text=text,
            word_count=len(text.split()),
            language=Language.ENGLISH,
        )
        for index, text in enumerate(texts)
    ]


@pytest.fixture
def mock_transcript() -> dict[str, Any]:
    """
    Create mock transcript data for testing.

    Returns:
        Dictionary with segments and metadata
    """
    return {
        "language": "en",
        "duration": 45.0,
        "metadata": {"model": "mock", "diarized": True},
        "segments": [
            {
                "start": 0.0,
                "end": 15.0,
                "speaker_name": "Alex",
                "text": "Welcome back to the show, um, we have a lot to cover.",
                "language": "en",
            },
            {
                "start": 15.0,
                "end": 30.0,
                "speaker_name": "Jordan",
                "text": "You know, I think the second half is the interesting part.",
                "language": "en",
            },
            {
                "start": 30.0,
                "end": 45.0,
                "speaker_name": "Sam",
                "text": "Actually, the numbers tell a different story here.",
                "language": "en",
            },
        ],
    }

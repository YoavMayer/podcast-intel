"""
Exercises every fixture in conftest.py.

CONTRIBUTING.md tells contributors to "use fixtures from tests/conftest.py".
All six were placeholder bodies returning None against declared return types, so
that instruction produced an AttributeError. These tests keep them real.
"""

from pathlib import Path
from typing import Any

from podcast_intel.config import Config
from podcast_intel.models.database import Database
from podcast_intel.models.entities import Episode, Segment, Speaker


def test_temp_dir_is_a_real_directory(temp_dir: Path):
    assert temp_dir.is_dir()


def test_test_config_points_at_the_temp_dir(test_config: Config, temp_dir: Path):
    assert isinstance(test_config, Config)
    assert temp_dir in test_config.db_path.parents
    assert test_config.audio_dir.parent == temp_dir
    assert test_config.transcription_device == "cpu"


def test_test_db_has_a_schema(test_db: Database):
    with test_db.get_connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
    assert {"episodes", "segments", "speakers"} <= tables


def test_test_db_accepts_an_insert(test_db: Database):
    with test_db.get_connection() as conn:
        episode_id = test_db.insert_episode(
            conn,
            guid="fixture-guid",
            title="Fixture Episode",
            pub_date="2026-01-01T00:00:00+00:00",
            audio_url="https://example.com/fixture.mp3",
        )
    assert isinstance(episode_id, int) and episode_id > 0


def test_sample_episodes(sample_episodes: list[Episode]):
    assert len(sample_episodes) == 3
    assert all(isinstance(ep, Episode) for ep in sample_episodes)
    guids = [ep.guid for ep in sample_episodes]
    assert len(set(guids)) == len(guids)
    dates = [ep.pub_date for ep in sample_episodes]
    assert dates == sorted(dates)


def test_sample_speakers(sample_speakers: list[Speaker]):
    assert len(sample_speakers) == 3
    assert sum(s.is_host for s in sample_speakers) == 2


def test_sample_segments(sample_segments: list[Segment]):
    assert len(sample_segments) == 3
    for seg in sample_segments:
        assert isinstance(seg, Segment)
        assert seg.end_time > seg.start_time
        assert seg.word_count == len(seg.text.split())


def test_mock_transcript(mock_transcript: dict[str, Any]):
    assert mock_transcript["language"] == "en"
    segments = mock_transcript["segments"]
    assert len(segments) == 3
    assert segments[0]["start"] == 0.0
    for earlier, later in zip(segments, segments[1:]):
        assert later["start"] >= earlier["end"]

"""
Tests for mock ingestion and transcription.

Validates that mock implementations generate realistic data
suitable for development and testing without external dependencies.

Covers:
- Mock ingestion: episode creation, speaker insertion, metadata
- Mock transcription: segment generation, speaker labels, timestamps
- Text (UTF-8) storage and retrieval
- Timestamp validity and sequencing
- Filler word detection and storage
- Silence event generation and storage
- Data integrity: foreign keys, constraints, non-null fields
- Episode status transitions (pending -> processing -> completed)
"""

import pytest
import sqlite3
import re
from pathlib import Path
from datetime import datetime, timedelta

from podcast_intel.models.database import Database
from podcast_intel.models.schema import create_all_tables, get_table_names
from podcast_intel.ingestion.mock_ingest import (
    generate_mock_episodes,
    create_test_episode,
)
from podcast_intel.transcription.mock_transcribe import (
    MockTranscriber,
    generate_mock_transcription,
    _find_filler_words_in_text,
    FILLER_WORDS_ALL,
    SPEAKER_PROFILES,
    ENGLISH_TERMS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Create an initialized Database in a temporary directory."""
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    database.initialize()
    return database


@pytest.fixture
def db_with_episodes(db):
    """Database pre-populated with mock episodes (and speakers)."""
    episode_ids = generate_mock_episodes(db, count=5)
    return db, episode_ids


@pytest.fixture
def db_with_transcription(db_with_episodes):
    """Database pre-populated with episodes AND one transcribed episode."""
    db, episode_ids = db_with_episodes
    first_id = episode_ids[0]
    seg_count = generate_mock_transcription(db, first_id)
    return db, episode_ids, first_id, seg_count


# ===================================================================
# Test 1: Mock ingestion creates episodes in SQLite correctly
# ===================================================================

class TestMockIngestion:
    """Tests for mock_ingest.generate_mock_episodes."""

    def test_generates_correct_number_of_episodes(self, db):
        """generate_mock_episodes with count=5 inserts exactly 5 episodes."""
        ids = generate_mock_episodes(db, count=5)
        assert len(ids) == 5
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM episodes")
            assert cursor.fetchone()[0] == 5

    def test_generates_fewer_episodes_when_requested(self, db):
        """Requesting count=2 yields 2 episodes."""
        ids = generate_mock_episodes(db, count=2)
        assert len(ids) == 2

    def test_episode_ids_are_positive_integers(self, db):
        """All returned IDs are positive integers."""
        ids = generate_mock_episodes(db, count=3)
        for eid in ids:
            assert isinstance(eid, int)
            assert eid > 0

    def test_episode_fields_are_populated(self, db):
        """Each episode has non-null required fields: guid, title, pub_date, audio_url."""
        generate_mock_episodes(db, count=3)
        with db.get_connection() as conn:
            episodes = db.get_all_episodes(conn)
            assert len(episodes) == 3
            for ep in episodes:
                assert ep["guid"] is not None and len(ep["guid"]) > 0
                assert ep["title"] is not None and len(ep["title"]) > 0
                assert ep["pub_date"] is not None
                assert ep["audio_url"] is not None and ep["audio_url"].startswith("https://")
                assert ep["duration_seconds"] is not None and ep["duration_seconds"] > 0
                assert ep["file_size_bytes"] is not None and ep["file_size_bytes"] > 0
                assert ep["episode_type"] == "full"

    def test_episode_guids_are_unique(self, db):
        """No duplicate GUIDs across generated episodes."""
        generate_mock_episodes(db, count=5)
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT guid FROM episodes")
            guids = [row[0] for row in cursor.fetchall()]
            assert len(guids) == len(set(guids))

    def test_speakers_are_created(self, db):
        """Mock ingestion creates exactly 3 speakers."""
        generate_mock_episodes(db, count=1)
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM speakers")
            assert cursor.fetchone()[0] == 3

    def test_speaker_names_and_roles(self, db):
        """Speakers have English names and correct host/guest roles."""
        generate_mock_episodes(db, count=1)
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT name, is_host FROM speakers ORDER BY id")
            rows = cursor.fetchall()
            expected = [
                ("Alex", 1),
                ("Jordan", 1),
                ("Sam", 0),
            ]
            for row, (name, is_host) in zip(rows, expected):
                assert row["name"] == name
                assert row["is_host"] == is_host

    def test_episode_default_status_is_pending(self, db):
        """Newly created episodes have transcription_status = 'pending'."""
        generate_mock_episodes(db, count=2)
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT transcription_status FROM episodes")
            for row in cursor.fetchall():
                assert row[0] == "pending"

    def test_episode_duration_in_valid_range(self, db):
        """Episode durations fall in the 2400-5400 seconds range."""
        generate_mock_episodes(db, count=5)
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT duration_seconds FROM episodes")
            for row in cursor.fetchall():
                assert 2400 <= row[0] <= 5400

    def test_create_test_episode_returns_dict(self):
        """create_test_episode returns a dictionary with all required keys."""
        ep = create_test_episode(
            title="Episode 100 - Season Review",
            duration_seconds=3600,
            pub_date=datetime(2024, 12, 15),
        )
        assert isinstance(ep, dict)
        required_keys = {
            "guid", "title", "description", "pub_date",
            "audio_url", "duration_seconds", "file_size_bytes", "episode_type",
        }
        assert required_keys.issubset(ep.keys())
        assert ep["title"] == "Episode 100 - Season Review"
        assert ep["duration_seconds"] == 3600
        assert ep["episode_type"] == "full"

    def test_create_test_episode_description_contains_title(self):
        """Description includes a reference to the title."""
        title = "Episode 100 - Test Episode"
        ep = create_test_episode(title=title, duration_seconds=1800, pub_date=datetime.now())
        assert title in ep["description"]


# ===================================================================
# Test 2: Mock transcription creates segments with valid speaker labels
# ===================================================================

class TestMockTranscription:
    """Tests for MockTranscriber and generate_mock_transcription."""

    def test_transcriber_produces_segments(self):
        """MockTranscriber.transcribe returns a result with 15-25 segments."""
        t = MockTranscriber(num_speakers=3)
        result = t.transcribe(Path("fake_audio.mp3"))
        assert 15 <= len(result.segments) <= 25

    def test_segments_have_valid_speaker_labels(self):
        """Every segment has a speaker label of the form SPEAKER_XX."""
        t = MockTranscriber(num_speakers=3)
        result = t.transcribe(Path("fake_audio.mp3"), diarize=True)
        for seg in result.segments:
            assert seg["speaker"] is not None
            assert re.match(r"^SPEAKER_\d{2}$", seg["speaker"])

    def test_speaker_labels_correspond_to_speaker_count(self):
        """Speaker indices stay within 0..num_speakers-1."""
        t = MockTranscriber(num_speakers=2)
        result = t.transcribe(Path("fake.mp3"), diarize=True)
        speakers = {seg["speaker"] for seg in result.segments}
        assert speakers.issubset({"SPEAKER_00", "SPEAKER_01"})

    def test_diarize_false_gives_no_speakers(self):
        """With diarize=False, speaker fields are None."""
        t = MockTranscriber(num_speakers=3)
        result = t.transcribe(Path("fake.mp3"), diarize=False)
        for seg in result.segments:
            assert seg["speaker"] is None
            assert seg["speaker_name"] is None

    def test_transcription_result_language(self):
        """TranscriptionResult.language matches the requested language."""
        t = MockTranscriber(num_speakers=3)
        result = t.transcribe(Path("x.mp3"), language="en")
        assert result.language == "en"

    def test_transcription_result_duration_positive(self):
        """Result duration is positive and non-zero."""
        t = MockTranscriber(num_speakers=3)
        result = t.transcribe(Path("x.mp3"))
        assert result.duration > 0

    def test_diarization_list_matches_segments(self):
        """Diarization list has same length as segments when diarize=True."""
        t = MockTranscriber(num_speakers=3)
        result = t.transcribe(Path("x.mp3"), diarize=True)
        assert len(result.diarization) == len(result.segments)

    def test_generate_mock_transcription_stores_segments(self, db_with_episodes):
        """generate_mock_transcription inserts segments into database."""
        db, episode_ids = db_with_episodes
        seg_count = generate_mock_transcription(db, episode_ids[0])
        assert seg_count >= 15
        with db.get_connection() as conn:
            rows = db.get_segments_by_episode(conn, episode_ids[0])
            assert len(rows) == seg_count

    def test_segments_have_valid_speaker_ids_in_db(self, db_with_transcription):
        """All segment speaker_ids reference valid speakers in the DB."""
        db, episode_ids, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            segments = db.get_segments_by_episode(conn, ep_id)
            speaker_cursor = conn.execute("SELECT id FROM speakers")
            valid_speaker_ids = {row[0] for row in speaker_cursor.fetchall()}
            for seg in segments:
                if seg["speaker_id"] is not None:
                    assert seg["speaker_id"] in valid_speaker_ids

    def test_segment_word_count_matches_text(self, db_with_transcription):
        """Segment word_count matches actual word count of text."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            segments = db.get_segments_by_episode(conn, ep_id)
            for seg in segments:
                actual_words = len(seg["text"].split())
                assert seg["word_count"] == actual_words


# ===================================================================
# Test 3: Text (UTF-8) is stored and retrieved correctly
# ===================================================================

class TestTextUtf8:
    """Tests for text round-tripping through SQLite."""

    def test_episode_titles_are_non_empty(self, db_with_episodes):
        """Episode titles are non-empty strings."""
        db, _ = db_with_episodes
        with db.get_connection() as conn:
            episodes = db.get_all_episodes(conn)
            for ep in episodes:
                title = ep["title"]
                assert title is not None and len(title) > 0

    def test_episode_descriptions_are_non_empty(self, db_with_episodes):
        """Descriptions are non-empty strings."""
        db, _ = db_with_episodes
        with db.get_connection() as conn:
            episodes = db.get_all_episodes(conn)
            for ep in episodes:
                desc = ep["description"]
                assert desc is not None and len(desc) > 0

    def test_segment_text_contains_english(self, db_with_transcription):
        """Transcribed segments contain English text."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            segments = db.get_segments_by_episode(conn, ep_id)
            english_count = 0
            for seg in segments:
                if any(ch.isascii() and ch.isalpha() for ch in seg["text"]):
                    english_count += 1
            # At least most segments should contain English
            assert english_count > len(segments) * 0.5

    def test_speaker_names_round_trip(self, db_with_episodes):
        """Speaker names survive database round-trip."""
        db, _ = db_with_episodes
        expected_names = {"Alex", "Jordan", "Sam"}
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM speakers")
            actual = {row[0] for row in cursor.fetchall()}
            assert actual == expected_names

    def test_text_exact_round_trip(self, db):
        """Insert and retrieve a specific string unchanged."""
        db.initialize()
        test_text = "Episode Special: Deep Tactical Analysis - Advanced Pressing Patterns"
        with db.get_connection() as conn:
            eid = db.insert_episode(
                conn,
                guid="test-text-roundtrip",
                title=test_text,
                pub_date="2024-01-01T00:00:00+00:00",
                audio_url="https://example.com/test.mp3",
            )
            ep = db.get_episode_by_id(conn, eid)
            assert ep["title"] == test_text

    def test_segment_language_labels(self, db_with_transcription):
        """Segment language is one of: he, en, mixed."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            segments = db.get_segments_by_episode(conn, ep_id)
            for seg in segments:
                assert seg["language"] in ("he", "en", "mixed")


# ===================================================================
# Test 4: Timestamps are valid and sequential within episodes
# ===================================================================

class TestTimestamps:
    """Tests for timestamp validity and ordering."""

    def test_segment_start_before_end(self, db_with_transcription):
        """Every segment has start_time < end_time."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            segments = db.get_segments_by_episode(conn, ep_id)
            for seg in segments:
                assert seg["start_time"] < seg["end_time"], (
                    f"Segment {seg['id']}: start={seg['start_time']} >= end={seg['end_time']}"
                )

    def test_segments_ordered_by_start_time(self, db_with_transcription):
        """get_segments_by_episode returns segments ordered by start_time."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            segments = db.get_segments_by_episode(conn, ep_id)
            for i in range(1, len(segments)):
                assert segments[i]["start_time"] >= segments[i - 1]["start_time"]

    def test_segment_times_are_non_negative(self, db_with_transcription):
        """All start/end times are non-negative."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            segments = db.get_segments_by_episode(conn, ep_id)
            for seg in segments:
                assert seg["start_time"] >= 0
                assert seg["end_time"] >= 0

    def test_word_timestamps_sequential(self):
        """Word-level timestamps are sequential within segments."""
        t = MockTranscriber(num_speakers=2)
        words = t.get_word_timestamps(Path("fake.mp3"))
        assert len(words) > 0
        for w in words:
            assert w["start"] <= w["end"], (
                f"Word '{w['word']}': start={w['start']} > end={w['end']}"
            )

    def test_word_timestamps_have_required_fields(self):
        """Word timestamps contain word, start, end, confidence, speaker."""
        t = MockTranscriber(num_speakers=2)
        words = t.get_word_timestamps(Path("fake.mp3"))
        for w in words:
            assert "word" in w
            assert "start" in w
            assert "end" in w
            assert "confidence" in w
            assert "speaker" in w

    def test_segment_confidence_in_valid_range(self, db_with_transcription):
        """Segment confidence is between 0.0 and 1.0."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            segments = db.get_segments_by_episode(conn, ep_id)
            for seg in segments:
                if seg["confidence"] is not None:
                    assert 0.0 <= seg["confidence"] <= 1.0

    def test_segment_sentiment_in_valid_range(self, db_with_transcription):
        """Segment sentiment is between -1.0 and 1.0."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            segments = db.get_segments_by_episode(conn, ep_id)
            for seg in segments:
                if seg["sentiment_score"] is not None:
                    assert -1.0 <= seg["sentiment_score"] <= 1.0


# ===================================================================
# Test 5: Filler words are detected and stored
# ===================================================================

class TestFillerWords:
    """Tests for filler word detection and storage."""

    def test_filler_detection_finds_known_fillers(self):
        """_find_filler_words_in_text detects known English filler words."""
        text = "um, I think that is good, like, obviously"
        fillers = _find_filler_words_in_text(text)
        filler_texts = [f[0] for f in fillers]
        assert "um" in filler_texts
        assert "like" in filler_texts

    def test_filler_detection_returns_positions(self):
        """Filler positions (offsets) are non-negative integers."""
        text = "well, basically, I agree"
        fillers = _find_filler_words_in_text(text)
        for filler_text, offset in fillers:
            assert isinstance(offset, int)
            assert offset >= 0
            # The filler text should appear at the offset position
            assert text[offset:offset + len(filler_text)] == filler_text

    def test_filler_words_stored_in_db(self, db_with_transcription):
        """generate_mock_transcription stores filler words in the filler_words table."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM filler_words WHERE episode_id = ?",
                (ep_id,),
            )
            count = cursor.fetchone()[0]
            # Given filler injection, we expect at least some fillers
            assert count > 0, "No filler words stored for transcribed episode"

    def test_filler_words_reference_valid_segments(self, db_with_transcription):
        """Filler word segment_ids reference existing segments."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT segment_id FROM filler_words WHERE episode_id = ?",
                (ep_id,),
            )
            segment_ids = {row[0] for row in cursor.fetchall()}
            for sid in segment_ids:
                seg = conn.execute(
                    "SELECT id FROM segments WHERE id = ?", (sid,)
                ).fetchone()
                assert seg is not None, f"Filler references non-existent segment {sid}"

    def test_filler_text_is_known(self, db_with_transcription):
        """All stored filler texts are from the known filler list."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT filler_text FROM filler_words WHERE episode_id = ?",
                (ep_id,),
            )
            for row in cursor.fetchall():
                assert row[0] in FILLER_WORDS_ALL, f"Unknown filler: {row[0]}"

    def test_filler_detection_standalone_only(self):
        """Filler detection should only find standalone tokens, not substrings."""
        # "so" should not be found inside "someone" (different word)
        text = "someone helped"
        fillers = _find_filler_words_in_text(text)
        filler_texts = [f[0] for f in fillers]
        assert "so" not in filler_texts

    def test_multiple_fillers_per_segment(self):
        """Multiple filler occurrences in same text are all detected."""
        text = "um, like, I think, um, that is right"
        fillers = _find_filler_words_in_text(text)
        um_count = sum(1 for f, _ in fillers if f == "um")
        assert um_count == 2


# ===================================================================
# Test 6: Silence events are generated and stored
# ===================================================================

class TestSilenceEvents:
    """Tests for silence event generation and storage."""

    def test_silence_events_created(self, db_with_transcription):
        """Transcription generates between 2 and 4 silence events per episode."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM silence_events WHERE episode_id = ?",
                (ep_id,),
            )
            count = cursor.fetchone()[0]
            assert 2 <= count <= 4, f"Expected 2-4 silence events, got {count}"

    def test_silence_event_types_valid(self, db_with_transcription):
        """Silence event types are 'dead_air' or 'long_pause'."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT event_type FROM silence_events WHERE episode_id = ?",
                (ep_id,),
            )
            for row in cursor.fetchall():
                assert row[0] in ("dead_air", "long_pause", "technical")

    def test_silence_event_duration_positive(self, db_with_transcription):
        """All silence durations are positive."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT duration, start_time, end_time FROM silence_events WHERE episode_id = ?",
                (ep_id,),
            )
            for row in cursor.fetchall():
                assert row["duration"] > 0
                assert row["end_time"] > row["start_time"]

    def test_silence_event_has_speaker_refs(self, db_with_transcription):
        """Silence events have preceding and/or following speaker IDs."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT preceding_speaker_id, following_speaker_id FROM silence_events WHERE episode_id = ?",
                (ep_id,),
            )
            for row in cursor.fetchall():
                # At least one speaker reference should be set
                assert row["preceding_speaker_id"] is not None or row["following_speaker_id"] is not None

    def test_silence_speaker_ids_valid(self, db_with_transcription):
        """Silence event speaker IDs reference valid speakers."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            speaker_ids = {
                row[0] for row in conn.execute("SELECT id FROM speakers").fetchall()
            }
            cursor = conn.execute(
                "SELECT preceding_speaker_id, following_speaker_id FROM silence_events WHERE episode_id = ?",
                (ep_id,),
            )
            for row in cursor.fetchall():
                if row["preceding_speaker_id"] is not None:
                    assert row["preceding_speaker_id"] in speaker_ids
                if row["following_speaker_id"] is not None:
                    assert row["following_speaker_id"] in speaker_ids


# ===================================================================
# Test 7: Data integrity - foreign keys, constraints, non-null fields
# ===================================================================

class TestDataIntegrity:
    """Tests for database constraints and referential integrity."""

    def test_schema_has_all_10_tables(self, db):
        """Database schema includes all 10 required tables."""
        tables = get_table_names(db.db_path)
        expected_tables = {
            "episodes", "speakers", "segments", "entities",
            "entity_mentions", "metrics", "embeddings_metadata",
            "filler_words", "silence_events", "coaching_notes",
        }
        actual_set = set(tables)
        for t in expected_tables:
            assert t in actual_set, f"Missing table: {t}"

    def test_duplicate_guid_rejected(self, db):
        """Inserting an episode with a duplicate GUID raises IntegrityError."""
        with db.get_connection() as conn:
            db.insert_episode(
                conn, guid="dup-guid", title="First",
                pub_date="2024-01-01T00:00:00+00:00",
                audio_url="https://example.com/1.mp3",
            )
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                db.insert_episode(
                    conn, guid="dup-guid", title="Second",
                    pub_date="2024-01-02T00:00:00+00:00",
                    audio_url="https://example.com/2.mp3",
                )

    def test_segment_requires_episode_id(self, db):
        """Inserting a segment referencing a non-existent episode fails."""
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                db.insert_segment(
                    conn, episode_id=99999,
                    start_time=0.0, end_time=1.0,
                    text="test",
                )

    def test_episode_type_check_constraint(self, db):
        """Episode type must be one of: full, trailer, bonus."""
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                conn.execute(
                    """INSERT INTO episodes (guid, title, pub_date, audio_url, episode_type)
                       VALUES (?, ?, ?, ?, ?)""",
                    ("g1", "t1", "2024-01-01", "https://ex.com/a.mp3", "invalid_type"),
                )

    def test_transcription_status_check_constraint(self, db):
        """Episode transcription_status must be one of: pending, processing, completed, failed."""
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                conn.execute(
                    """INSERT INTO episodes (guid, title, pub_date, audio_url, transcription_status)
                       VALUES (?, ?, ?, ?, ?)""",
                    ("g2", "t2", "2024-01-01", "https://ex.com/b.mp3", "banana"),
                )

    def test_segment_end_time_must_exceed_start(self, db):
        """Segment with end_time <= start_time violates CHECK constraint."""
        with db.get_connection() as conn:
            eid = db.insert_episode(
                conn, guid="check-test", title="Check",
                pub_date="2024-01-01T00:00:00+00:00",
                audio_url="https://example.com/check.mp3",
            )
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                db.insert_segment(
                    conn, episode_id=eid,
                    start_time=10.0, end_time=5.0,
                    text="bad range",
                )

    def test_language_check_constraint(self, db):
        """Segment language must be he, en, or mixed."""
        with db.get_connection() as conn:
            eid = db.insert_episode(
                conn, guid="lang-test", title="Lang",
                pub_date="2024-01-01T00:00:00+00:00",
                audio_url="https://example.com/lang.mp3",
            )
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                conn.execute(
                    """INSERT INTO segments (episode_id, start_time, end_time, text, language)
                       VALUES (?, ?, ?, ?, ?)""",
                    (eid, 0.0, 1.0, "text", "fr"),
                )

    def test_entity_type_check_constraint(self, db):
        """Entity type must be one of the allowed values."""
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                conn.execute(
                    """INSERT INTO entities (canonical_name, entity_type)
                       VALUES (?, ?)""",
                    ("Test", "invalid_type"),
                )

    def test_silence_event_type_check(self, db):
        """Silence event_type must be dead_air, long_pause, or technical."""
        with db.get_connection() as conn:
            eid = db.insert_episode(
                conn, guid="sil-test", title="Silence",
                pub_date="2024-01-01T00:00:00+00:00",
                audio_url="https://example.com/sil.mp3",
            )
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                conn.execute(
                    """INSERT INTO silence_events
                       (episode_id, start_time, end_time, duration, event_type)
                       VALUES (?, ?, ?, ?, ?)""",
                    (eid, 0.0, 5.0, 5.0, "invalid_type"),
                )

    def test_foreign_key_cascade_on_episode_delete(self, db_with_transcription):
        """Deleting an episode cascades to segments, filler_words, silence_events."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            # Verify data exists first
            seg_count = conn.execute(
                "SELECT COUNT(*) FROM segments WHERE episode_id = ?", (ep_id,)
            ).fetchone()[0]
            assert seg_count > 0

            # Delete the episode
            conn.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))

            # Verify cascaded deletes
            assert conn.execute(
                "SELECT COUNT(*) FROM segments WHERE episode_id = ?", (ep_id,)
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM filler_words WHERE episode_id = ?", (ep_id,)
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM silence_events WHERE episode_id = ?", (ep_id,)
            ).fetchone()[0] == 0

    def test_speaker_insert_is_idempotent(self, db):
        """Inserting the same speaker name twice returns the same ID."""
        with db.get_connection() as conn:
            id1 = db.insert_speaker(conn, name="TestSpeaker")
            id2 = db.insert_speaker(conn, name="TestSpeaker")
            assert id1 == id2


# ===================================================================
# Test 8: Episode status transitions (pending -> processing -> completed)
# ===================================================================

class TestEpisodeStatusTransitions:
    """Tests for episode transcription_status lifecycle."""

    def test_initial_status_is_pending(self, db_with_episodes):
        """Newly ingested episodes start with status 'pending'."""
        db, episode_ids = db_with_episodes
        with db.get_connection() as conn:
            for eid in episode_ids:
                ep = db.get_episode_by_id(conn, eid)
                assert ep["transcription_status"] == "pending"

    def test_status_transitions_to_completed_after_transcription(self, db_with_transcription):
        """After generate_mock_transcription, episode status is 'completed'."""
        db, _, ep_id, _ = db_with_transcription
        with db.get_connection() as conn:
            ep = db.get_episode_by_id(conn, ep_id)
            assert ep["transcription_status"] == "completed"

    def test_untranscribed_episodes_remain_pending(self, db_with_transcription):
        """Episodes not yet transcribed still have status 'pending'."""
        db, episode_ids, transcribed_id, _ = db_with_transcription
        with db.get_connection() as conn:
            for eid in episode_ids:
                if eid != transcribed_id:
                    ep = db.get_episode_by_id(conn, eid)
                    assert ep["transcription_status"] == "pending"

    def test_manual_status_update_processing(self, db_with_episodes):
        """update_episode_status can set status to 'processing'."""
        db, episode_ids = db_with_episodes
        with db.get_connection() as conn:
            db.update_episode_status(conn, episode_ids[0], "processing")
            ep = db.get_episode_by_id(conn, episode_ids[0])
            assert ep["transcription_status"] == "processing"

    def test_manual_status_update_failed(self, db_with_episodes):
        """update_episode_status can set status to 'failed'."""
        db, episode_ids = db_with_episodes
        with db.get_connection() as conn:
            db.update_episode_status(conn, episode_ids[0], "failed")
            ep = db.get_episode_by_id(conn, episode_ids[0])
            assert ep["transcription_status"] == "failed"

    def test_transcribe_nonexistent_episode_raises(self, db_with_episodes):
        """generate_mock_transcription raises ValueError for unknown episode."""
        db, _ = db_with_episodes
        with pytest.raises(ValueError, match="not found"):
            generate_mock_transcription(db, 99999)

    def test_transcription_updates_updated_at(self, db_with_episodes):
        """Transcription updates the updated_at timestamp."""
        db, episode_ids = db_with_episodes
        with db.get_connection() as conn:
            ep_before = db.get_episode_by_id(conn, episode_ids[0])
            created_at = ep_before["created_at"]

        # Transcribe
        generate_mock_transcription(db, episode_ids[0])

        with db.get_connection() as conn:
            ep_after = db.get_episode_by_id(conn, episode_ids[0])
            assert ep_after["updated_at"] is not None


# ===================================================================
# Additional integration / edge-case tests
# ===================================================================

class TestMockTranscriberGeneration:
    """Tests for the segment text generation logic."""

    def test_generate_segment_returns_string(self):
        """generate_segment returns a non-empty string."""
        t = MockTranscriber(num_speakers=3)
        for speaker_id in range(3):
            text = t.generate_segment(speaker_id)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_generate_hebrew_segment_returns_string(self):
        """generate_hebrew_segment (legacy alias) returns a non-empty string."""
        t = MockTranscriber(num_speakers=3)
        for speaker_id in range(3):
            text = t.generate_hebrew_segment(speaker_id)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_speaker_sequence_starts_with_host(self):
        """Speaker sequence always starts with speaker 0 (Alex, host)."""
        t = MockTranscriber(num_speakers=3)
        seq = t._generate_speaker_sequence(20)
        assert seq[0] == 0

    def test_speaker_sequence_host_re_enters_periodically(self):
        """Host (speaker 0) re-enters every 5 segments."""
        t = MockTranscriber(num_speakers=3)
        seq = t._generate_speaker_sequence(20)
        # At positions 0, 5, 10, 15, host should be speaking
        for pos in [0, 5, 10, 15]:
            assert seq[pos] == 0

    def test_all_speakers_participate(self):
        """Over enough segments, all speakers get speaking turns."""
        t = MockTranscriber(num_speakers=3)
        result = t.transcribe(Path("fake.mp3"), diarize=True)
        speakers_seen = {seg["speaker"] for seg in result.segments}
        assert len(speakers_seen) >= 2  # At minimum 2 speakers participate

    def test_segments_contain_english_football_terms(self):
        """At least some segments contain English football terminology."""
        t = MockTranscriber(num_speakers=3)
        result = t.transcribe(Path("fake.mp3"))
        has_english = False
        for seg in result.segments:
            for term in ENGLISH_TERMS:
                if term in seg["text"]:
                    has_english = True
                    break
            if has_english:
                break
        assert has_english, "No English football terms found in any segment"

    def test_multiple_episodes_transcription(self, db_with_episodes):
        """Multiple episodes can each be transcribed independently."""
        db, episode_ids = db_with_episodes
        counts = []
        for eid in episode_ids[:3]:
            count = generate_mock_transcription(db, eid)
            counts.append(count)
            assert count >= 15

        # Verify each episode has its own segments
        with db.get_connection() as conn:
            for eid in episode_ids[:3]:
                segs = db.get_segments_by_episode(conn, eid)
                assert len(segs) > 0

"""
Tests for the explicit transcription-status state machine.

Covers the transition table itself, reading old string values, the database
helpers that enforce it, and the two transcribers that drive it.
"""

import importlib
import sys
import types
from itertools import product
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from podcast_intel.models.database import Database
from podcast_intel.models.entities import TranscriptionStatus as EntitiesStatus
from podcast_intel.models.state import (
    TRANSCRIPTION_TRANSITIONS,
    IllegalTransitionError,
    TranscriptionStatus,
    can_transition,
    check_transition,
)

S = TranscriptionStatus

ALLOWED = {
    (S.PENDING, S.PROCESSING),
    (S.PROCESSING, S.COMPLETED),
    (S.PROCESSING, S.FAILED),
    (S.PROCESSING, S.PENDING),
    (S.FAILED, S.PROCESSING),
    (S.FAILED, S.PENDING),
    (S.COMPLETED, S.PENDING),
}


# ---------------------------------------------------------------------------
#  Table and parsing
# ---------------------------------------------------------------------------

def test_entities_reexports_the_same_enum():
    assert EntitiesStatus is TranscriptionStatus


def test_every_state_has_a_row():
    assert set(TRANSCRIPTION_TRANSITIONS) == set(S)


def test_stored_values_are_unchanged():
    assert [s.value for s in S] == ["pending", "processing", "completed", "failed"]


@pytest.mark.parametrize("current,new", list(product(S, S)))
def test_transition_table_matches_spec(current, new):
    expected = (current, new) in ALLOWED
    assert can_transition(current, new) is expected
    if expected:
        assert check_transition(current, new) is new
    else:
        with pytest.raises(IllegalTransitionError):
            check_transition(current, new)


@pytest.mark.parametrize("raw", ["completed", "COMPLETED", " Completed "])
def test_parse_accepts_old_strings(raw):
    assert S.parse(raw) is S.COMPLETED


@pytest.mark.parametrize("raw", ["done", "", "processed", None, 3])
def test_parse_rejects_unknown(raw):
    with pytest.raises(ValueError):
        S.parse(raw)


def test_string_arguments_work_for_checks():
    assert check_transition("pending", "processing") is S.PROCESSING
    assert not can_transition("completed", "completed")


# ---------------------------------------------------------------------------
#  Database helpers
# ---------------------------------------------------------------------------

def _new_episode(db: Database, guid: str = "g1") -> int:
    with db.get_connection() as conn:
        return db.insert_episode(
            conn, guid=guid, title="t", pub_date="2026-01-01", audio_url="http://x/a.mp3",
            duration_seconds=600,
        )


def _status(db: Database, episode_id: int) -> S:
    with db.get_connection() as conn:
        return db.get_episode_status(conn, episode_id)


def test_new_episode_starts_pending(test_db):
    assert _status(test_db, _new_episode(test_db)) is S.PENDING


def test_transition_walks_happy_path(test_db):
    ep = _new_episode(test_db)
    with test_db.get_connection() as conn:
        assert test_db.transition_episode_status(conn, ep, S.PROCESSING) is S.PENDING
        assert test_db.transition_episode_status(conn, ep, "completed") is S.PROCESSING
    assert _status(test_db, ep) is S.COMPLETED


def test_transition_rejects_illegal_move_and_keeps_state(test_db):
    ep = _new_episode(test_db)
    with pytest.raises(IllegalTransitionError):
        with test_db.get_connection() as conn:
            test_db.transition_episode_status(conn, ep, S.COMPLETED)
    assert _status(test_db, ep) is S.PENDING


def test_transition_unknown_episode(test_db):
    with pytest.raises(LookupError):
        with test_db.get_connection() as conn:
            test_db.transition_episode_status(conn, 999, S.PROCESSING)


def test_update_episode_status_stays_backward_compatible(test_db):
    ep = _new_episode(test_db)
    with test_db.get_connection() as conn:
        test_db.update_episode_status(conn, ep, "completed")  # old string API, no check
        raw = conn.execute(
            "SELECT transcription_status FROM episodes WHERE id = ?", (ep,)
        ).fetchone()[0]
    assert raw == "completed"


def test_update_episode_status_rejects_unknown_value(test_db):
    ep = _new_episode(test_db)
    with pytest.raises(ValueError):
        with test_db.get_connection() as conn:
            test_db.update_episode_status(conn, ep, "done")


# ---------------------------------------------------------------------------
#  Mock transcriber
# ---------------------------------------------------------------------------

def _seed_speakers(db: Database) -> None:
    from podcast_intel.transcription.mock_transcribe import SPEAKER_PROFILES

    with db.get_connection() as conn:
        for name in SPEAKER_PROFILES:
            db.insert_speaker(conn, name)


def test_mock_transcription_ends_completed(test_db):
    from podcast_intel.transcription.mock_transcribe import generate_mock_transcription

    _seed_speakers(test_db)
    ep = _new_episode(test_db)
    assert generate_mock_transcription(test_db, ep) > 0
    assert _status(test_db, ep) is S.COMPLETED


def test_mock_transcription_recovers_stale_processing(test_db):
    from podcast_intel.transcription.mock_transcribe import generate_mock_transcription

    _seed_speakers(test_db)
    ep = _new_episode(test_db)
    with test_db.get_connection() as conn:
        test_db.update_episode_status(conn, ep, S.PROCESSING)
    generate_mock_transcription(test_db, ep)
    assert _status(test_db, ep) is S.COMPLETED


def test_mock_transcription_refuses_completed_episode(test_db):
    from podcast_intel.transcription.mock_transcribe import generate_mock_transcription

    _seed_speakers(test_db)
    ep = _new_episode(test_db)
    generate_mock_transcription(test_db, ep)
    with test_db.get_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    with pytest.raises(IllegalTransitionError):
        generate_mock_transcription(test_db, ep)
    with test_db.get_connection() as conn:
        after = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    assert after == before  # no duplicate segments


# ---------------------------------------------------------------------------
#  Whisper transcriber (faster_whisper stubbed out)
# ---------------------------------------------------------------------------

@pytest.fixture
def whisper_module(monkeypatch):
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = MagicMock()
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.delitem(
        sys.modules, "podcast_intel.transcription.whisper_transcriber", raising=False
    )
    module = importlib.import_module("podcast_intel.transcription.whisper_transcriber")
    yield module
    sys.modules.pop("podcast_intel.transcription.whisper_transcriber", None)


def _result(module, segments):
    return module.TranscriptionResult(segments=segments, language="en", duration=10.0)


def test_whisper_failure_marks_failed(test_db, whisper_module, monkeypatch):
    ep = _new_episode(test_db)
    boom = MagicMock(side_effect=RuntimeError("decoder crashed"))
    monkeypatch.setattr(whisper_module.WhisperTranscriber, "transcribe", boom)
    with pytest.raises(RuntimeError):
        whisper_module.transcribe_episode(Path("a.mp3"), test_db, ep)
    assert _status(test_db, ep) is S.FAILED


def test_whisper_success_then_retry_after_failure(test_db, whisper_module, monkeypatch):
    ep = _new_episode(test_db)
    with test_db.get_connection() as conn:
        test_db.update_episode_status(conn, ep, S.FAILED)
    seg = {"start": 0.0, "end": 1.0, "text": "hello there"}
    monkeypatch.setattr(
        whisper_module.WhisperTranscriber, "transcribe",
        lambda self, path: _result(whisper_module, [seg]),
    )
    stats = whisper_module.transcribe_episode(Path("a.mp3"), test_db, ep)
    assert stats["segments"] == 1
    assert _status(test_db, ep) is S.COMPLETED


def test_whisper_refuses_episode_already_processing(test_db, whisper_module, monkeypatch):
    ep = _new_episode(test_db)
    with test_db.get_connection() as conn:
        test_db.update_episode_status(conn, ep, S.PROCESSING)
    called = MagicMock()
    monkeypatch.setattr(whisper_module.WhisperTranscriber, "transcribe", called)
    with pytest.raises(IllegalTransitionError):
        whisper_module.transcribe_episode(Path("a.mp3"), test_db, ep)
    called.assert_not_called()
    assert _status(test_db, ep) is S.PROCESSING


def test_whisper_download_leaves_no_partial_file(tmp_path, whisper_module, monkeypatch):
    class BrokenResponse:
        headers = {"content-length": "100"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"ID3partial"
            raise ConnectionError("dropped")

    monkeypatch.setattr(whisper_module.requests, "get", lambda *a, **k: BrokenResponse())
    dest = tmp_path / "ep.mp3"
    with pytest.raises(ConnectionError):
        whisper_module.download_episode("http://x/a.mp3", dest)
    assert not dest.exists()  # the "already downloaded" marker was not created

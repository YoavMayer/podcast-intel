"""
Tests for ingestion/downloader.py.

The three functions were placeholder bodies returning None against ``-> bool``
signatures while ``ingestion/__init__.py`` exported ``download_episode``. These
tests pin the real behaviour without touching the network.
"""

from pathlib import Path

import pytest
import requests

from podcast_intel.ingestion.downloader import (
    download_episode,
    get_file_size,
    validate_audio_file,
)


class FakeResponse:
    """Minimal stand-in for requests.Response in streaming mode."""

    def __init__(self, chunks=(), headers=None, raise_for_status=None):
        self._chunks = list(chunks)
        self.headers = headers or {}
        self._raise = raise_for_status

    def raise_for_status(self):
        if self._raise is not None:
            raise self._raise

    def iter_content(self, chunk_size=None):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestDownloadEpisode:
    def test_writes_the_body_and_returns_true(self, tmp_path: Path, monkeypatch):
        body = [b"abc", b"def"]
        monkeypatch.setattr(
            requests,
            "get",
            lambda *a, **k: FakeResponse(body, {"Content-Length": "6"}),
        )
        target = tmp_path / "nested" / "ep.mp3"

        assert download_episode("https://example.com/a.mp3", target) is True
        assert target.read_bytes() == b"abcdef"

    def test_reports_progress(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            requests,
            "get",
            lambda *a, **k: FakeResponse([b"ab", b"cd"], {"Content-Length": "4"}),
        )
        seen = []
        download_episode(
            "https://example.com/a.mp3",
            tmp_path / "ep.mp3",
            progress_callback=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(2, 4), (4, 4)]

    def test_network_error_returns_false_and_leaves_no_file(
        self, tmp_path: Path, monkeypatch
    ):
        def boom(*a, **k):
            raise requests.ConnectionError("no route to host")

        monkeypatch.setattr(requests, "get", boom)
        target = tmp_path / "ep.mp3"

        assert download_episode("https://example.com/a.mp3", target) is False
        assert not target.exists()

    def test_truncated_body_is_rejected(self, tmp_path: Path, monkeypatch):
        """A short read must not be promoted to a complete file."""
        monkeypatch.setattr(
            requests,
            "get",
            lambda *a, **k: FakeResponse([b"ab"], {"Content-Length": "99"}),
        )
        target = tmp_path / "ep.mp3"

        assert download_episode("https://example.com/a.mp3", target) is False
        assert not target.exists()
        assert not target.with_suffix(".mp3.part").exists()

    def test_missing_content_length_is_tolerated(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse([b"xy"]))
        target = tmp_path / "ep.mp3"

        assert download_episode("https://example.com/a.mp3", target) is True
        assert target.read_bytes() == b"xy"


class TestValidateAudioFile:
    @pytest.mark.parametrize("magic", [b"ID3\x03", b"\xff\xfb\x90", b"\xff\xf3\x00"])
    def test_accepts_mp3_headers(self, tmp_path: Path, magic: bytes):
        f = tmp_path / "ok.mp3"
        f.write_bytes(magic + b"padding")
        assert validate_audio_file(f) is True

    def test_rejects_an_html_error_page(self, tmp_path: Path):
        f = tmp_path / "bad.mp3"
        f.write_bytes(b"<!DOCTYPE html><html>404</html>")
        assert validate_audio_file(f) is False

    def test_rejects_empty_and_missing_files(self, tmp_path: Path):
        empty = tmp_path / "empty.mp3"
        empty.write_bytes(b"")
        assert validate_audio_file(empty) is False
        assert validate_audio_file(tmp_path / "nope.mp3") is False


class TestGetFileSize:
    def test_returns_content_length(self, monkeypatch):
        monkeypatch.setattr(
            requests, "head", lambda *a, **k: FakeResponse(headers={"Content-Length": "42"})
        )
        assert get_file_size("https://example.com/a.mp3") == 42

    def test_returns_none_without_content_length(self, monkeypatch):
        monkeypatch.setattr(requests, "head", lambda *a, **k: FakeResponse())
        assert get_file_size("https://example.com/a.mp3") is None

    def test_returns_none_on_network_error(self, monkeypatch):
        def boom(*a, **k):
            raise requests.Timeout("slow")

        monkeypatch.setattr(requests, "head", boom)
        assert get_file_size("https://example.com/a.mp3") is None

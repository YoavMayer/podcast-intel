"""
Tests for the football-data.org provider.

Validates the FootballProvider implementation including:
- Match parsing from API responses (FINISHED, SCHEDULED status)
- Score extraction (full-time, half-time, missing)
- Competition filtering
- API key handling (missing key, env var lookup)
- HTTP error handling (timeout, 4xx, 5xx)
- CommunityEvent mapping correctness

Uses unittest.mock to mock requests.get responses.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from podcast_intel.triggers.community_events import CommunityEvent
from podcast_intel.triggers.providers.football import (
    FootballProvider,
    _parse_match,
    _extract_score,
    API_BASE_URL,
)


# ---------------------------------------------------------------------------
#  Helpers -- build realistic API responses
# ---------------------------------------------------------------------------

def _make_match_response(
    match_id: int = 12345,
    home_name: str = "Team Alpha",
    away_name: str = "Team Beta",
    home_score: int = 2,
    away_score: int = 1,
    status: str = "FINISHED",
    competition_name: str = "Premier League",
    competition_id: int = 2021,
    utc_date: str = "2026-02-14T15:00:00Z",
    home_short: str = "ALF",
    away_short: str = "BET",
) -> dict:
    """Build a football-data.org match API response object."""
    return {
        "id": match_id,
        "utcDate": utc_date,
        "status": status,
        "competition": {
            "id": competition_id,
            "name": competition_name,
        },
        "homeTeam": {
            "id": 100,
            "name": home_name,
            "shortName": home_short,
        },
        "awayTeam": {
            "id": 200,
            "name": away_name,
            "shortName": away_short,
        },
        "score": {
            "winner": "HOME_TEAM" if home_score > away_score else (
                "AWAY_TEAM" if away_score > home_score else "DRAW"
            ),
            "duration": "REGULAR",
            "fullTime": {"home": home_score, "away": away_score},
            "halfTime": {"home": 1, "away": 0},
        },
    }


def _make_scheduled_match(
    match_id: int = 99999,
    home_name: str = "Team Alpha",
    away_name: str = "Team Gamma",
    competition_name: str = "Premier League",
    competition_id: int = 2021,
    utc_date: str = "2026-02-21T17:30:00Z",
) -> dict:
    """Build a scheduled match (no score)."""
    return {
        "id": match_id,
        "utcDate": utc_date,
        "status": "SCHEDULED",
        "competition": {
            "id": competition_id,
            "name": competition_name,
        },
        "homeTeam": {
            "id": 100,
            "name": home_name,
            "shortName": "ALF",
        },
        "awayTeam": {
            "id": 300,
            "name": away_name,
            "shortName": "GAM",
        },
        "score": {
            "winner": None,
            "duration": "REGULAR",
            "fullTime": {"home": None, "away": None},
            "halfTime": {"home": None, "away": None},
        },
    }


def _make_api_response(matches: list) -> MagicMock:
    """Build a mock requests.Response with JSON data."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"matches": matches}
    mock_response.raise_for_status.return_value = None
    return mock_response


# ---------------------------------------------------------------------------
#  _parse_match tests
# ---------------------------------------------------------------------------

class TestParseMatch:
    """Tests for the _parse_match helper."""

    def test_parse_finished_match(self):
        """Finished match is parsed with correct fields."""
        match_data = _make_match_response()
        event = _parse_match(match_data)

        assert event is not None
        assert event.event_id == "12345"
        assert event.event_type == "match"
        assert event.status == "FINISHED"
        assert event.teams == ["Team Alpha", "Team Beta"]
        assert event.score == "2-1"
        assert event.competition == "Premier League"

    def test_parse_scheduled_match(self):
        """Scheduled match is parsed as fixture with no score."""
        match_data = _make_scheduled_match()
        event = _parse_match(match_data)

        assert event is not None
        assert event.event_id == "99999"
        assert event.event_type == "fixture"
        assert event.status == "SCHEDULED"
        assert event.score is None
        assert "Team Gamma" in event.teams

    def test_parse_match_preserves_raw_data(self):
        """Parsed event contains original raw_data."""
        match_data = _make_match_response()
        event = _parse_match(match_data)

        assert event.raw_data == match_data
        assert event.raw_data["id"] == 12345

    def test_parse_match_missing_id_returns_none(self):
        """Match without ID returns None."""
        match_data = {"status": "FINISHED", "homeTeam": {}, "awayTeam": {}}
        event = _parse_match(match_data)
        assert event is None

    def test_parse_match_summary_with_score(self):
        """Summary includes score for finished matches."""
        match_data = _make_match_response(
            home_name="Liverpool", away_name="Chelsea",
            home_score=3, away_score=0,
        )
        event = _parse_match(match_data)
        assert "Liverpool" in event.summary
        assert "3-0" in event.summary
        assert "Chelsea" in event.summary

    def test_parse_match_summary_without_score(self):
        """Summary uses 'vs' for scheduled matches."""
        match_data = _make_scheduled_match(
            home_name="Arsenal", away_name="Brighton",
        )
        event = _parse_match(match_data)
        assert "vs" in event.summary
        assert "Arsenal" in event.summary

    def test_parse_live_match(self):
        """In-play match is parsed as live_match type."""
        match_data = _make_match_response(status="IN_PLAY")
        event = _parse_match(match_data)
        assert event.event_type == "live_match"
        assert event.status == "IN_PLAY"


# ---------------------------------------------------------------------------
#  _extract_score tests
# ---------------------------------------------------------------------------

class TestExtractScore:
    """Tests for the _extract_score helper."""

    def test_extract_full_time_score(self):
        """Extracts full-time score correctly."""
        match_data = _make_match_response(home_score=3, away_score=2)
        score = _extract_score(match_data)
        assert score == "3-2"

    def test_extract_score_nil_nil(self):
        """Extracts 0-0 score."""
        match_data = _make_match_response(home_score=0, away_score=0)
        score = _extract_score(match_data)
        assert score == "0-0"

    def test_extract_score_no_fulltime_uses_halftime(self):
        """Falls back to half-time when full-time is missing."""
        match_data = {
            "score": {
                "fullTime": {"home": None, "away": None},
                "halfTime": {"home": 1, "away": 0},
            }
        }
        score = _extract_score(match_data)
        assert score == "1-0 (HT)"

    def test_extract_score_no_score_data(self):
        """Returns None when no score data is present."""
        score = _extract_score({})
        assert score is None

    def test_extract_score_empty_score_dict(self):
        """Returns None for empty score dictionary."""
        score = _extract_score({"score": {}})
        assert score is None


# ---------------------------------------------------------------------------
#  FootballProvider tests
# ---------------------------------------------------------------------------

class TestFootballProvider:
    """Tests for the FootballProvider class."""

    def test_init_reads_config(self):
        """Provider initializes from config dict."""
        config = {
            "team_id": 42,
            "lookback_days": 5,
            "lookahead_days": 14,
            "competition_ids": [2021, 2001],
        }
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key-123"}):
            provider = FootballProvider(config)

        assert provider.team_id == 42
        assert provider.lookback_days == 5
        assert provider.lookahead_days == 14
        assert provider.competition_ids == [2021, 2001]
        assert provider.api_key == "test-key-123"

    def test_init_missing_api_key_warns(self):
        """Provider warns but doesn't fail when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("FOOTBALL_DATA_API_KEY", None)
            provider = FootballProvider({"team_id": 73})

        assert provider.api_key == ""

    def test_init_custom_api_key_env(self):
        """Provider reads API key from custom env variable name."""
        with patch.dict(os.environ, {"MY_CUSTOM_KEY": "custom-secret"}):
            provider = FootballProvider({
                "team_id": 73,
                "api_key_env": "MY_CUSTOM_KEY",
            })
        assert provider.api_key == "custom-secret"

    @patch("podcast_intel.triggers.providers.football.requests.get")
    def test_fetch_recent_events(self, mock_get):
        """fetch_recent_events returns CommunityEvent list from API."""
        mock_get.return_value = _make_api_response([
            _make_match_response(match_id=1, home_score=2, away_score=1),
            _make_match_response(match_id=2, home_score=0, away_score=3),
        ])

        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key"}):
            provider = FootballProvider({"team_id": 73})
            events = provider.fetch_recent_events()

        assert len(events) == 2
        assert all(isinstance(e, CommunityEvent) for e in events)
        assert events[0].event_id == "1"
        assert events[1].score == "0-3"

        # Verify API was called with correct parameters
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert "X-Auth-Token" in call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))

    @patch("podcast_intel.triggers.providers.football.requests.get")
    def test_fetch_upcoming_events(self, mock_get):
        """fetch_upcoming_events returns scheduled fixtures."""
        mock_get.return_value = _make_api_response([
            _make_scheduled_match(match_id=555),
        ])

        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key"}):
            provider = FootballProvider({"team_id": 73})
            events = provider.fetch_upcoming_events()

        assert len(events) == 1
        assert events[0].event_type == "fixture"
        assert events[0].status == "SCHEDULED"

    def test_fetch_without_api_key_returns_empty(self):
        """Fetching without API key returns empty list gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FOOTBALL_DATA_API_KEY", None)
            provider = FootballProvider({"team_id": 73})
            events = provider.fetch_recent_events()

        assert events == []

    @patch("podcast_intel.triggers.providers.football.requests.get")
    def test_competition_filter(self, mock_get):
        """Provider filters events by competition_ids."""
        mock_get.return_value = _make_api_response([
            _make_match_response(match_id=1, competition_id=2021),
            _make_match_response(match_id=2, competition_id=2001),
            _make_match_response(match_id=3, competition_id=9999),
        ])

        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key"}):
            provider = FootballProvider({
                "team_id": 73,
                "competition_ids": [2021],
            })
            events = provider.fetch_recent_events()

        assert len(events) == 1
        assert events[0].event_id == "1"

    @patch("podcast_intel.triggers.providers.football.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        """HTTP errors are handled gracefully with empty result."""
        import requests as req
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.exceptions.HTTPError(
            "403 Forbidden"
        )
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "bad-key"}):
            provider = FootballProvider({"team_id": 73})
            events = provider.fetch_recent_events()

        assert events == []

    @patch("podcast_intel.triggers.providers.football.requests.get")
    def test_timeout_returns_empty(self, mock_get):
        """Timeout errors are handled gracefully."""
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout("Connection timed out")

        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key"}):
            provider = FootballProvider({"team_id": 73})
            events = provider.fetch_recent_events()

        assert events == []

    @patch("podcast_intel.triggers.providers.football.requests.get")
    def test_connection_error_returns_empty(self, mock_get):
        """Connection errors are handled gracefully."""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("DNS failure")

        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key"}):
            provider = FootballProvider({"team_id": 73})
            events = provider.fetch_recent_events()

        assert events == []

    def test_format_event_finished(self):
        """format_event produces readable string for finished match."""
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "key"}):
            provider = FootballProvider({"team_id": 73})

        event = CommunityEvent(
            event_id="1",
            event_type="match",
            status="FINISHED",
            teams=["Liverpool", "Chelsea"],
            score="3-0",
            date="2026-02-14T15:00:00Z",
            competition="Premier League",
        )

        formatted = provider.format_event(event)
        assert "Liverpool" in formatted
        assert "Chelsea" in formatted
        assert "3-0" in formatted
        assert "FINISHED" in formatted

    def test_format_event_scheduled(self):
        """format_event produces readable string for scheduled fixture."""
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "key"}):
            provider = FootballProvider({"team_id": 73})

        event = CommunityEvent(
            event_id="2",
            event_type="fixture",
            status="SCHEDULED",
            teams=["Arsenal", "Brighton"],
            date="2026-02-21T17:30:00Z",
            competition="FA Cup",
        )

        formatted = provider.format_event(event)
        assert "Arsenal" in formatted
        assert "Brighton" in formatted
        assert "SCHEDULED" in formatted

    @patch("podcast_intel.triggers.providers.football.requests.get")
    def test_api_url_includes_team_id(self, mock_get):
        """API request URL includes the configured team_id."""
        mock_get.return_value = _make_api_response([])

        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key"}):
            provider = FootballProvider({"team_id": 73})
            provider.fetch_recent_events()

        call_args = mock_get.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "/teams/73/matches" in url

    @patch("podcast_intel.triggers.providers.football.requests.get")
    def test_api_sends_auth_header(self, mock_get):
        """API request includes X-Auth-Token header."""
        mock_get.return_value = _make_api_response([])

        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "my-secret-key"}):
            provider = FootballProvider({"team_id": 73})
            provider.fetch_recent_events()

        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert headers.get("X-Auth-Token") == "my-secret-key"

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

import os
from unittest.mock import MagicMock, patch

from podcast_intel.triggers.briefing_generator import generate_briefing
from podcast_intel.triggers.community_events import CommunityEvent
from podcast_intel.triggers.providers import get_provider
from podcast_intel.triggers.providers.football import (
    FootballProvider,
    _extract_score,
    _parse_goals,
    _parse_match,
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
        assert event.participants == ["Team Alpha", "Team Beta"]
        assert event.result == "2-1"
        assert event.context == "Premier League"

    def test_parse_scheduled_match(self):
        """Scheduled match is parsed as fixture with no score."""
        match_data = _make_scheduled_match()
        event = _parse_match(match_data)

        assert event is not None
        assert event.event_id == "99999"
        assert event.event_type == "fixture"
        assert event.status == "SCHEDULED"
        assert event.result is None
        assert "Team Gamma" in event.participants

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
#  _parse_goals tests
# ---------------------------------------------------------------------------

class TestParseGoals:
    """Tests for the _parse_goals helper that reads a scoreline back."""

    def test_parses_full_time_scoreline(self):
        """A plain "2-1" splits into two ints."""
        assert _parse_goals("2-1") == (2, 1)

    def test_parses_half_time_scoreline(self):
        """The "(HT)" suffix is stripped before parsing."""
        assert _parse_goals("1-0 (HT)") == (1, 0)

    def test_unparseable_scoreline_is_nil_nil(self):
        """A result string that is not a scoreline degrades to (0, 0)."""
        assert _parse_goals("postponed") == (0, 0)
        assert _parse_goals("") == (0, 0)


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
        assert events[1].result == "0-3"

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
            participants=["Liverpool", "Chelsea"],
            result="3-0",
            date="2026-02-14T15:00:00Z",
            context="Premier League",
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
            participants=["Arsenal", "Brighton"],
            date="2026-02-21T17:30:00Z",
            context="FA Cup",
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


# ---------------------------------------------------------------------------
#  The provider survives the de-sporting of the generic layer
# ---------------------------------------------------------------------------

class TestFootballStaysAWorkingPlugIn:
    """
    Football is one optional provider, not the shape of the framework.

    These tests pin the contract from the other side: after the generic
    ``CommunityEvent`` stopped being a fixture and the match-shaped talking
    points left the briefing generator, the football plug-in still produces
    football output end to end.
    """

    def _provider(self) -> FootballProvider:
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "key"}):
            return FootballProvider({"team_id": 73})

    def test_provider_fills_the_neutral_fields(self):
        """A parsed match populates participants/result/context, not teams/score."""
        event = _parse_match(_make_match_response())
        assert event.participants == ["Team Alpha", "Team Beta"]
        assert event.result == "2-1"
        assert event.context == "Premier League"
        assert "teams" not in event.to_dict()

    def test_talking_points_for_finished_match_are_football_shaped(self):
        """The provider hook still yields match analysis for a finished game."""
        event = _parse_match(_make_match_response(
            home_name="Team Alpha", away_name="Team Beta",
            home_score=3, away_score=1,
        ))
        points = self._provider().talking_points(event)

        assert len(points) == 5
        joined = " ".join(points)
        assert "Team Alpha won 3-1" in joined
        assert "Player ratings" in joined
        assert "formations" in joined

    def test_talking_points_for_draw_and_away_win(self):
        """Both other scorelines get their own framing."""
        provider = self._provider()

        draw = _parse_match(_make_match_response(home_score=1, away_score=1))
        assert "fair result" in " ".join(provider.talking_points(draw))

        away_win = _parse_match(_make_match_response(home_score=0, away_score=2))
        assert "away from home" in " ".join(provider.talking_points(away_win))

    def test_talking_points_for_fixture_are_a_preview(self):
        """A scheduled fixture gets preview prompts, not a post-mortem."""
        event = _parse_match(_make_scheduled_match())
        points = self._provider().talking_points(event)
        joined = " ".join(points)
        assert "Preview" in joined
        assert "Predicted lineups" in joined

    def test_talking_points_respects_max_points(self):
        """The hook honours the caller's budget."""
        event = _parse_match(_make_match_response())
        assert len(self._provider().talking_points(event, max_points=2)) == 2

    def test_talking_points_empty_for_unknown_status(self):
        """A status the provider cannot read falls back to the generic prompts."""
        event = _parse_match(_make_match_response(status="POSTPONED"))
        assert self._provider().talking_points(event) == []

    def test_briefing_with_provider_is_football_shaped(self):
        """generate_briefing + FootballProvider still renders a match briefing."""
        event = _parse_match(_make_match_response(
            home_name="Team Alpha", away_name="Team Beta",
            home_score=2, away_score=1,
        ))
        briefing = generate_briefing(
            event,
            {"podcast": {"name": "Show", "language": "en"}},
            formats=["html", "whatsapp"],
            provider=self._provider(),
        )

        for text in briefing.values():
            assert "Team Alpha" in text
            assert "2-1" in text
            assert "Player ratings" in text

    def test_briefing_without_provider_is_neutral(self):
        """The same event with no provider gets no football vocabulary."""
        event = _parse_match(_make_match_response())
        briefing = generate_briefing(
            event,
            {"podcast": {"name": "Show", "language": "en"}},
            formats=["whatsapp"],
        )
        text = briefing["whatsapp"]
        assert "Team Alpha" in text
        for football_word in ("Player ratings", "formations", "lineups", "standings"):
            assert football_word not in text

    def test_registry_still_resolves_football(self):
        """The provider is still reachable by name from podcast.yaml."""
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "key"}):
            provider = get_provider("football", {"team_id": 73})
        assert isinstance(provider, FootballProvider)

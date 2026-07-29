"""
Football-data.org provider for community events.

Fetches match results and upcoming fixtures from the football-data.org v4 API.
Maps API responses to generic CommunityEvent objects for use by the
briefing generator and CLI.

API Documentation: https://docs.football-data.org/general/v4/

Configuration (via podcast.yaml provider_config):
    api: "football-data.org"
    api_key_env: "FOOTBALL_DATA_API_KEY"
    team_id: 73
    competition_ids: [2021]
    lookback_days: 2
    lookahead_days: 7

Example:
    >>> from podcast_intel.triggers.providers.football import FootballProvider
    >>> provider = FootballProvider({
    ...     "api_key_env": "FOOTBALL_DATA_API_KEY",
    ...     "team_id": 73,
    ... })
    >>> recent = provider.fetch_recent_events()
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from podcast_intel.triggers.community_events import (
    CommunityEvent,
    CommunityEventProvider,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

API_BASE_URL = "https://api.football-data.org/v4"
DEFAULT_LOOKBACK_DAYS = 2
DEFAULT_LOOKAHEAD_DAYS = 7
REQUEST_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
#  Provider implementation
# ---------------------------------------------------------------------------

class FootballProvider(CommunityEventProvider):
    """
    Community event provider backed by the football-data.org v4 API.

    Fetches recent match results and upcoming fixtures for a configured
    team. Normalizes the API responses into CommunityEvent objects.

    The API key is read from an environment variable whose name is
    specified in ``api_key_env`` (default: ``FOOTBALL_DATA_API_KEY``).

    Attributes:
        team_id: Football-data.org team identifier
        api_key: API key for authentication
        competition_ids: Optional list of competition IDs to filter by
        lookback_days: Number of days to look back for recent matches
        lookahead_days: Number of days to look ahead for fixtures
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the football provider from config dict.

        Args:
            config: Provider configuration from podcast.yaml provider_config
        """
        self.team_id: int = config.get("team_id", 73)
        self.competition_ids: list[int] = config.get("competition_ids", [])
        self.lookback_days: int = config.get("lookback_days", DEFAULT_LOOKBACK_DAYS)
        self.lookahead_days: int = config.get("lookahead_days", DEFAULT_LOOKAHEAD_DAYS)

        # Resolve API key from environment variable
        api_key_env = config.get("api_key_env", "FOOTBALL_DATA_API_KEY")
        self.api_key: str = os.environ.get(api_key_env, "")

        if not self.api_key:
            logger.warning(
                "No API key found in environment variable '%s'. "
                "Football-data.org requests will likely fail. "
                "Set %s to your API key.",
                api_key_env,
                api_key_env,
            )

    # -------------------------------------------------------------------
    #  CommunityEventProvider interface
    # -------------------------------------------------------------------

    def fetch_recent_events(self) -> list[CommunityEvent]:
        """
        Fetch recently completed matches for the configured team.

        Queries the football-data.org matches endpoint with a date range
        from (today - lookback_days) to today, filtering for finished
        matches.

        Returns:
            List of CommunityEvent objects for completed matches
        """
        return self._fetch_matches(
            date_from=datetime.now(timezone.utc) - timedelta(days=self.lookback_days),
            date_to=datetime.now(timezone.utc),
            status_filter="FINISHED",
        )

    def fetch_upcoming_events(self) -> list[CommunityEvent]:
        """
        Fetch upcoming scheduled matches for the configured team.

        Queries the football-data.org matches endpoint with a date range
        from today to (today + lookahead_days), filtering for scheduled
        or timed matches.

        Returns:
            List of CommunityEvent objects for upcoming matches
        """
        return self._fetch_matches(
            date_from=datetime.now(timezone.utc),
            date_to=datetime.now(timezone.utc) + timedelta(days=self.lookahead_days),
            status_filter="SCHEDULED",
        )

    def format_event(self, event: CommunityEvent) -> str:
        """
        Format a football event as a human-readable string.

        Args:
            event: The community event to format

        Returns:
            Formatted string like "Home 2-1 Away (Competition, 2026-02-08)"
        """
        teams_str = " vs ".join(event.participants) if event.participants else "Unknown"
        score_str = f" {event.result}" if event.result else ""
        comp_str = f" ({event.context})" if event.context else ""
        date_str = f" [{event.date[:10]}]" if event.date else ""
        status_str = f" [{event.status}]" if event.status else ""

        return f"{teams_str}{score_str}{comp_str}{date_str}{status_str}"

    def talking_points(
        self,
        event: CommunityEvent,
        max_points: int = 5,
    ) -> list[str]:
        """
        Produce football-shaped discussion prompts for a match or fixture.

        These templates used to live in the generic briefing generator, which
        meant every podcast got asked about lineups and formations. They are
        football knowledge, so they live in the football provider: a show that
        does not load this provider never sees them.

        Args:
            event: The match or fixture to generate prompts for
            max_points: Maximum number of prompts to return

        Returns:
            List of talking point strings (empty for events this provider
            has nothing football-specific to say about).
        """
        teams = event.participants
        home = teams[0] if len(teams) > 0 else "Home team"
        away = teams[1] if len(teams) > 1 else "Away team"
        competition = event.context

        points: list[str] = []

        if event.status == "FINISHED" and event.result:
            home_goals, away_goals = _parse_goals(event.result)

            if home_goals > away_goals:
                points.append(f"{home} won {event.result} -- what were the key moments?")
                points.append(f"How did {away} lose control of this match?")
            elif away_goals > home_goals:
                points.append(
                    f"{away} won {event.result} away from home -- analyze the performance."
                )
                points.append(f"Where did {home} go wrong at home?")
            else:
                points.append(f"The match ended {event.result} -- was this a fair result?")
                points.append("Which side should feel more disappointed with the draw?")

            points.append(f"Player ratings: who stood out in {home} vs {away}?")
            points.append(
                f"Tactical analysis: what formations and strategies shaped this "
                f"{competition} match?"
            )
            points.append("What does this result mean for the season standings?")

        elif event.status in ("SCHEDULED", "TIMED"):
            points.append(f"Preview: {home} vs {away} -- what to expect?")
            points.append("Predicted lineups and key matchups to watch.")
            points.append(f"Recent form: how are both sides performing in {competition}?")
            points.append("Key players to watch and potential game-changers.")
            points.append("Score prediction and tactical approach.")

        return points[:max_points]

    # -------------------------------------------------------------------
    #  Internal API methods
    # -------------------------------------------------------------------

    def _fetch_matches(
        self,
        date_from: datetime,
        date_to: datetime,
        status_filter: str = "",
    ) -> list[CommunityEvent]:
        """
        Fetch matches from the football-data.org API.

        Args:
            date_from: Start of date range (inclusive)
            date_to: End of date range (inclusive)
            status_filter: Optional status to filter by (FINISHED, SCHEDULED)

        Returns:
            List of CommunityEvent objects for matching matches
        """
        if not self.api_key:
            logger.warning(
                "Skipping football API call -- no API key configured."
            )
            return []

        url = f"{API_BASE_URL}/teams/{self.team_id}/matches"
        params: dict[str, str] = {
            "dateFrom": date_from.strftime("%Y-%m-%d"),
            "dateTo": date_to.strftime("%Y-%m-%d"),
        }

        if status_filter:
            params["status"] = status_filter

        headers = {
            "X-Auth-Token": self.api_key,
        }

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("Football API request timed out for team %d", self.team_id)
            return []
        except requests.exceptions.HTTPError as exc:
            logger.error(
                "Football API HTTP error for team %d: %s",
                self.team_id,
                exc,
            )
            return []
        except requests.exceptions.RequestException as exc:
            logger.error(
                "Football API request failed for team %d: %s",
                self.team_id,
                exc,
            )
            return []

        data = response.json()
        matches = data.get("matches", [])

        events = []
        for match in matches:
            event = _parse_match(match)
            if event is None:
                continue

            # Filter by competition if configured
            if self.competition_ids:
                competition_id = match.get("competition", {}).get("id")
                if competition_id and competition_id not in self.competition_ids:
                    continue

            events.append(event)

        logger.info(
            "Fetched %d match(es) for team %d (%s to %s, status=%s)",
            len(events),
            self.team_id,
            params["dateFrom"],
            params["dateTo"],
            status_filter or "any",
        )

        return events


# ---------------------------------------------------------------------------
#  Response parsing helpers
# ---------------------------------------------------------------------------

def _parse_match(match_data: dict[str, Any]) -> CommunityEvent | None:
    """
    Parse a single match object from the football-data.org API response.

    Args:
        match_data: Match dictionary from the API ``matches`` array

    Returns:
        CommunityEvent for the match, or None if the data is malformed
    """
    match_id = match_data.get("id")
    if match_id is None:
        return None

    # Teams
    home_team = match_data.get("homeTeam", {})
    away_team = match_data.get("awayTeam", {})
    home_name = home_team.get("name", home_team.get("shortName", "Unknown"))
    away_name = away_team.get("name", away_team.get("shortName", "Unknown"))

    # Status
    status = match_data.get("status", "UNKNOWN")

    # Event type: finished matches are "match", scheduled are "fixture"
    if status == "FINISHED":
        event_type = "match"
    elif status in ("SCHEDULED", "TIMED"):
        event_type = "fixture"
    elif status in ("IN_PLAY", "PAUSED", "HALFTIME"):
        event_type = "live_match"
    else:
        event_type = "match"

    # Score
    score = _extract_score(match_data)

    # Date
    utc_date = match_data.get("utcDate", "")

    # Competition
    competition = match_data.get("competition", {})
    competition_name = competition.get("name", "")

    # Summary
    if score:
        summary = f"{home_name} {score} {away_name}"
    else:
        summary = f"{home_name} vs {away_name}"

    if competition_name:
        summary += f" ({competition_name})"

    return CommunityEvent(
        event_id=str(match_id),
        event_type=event_type,
        status=status,
        participants=[home_name, away_name],
        result=score,
        date=utc_date,
        context=competition_name,
        summary=summary,
        raw_data=match_data,
    )


def _parse_goals(score: str) -> tuple[int, int]:
    """
    Split a "2-1" / "1-0 (HT)" score string into home and away goals.

    Args:
        score: Score string as produced by :func:`_extract_score`

    Returns:
        ``(home_goals, away_goals)``, or ``(0, 0)`` if the string is not
        a parseable scoreline.
    """
    parts = score.replace(" (HT)", "").split("-")
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        return 0, 0


def _extract_score(match_data: dict[str, Any]) -> str | None:
    """
    Extract the match score from the API response.

    Prefers full-time score. Falls back to half-time if full-time
    is not available. Returns None for scheduled matches.

    Args:
        match_data: Match dictionary from the API

    Returns:
        Score string like "2-1", or None if no score is available
    """
    score_data = match_data.get("score", {})
    if not score_data:
        return None

    # Try full-time first
    full_time = score_data.get("fullTime", {})
    if full_time:
        home = full_time.get("home")
        away = full_time.get("away")
        if home is not None and away is not None:
            return f"{home}-{away}"

    # Fall back to half-time
    half_time = score_data.get("halfTime", {})
    if half_time:
        home = half_time.get("home")
        away = half_time.get("away")
        if home is not None and away is not None:
            return f"{home}-{away} (HT)"

    return None


__all__ = [
    "FootballProvider",
]

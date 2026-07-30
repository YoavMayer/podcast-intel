"""
Community events framework for automated podcast content triggers.

Provides a generic, provider-based system for detecting community events
(e.g., sports matches, meetups, releases) that should trigger podcast
content generation such as briefings, social cards, or discussion prompts.

This module is designed to be used in three ways:

1. **Programmatic** -- call ``check_community_events()`` from Python.
2. **CLI** -- invoked via ``podcast-intel events check``.
3. **Scheduled** -- called periodically by an external scheduler (cron,
   systemd timer, GitHub Actions, etc.).

Example:
    >>> from podcast_intel.triggers.community_events import check_community_events
    >>> result = check_community_events(config)
    >>> if result.has_events:
    ...     for event in result.events:
    ...         print(f"{event.event_type}: {event.summary}")
"""

import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data models
# ---------------------------------------------------------------------------

#: Field names that were football-shaped until 0.4.0, mapped to the neutral
#: names that replaced them. The old names still work -- reading, writing and
#: passing them to the constructor -- but each use raises a
#: ``DeprecationWarning``. They are scheduled for removal one release after
#: 0.4.0; nothing inside this package uses them any more.
DEPRECATED_FIELD_ALIASES: dict[str, str] = {
    "teams": "participants",
    "score": "result",
    "competition": "context",
}


def _warn_alias(old: str) -> None:
    """Emit the one-release deprecation warning for a renamed field."""
    warnings.warn(
        f"CommunityEvent.{old} is deprecated; use "
        f"CommunityEvent.{DEPRECATED_FIELD_ALIASES[old]} instead. "
        f"The alias will be removed one release after 0.4.0.",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(init=False)
class CommunityEvent:
    """
    A single community event detected by a provider.

    Represents any event relevant to the podcast's community -- a meetup, a
    product release, a conference talk, a council vote, a match result. The
    field names are deliberately domain-neutral: a provider decides what
    ``participants``, ``result`` and ``context`` mean for its own source.

    Attributes:
        event_id: Unique identifier for this event (provider-specific)
        event_type: Category of event (e.g., "release", "meetup", "match")
        status: Current status (e.g., "FINISHED", "SCHEDULED", "LIVE")
        participants: Names involved -- people, teams, organisations, projects
        result: Outcome string, if the event has one (e.g., "2-1", "passed")
        date: Event date as ISO-8601 string
        context: The series, venue, competition or programme the event sits in
        summary: Human-readable one-line summary
        raw_data: Full raw data from the provider API

    Deprecated aliases (``teams``, ``score``, ``competition``) still read,
    write and construct; see :data:`DEPRECATED_FIELD_ALIASES`.

    Example:
        >>> event = CommunityEvent(
        ...     event_id="42",
        ...     event_type="release",
        ...     status="FINISHED",
        ...     participants=["Project Aurora"],
        ...     result="v2.0 shipped",
        ...     context="Q1 roadmap",
        ... )
        >>> event.participants
        ['Project Aurora']
    """

    event_id: str
    event_type: str
    status: str
    participants: list[str] = field(default_factory=list)
    result: str | None = None
    date: str = ""
    context: str = ""
    summary: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        event_id: str,
        event_type: str,
        status: str,
        participants: list[str] | None = None,
        result: str | None = None,
        date: str = "",
        context: str = "",
        summary: str = "",
        raw_data: dict[str, Any] | None = None,
        *,
        teams: list[str] | None = None,
        score: str | None = None,
        competition: str | None = None,
    ) -> None:
        """
        Build an event, accepting the deprecated field names for one release.

        Args:
            event_id: Provider-specific unique identifier
            event_type: Category of event
            status: Current status string
            participants: Names involved in the event
            result: Outcome string, if any
            date: ISO-8601 event date
            context: Series/venue/competition the event belongs to
            summary: Human-readable one-line summary
            raw_data: Untouched provider payload
            teams: Deprecated alias for ``participants``
            score: Deprecated alias for ``result``
            competition: Deprecated alias for ``context``
        """
        if teams is not None:
            _warn_alias("teams")
            if participants is None:
                participants = teams
        if score is not None:
            _warn_alias("score")
            if result is None:
                result = score
        if competition is not None:
            _warn_alias("competition")
            if not context:
                context = competition

        self.event_id = event_id
        self.event_type = event_type
        self.status = status
        self.participants = [] if participants is None else participants
        self.result = result
        self.date = date
        self.context = context
        self.summary = summary
        self.raw_data = {} if raw_data is None else raw_data

    # -- deprecated aliases, one release only -------------------------------

    @property
    def teams(self) -> list[str]:
        """Deprecated alias for :attr:`participants`."""
        _warn_alias("teams")
        return self.participants

    @teams.setter
    def teams(self, value: list[str]) -> None:
        _warn_alias("teams")
        self.participants = value

    @property
    def score(self) -> str | None:
        """Deprecated alias for :attr:`result`."""
        _warn_alias("score")
        return self.result

    @score.setter
    def score(self, value: str | None) -> None:
        _warn_alias("score")
        self.result = value

    @property
    def competition(self) -> str:
        """Deprecated alias for :attr:`context`."""
        _warn_alias("competition")
        return self.context

    @competition.setter
    def competition(self, value: str) -> None:
        _warn_alias("competition")
        self.context = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class EventCheckResult:
    """
    Result of a community event check operation.

    Attributes:
        has_events: True if at least one event was found
        events: List of detected community events
        provider_name: Name of the provider that ran the check
        checked_at: ISO-8601 timestamp of when the check was performed
        errors: List of error messages encountered during the check
    """

    has_events: bool = False
    events: list[CommunityEvent] = field(default_factory=list)
    provider_name: str = ""
    checked_at: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_events": self.has_events,
            "events": [ev.to_dict() for ev in self.events],
            "provider_name": self.provider_name,
            "checked_at": self.checked_at,
            "errors": self.errors,
            "event_count": len(self.events),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
#  Abstract provider interface
# ---------------------------------------------------------------------------

class CommunityEventProvider(ABC):
    """
    Abstract base class for community event providers.

    Each provider connects to a specific data source (API, scraper, etc.)
    and returns normalized ``CommunityEvent`` objects. Providers are
    registered in the provider registry and selected via podcast.yaml
    configuration.

    Subclasses must implement:
        - ``fetch_recent_events()`` -- events that already happened
        - ``fetch_upcoming_events()`` -- events scheduled in the future
        - ``format_event()`` -- human-readable formatting of an event

    Example:
        >>> class MyProvider(CommunityEventProvider):
        ...     def fetch_recent_events(self) -> List[CommunityEvent]:
        ...         return [...]
    """

    @abstractmethod
    def fetch_recent_events(self) -> list[CommunityEvent]:
        """
        Fetch events that have already occurred within the lookback window.

        Returns:
            List of CommunityEvent objects for recent events
        """

    @abstractmethod
    def fetch_upcoming_events(self) -> list[CommunityEvent]:
        """
        Fetch events scheduled within the lookahead window.

        Returns:
            List of CommunityEvent objects for upcoming events
        """

    @abstractmethod
    def format_event(self, event: CommunityEvent) -> str:
        """
        Format a single event as a human-readable string.

        Args:
            event: The community event to format

        Returns:
            Formatted string representation of the event
        """

    def talking_points(
        self,
        event: CommunityEvent,
        max_points: int = 5,
    ) -> list[str]:
        """
        Optional hook: domain-specific discussion prompts for an event.

        This is where knowledge of what an event *means* belongs. The base
        implementation returns an empty list, which makes the briefing
        generator fall back to its domain-neutral prompts -- so a provider
        only implements this if it has something better to say.

        Args:
            event: The community event to generate prompts for
            max_points: Maximum number of prompts to return

        Returns:
            List of talking point strings, or an empty list to use the
            generic prompts.
        """
        return []


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def check_community_events(
    config: dict[str, Any],
) -> EventCheckResult:
    """
    Check for community events using the configured provider.

    Reads the ``community_events`` section from the trigger configuration,
    instantiates the appropriate provider, and fetches both recent and
    upcoming events.

    Args:
        config: The ``community_events`` section from podcast.yaml triggers

    Returns:
        EventCheckResult with all detected events and any errors

    Example:
        >>> yaml_config = load_podcast_yaml()
        >>> ce_config = yaml_config["triggers"]["community_events"]
        >>> result = check_community_events(ce_config)
        >>> print(f"Found {len(result.events)} events")
    """
    result = EventCheckResult(
        checked_at=datetime.now().isoformat(),
    )

    provider_name = config.get("provider", "")
    if not provider_name:
        result.errors.append(
            "No provider configured for community_events. "
            "Set 'provider' in the community_events trigger config."
        )
        return result

    result.provider_name = provider_name

    # Import provider registry and get provider
    try:
        from podcast_intel.triggers.providers import get_provider

        provider_config = config.get("provider_config", {})
        provider = get_provider(provider_name, provider_config)
    except ValueError as exc:
        result.errors.append(str(exc))
        return result
    except Exception as exc:
        result.errors.append(f"Failed to initialize provider '{provider_name}': {exc}")
        return result

    # Fetch recent events
    try:
        recent = provider.fetch_recent_events()
        result.events.extend(recent)
        logger.info(
            "Provider '%s' returned %d recent event(s)",
            provider_name,
            len(recent),
        )
    except Exception as exc:
        error_msg = f"Error fetching recent events from '{provider_name}': {exc}"
        result.errors.append(error_msg)
        logger.error(error_msg)

    # Fetch upcoming events
    try:
        upcoming = provider.fetch_upcoming_events()
        result.events.extend(upcoming)
        logger.info(
            "Provider '%s' returned %d upcoming event(s)",
            provider_name,
            len(upcoming),
        )
    except Exception as exc:
        error_msg = f"Error fetching upcoming events from '{provider_name}': {exc}"
        result.errors.append(error_msg)
        logger.error(error_msg)

    result.has_events = len(result.events) > 0

    return result


def check_recent_events(
    config: dict[str, Any],
) -> EventCheckResult:
    """
    Check for recent (past) events only.

    A convenience wrapper that only fetches events that have already
    occurred, skipping the upcoming events check.

    Args:
        config: The ``community_events`` section from podcast.yaml triggers

    Returns:
        EventCheckResult with recent events only
    """
    result = EventCheckResult(
        checked_at=datetime.now().isoformat(),
    )

    provider_name = config.get("provider", "")
    if not provider_name:
        result.errors.append("No provider configured for community_events.")
        return result

    result.provider_name = provider_name

    try:
        from podcast_intel.triggers.providers import get_provider

        provider_config = config.get("provider_config", {})
        provider = get_provider(provider_name, provider_config)
        recent = provider.fetch_recent_events()
        result.events = recent
        result.has_events = len(recent) > 0
    except Exception as exc:
        result.errors.append(f"Error: {exc}")

    return result


def check_upcoming_events(
    config: dict[str, Any],
) -> EventCheckResult:
    """
    Check for upcoming (future) events only.

    A convenience wrapper that only fetches scheduled events,
    skipping the recent events check.

    Args:
        config: The ``community_events`` section from podcast.yaml triggers

    Returns:
        EventCheckResult with upcoming events only
    """
    result = EventCheckResult(
        checked_at=datetime.now().isoformat(),
    )

    provider_name = config.get("provider", "")
    if not provider_name:
        result.errors.append("No provider configured for community_events.")
        return result

    result.provider_name = provider_name

    try:
        from podcast_intel.triggers.providers import get_provider

        provider_config = config.get("provider_config", {})
        provider = get_provider(provider_name, provider_config)
        upcoming = provider.fetch_upcoming_events()
        result.events = upcoming
        result.has_events = len(upcoming) > 0
    except Exception as exc:
        result.errors.append(f"Error: {exc}")

    return result


__all__ = [
    "DEPRECATED_FIELD_ALIASES",
    "CommunityEvent",
    "CommunityEventProvider",
    "EventCheckResult",
    "check_community_events",
    "check_recent_events",
    "check_upcoming_events",
]

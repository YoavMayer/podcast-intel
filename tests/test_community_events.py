"""
Tests for the community events trigger framework.

Validates the generic community event system including:
- CommunityEvent dataclass and serialization
- EventCheckResult structure and JSON output
- Provider registry (get_provider, list_providers)
- check_community_events with mock providers
- check_recent_events and check_upcoming_events
- Error handling for missing providers and config
- Briefing generator output formats

Uses unittest.mock to isolate from real API calls.
"""

import json
import warnings
from dataclasses import fields
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from podcast_intel.triggers.community_events import (
    DEPRECATED_FIELD_ALIASES,
    CommunityEvent,
    CommunityEventProvider,
    EventCheckResult,
    check_community_events,
    check_recent_events,
    check_upcoming_events,
)
from podcast_intel.triggers.providers import (
    get_provider,
    list_providers,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_id: str = "12345",
    event_type: str = "panel",
    status: str = "FINISHED",
    participants: list = None,
    result: str = "2-1",
    date: str = "2026-02-14T15:00:00Z",
    context: str = "Season One",
    summary: str = "Group A 2-1 Group B",
) -> CommunityEvent:
    """Create a sample CommunityEvent for testing."""
    return CommunityEvent(
        event_id=event_id,
        event_type=event_type,
        status=status,
        participants=participants or ["Group A", "Group B"],
        result=result,
        date=date,
        context=context,
        summary=summary,
    )


class MockProvider(CommunityEventProvider):
    """Mock provider for testing the framework."""

    def __init__(self, config=None):
        self.config = config or {}
        self.recent_events = []
        self.upcoming_events = []

    def fetch_recent_events(self) -> list[CommunityEvent]:
        return self.recent_events

    def fetch_upcoming_events(self) -> list[CommunityEvent]:
        return self.upcoming_events

    def format_event(self, event: CommunityEvent) -> str:
        return f"[{event.status}] {event.summary}"


# ---------------------------------------------------------------------------
#  CommunityEvent tests
# ---------------------------------------------------------------------------

class TestCommunityEvent:
    """Tests for the CommunityEvent dataclass."""

    def test_create_event_with_defaults(self):
        """CommunityEvent can be created with minimal fields."""
        event = CommunityEvent(
            event_id="1",
            event_type="release",
            status="FINISHED",
        )
        assert event.event_id == "1"
        assert event.participants == []
        assert event.result is None
        assert event.context == ""
        assert event.date == ""
        assert event.raw_data == {}

    def test_create_event_with_all_fields(self):
        """CommunityEvent stores all provided fields."""
        event = _make_event()
        assert event.event_id == "12345"
        assert event.event_type == "panel"
        assert event.status == "FINISHED"
        assert event.participants == ["Group A", "Group B"]
        assert event.result == "2-1"
        assert event.context == "Season One"

    def test_default_mutables_are_not_shared(self):
        """Two default-constructed events do not share their containers."""
        first = CommunityEvent(event_id="1", event_type="meetup", status="SCHEDULED")
        second = CommunityEvent(event_id="2", event_type="meetup", status="SCHEDULED")
        first.participants.append("Someone")
        first.raw_data["k"] = "v"
        assert second.participants == []
        assert second.raw_data == {}

    def test_event_to_dict(self):
        """to_dict produces a serializable dictionary with the neutral names."""
        event = _make_event()
        d = event.to_dict()
        assert isinstance(d, dict)
        assert d["event_id"] == "12345"
        assert d["participants"] == ["Group A", "Group B"]
        assert d["result"] == "2-1"
        assert d["context"] == "Season One"
        # The football-shaped names are gone from the wire format.
        assert "teams" not in d
        assert "score" not in d
        assert "competition" not in d

    def test_event_to_json(self):
        """to_json produces valid JSON string."""
        event = _make_event()
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "12345"
        assert parsed["status"] == "FINISHED"

    def test_event_to_json_unicode(self):
        """to_json handles non-ASCII characters correctly."""
        event = _make_event(
            participants=["קבוצה א", "Équipe B"],
            summary="קבוצה א 3-0 Équipe B",
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert "Équipe B" in parsed["participants"]

    def test_event_is_not_shaped_like_a_fixture(self):
        """No football-specific field survives on the generic dataclass."""
        field_names = {f.name for f in fields(CommunityEvent)}
        assert field_names == {
            "event_id", "event_type", "status", "participants", "result",
            "date", "context", "summary", "raw_data",
        }


# ---------------------------------------------------------------------------
#  Deprecated alias tests -- remove together with the aliases
# ---------------------------------------------------------------------------

class TestDeprecatedFieldAliases:
    """The old football-shaped names keep working for one release."""

    def test_alias_map_is_declared(self):
        """The alias map documents exactly the three renamed fields."""
        assert DEPRECATED_FIELD_ALIASES == {
            "teams": "participants",
            "score": "result",
            "competition": "context",
        }

    @pytest.mark.parametrize(
        ("old", "new", "value"),
        [
            ("teams", "participants", ["A", "B"]),
            ("score", "result", "2-1"),
            ("competition", "context", "Season One"),
        ],
    )
    def test_old_constructor_kwarg_fills_new_field(self, old, new, value):
        """Constructing with an old name populates the new field and warns."""
        with pytest.warns(DeprecationWarning, match=old):
            event = CommunityEvent(
                event_id="1", event_type="match", status="FINISHED",
                **{old: value},
            )
        assert getattr(event, new) == value

    @pytest.mark.parametrize(
        ("old", "new", "value"),
        [
            ("teams", "participants", ["A", "B"]),
            ("score", "result", "2-1"),
            ("competition", "context", "Season One"),
        ],
    )
    def test_old_attribute_reads_new_field(self, old, new, value):
        """Reading an old name returns the new field's value and warns."""
        event = CommunityEvent(
            event_id="1", event_type="match", status="FINISHED",
            **{new: value},
        )
        with pytest.warns(DeprecationWarning, match=old):
            assert getattr(event, old) == value

    @pytest.mark.parametrize(
        ("old", "new", "value"),
        [
            ("teams", "participants", ["C"]),
            ("score", "result", "0-0"),
            ("competition", "context", "Cup"),
        ],
    )
    def test_old_attribute_writes_new_field(self, old, new, value):
        """Assigning through an old name updates the new field and warns."""
        event = CommunityEvent(event_id="1", event_type="match", status="FINISHED")
        with pytest.warns(DeprecationWarning, match=old):
            setattr(event, old, value)
        assert getattr(event, new) == value

    def test_new_name_wins_when_both_are_given(self):
        """An explicit new name is not overwritten by its deprecated alias."""
        with pytest.warns(DeprecationWarning):
            event = CommunityEvent(
                event_id="1", event_type="match", status="FINISHED",
                participants=["New"], teams=["Old"],
            )
        assert event.participants == ["New"]

    def test_new_names_do_not_warn(self):
        """The supported names are warning-free."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            event = CommunityEvent(
                event_id="1", event_type="match", status="FINISHED",
                participants=["A", "B"], result="1-0", context="Cup",
            )
            assert event.participants == ["A", "B"]
            assert event.result == "1-0"
            assert event.context == "Cup"


# ---------------------------------------------------------------------------
#  EventCheckResult tests
# ---------------------------------------------------------------------------

class TestEventCheckResult:
    """Tests for the EventCheckResult dataclass."""

    def test_empty_result(self):
        """Empty result has correct defaults."""
        result = EventCheckResult()
        assert result.has_events is False
        assert result.events == []
        assert result.provider_name == ""
        assert result.errors == []

    def test_result_with_events(self):
        """Result correctly reports events."""
        events = [_make_event(), _make_event(event_id="67890")]
        result = EventCheckResult(
            has_events=True,
            events=events,
            provider_name="football",
            checked_at="2026-02-14T10:00:00",
        )
        assert result.has_events is True
        assert len(result.events) == 2
        assert result.provider_name == "football"

    def test_result_to_dict(self):
        """to_dict includes event_count."""
        events = [_make_event()]
        result = EventCheckResult(
            has_events=True,
            events=events,
            provider_name="football",
        )
        d = result.to_dict()
        assert d["event_count"] == 1
        assert d["has_events"] is True
        assert len(d["events"]) == 1

    def test_result_to_json(self):
        """to_json produces valid JSON."""
        result = EventCheckResult(
            has_events=False,
            provider_name="football",
            checked_at="2026-02-14T10:00:00",
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["provider_name"] == "football"
        assert parsed["event_count"] == 0

    def test_result_with_errors(self):
        """Result can accumulate errors."""
        result = EventCheckResult(
            errors=["API timeout", "Rate limited"],
        )
        assert len(result.errors) == 2
        d = result.to_dict()
        assert "API timeout" in d["errors"]


# ---------------------------------------------------------------------------
#  Provider registry tests
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    """Tests for the provider registry."""

    def test_list_providers_returns_dict(self):
        """list_providers returns a dictionary of registered providers."""
        providers = list_providers()
        assert isinstance(providers, dict)
        assert "football" in providers

    def test_get_provider_football(self):
        """get_provider returns a FootballProvider for 'football'."""
        with patch.dict("os.environ", {"FOOTBALL_DATA_API_KEY": "test-key"}):
            provider = get_provider("football", {"team_id": 73})
        assert provider is not None
        assert isinstance(provider, CommunityEventProvider)

    def test_get_provider_unknown_raises(self):
        """get_provider raises ValueError for unknown provider names."""
        with pytest.raises(ValueError, match="Unknown community event provider"):
            get_provider("basketball", {})

    def test_get_provider_error_message_lists_available(self):
        """Error message from get_provider lists available providers."""
        with pytest.raises(ValueError, match="football"):
            get_provider("invalid_provider", {})


# ---------------------------------------------------------------------------
#  check_community_events tests
# ---------------------------------------------------------------------------

class TestCheckCommunityEvents:
    """Tests for the check_community_events function."""

    def test_missing_provider_returns_error(self):
        """check_community_events returns error when no provider is configured."""
        result = check_community_events({})
        assert result.has_events is False
        assert len(result.errors) > 0
        assert "No provider" in result.errors[0]

    def test_unknown_provider_returns_error(self):
        """check_community_events handles unknown provider gracefully."""
        result = check_community_events({"provider": "nonexistent"})
        assert result.has_events is False
        assert len(result.errors) > 0

    @patch("podcast_intel.triggers.providers.get_provider")
    def test_check_with_mock_provider(self, mock_get_provider):
        """check_community_events uses provider to fetch events."""
        mock_provider = MockProvider()
        mock_provider.recent_events = [_make_event()]
        mock_provider.upcoming_events = [
            _make_event(event_id="99", status="SCHEDULED", event_type="fixture")
        ]
        mock_get_provider.return_value = mock_provider

        result = check_community_events({
            "provider": "football",
            "provider_config": {"team_id": 73},
        })

        assert result.has_events is True
        assert len(result.events) == 2
        assert result.provider_name == "football"

    def test_check_sets_checked_at_timestamp(self):
        """check_community_events sets checked_at to current time."""
        result = check_community_events({"provider": "nonexistent"})
        assert result.checked_at != ""
        # Verify it's a valid ISO format
        datetime.fromisoformat(result.checked_at)

    @patch("podcast_intel.triggers.providers.get_provider")
    def test_check_handles_provider_exception(self, mock_get):
        """check_community_events handles exceptions from providers."""
        mock_prov = MockProvider()
        mock_prov.fetch_recent_events = MagicMock(
            side_effect=RuntimeError("API down")
        )
        mock_prov.fetch_upcoming_events = MagicMock(return_value=[])
        mock_get.return_value = mock_prov

        result = check_community_events({
            "provider": "football",
            "provider_config": {},
        })

        assert len(result.errors) > 0
        assert "API down" in result.errors[0]


# ---------------------------------------------------------------------------
#  check_recent_events / check_upcoming_events tests
# ---------------------------------------------------------------------------

class TestCheckRecentAndUpcoming:
    """Tests for convenience functions."""

    @patch("podcast_intel.triggers.providers.get_provider")
    def test_check_recent_only(self, mock_get):
        """check_recent_events only calls fetch_recent_events."""
        mock_prov = MockProvider()
        mock_prov.recent_events = [_make_event()]
        mock_get.return_value = mock_prov

        result = check_recent_events({
            "provider": "football",
            "provider_config": {},
        })

        assert result.has_events is True
        assert len(result.events) == 1
        assert result.events[0].status == "FINISHED"

    @patch("podcast_intel.triggers.providers.get_provider")
    def test_check_upcoming_only(self, mock_get):
        """check_upcoming_events only calls fetch_upcoming_events."""
        mock_prov = MockProvider()
        mock_prov.upcoming_events = [
            _make_event(status="SCHEDULED", event_type="fixture")
        ]
        mock_get.return_value = mock_prov

        result = check_upcoming_events({
            "provider": "football",
            "provider_config": {},
        })

        assert result.has_events is True
        assert len(result.events) == 1
        assert result.events[0].status == "SCHEDULED"

    def test_recent_missing_provider(self):
        """check_recent_events returns error for missing provider."""
        result = check_recent_events({})
        assert result.has_events is False
        assert len(result.errors) > 0

    def test_upcoming_missing_provider(self):
        """check_upcoming_events returns error for missing provider."""
        result = check_upcoming_events({})
        assert result.has_events is False
        assert len(result.errors) > 0

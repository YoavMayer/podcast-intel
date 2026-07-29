"""
Tests for the domain-neutral briefing generator.

The briefing generator used to read ``teams``/``score``/``competition`` and
emit match-shaped talking points for every podcast. These tests pin the
generalised behaviour:

- ``headline_slots`` lays an event out from participant count alone
- the built-in talking points carry no sport vocabulary
- domain-specific prompts arrive only through the provider hook
- a provider that raises cannot break a briefing
- saved filenames are built from participants, not from "home vs away"

Rendering assertions stay on the plain-text formats where possible; the HTML
checks look for the generic class names that replaced the fixture-shaped ones.
"""

from podcast_intel.triggers.briefing_generator import (
    NEUTRAL_SEPARATOR,
    _neutral_talking_points,
    generate_briefing,
    headline_slots,
)
from podcast_intel.triggers.community_events import (
    CommunityEvent,
    CommunityEventProvider,
)

#: Words that only make sense if the framework assumes football.
FOOTBALL_VOCABULARY = (
    "Player ratings",
    "formations",
    "lineups",
    "standings",
    "match",
    "draw",
    "away from home",
    "both sides",
)

EN_CONFIG = {"podcast": {"name": "The Show", "name_en": "The Show", "language": "en"}}


def _event(**overrides) -> CommunityEvent:
    """Build a neutral community event, overriding whatever the test needs."""
    fields = {
        "event_id": "e1",
        "event_type": "release",
        "status": "FINISHED",
        "participants": ["Project Aurora"],
        "result": "v2.0 shipped",
        "date": "2026-03-04T09:00:00Z",
        "context": "Q1 roadmap",
        "summary": "Project Aurora shipped v2.0",
    }
    fields.update(overrides)
    return CommunityEvent(**fields)


# ---------------------------------------------------------------------------
#  headline_slots
# ---------------------------------------------------------------------------

class TestHeadlineSlots:
    """The header layout depends on participant count, nothing else."""

    def test_no_participants_uses_the_summary(self):
        """An event with no named participants still gets a headline."""
        event = _event(participants=[], summary="Annual meetup announced")
        assert headline_slots(event) == ("Annual meetup announced", "v2.0 shipped", "")

    def test_no_participants_and_no_summary_falls_back_to_type(self):
        """The event type is the last resort, never an invented team name."""
        event = _event(participants=[], summary="", result=None)
        assert headline_slots(event) == ("release", "", "")

    def test_one_participant_leaves_the_trail_empty(self):
        """A single-subject event does not invent an opponent."""
        lead, divider, trail = headline_slots(_event())
        assert (lead, divider, trail) == ("Project Aurora", "v2.0 shipped", "")

    def test_two_participants_sit_either_side_of_the_result(self):
        """Two participants keep the familiar three-slot layout."""
        event = _event(participants=["Group A", "Group B"], result="2-1")
        assert headline_slots(event) == ("Group A", "2-1", "Group B")

    def test_two_participants_without_a_result_get_a_neutral_separator(self):
        """No result means a neutral dot -- not "vs", which implies a contest."""
        event = _event(participants=["Group A", "Group B"], result=None)
        assert headline_slots(event) == ("Group A", NEUTRAL_SEPARATOR, "Group B")

    def test_many_participants_are_listed_in_the_lead(self):
        """A panel of four is a list, not a fixture."""
        event = _event(participants=["A", "B", "C", "D"], result="agreed")
        assert headline_slots(event) == ("A, B, C, D", "agreed", "")

    def test_blank_participant_names_are_ignored(self):
        """Empty strings from a sloppy provider do not open a slot."""
        event = _event(participants=["", "Real Name", ""])
        lead, _, trail = headline_slots(event)
        assert lead == "Real Name"
        assert trail == ""


# ---------------------------------------------------------------------------
#  Neutral talking points
# ---------------------------------------------------------------------------

class TestNeutralTalkingPoints:
    """The built-in prompts must read sensibly for any kind of event."""

    def test_points_are_generated_for_a_non_sport_event(self):
        """A software release gets usable prompts."""
        points = _neutral_talking_points(_event())
        assert 1 <= len(points) <= 5
        assert all(isinstance(p, str) and p for p in points)

    def test_points_carry_no_football_vocabulary(self):
        """Nothing in the neutral prompts assumes a match."""
        joined = " ".join(_neutral_talking_points(_event()))
        for word in FOOTBALL_VOCABULARY:
            assert word not in joined

    def test_points_name_the_participants(self):
        """Participants are addressed by name, not as "home"/"away"."""
        event = _event(participants=["Group A", "Group B"], summary="")
        joined = " ".join(_neutral_talking_points(event))
        assert "Group A" in joined
        assert "Group B" in joined
        assert "Home team" not in joined
        assert "Away team" not in joined

    def test_scheduled_event_asks_forward_looking_questions(self):
        """An event that has not happened yet is framed as a preview."""
        event = _event(status="SCHEDULED", result=None, summary="")
        joined = " ".join(_neutral_talking_points(event))
        assert "expect" in joined

    def test_result_is_quoted_verbatim(self):
        """A non-numeric result is used as-is, never parsed as a scoreline."""
        event = _event(result="motion carried", summary="")
        assert any("motion carried" in p for p in _neutral_talking_points(event))

    def test_empty_event_still_produces_points(self):
        """A bare event never yields an empty talking-points list."""
        bare = CommunityEvent(event_id="x", event_type="update", status="UNKNOWN")
        assert _neutral_talking_points(bare)

    def test_max_points_is_respected(self):
        """The caller's budget caps the output."""
        assert len(_neutral_talking_points(_event(), max_points=2)) == 2


# ---------------------------------------------------------------------------
#  Provider hook
# ---------------------------------------------------------------------------

class _StubProvider(CommunityEventProvider):
    """Minimal provider used to exercise the talking-points hook."""

    def __init__(self, points=None, raises=False):
        self._points = points or []
        self._raises = raises

    def fetch_recent_events(self):
        return []

    def fetch_upcoming_events(self):
        return []

    def format_event(self, event):
        return event.summary

    def talking_points(self, event, max_points=5):
        if self._raises:
            raise RuntimeError("provider exploded")
        return self._points[:max_points]


class TestProviderTalkingPointsHook:
    """Domain knowledge reaches the briefing only through the provider."""

    def test_base_provider_supplies_nothing(self):
        """The default hook returns an empty list, so the neutral prompts win."""
        assert _StubProvider().talking_points(_event()) == []

    def test_provider_points_replace_the_neutral_ones(self):
        """A provider with something to say owns the talking-points section."""
        provider = _StubProvider(points=["Domain point one", "Domain point two"])
        text = generate_briefing(
            _event(), EN_CONFIG, formats=["whatsapp"], provider=provider,
        )["whatsapp"]
        assert "Domain point one" in text
        assert "Recap:" not in text

    def test_no_provider_means_neutral_points(self):
        """Without a provider the briefing uses its own prompts."""
        text = generate_briefing(_event(), EN_CONFIG, formats=["whatsapp"])["whatsapp"]
        assert "Recap:" in text

    def test_provider_returning_nothing_falls_back(self):
        """An empty hook result is a fallback signal, not an empty section."""
        text = generate_briefing(
            _event(), EN_CONFIG, formats=["whatsapp"], provider=_StubProvider(),
        )["whatsapp"]
        assert "Recap:" in text

    def test_provider_exception_does_not_break_the_briefing(self):
        """A broken provider degrades to the neutral prompts."""
        text = generate_briefing(
            _event(),
            EN_CONFIG,
            formats=["whatsapp"],
            provider=_StubProvider(raises=True),
        )["whatsapp"]
        assert "Recap:" in text

    def test_provider_points_are_capped_for_the_social_card(self):
        """The social card asks for 3 points and gets no more."""
        provider = _StubProvider(points=[f"Point {i}" for i in range(1, 9)])
        card = generate_briefing(
            _event(), EN_CONFIG, formats=["social_card"], provider=provider,
        )["social_card"]
        assert "Point 3" in card
        assert "Point 4" not in card


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------

class TestNeutralRendering:
    """The rendered output no longer describes every event as a fixture."""

    def test_html_uses_the_generic_class_names(self):
        """The fixture-shaped CSS hooks are gone."""
        html = generate_briefing(_event(), EN_CONFIG, formats=["html"])["html"]
        assert 'class="headline"' in html
        assert 'class="participant lead"' in html
        assert 'class="result"' in html
        assert "score-display" not in html
        assert "team away" not in html

    def test_html_labels_the_context_field(self):
        """The details section says Context, not Competition."""
        html = generate_briefing(_event(), EN_CONFIG, formats=["html"])["html"]
        assert "<strong>Context:</strong> Q1 roadmap" in html
        assert "Competition:" not in html

    def test_social_card_uses_the_generic_class_names(self):
        """The card markup is generic too."""
        card = generate_briefing(_event(), EN_CONFIG, formats=["social_card"])["social_card"]
        assert 'class="participants-row"' in card
        assert 'class="participant-name lead"' in card
        assert 'class="result-box"' in card
        assert "teams-row" not in card

    def test_single_participant_renders_without_an_invented_opponent(self):
        """A one-subject event must not print "Team B"."""
        text = generate_briefing(_event(), EN_CONFIG, formats=["whatsapp"])["whatsapp"]
        assert "Project Aurora" in text
        assert "Team A" not in text
        assert "Team B" not in text

    def test_whatsapp_headline_has_no_dangling_separators(self):
        """Empty slots collapse instead of leaving stray spaces."""
        text = generate_briefing(_event(), EN_CONFIG, formats=["whatsapp"])["whatsapp"]
        assert "*Project Aurora v2.0 shipped*" in text

    def test_all_three_formats_render_together(self):
        """The multi-format path still returns every requested format."""
        out = generate_briefing(
            _event(), EN_CONFIG, formats=["html", "whatsapp", "social_card"],
        )
        assert set(out) == {"html", "whatsapp", "social_card"}
        assert all(out.values())


# ---------------------------------------------------------------------------
#  File output
# ---------------------------------------------------------------------------

class TestBriefingFilenames:
    """Saved filenames are built from participants, not from a fixture."""

    def test_filename_uses_participant_slugs(self, tmp_path):
        """Two participants produce one slug each, joined by an underscore."""
        event = _event(participants=["Group A", "Group B"])
        paths = generate_briefing(
            event, EN_CONFIG, formats=["html"], output_dir=str(tmp_path),
        )
        assert "20260304_group_a_group_b_html.html" in paths["html"]
        assert "_vs_" not in paths["html"]

    def test_filename_falls_back_to_event(self, tmp_path):
        """No participants means a neutral "event" prefix."""
        event = _event(participants=[])
        paths = generate_briefing(
            event, EN_CONFIG, formats=["html"], output_dir=str(tmp_path),
        )
        assert "20260304_event_html.html" in paths["html"]

    def test_non_latin_participants_do_not_break_the_filename(self, tmp_path):
        """Hebrew participant names slug to nothing, so the fallback applies."""
        event = _event(participants=["קבוצה א"])
        paths = generate_briefing(
            event, EN_CONFIG, formats=["html"], output_dir=str(tmp_path),
        )
        assert "20260304_event_html.html" in paths["html"]

    def test_manifest_stores_the_neutral_field_names(self, tmp_path):
        """The written manifest carries participants/result/context."""
        import json

        paths = generate_briefing(
            _event(), EN_CONFIG, formats=["html"], output_dir=str(tmp_path),
        )
        manifest = json.loads(
            (tmp_path / paths["manifest"].split("/")[-1]).read_text(encoding="utf-8")
        )
        assert manifest["event"]["participants"] == ["Project Aurora"]
        assert manifest["event"]["result"] == "v2.0 shipped"
        assert "teams" not in manifest["event"]

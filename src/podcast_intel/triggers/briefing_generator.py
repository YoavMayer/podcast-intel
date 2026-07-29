"""
Generic briefing generator for community events.

Generates multi-format briefing content (HTML, WhatsApp, social card)
from community events, using podcast.yaml branding configuration for
styling.

The renderer is domain-neutral. It reads only the generic ``participants``,
``result`` and ``context`` fields of ``CommunityEvent``, and its own talking
points are neutral prompts. Domain-shaped prompts (a football preview, a
release checklist) come from the provider's optional
``CommunityEventProvider.talking_points()`` hook -- pass the provider to
``generate_briefing()`` to use them.

Defaults are Hebrew-first, matching ``presets.DEFAULT_LANGUAGE``: an
unconfigured briefing renders ``dir="rtl"`` with a Heebo-first font stack.
``direction`` follows ``podcast.language`` unless podcast.yaml states it
explicitly, so an English show only has to set the language.

Branding and voice come entirely from podcast.yaml configuration:
    podcast:
      name: "My Podcast"
      name_en: "My Podcast"
      link: "https://example.com"
      direction: "rtl"
      language: "he"
      branding:
        primary_color: "#1a1a2e"
        secondary_color: "#FFFFFF"
        accent_color: "#4A90D9"
        highlight_color: "#C4A747"
        font_family: "Heebo"

Example:
    >>> from podcast_intel.triggers.briefing_generator import generate_briefing
    >>> from podcast_intel.triggers.community_events import CommunityEvent
    >>> event = CommunityEvent(
    ...     event_id="12345",
    ...     event_type="panel",
    ...     status="FINISHED",
    ...     participants=["Group A", "Group B"],
    ...     result="agreed",
    ...     context="Annual review",
    ... )
    >>> result = generate_briefing(event, config)
    >>> print(result["html"])
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from podcast_intel.presets import DEFAULT_LANGUAGE
from podcast_intel.triggers.community_events import (
    CommunityEvent,
    CommunityEventProvider,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Default branding (overridden by podcast.yaml)
# ---------------------------------------------------------------------------

_DEFAULT_BRANDING = {
    "primary_color": "#1a1a2e",
    "secondary_color": "#ffffff",
    "accent_color": "#4a90d9",
    "highlight_color": "#c4a747",
    # Heebo covers Hebrew and Latin; the generic families are the fallback chain
    # for anyone who has not installed it.
    "font_family": "Heebo, system-ui, -apple-system, sans-serif",
}

# Hebrew-first, matching presets.DEFAULT_LANGUAGE: an unconfigured briefing
# renders RTL Hebrew markup. An English show sets `direction: "ltr"` and
# `language: "en"` under `podcast:` in podcast.yaml.
_DEFAULT_PODCAST = {
    "name": "Podcast",
    "name_en": "Podcast",
    "link": "",
    "direction": "rtl",
    "language": DEFAULT_LANGUAGE,
}


# ---------------------------------------------------------------------------
#  Configuration helpers
# ---------------------------------------------------------------------------

def _get_branding(config: dict[str, Any]) -> dict[str, str]:
    """
    Extract branding configuration, falling back to defaults.

    Args:
        config: Full podcast.yaml configuration dictionary

    Returns:
        Branding dictionary with color and font values
    """
    branding = dict(_DEFAULT_BRANDING)
    yaml_branding = config.get("podcast", {}).get("branding", {})
    if isinstance(yaml_branding, dict):
        branding.update(yaml_branding)
    return branding


#: Languages written right-to-left, by ISO 639-1 code.
_RTL_LANGUAGES = frozenset({"he", "ar", "fa", "ur", "yi", "ps", "dv"})


def text_direction(language: str) -> str:
    """Return ``"rtl"`` or ``"ltr"`` for an ISO 639-1 language code.

    Args:
        language: Language code, e.g. ``"he"``. A region suffix is ignored.

    Returns:
        ``"rtl"`` for a right-to-left language, ``"ltr"`` otherwise.

    Example:
        >>> text_direction("he"), text_direction("en")
        ('rtl', 'ltr')
    """
    code = str(language or "").strip().lower().replace("_", "-").split("-")[0]
    return "rtl" if code in _RTL_LANGUAGES else "ltr"


def _get_podcast_info(config: dict[str, Any]) -> dict[str, str]:
    """
    Extract podcast identity from configuration.

    ``direction`` FOLLOWS ``language`` unless podcast.yaml states it explicitly.
    Requiring a separate key would mean an English show that set only
    ``podcast.language: "en"`` still rendered RTL markup -- the acceptance
    contract is four keys, and ``direction`` is not one of them.

    Args:
        config: Full podcast.yaml configuration dictionary

    Returns:
        Dictionary with name, name_en, link, direction, language
    """
    info = dict(_DEFAULT_PODCAST)
    podcast_section = config.get("podcast", {})
    if isinstance(podcast_section, dict):
        for key in _DEFAULT_PODCAST:
            if key in podcast_section:
                info[key] = podcast_section[key]
        if "direction" not in podcast_section:
            info["direction"] = text_direction(info["language"])
    return info


# ---------------------------------------------------------------------------
#  Event shape helpers
# ---------------------------------------------------------------------------

#: Separator between two participants when the event has no result yet.
#: Deliberately not "vs" -- a release or a meetup is not a contest.
NEUTRAL_SEPARATOR = "·"


def headline_slots(event: CommunityEvent) -> tuple[str, str, str]:
    """
    Split an event into the three header slots the briefing layouts use.

    The layouts are ``lead | divider | trail``. What fills them depends only
    on how many participants the provider supplied, never on what kind of
    event it is:

    * no participants -- the summary carries the header on its own
    * one participant -- name in the lead, result as the divider
    * two participants -- one each side, result (or a neutral dot) between
    * three or more -- all names in the lead, result as the divider

    Args:
        event: The community event to lay out

    Returns:
        ``(lead, divider, trail)`` -- any slot may be an empty string.

    Example:
        >>> ev = CommunityEvent("1", "vote", "FINISHED",
        ...                     participants=["Motion 4"], result="passed")
        >>> headline_slots(ev)
        ('Motion 4', 'passed', '')
    """
    participants = [p for p in event.participants if p]
    result = event.result or ""

    if not participants:
        return (event.summary or event.event_type, result, "")
    if len(participants) == 1:
        return (participants[0], result, "")
    if len(participants) == 2:
        return (participants[0], result or NEUTRAL_SEPARATOR, participants[1])
    return (", ".join(participants), result, "")


def _slugify(text: str) -> str:
    """
    Reduce a participant name to a filename-safe token.

    Args:
        text: Arbitrary participant or event name

    Returns:
        Lowercase token with runs of non-alphanumerics collapsed to ``_``,
        or ``""`` if nothing survives.
    """
    return re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


# ---------------------------------------------------------------------------
#  Briefing generation
# ---------------------------------------------------------------------------

def generate_briefing(
    event: CommunityEvent,
    config: dict[str, Any],
    formats: list[str] | None = None,
    output_dir: str | None = None,
    provider: CommunityEventProvider | None = None,
) -> dict[str, str]:
    """
    Generate briefing content for a community event.

    Creates content in the requested formats (html, whatsapp, social_card)
    and optionally saves files to the output directory.

    Args:
        event: The community event to generate a briefing for
        config: Full podcast.yaml configuration dictionary
        formats: List of output formats (default: ["html"])
        output_dir: Directory to save files to (optional)
        provider: Optional provider whose ``talking_points()`` hook supplies
            domain-specific prompts. Without it, the neutral prompts are used.

    Returns:
        Dictionary mapping format name to content string (or file path
        if output_dir is specified)

    Example:
        >>> result = generate_briefing(event, yaml_config, formats=["html", "whatsapp"])
        >>> print(result["html"][:100])
    """
    if formats is None:
        formats = ["html"]

    branding = _get_branding(config)
    podcast = _get_podcast_info(config)

    results: dict[str, str] = {}

    if "html" in formats:
        results["html"] = _generate_html_briefing(event, branding, podcast, provider)

    if "whatsapp" in formats:
        results["whatsapp"] = _generate_whatsapp_briefing(event, podcast, provider)

    if "social_card" in formats:
        results["social_card"] = _generate_social_card(event, branding, podcast, provider)

    # Save to files if output_dir is specified
    if output_dir:
        saved = _save_briefing_files(event, results, output_dir)
        # Return file paths instead of content
        results = saved

    return results


def _save_briefing_files(
    event: CommunityEvent,
    content: dict[str, str],
    output_dir: str,
) -> dict[str, str]:
    """
    Save briefing content to files and return file paths.

    Args:
        event: The community event (used for filename generation)
        content: Dictionary mapping format to content string
        output_dir: Directory to save files in

    Returns:
        Dictionary mapping format to file path
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build filename prefix
    date_str = event.date[:10].replace("-", "") if event.date else datetime.now().strftime("%Y%m%d")
    slugs = [s for s in (_slugify(p) for p in event.participants[:2]) if s]
    prefix = f"{date_str}_{'_'.join(slugs) if slugs else 'event'}"

    file_paths: dict[str, str] = {}

    extension_map = {
        "html": ".html",
        "whatsapp": ".txt",
        "social_card": "_card.html",
    }

    for fmt, text in content.items():
        ext = extension_map.get(fmt, f".{fmt}")
        file_path = out_path / f"{prefix}_{fmt}{ext}"
        file_path.write_text(text, encoding="utf-8")
        file_paths[fmt] = str(file_path)
        logger.info("Saved %s briefing: %s", fmt, file_path)

    # Save manifest
    manifest = {
        "event": event.to_dict(),
        "files": file_paths,
        "generated_at": datetime.now().isoformat(),
    }
    manifest_path = out_path / f"{prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    file_paths["manifest"] = str(manifest_path)

    return file_paths


# ---------------------------------------------------------------------------
#  HTML briefing
# ---------------------------------------------------------------------------

def _generate_html_briefing(
    event: CommunityEvent,
    branding: dict[str, str],
    podcast: dict[str, str],
    provider: CommunityEventProvider | None = None,
) -> str:
    """
    Generate a responsive HTML briefing page for a community event.

    Uses podcast branding for colors, fonts, and text direction.
    Mobile-first design with flexbox layout.

    Args:
        event: The community event
        branding: Branding configuration (colors, fonts)
        podcast: Podcast identity (name, direction, language)
        provider: Optional provider supplying domain-specific talking points

    Returns:
        Complete HTML document string
    """
    primary = branding["primary_color"]
    secondary = branding["secondary_color"]
    accent = branding["accent_color"]
    font = branding["font_family"]
    direction = podcast["direction"]
    lang = podcast["language"]
    pod_name = podcast["name"]
    pod_name_en = podcast["name_en"]
    pod_link = podcast["link"]

    # Event data -- generic slots, filled by whatever the provider supplied
    lead, divider, trail = headline_slots(event)
    context = event.context or ""
    event_date = event.date[:10] if event.date else datetime.now().strftime("%Y-%m-%d")
    headline = " ".join(part for part in (lead, divider, trail) if part)

    # Status-based styling
    status_label = _status_display(event.status)

    # Talking points
    talking_points = _generate_talking_points(event, provider=provider)
    talking_points_html = ""
    for i, point in enumerate(talking_points, 1):
        talking_points_html += f"""
            <div class="point-card">
                <span class="point-num">{i}</span>
                <p>{point}</p>
            </div>"""

    # Border direction for RTL/LTR
    border_side = "right" if direction == "rtl" else "left"

    html = f"""<!DOCTYPE html>
<html lang="{lang}" dir="{direction}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{headline} | {pod_name}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family={font.split(",")[0].strip()}:wght@300;400;500;600;700;800;900&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    font-family: '{font}', system-ui, -apple-system, sans-serif;
    font-size: 16px;
    line-height: 1.7;
    color: #1a1a1a;
    background: #f5f5f7;
    direction: {direction};
    -webkit-text-size-adjust: 100%;
}}

.container {{
    max-width: 780px;
    margin: 0 auto;
    padding: 20px;
}}

/* Header */
.event-header {{
    background: linear-gradient(135deg, {primary} 0%, {_lighten_color(primary, 0.15)} 100%);
    border-radius: 16px;
    padding: 36px;
    color: {secondary};
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}}

.brand {{
    font-size: 18px;
    font-weight: 600;
    color: rgba(255,255,255,0.6);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 20px;
}}

.event-meta {{
    font-size: 14px;
    color: rgba(255,255,255,0.5);
    margin-bottom: 16px;
    letter-spacing: 1px;
}}

.headline {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 28px;
    margin: 20px 0;
}}

.headline .participant {{
    font-size: 28px;
    font-weight: 800;
    min-width: 200px;
}}

.headline .participant.lead {{ text-align: {("left" if direction == "rtl" else "right")}; }}
.headline .participant.trail {{ text-align: {("right" if direction == "rtl" else "left")}; }}

.headline .result {{
    font-size: 56px;
    font-weight: 900;
    color: {accent};
    direction: ltr;
    min-width: 100px;
    text-align: center;
}}

.status-badge {{
    display: inline-block;
    padding: 6px 24px;
    border-radius: 20px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 8px;
    background: rgba(255,255,255,0.15);
    color: {accent};
    border: 2px solid rgba(255,255,255,0.2);
}}

/* Content sections */
section {{
    background: white;
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}}

h2 {{
    font-size: 22px;
    font-weight: 800;
    color: {primary};
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 3px solid {accent};
    display: inline-block;
}}

/* Talking points */
.point-card {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 16px;
    margin: 12px 0;
    background: #f8f9fb;
    border-radius: 10px;
    border-{border_side}: 4px solid {accent};
}}

.point-num {{
    font-size: 24px;
    font-weight: 900;
    color: {accent};
    min-width: 32px;
    text-align: center;
}}

.point-card p {{
    font-size: 16px;
    line-height: 1.6;
    color: #333;
}}

/* Raw data section */
.raw-data {{
    background: #f8f9fb;
    border-radius: 10px;
    padding: 16px;
    font-family: monospace;
    font-size: 13px;
    color: #555;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}}

/* Footer */
.brief-footer {{
    text-align: center;
    padding: 24px;
    color: #888;
    font-size: 14px;
}}

.brief-footer a {{
    color: {primary};
    text-decoration: none;
    font-weight: 600;
}}

@media (max-width: 600px) {{
    .container {{ padding: 12px; }}
    .event-header {{ padding: 24px 16px; }}
    .headline .participant {{ font-size: 20px; min-width: 100px; }}
    .headline .result {{ font-size: 40px; min-width: 70px; }}
    section {{ padding: 20px 16px; }}
}}
</style>
</head>
<body>
<div class="container">

    <!-- Event Header -->
    <div class="event-header">
        <div class="brand">{pod_name}</div>
        <div class="event-meta">{context} | {event_date}</div>
        <div class="headline">
            <div class="participant lead">{lead}</div>
            <div class="result">{divider}</div>
            <div class="participant trail">{trail}</div>
        </div>
        <div class="status-badge">{status_label}</div>
    </div>

    <!-- Talking Points -->
    <section>
        <h2>{"Talking Points" if lang == "en" else "Talking Points"}</h2>
        {talking_points_html}
    </section>

    <!-- Event Details -->
    <section>
        <h2>{"Details" if lang == "en" else "Details"}</h2>
        <p><strong>Status:</strong> {status_label}</p>
        <p><strong>Context:</strong> {context}</p>
        <p><strong>Date:</strong> {event_date}</p>
        <p><strong>Event ID:</strong> {event.event_id}</p>
    </section>

    <!-- Footer -->
    <div class="brief-footer">
        <p>
            {"<a href='" + pod_link + "'>" + pod_name_en + "</a> | " if pod_link else pod_name_en + " | "}
            Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </p>
    </div>

</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
#  WhatsApp briefing
# ---------------------------------------------------------------------------

def _generate_whatsapp_briefing(
    event: CommunityEvent,
    podcast: dict[str, str],
    provider: CommunityEventProvider | None = None,
) -> str:
    """
    Generate a plain-text WhatsApp message for a community event.

    Formatted for copy-paste into WhatsApp with bold markers.

    Args:
        event: The community event
        podcast: Podcast identity information
        provider: Optional provider supplying domain-specific talking points

    Returns:
        Plain text string ready for WhatsApp
    """
    pod_name = podcast["name"]
    pod_link = podcast["link"]

    lead, divider, trail = headline_slots(event)
    headline = " ".join(part for part in (lead, divider, trail) if part)

    status_label = _status_display(event.status)
    event_date = event.date[:10] if event.date else ""

    talking_points = _generate_talking_points(event, provider=provider)
    points_text = ""
    for i, point in enumerate(talking_points, 1):
        points_text += f"\n{i}. {point}"

    link_line = f"\n{pod_link}" if pod_link else ""

    msg = f"""*{pod_name}*

*{headline}*
{event.context} | {event_date} | {status_label}

---

*Talking Points:*
{points_text}

---
{link_line}"""

    return msg.strip()


# ---------------------------------------------------------------------------
#  Social card
# ---------------------------------------------------------------------------

def _generate_social_card(
    event: CommunityEvent,
    branding: dict[str, str],
    podcast: dict[str, str],
    provider: CommunityEventProvider | None = None,
) -> str:
    """
    Generate a square HTML social card (1080x1080) for a community event.

    Designed for screenshots to use on Instagram, Twitter/X, etc.

    Args:
        event: The community event
        branding: Branding configuration (colors, fonts)
        podcast: Podcast identity information
        provider: Optional provider supplying domain-specific talking points

    Returns:
        Complete HTML document string (1080x1080 viewport)
    """
    primary = branding["primary_color"]
    secondary = branding["secondary_color"]
    accent = branding["accent_color"]
    font = branding["font_family"]
    direction = podcast["direction"]
    lang = podcast["language"]
    pod_name = podcast["name"]
    pod_name_en = podcast["name_en"]

    lead, divider, trail = headline_slots(event)
    headline = " ".join(part for part in (lead, divider, trail) if part)
    status_label = _status_display(event.status)

    talking_points = _generate_talking_points(event, max_points=3, provider=provider)
    points_html = ""
    for i, point in enumerate(talking_points, 1):
        points_html += f"""
        <div class="point">
            <div class="point-num">{i}</div>
            <div class="point-text">{point}</div>
        </div>"""

    # Border direction for RTL/LTR
    border_side = "right" if direction == "rtl" else "left"

    html = f"""<!DOCTYPE html>
<html lang="{lang}" dir="{direction}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=1080">
<title>{headline} | {pod_name}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family={font.split(",")[0].strip()}:wght@400;500;700;800;900&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    font-family: '{font}', system-ui, sans-serif;
    width: 1080px;
    height: 1080px;
    overflow: hidden;
    background: linear-gradient(135deg, {primary} 0%, {_lighten_color(primary, 0.2)} 100%);
    color: {secondary};
    direction: {direction};
}}

.card {{
    width: 1080px;
    height: 1080px;
    display: flex;
    flex-direction: column;
    padding: 48px;
    position: relative;
}}

.brand-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}}

.brand-name {{
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 1px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}}

.brand-tag {{
    font-size: 22px;
    font-weight: 500;
    color: rgba(255,255,255,0.6);
    background: rgba(255,255,255,0.08);
    padding: 6px 18px;
    border-radius: 20px;
}}

.event-header {{
    text-align: center;
    margin-bottom: 28px;
    padding: 28px 0;
    border-top: 2px solid rgba(255,255,255,0.15);
    border-bottom: 2px solid rgba(255,255,255,0.15);
}}

.context {{
    font-size: 22px;
    font-weight: 500;
    color: rgba(255,255,255,0.55);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 2px;
}}

.participants-row {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 32px;
    margin-bottom: 12px;
}}

.participant-name {{
    font-size: 42px;
    font-weight: 800;
    min-width: 260px;
}}

.participant-name.lead {{ text-align: {("left" if direction == "rtl" else "right")}; }}
.participant-name.trail {{ text-align: {("right" if direction == "rtl" else "left")}; }}

.result-box {{
    font-size: 72px;
    font-weight: 900;
    color: {accent};
    text-shadow: 0 0 30px rgba(255,255,255,0.15);
    direction: ltr;
    min-width: 140px;
    text-align: center;
}}

.status-badge {{
    display: inline-block;
    font-size: 24px;
    font-weight: 700;
    padding: 6px 28px;
    border-radius: 24px;
    margin-top: 12px;
    background: rgba(255,255,255,0.1);
    color: {accent};
    border: 2px solid rgba(255,255,255,0.2);
}}

.points-section {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin: 16px 0;
}}

.points-title {{
    font-size: 26px;
    font-weight: 700;
    color: {accent};
    margin-bottom: 4px;
}}

.point {{
    display: flex;
    align-items: flex-start;
    gap: 16px;
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 22px;
    border-{border_side}: 4px solid {accent};
}}

.point-num {{
    font-size: 28px;
    font-weight: 900;
    color: {accent};
    min-width: 36px;
    text-align: center;
    line-height: 1.1;
}}

.point-text {{
    font-size: 23px;
    font-weight: 500;
    line-height: 1.5;
    color: rgba(255,255,255,0.92);
}}

.footer {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-top: 2px solid rgba(255,255,255,0.15);
    padding-top: 20px;
    margin-top: auto;
}}

.footer-text {{
    font-size: 20px;
    font-weight: 400;
    color: rgba(255,255,255,0.4);
}}
</style>
</head>
<body>
<div class="card">
    <div class="brand-bar">
        <div class="brand-name">{pod_name}</div>
        <div class="brand-tag">{status_label}</div>
    </div>

    <div class="event-header">
        <div class="context">{event.context} | {event.date[:10] if event.date else ""}</div>
        <div class="participants-row">
            <div class="participant-name lead">{lead}</div>
            <div class="result-box">{divider}</div>
            <div class="participant-name trail">{trail}</div>
        </div>
        <div class="status-badge">{status_label}</div>
    </div>

    <div class="points-section">
        <div class="points-title">Talking Points</div>
        {points_html}
    </div>

    <div class="footer">
        <div class="footer-text">{pod_name_en}</div>
        <div class="footer-text">{datetime.now().strftime("%Y-%m-%d")}</div>
    </div>
</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _status_display(status: str) -> str:
    """
    Convert an API status code to a human-readable label.

    Args:
        status: Status string from the event (e.g., "FINISHED")

    Returns:
        Human-readable status label
    """
    status_map = {
        "FINISHED": "Final",
        "SCHEDULED": "Upcoming",
        "TIMED": "Upcoming",
        "IN_PLAY": "Live",
        "PAUSED": "Halftime",
        "HALFTIME": "Halftime",
        "POSTPONED": "Postponed",
        "CANCELLED": "Cancelled",
        "SUSPENDED": "Suspended",
    }
    return status_map.get(status, status)


def _generate_talking_points(
    event: CommunityEvent,
    max_points: int = 5,
    provider: CommunityEventProvider | None = None,
) -> list[str]:
    """
    Generate talking points for a community event.

    A provider that knows its domain gets first refusal via its
    ``talking_points()`` hook -- that is where football previews, release
    checklists or council-vote framings belong. If no provider is supplied,
    or it returns nothing, the neutral prompts below are used. They reference
    only ``participants``, ``result``, ``context`` and ``status``, so they
    read sensibly for any kind of event.

    Args:
        event: The community event
        max_points: Maximum number of talking points to generate
        provider: Optional provider offering domain-specific prompts

    Returns:
        List of talking point strings
    """
    if provider is not None:
        try:
            domain_points = provider.talking_points(event, max_points=max_points)
        except Exception as exc:  # a bad provider must not break the briefing
            logger.warning(
                "Provider %s raised in talking_points(); using neutral prompts: %s",
                type(provider).__name__,
                exc,
            )
            domain_points = []
        if domain_points:
            return list(domain_points)[:max_points]

    return _neutral_talking_points(event, max_points)


def _neutral_talking_points(
    event: CommunityEvent,
    max_points: int = 5,
) -> list[str]:
    """
    Domain-neutral discussion prompts, used when no provider supplies any.

    Args:
        event: The community event
        max_points: Maximum number of talking points to generate

    Returns:
        List of talking point strings
    """
    points: list[str] = []
    participants = [p for p in event.participants if p]
    status_label = _status_display(event.status)

    if event.summary:
        points.append(f"Recap: {event.summary}")

    if event.result:
        points.append(f"The outcome was {event.result} -- was that the expected one?")
    elif event.status in ("SCHEDULED", "TIMED"):
        points.append("What outcome should the community expect, and why?")

    for name in participants[:2]:
        points.append(f"What does this change for {name}?")

    if event.context:
        points.append(f"How does it sit in the wider picture of {event.context}?")

    points.append(f"Status is {status_label} -- what are the implications?")
    points.append("What should the community watch for next?")

    return points[:max_points]


def _lighten_color(hex_color: str, factor: float = 0.15) -> str:
    """
    Lighten a hex color by a factor.

    Args:
        hex_color: Hex color string (e.g., "#1a1a2e")
        factor: Lightening factor (0.0 = no change, 1.0 = white)

    Returns:
        Lightened hex color string
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"#{hex_color}"

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return f"#{hex_color}"

    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))

    return f"#{r:02x}{g:02x}{b:02x}"


__all__ = [
    "NEUTRAL_SEPARATOR",
    "generate_briefing",
    "headline_slots",
    "text_direction",
]

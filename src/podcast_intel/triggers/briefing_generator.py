"""
Generic briefing generator for community events.

Generates multi-format briefing content (HTML, WhatsApp, social card)
from community events, using podcast.yaml branding configuration for
styling. This module is intentionally GENERIC -- no team-specific or
sport-specific content is hardcoded.

Branding and voice come entirely from podcast.yaml configuration:
    podcast:
      name: "My Podcast"
      name_en: "My Podcast"
      link: "https://example.com"
      direction: "rtl"
      language: "he"
      branding:
        primary_color: "#132257"
        secondary_color: "#FFFFFF"
        accent_color: "#4A90D9"
        highlight_color: "#C4A747"
        font_family: "Heebo"

Example:
    >>> from podcast_intel.triggers.briefing_generator import generate_briefing
    >>> from podcast_intel.triggers.community_events import CommunityEvent
    >>> event = CommunityEvent(
    ...     event_id="12345",
    ...     event_type="match",
    ...     status="FINISHED",
    ...     teams=["Team A", "Team B"],
    ...     score="2-1",
    ...     competition="League",
    ... )
    >>> result = generate_briefing(event, config)
    >>> print(result["html"])
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from podcast_intel.triggers.community_events import CommunityEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Default branding (overridden by podcast.yaml)
# ---------------------------------------------------------------------------

_DEFAULT_BRANDING = {
    "primary_color": "#1a1a2e",
    "secondary_color": "#ffffff",
    "accent_color": "#4a90d9",
    "highlight_color": "#c4a747",
    "font_family": "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
}

_DEFAULT_PODCAST = {
    "name": "Podcast",
    "name_en": "Podcast",
    "link": "",
    "direction": "ltr",
    "language": "en",
}


# ---------------------------------------------------------------------------
#  Configuration helpers
# ---------------------------------------------------------------------------

def _get_branding(config: Dict[str, Any]) -> Dict[str, str]:
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


def _get_podcast_info(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract podcast identity from configuration.

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
    return info


# ---------------------------------------------------------------------------
#  Briefing generation
# ---------------------------------------------------------------------------

def generate_briefing(
    event: CommunityEvent,
    config: Dict[str, Any],
    formats: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate briefing content for a community event.

    Creates content in the requested formats (html, whatsapp, social_card)
    and optionally saves files to the output directory.

    Args:
        event: The community event to generate a briefing for
        config: Full podcast.yaml configuration dictionary
        formats: List of output formats (default: ["html"])
        output_dir: Directory to save files to (optional)

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

    results: Dict[str, str] = {}

    if "html" in formats:
        results["html"] = _generate_html_briefing(event, branding, podcast)

    if "whatsapp" in formats:
        results["whatsapp"] = _generate_whatsapp_briefing(event, podcast)

    if "social_card" in formats:
        results["social_card"] = _generate_social_card(event, branding, podcast)

    # Save to files if output_dir is specified
    if output_dir:
        saved = _save_briefing_files(event, results, output_dir)
        # Return file paths instead of content
        results = saved

    return results


def _save_briefing_files(
    event: CommunityEvent,
    content: Dict[str, str],
    output_dir: str,
) -> Dict[str, str]:
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
    teams_slug = "_vs_".join(
        t.lower().replace(" ", "_") for t in event.teams[:2]
    ) if event.teams else "event"
    prefix = f"{date_str}_{teams_slug}"

    file_paths: Dict[str, str] = {}

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
    branding: Dict[str, str],
    podcast: Dict[str, str],
) -> str:
    """
    Generate a responsive HTML briefing page for a community event.

    Uses podcast branding for colors, fonts, and text direction.
    Mobile-first design with flexbox layout.

    Args:
        event: The community event
        branding: Branding configuration (colors, fonts)
        podcast: Podcast identity (name, direction, language)

    Returns:
        Complete HTML document string
    """
    primary = branding["primary_color"]
    secondary = branding["secondary_color"]
    accent = branding["accent_color"]
    highlight = branding["highlight_color"]
    font = branding["font_family"]
    direction = podcast["direction"]
    lang = podcast["language"]
    pod_name = podcast["name"]
    pod_name_en = podcast["name_en"]
    pod_link = podcast["link"]

    # Event data
    teams = event.teams if event.teams else ["Team A", "Team B"]
    home_team = teams[0] if len(teams) > 0 else "Home"
    away_team = teams[1] if len(teams) > 1 else "Away"
    score_display = event.score or "vs"
    competition = event.competition or ""
    event_date = event.date[:10] if event.date else datetime.now().strftime("%Y-%m-%d")

    # Status-based styling
    is_finished = event.status == "FINISHED"
    status_label = _status_display(event.status)

    # Talking points
    talking_points = _generate_talking_points(event)
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
<title>{home_team} {score_display} {away_team} | {pod_name}</title>
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

.score-display {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 28px;
    margin: 20px 0;
}}

.score-display .team {{
    font-size: 28px;
    font-weight: 800;
    min-width: 200px;
}}

.score-display .team.home {{ text-align: {("left" if direction == "rtl" else "right")}; }}
.score-display .team.away {{ text-align: {("right" if direction == "rtl" else "left")}; }}

.score-display .score {{
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
    .score-display .team {{ font-size: 20px; min-width: 100px; }}
    .score-display .score {{ font-size: 40px; min-width: 70px; }}
    section {{ padding: 20px 16px; }}
}}
</style>
</head>
<body>
<div class="container">

    <!-- Event Header -->
    <div class="event-header">
        <div class="brand">{pod_name}</div>
        <div class="event-meta">{competition} | {event_date}</div>
        <div class="score-display">
            <div class="team home">{home_team}</div>
            <div class="score">{score_display}</div>
            <div class="team away">{away_team}</div>
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
        <p><strong>Competition:</strong> {competition}</p>
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
    podcast: Dict[str, str],
) -> str:
    """
    Generate a plain-text WhatsApp message for a community event.

    Formatted for copy-paste into WhatsApp with bold markers.

    Args:
        event: The community event
        podcast: Podcast identity information

    Returns:
        Plain text string ready for WhatsApp
    """
    pod_name = podcast["name"]
    pod_link = podcast["link"]

    teams = event.teams if event.teams else ["Team A", "Team B"]
    home_team = teams[0] if len(teams) > 0 else "Home"
    away_team = teams[1] if len(teams) > 1 else "Away"
    score = event.score or "vs"

    status_label = _status_display(event.status)
    event_date = event.date[:10] if event.date else ""

    talking_points = _generate_talking_points(event)
    points_text = ""
    for i, point in enumerate(talking_points, 1):
        points_text += f"\n{i}. {point}"

    link_line = f"\n{pod_link}" if pod_link else ""

    msg = f"""*{pod_name}*

*{home_team} {score} {away_team}*
{event.competition} | {event_date} | {status_label}

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
    branding: Dict[str, str],
    podcast: Dict[str, str],
) -> str:
    """
    Generate a square HTML social card (1080x1080) for a community event.

    Designed for screenshots to use on Instagram, Twitter/X, etc.

    Args:
        event: The community event
        branding: Branding configuration (colors, fonts)
        podcast: Podcast identity information

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

    teams = event.teams if event.teams else ["Team A", "Team B"]
    home_team = teams[0] if len(teams) > 0 else "Home"
    away_team = teams[1] if len(teams) > 1 else "Away"
    score_display = event.score or "vs"
    status_label = _status_display(event.status)

    talking_points = _generate_talking_points(event, max_points=3)
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
<title>{home_team} {score_display} {away_team} | {pod_name}</title>
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

.competition {{
    font-size: 22px;
    font-weight: 500;
    color: rgba(255,255,255,0.55);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 2px;
}}

.teams-row {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 32px;
    margin-bottom: 12px;
}}

.team-name {{
    font-size: 42px;
    font-weight: 800;
    min-width: 260px;
}}

.team-name.home {{ text-align: {("left" if direction == "rtl" else "right")}; }}
.team-name.away {{ text-align: {("right" if direction == "rtl" else "left")}; }}

.score-box {{
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
        <div class="competition">{event.competition} | {event.date[:10] if event.date else ""}</div>
        <div class="teams-row">
            <div class="team-name home">{home_team}</div>
            <div class="score-box">{score_display}</div>
            <div class="team-name away">{away_team}</div>
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
) -> List[str]:
    """
    Generate generic talking points for a community event.

    Creates discussion prompts based on the event type, status,
    teams, and score. These are intentionally generic and do not
    contain any team-specific opinions or takes.

    Args:
        event: The community event
        max_points: Maximum number of talking points to generate

    Returns:
        List of talking point strings
    """
    points: List[str] = []
    teams = event.teams if event.teams else []
    home = teams[0] if len(teams) > 0 else "Home team"
    away = teams[1] if len(teams) > 1 else "Away team"

    if event.status == "FINISHED" and event.score:
        # Parse score for context
        score_parts = event.score.replace(" (HT)", "").split("-")
        try:
            home_goals = int(score_parts[0].strip())
            away_goals = int(score_parts[1].strip())
        except (ValueError, IndexError):
            home_goals = 0
            away_goals = 0

        if home_goals > away_goals:
            points.append(
                f"{home} won {event.score} -- what were the key moments?"
            )
            points.append(
                f"How did {away} lose control of this match?"
            )
        elif away_goals > home_goals:
            points.append(
                f"{away} won {event.score} away from home -- analyze the performance."
            )
            points.append(
                f"Where did {home} go wrong at home?"
            )
        else:
            points.append(
                f"The match ended {event.score} -- was this a fair result?"
            )
            points.append(
                "Which side should feel more disappointed with the draw?"
            )

        points.append(
            f"Player ratings: who stood out in {home} vs {away}?"
        )
        points.append(
            f"Tactical analysis: what formations and strategies shaped this {event.competition} match?"
        )
        points.append(
            "What does this result mean for the season standings?"
        )

    elif event.status in ("SCHEDULED", "TIMED"):
        points.append(
            f"Preview: {home} vs {away} -- what to expect?"
        )
        points.append(
            "Predicted lineups and key matchups to watch."
        )
        points.append(
            f"Recent form: how are both sides performing in {event.competition}?"
        )
        points.append(
            "Key players to watch and potential game-changers."
        )
        points.append(
            "Score prediction and tactical approach."
        )

    else:
        points.append(f"Event update: {event.summary}")
        points.append(f"Status: {_status_display(event.status)}")
        points.append("Discussion: what are the implications?")

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
    "generate_briefing",
]

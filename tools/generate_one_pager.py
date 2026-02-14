#!/usr/bin/env python3
"""
Generate a pre-recording one-pager brief for podcast panelists.

Usage:
    python tools/generate_one_pager.py --episode 203
    python tools/generate_one_pager.py --episode 203 --date "February 9, 2026"
    python tools/generate_one_pager.py --episode 203 --output /custom/path/brief.html

The script reads PQS v3 score data from the reports directory and generates
a mobile-first HTML brief for panelists to review in 2 minutes before recording.

Output: reports/one_pagers/pre_recording_brief_ep{N}.html
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "one_pagers"


def _load_config() -> dict:
    """Load podcast.yaml configuration if available."""
    config_path = PROJECT_ROOT / "podcast.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
    return {}


# Tier labels
TIER_LABELS = {
    "S": "Excellent",
    "A": "Very Good",
    "A-": "Very Good",
    "Good": "Good",
    "B": "Good",
    "B+": "Good",
    "B-": "Above Average",
    "C": "Average",
    "C+": "Above Average",
    "C-": "Below Average",
    "C - Average": "Average",
    "D": "Weak",
    "F": "Failed",
}

# Component name display labels
COMPONENT_NAMES = {
    "audio_quality": "Audio",
    "audioQuality": "Audio",
    "audio": "Audio",
    "delivery": "Delivery",
    "delivery_dynamics": "Delivery",
    "delivery_score": "Delivery",
    "structure": "Structure",
    "structure_flow": "Structure",
    "structure_score": "Structure",
    "content_depth": "Content",
    "content": "Content",
    "content_score": "Content",
    "engagement": "Engagement",
    "engagement_proxies": "Engagement",
    "engagement_score": "Engagement",
}


# ---------------------------------------------------------------------------
# Data Extraction -- handles varying JSON schemas across episodes
# ---------------------------------------------------------------------------

def find_episode_dir(episode_num: int) -> Path | None:
    """Find the report directory for a given episode number."""
    patterns = [
        f"episode_{episode_num}",
        f"ep_{episode_num}",
    ]
    for pattern in patterns:
        candidate = REPORTS_DIR / pattern
        if candidate.is_dir():
            return candidate
    return None


def load_pqs_data(episode_num: int) -> dict | None:
    """Load PQS v3 scores JSON for a given episode."""
    ep_dir = find_episode_dir(episode_num)
    if ep_dir is None:
        return None
    scores_file = ep_dir / "pqs_v3_scores.json"
    if not scores_file.exists():
        return None
    with open(scores_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_composite_score(data: dict) -> float:
    """Extract composite score from varying JSON schemas."""
    for key in ("composite_score", "compositeScore", "composite_pqs_v3"):
        if key in data:
            return float(data[key])
    return 0.0


def extract_tier(data: dict) -> str:
    """Extract tier from varying JSON schemas."""
    raw_tier = data.get("tier", "?")
    return str(raw_tier).strip()


def extract_component_scores(data: dict) -> dict:
    """
    Extract component scores into a normalized dict.
    Handles all known JSON schema variations.
    """
    result = {}

    # Schema 1: component_scores with nested {score, weight}
    cs = data.get("component_scores", {})
    if cs:
        for key, val in cs.items():
            if isinstance(val, dict) and "score" in val:
                result[key] = float(val["score"])
            elif isinstance(val, (int, float)):
                result[key] = float(val)

    # Schema 2: domainScores with nested {score, weight, contribution}
    ds = data.get("domainScores", {})
    if ds:
        for key, val in ds.items():
            if isinstance(val, dict) and "score" in val:
                result[key] = float(val["score"])

    return result


def extract_bottlenecks(data: dict) -> list[dict]:
    """Extract bottleneck items."""
    raw = data.get("bottlenecks", data.get("top_3_bottlenecks", []))
    bottlenecks = []
    for item in raw:
        if isinstance(item, dict):
            metric = item.get("metric", "")
            score = item.get("score", 0)
            bottlenecks.append({"metric": metric, "score": float(score)})
        elif isinstance(item, str):
            bottlenecks.append({"metric": item, "score": 0})
    return bottlenecks


def extract_strengths(data: dict) -> list[dict]:
    """Extract strength items."""
    raw = data.get("strengths", data.get("top_3_strengths", []))
    strengths = []
    for item in raw:
        if isinstance(item, dict):
            metric = item.get("metric", "")
            score = item.get("score", 100)
            strengths.append({"metric": metric, "score": float(score)})
        elif isinstance(item, str):
            strengths.append({"metric": item, "score": 100})
    return strengths


def extract_episode_context(data: dict) -> dict:
    """Extract episode context like format, duration, etc."""
    ctx = data.get("episode_context", {})
    return {
        "format": ctx.get("format", ""),
        "duration": ctx.get("duration_minutes", 0),
        "title": data.get("episode", ""),
    }


# ---------------------------------------------------------------------------
# Score color classification
# ---------------------------------------------------------------------------

def score_color(score: float) -> str:
    """Return CSS color class based on score."""
    if score >= 75:
        return "green"
    elif score >= 60:
        return "yellow"
    else:
        return "red"


def tier_display(tier: str) -> str:
    """Get display label for a tier."""
    return TIER_LABELS.get(tier, tier)


# ---------------------------------------------------------------------------
# Generate improvement recommendations based on data
# ---------------------------------------------------------------------------

def generate_fix_items(data: dict) -> list[dict]:
    """
    Generate 3 actionable fix items based on bottlenecks and component scores.
    Returns list of {priority: red/yellow/green, icon: emoji, text: str}
    """
    components = extract_component_scores(data)
    context = extract_episode_context(data)

    fixes = []

    # Duration check
    duration = context.get("duration", 0)
    if duration > 0 and duration < 50:
        fixes.append({
            "priority": "red",
            "icon": "\U0001F534",
            "text": f"<strong>Episode length:</strong> {duration:.0f} minutes is too short. Aim for 55-65 minutes -- prepare enough material."
        })
    elif duration > 70:
        fixes.append({
            "priority": "red",
            "icon": "\U0001F534",
            "text": f"<strong>Episode length:</strong> {duration:.0f} minutes is too long. Stop at 60 minutes -- use a timer."
        })

    # Engagement check
    engagement_score = None
    for key, val in components.items():
        if "engage" in key.lower():
            engagement_score = val
            break
    if engagement_score is not None and engagement_score < 50:
        fixes.append({
            "priority": "red",
            "icon": "\U0001F534",
            "text": f"<strong>Engagement:</strong> Score {engagement_score:.1f} is low. Add interaction, debate, and listener questions."
        })
    elif engagement_score is not None and engagement_score < 70:
        fixes.append({
            "priority": "yellow",
            "icon": "\U0001F7E1",
            "text": f"<strong>Engagement:</strong> Score {engagement_score:.1f} is average. Prepare debate questions and break up monologues."
        })

    # Delivery check
    delivery_score = None
    for key, val in components.items():
        if "deliver" in key.lower():
            delivery_score = val
            break
    if delivery_score is not None and delivery_score < 65:
        fixes.append({
            "priority": "red",
            "icon": "\U0001F534",
            "text": f"<strong>Delivery:</strong> Score {delivery_score:.1f} is low. Reduce filler words, work on pacing, replace 'um' with a brief pause."
        })
    elif delivery_score is not None and delivery_score < 75:
        fixes.append({
            "priority": "yellow",
            "icon": "\U0001F7E1",
            "text": f"<strong>Fillers:</strong> Reduce 'um', 'like', 'you know'. Replace with a short pause -- it sounds more confident."
        })

    # Structure check
    structure_score = None
    for key, val in components.items():
        if "struct" in key.lower():
            structure_score = val
            break
    if structure_score is not None and structure_score < 70:
        fixes.append({
            "priority": "yellow",
            "icon": "\U0001F7E1",
            "text": f"<strong>Structure:</strong> Score {structure_score:.1f}. Prepare clear transitions between topics and ensure a strong open and close."
        })

    # Content check
    content_score = None
    for key, val in components.items():
        if "content" in key.lower():
            content_score = val
            break
    if content_score is not None and content_score < 70:
        fixes.append({
            "priority": "yellow",
            "icon": "\U0001F7E1",
            "text": f"<strong>Content:</strong> Score {content_score:.1f}. Go deeper on analysis and add more evidence-based opinions."
        })

    # Always add closing fix if we have room
    if len(fixes) < 3:
        fixes.append({
            "priority": "green",
            "icon": "\U0001F7E2",
            "text": "<strong>Strong close:</strong> End with a 30-second summary + preview of next episode + call to action."
        })

    # Always add opening hook if we have room
    if len(fixes) < 3:
        fixes.append({
            "priority": "green",
            "icon": "\U0001F7E2",
            "text": "<strong>Strong open:</strong> Start with a hook -- a sharp take or provocative question within 30 seconds."
        })

    return fixes[:3]


# ---------------------------------------------------------------------------
# Generate episode summary line
# ---------------------------------------------------------------------------

def generate_summary_line(data: dict) -> str:
    """Generate a one-sentence summary of the last episode."""
    components = extract_component_scores(data)
    context = extract_episode_context(data)

    scored = []
    for key, val in components.items():
        name = COMPONENT_NAMES.get(key, key)
        scored.append((name, val))
    scored.sort(key=lambda x: x[1], reverse=True)

    if len(scored) >= 2:
        best_name, best_val = scored[0]
        worst_name, worst_val = scored[-1]
    else:
        return "Not enough data for a summary."

    fmt = context.get("format", "")
    duration = context.get("duration", 0)

    parts = []
    if fmt:
        if "solo" in fmt.lower() or "monologue" in fmt.lower():
            parts.append("Solo format")
        elif "panel" in fmt.lower():
            parts.append("Panel format")

    parts.append(f"with strong {best_name} ({best_val:.1f})")
    parts.append(f"but low {worst_name} ({worst_val:.1f})")

    if duration > 0:
        if duration < 50:
            parts.append(f"and episode too short ({duration:.0f} min)")
        elif duration > 70:
            parts.append(f"and episode too long ({duration:.0f} min)")

    return " ".join(parts) + "."


# ---------------------------------------------------------------------------
# Calculate target score
# ---------------------------------------------------------------------------

def calculate_target(current: float) -> float:
    """Calculate a reasonable target score (current + ~8, capped at 95)."""
    return min(current + 8, 95.0)


# ---------------------------------------------------------------------------
# HTML Generation
# ---------------------------------------------------------------------------

def generate_html(
    next_episode: int,
    prev_data: dict,
    trend_data: list[tuple[int, float]],
    recording_date: str,
    podcast_name: str = "Podcast",
    accent_color: str = "#132257",
) -> str:
    """Generate the full one-pager HTML string."""

    prev_ep = next_episode - 1
    composite = extract_composite_score(prev_data)
    tier = extract_tier(prev_data)
    tier_label = tier_display(tier)
    color = score_color(composite)
    components = extract_component_scores(prev_data)
    summary = generate_summary_line(prev_data)
    fixes = generate_fix_items(prev_data)
    target = calculate_target(composite)
    improvement = target - composite

    # Build component bars HTML
    comp_items = []
    for key, val in components.items():
        name = COMPONENT_NAMES.get(key, key)
        comp_items.append((name, val))
    comp_items.sort(key=lambda x: x[1], reverse=True)

    components_html = ""
    for name, val in comp_items:
        components_html += f"""    <div class="component">
      <div class="component-value">{val:.1f}</div>
      <div class="component-name">{name}</div>
    </div>\n"""

    # Build fix cards HTML
    fixes_html = ""
    for fix in fixes:
        fixes_html += f"""  <div class="fix-card priority-{fix['priority']}">
    <span class="fix-icon">{fix['icon']}</span>
    <div class="fix-text">{fix['text']}</div>
  </div>\n"""

    # Build trend bar HTML
    trend_html = ""
    if trend_data:
        trend_items = []
        for i, (ep_num, ep_score) in enumerate(trend_data):
            is_last = (i == len(trend_data) - 1)
            cls = ' class="trend-ep current"' if is_last else ' class="trend-ep"'
            trend_items.append(
                f'      <div{cls}>\n'
                f'        <div>{ep_num}</div>\n'
                f'        <div class="trend-score">{ep_score:.1f}</div>\n'
                f'      </div>'
            )

        trend_html = '\n      <span class="trend-arrow">&larr;</span>\n'.join(trend_items)
        trend_html = f"""
    <div class="trend-bar">
{trend_html}
    </div>"""

    # Meter calculations
    fill_pct = min(composite, 100)
    target_pct = min(target, 100)
    target_right_pct = 100 - target_pct

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=3.0, user-scalable=yes">
<title>Pre-Recording Brief -- Episode {next_episode}</title>
<style>
  :root {{
    --accent: {accent_color};
    --accent-light: {accent_color}cc;
    --gold: #d4a843;
    --red: #e74c3c;
    --yellow: #f39c12;
    --green: #27ae60;
    --bg: #f5f6fa;
    --card: #ffffff;
    --text: #1a1a1a;
    --text-muted: #6b7280;
    --shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    --radius: 12px;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 15px;
    line-height: 1.55;
    color: var(--text);
    background: var(--bg);
    max-width: 400px;
    margin: 0 auto;
    padding-bottom: 32px;
  }}

  .header {{
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%);
    color: #fff;
    padding: 24px 20px 20px;
    text-align: center;
  }}
  .header-logo {{
    font-size: 0.7em;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 6px;
  }}
  .header h1 {{
    font-size: 1.25em;
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .header-meta {{
    font-size: 0.8em;
    opacity: 0.75;
  }}

  .content {{ padding: 0 16px; }}

  .score-section {{
    background: var(--card);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    margin: -16px 16px 16px;
    padding: 20px;
    text-align: center;
  }}
  .score-label {{
    font-size: 0.75em;
    color: var(--text-muted);
    margin-bottom: 4px;
  }}
  .score-big {{
    font-size: 3.2em;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
  }}
  .score-big.green {{ color: var(--green); }}
  .score-big.yellow {{ color: var(--yellow); }}
  .score-big.red {{ color: var(--red); }}

  .tier-badge {{
    display: inline-block;
    padding: 2px 14px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 700;
  }}
  .tier-badge.green {{ background: #e8f8ef; color: var(--green); }}
  .tier-badge.yellow {{ background: #fef5e7; color: var(--yellow); }}
  .tier-badge.red {{ background: #fdedec; color: var(--red); }}

  .score-summary {{
    font-size: 0.85em;
    color: var(--text-muted);
    margin-top: 10px;
  }}

  .score-components {{
    display: flex;
    justify-content: space-between;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid #f0f0f0;
    gap: 4px;
  }}
  .component {{ text-align: center; flex: 1; }}
  .component-value {{ font-size: 1.05em; font-weight: 700; color: var(--accent); }}
  .component-name {{ font-size: 0.6em; color: var(--text-muted); margin-top: 2px; }}

  .section-title {{
    font-size: 0.95em;
    font-weight: 700;
    color: var(--accent);
    margin: 20px 0 10px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .section-title::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #d0d5e0, transparent);
  }}

  .fix-card {{
    background: var(--card);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    border-left: 4px solid transparent;
  }}
  .fix-card.priority-red {{ border-left-color: var(--red); }}
  .fix-card.priority-yellow {{ border-left-color: var(--yellow); }}
  .fix-card.priority-green {{ border-left-color: var(--green); }}
  .fix-icon {{ font-size: 1.1em; flex-shrink: 0; }}
  .fix-text {{ font-size: 0.88em; line-height: 1.45; }}
  .fix-text strong {{ color: var(--accent); }}

  .checklist {{
    background: var(--card);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 14px 16px;
    margin-bottom: 8px;
  }}
  .check-item {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    font-size: 0.88em;
    border-bottom: 1px solid #f5f5f5;
  }}
  .check-item:last-child {{ border-bottom: none; }}
  .check-box {{
    width: 20px; height: 20px;
    border: 2px solid #c8cdd5;
    border-radius: 4px;
    flex-shrink: 0;
  }}

  .target-section {{
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 16px 18px;
    margin-bottom: 8px;
    color: #fff;
    text-align: center;
  }}
  .target-title {{ font-size: 0.78em; opacity: 0.8; margin-bottom: 6px; }}
  .target-value {{ font-size: 1.4em; font-weight: 800; margin-bottom: 4px; }}
  .target-sub {{ font-size: 0.78em; opacity: 0.7; margin-bottom: 12px; }}

  .meter-track {{
    background: rgba(255,255,255,0.15);
    border-radius: 8px;
    height: 10px;
    position: relative;
  }}
  .meter-fill {{
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, var(--yellow), var(--green));
  }}
  .meter-target {{
    position: absolute;
    top: -4px;
    width: 3px; height: 18px;
    background: var(--gold);
    border-radius: 2px;
  }}
  .meter-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 0.65em;
    opacity: 0.6;
    margin-top: 4px;
  }}

  .trend-bar {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.15);
    font-size: 0.75em;
  }}
  .trend-ep {{ text-align: center; opacity: 0.7; }}
  .trend-ep.current {{ opacity: 1; font-weight: 700; }}
  .trend-score {{ font-size: 1.3em; font-weight: 700; }}
  .trend-arrow {{ opacity: 0.5; font-size: 1.2em; }}

  .footer {{
    text-align: center;
    padding: 20px 16px 8px;
    color: var(--text-muted);
    font-size: 0.75em;
  }}
  .footer-luck {{ font-size: 1.1em; font-weight: 700; color: var(--accent); margin-bottom: 2px; }}
</style>
</head>
<body>

<div class="header">
  <div class="header-logo">{podcast_name.upper()}</div>
  <h1>Pre-Recording Brief</h1>
  <div class="header-meta">Episode {next_episode} &middot; {recording_date}</div>
</div>

<div class="score-section">
  <div class="score-label">Previous Episode Score ({prev_ep})</div>
  <div class="score-big {color}">{composite:.1f}</div>
  <span class="tier-badge {color}">{tier} -- {tier_label}</span>
  <div class="score-summary">{summary}</div>
  <div class="score-components">
{components_html}  </div>
</div>

<div class="content">
  <div class="section-title">3 Things to Improve</div>

{fixes_html}
  <div class="section-title">Quick Checklist</div>

  <div class="checklist">
    <div class="check-item">
      <div class="check-box"></div>
      <div class="check-label">Opening hook ready -- sharp take within 30 seconds</div>
    </div>
    <div class="check-item">
      <div class="check-box"></div>
      <div class="check-label">Timer set: 60 minutes maximum</div>
    </div>
    <div class="check-item">
      <div class="check-box"></div>
      <div class="check-label">3 debate questions prepared</div>
    </div>
    <div class="check-item">
      <div class="check-box"></div>
      <div class="check-label">Closing planned: summary + preview + CTA</div>
    </div>
    <div class="check-item">
      <div class="check-box"></div>
      <div class="check-label">Filler awareness -- minimize "um" and "like"</div>
    </div>
  </div>

  <div class="section-title">Episode Target</div>

  <div class="target-section">
    <div class="target-title">PQS Target Score</div>
    <div class="target-value">{target:.0f}+</div>
    <div class="target-sub">Improvement of {improvement:.1f} points from last episode</div>

    <div class="meter-track">
      <div class="meter-fill" style="width: {fill_pct:.1f}%;"></div>
      <div class="meter-target" style="right: {target_right_pct:.1f}%; top: -4px; position: absolute;"></div>
    </div>
    <div class="meter-labels">
      <span>0</span>
      <span>{composite:.1f} current</span>
      <span>{target:.0f} target</span>
      <span>100</span>
    </div>
{trend_html}
  </div>
</div>

<div class="footer">
  <div class="footer-luck">Good luck!</div>
  <div>Recording: {recording_date} &middot; {podcast_name}</div>
</div>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a pre-recording one-pager brief for podcast panelists.",
    )
    parser.add_argument(
        "--episode", "-e", type=int, required=True,
        help="Episode number to prepare for. The brief is based on the PREVIOUS episode's data.",
    )
    parser.add_argument(
        "--date", "-d", type=str, default=None,
        help="Recording date. Defaults to today.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path. Defaults to reports/one_pagers/pre_recording_brief_ep{N}.html",
    )

    args = parser.parse_args()
    next_episode = args.episode
    prev_episode = next_episode - 1

    # Load config
    config = _load_config()
    podcast_name = config.get("podcast_name", "Podcast")
    accent_color = config.get("branding", {}).get("accent_color", "#132257")

    # Load previous episode data
    prev_data = load_pqs_data(prev_episode)
    if prev_data is None:
        print(f"ERROR: Could not find PQS data for episode {prev_episode}.")
        print(f"  Looked in: {REPORTS_DIR / f'episode_{prev_episode}' / 'pqs_v3_scores.json'}")
        sys.exit(1)

    print(f"Loaded PQS data for episode {prev_episode}: score {extract_composite_score(prev_data):.1f}")

    # Load trend data (up to 3 previous episodes)
    trend_data = []
    for ep_num in range(prev_episode - 2, prev_episode + 1):
        ep_data = load_pqs_data(ep_num)
        if ep_data is not None:
            score = extract_composite_score(ep_data)
            if score > 0:
                trend_data.append((ep_num, score))

    if trend_data:
        print(f"Trend data: {', '.join(f'ep{e}={s:.1f}' for e, s in trend_data)}")

    # Recording date
    recording_date = args.date or datetime.now().strftime("%B %d, %Y")

    # Generate HTML
    html = generate_html(
        next_episode=next_episode,
        prev_data=prev_data,
        trend_data=trend_data,
        recording_date=recording_date,
        podcast_name=podcast_name,
        accent_color=accent_color,
    )

    # Output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"pre_recording_brief_ep{next_episode}.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"One-pager generated: {output_path}")
    print(f"  Episode: {next_episode} (based on ep {prev_episode} data)")
    print(f"  Score: {extract_composite_score(prev_data):.1f} ({extract_tier(prev_data)})")
    print(f"  Target: {calculate_target(extract_composite_score(prev_data)):.0f}+")


if __name__ == "__main__":
    main()

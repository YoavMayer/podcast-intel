#!/usr/bin/env python3
"""
Panel Chemistry Analysis for Podcast Episodes.

Analyzes how different panelist combinations affect podcast quality metrics.
Reads PQS v3 data from episode reports and computes:
1. Panel Size Effect (1 vs 2 vs 3 vs 4 participants)
2. Host Presence Effect
3. Engagement by Configuration
4. Content by Configuration
5. Format Effect (Solo vs Duo vs Full Panel)
6. Context Effect (e.g., post-win vs post-loss for sports, etc.)
7. Optimal Configuration Recommendations

Unlike the podcast-specific version, this tool dynamically discovers
episode data from the reports directory rather than hardcoding episodes.

Usage:
    python tools/analyze_panel_chemistry.py
    python tools/analyze_panel_chemistry.py --reports-dir /path/to/reports
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


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


def load_json(path: str) -> dict | None:
    """Load a JSON file, return None if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return None


def discover_episodes(reports_dir: Path) -> dict[int, dict]:
    """
    Discover all episodes with PQS data from the reports directory.

    Scans for directories matching episode_N and loads available metrics.
    Returns a dict of episode_num -> episode_data.
    """
    episodes = {}

    for ep_dir in sorted(reports_dir.glob("episode_*")):
        try:
            ep_num = int(ep_dir.name.replace("episode_", ""))
        except ValueError:
            continue

        pqs_data = load_json(str(ep_dir / "pqs_v3_scores.json"))
        if not pqs_data:
            continue

        # Extract composite score
        composite = 0.0
        for key in ("composite_score", "compositeScore", "composite_pqs_v3"):
            if key in pqs_data:
                composite = float(pqs_data[key])
                break

        tier = pqs_data.get("tier", "?")

        # Extract component scores
        components = {}
        cs = pqs_data.get("component_scores", {})
        for key, val in cs.items():
            if isinstance(val, dict) and "score" in val:
                components[key] = float(val["score"])
            elif isinstance(val, (int, float)):
                components[key] = float(val)

        ds = pqs_data.get("domainScores", {})
        for key, val in ds.items():
            if isinstance(val, dict) and "score" in val:
                components[key] = float(val["score"])

        # Extract context info
        ctx = pqs_data.get("episode_context", {})
        duration_min = ctx.get("duration_minutes", 0)
        episode_format = ctx.get("format", "")

        # Load diarization data for panel info
        dia_summary = load_json(str(ep_dir / "diarization" / "diarization_summary.json"))
        n_speakers = dia_summary.get("n_speakers", 0) if dia_summary else 0
        speaker_names = list(dia_summary.get("speaker_name_map", {}).values()) if dia_summary else []

        # Load engagement metrics
        eng_data = load_json(str(ep_dir / "engagement_metrics.json"))

        # Load delivery metrics
        del_data = load_json(str(ep_dir / "delivery_metrics.json"))

        # Determine scores from components
        audio = 0.0
        delivery = 0.0
        structure = 0.0
        content = 0.0
        engagement = 0.0

        for key, val in components.items():
            key_lower = key.lower()
            if "audio" in key_lower:
                audio = val
            elif "deliver" in key_lower:
                delivery = val
            elif "struct" in key_lower:
                structure = val
            elif "content" in key_lower:
                content = val
            elif "engage" in key_lower:
                engagement = val

        # Determine format
        if not episode_format:
            if n_speakers == 1:
                episode_format = "solo"
            elif n_speakers == 2:
                episode_format = "duo"
            elif n_speakers == 3:
                episode_format = "trio"
            elif n_speakers >= 4:
                episode_format = "full_panel"
            else:
                episode_format = "unknown"

        episodes[ep_num] = {
            "number": ep_num,
            "title": pqs_data.get("episode", f"Episode {ep_num}"),
            "panel": speaker_names,
            "panel_size": n_speakers if n_speakers > 0 else 1,
            "format": episode_format,
            "pqs": composite,
            "tier": tier,
            "audio": audio,
            "delivery": delivery,
            "structure": structure,
            "content": content,
            "engagement": engagement,
            "duration_min": duration_min,
        }

    return episodes


def avg(values: list) -> float:
    """Return average of a list, or 0 if empty."""
    return round(sum(values) / len(values), 2) if values else 0


def compute_panel_size_analysis(episodes: dict) -> dict:
    """Group episodes by panel size and compute averages."""
    groups: dict[int, list] = {}
    for ep in episodes.values():
        size = ep["panel_size"]
        if size not in groups:
            groups[size] = []
        groups[size].append(ep)

    size_labels = {
        1: "Solo (1 person)",
        2: "Duo (2 people)",
        3: "Trio (3 people)",
        4: "Full Panel (4 people)",
    }

    result = {}
    for size in sorted(groups.keys()):
        eps = groups[size]
        result[size] = {
            "label": size_labels.get(size, f"{size} people"),
            "episode_count": len(eps),
            "episodes": [ep["number"] for ep in eps],
            "avg_pqs": avg([ep["pqs"] for ep in eps]),
            "avg_engagement": avg([ep["engagement"] for ep in eps]),
            "avg_content": avg([ep["content"] for ep in eps]),
            "avg_delivery": avg([ep["delivery"] for ep in eps]),
            "avg_structure": avg([ep["structure"] for ep in eps]),
            "avg_duration": avg([ep["duration_min"] for ep in eps]),
        }
    return result


def compute_format_analysis(episodes: dict) -> dict:
    """Determine which format is best for which type of content."""
    formats: dict[str, list] = {}
    for ep in episodes.values():
        fmt = ep["format"]
        if fmt not in formats:
            formats[fmt] = []
        formats[fmt].append(ep)

    results = {}
    for fmt, eps in formats.items():
        results[fmt] = {
            "format": fmt,
            "episode_count": len(eps),
            "avg_pqs": avg([e["pqs"] for e in eps]),
            "avg_engagement": avg([e["engagement"] for e in eps]),
            "avg_content": avg([e["content"] for e in eps]),
            "avg_delivery": avg([e["delivery"] for e in eps]),
            "avg_structure": avg([e["structure"] for e in eps]),
            "best_for": [],
            "worst_for": [],
        }

        r = results[fmt]
        if r["avg_content"] > 75:
            r["best_for"].append("Deep content analysis")
        if r["avg_engagement"] > 70:
            r["best_for"].append("Audience engagement")
        if r["avg_delivery"] > 80:
            r["best_for"].append("Delivery quality")
        if r["avg_engagement"] < 50:
            r["worst_for"].append("Engagement")
        if r["avg_structure"] < 65:
            r["worst_for"].append("Structure")

    return results


def compute_recommendations(
    episodes: dict,
    panel_size_data: dict,
    format_data: dict,
) -> list[dict]:
    """Generate optimal configuration recommendations."""
    recommendations = []

    if not panel_size_data:
        return recommendations

    # 1. Optimal overall configuration
    best_size = max(panel_size_data.items(), key=lambda x: x[1]["avg_pqs"])
    recommendations.append({
        "category": "optimal_overall",
        "recommendation": f"{best_size[1]['label']} produces the highest average PQS of {best_size[1]['avg_pqs']}",
        "confidence": "high" if best_size[1]["episode_count"] >= 2 else "moderate",
        "data_points": best_size[1]["episode_count"],
    })

    # 2. Best for engagement
    best_eng_size = max(panel_size_data.items(), key=lambda x: x[1]["avg_engagement"])
    recommendations.append({
        "category": "best_engagement",
        "recommendation": f"{best_eng_size[1]['label']} produces highest engagement ({best_eng_size[1]['avg_engagement']})",
        "confidence": "high",
        "data_points": best_eng_size[1]["episode_count"],
    })

    # 3. Best for content depth
    best_content_size = max(panel_size_data.items(), key=lambda x: x[1]["avg_content"])
    recommendations.append({
        "category": "best_content",
        "recommendation": f"{best_content_size[1]['label']} produces highest content score ({best_content_size[1]['avg_content']})",
        "confidence": "moderate",
        "data_points": best_content_size[1]["episode_count"],
    })

    # 4. Solo episodes warning
    if 1 in panel_size_data:
        solo = panel_size_data[1]
        recommendations.append({
            "category": "solo_warning",
            "recommendation": f"Solo episodes score significantly lower (PQS {solo['avg_pqs']}, Engagement {solo['avg_engagement']}). Use only when a panel is unavailable.",
            "confidence": "high",
            "data_points": solo["episode_count"],
        })

    # 5. Duration management
    long_eps = [ep for ep in episodes.values() if ep["duration_min"] > 70]
    if long_eps:
        recommendations.append({
            "category": "duration_management",
            "recommendation": f"{len(long_eps)} episodes exceeded 70 minutes. Use a timer to enforce the 55-65 minute target.",
            "confidence": "high",
            "data_points": len(long_eps),
        })

    return recommendations


def build_analysis_report(episodes: dict) -> dict:
    """Build the complete analysis report."""
    panel_size = compute_panel_size_analysis(episodes)
    format_analysis = compute_format_analysis(episodes)
    recommendations = compute_recommendations(episodes, panel_size, format_analysis)

    config = _load_config()
    podcast_name = config.get("podcast_name", "Podcast")

    report = {
        "report_metadata": {
            "title": f"Panel Chemistry Analysis - {podcast_name}",
            "generated_at": datetime.now().isoformat(),
            "framework": "PQS v3.0",
            "episodes_analyzed": len(episodes),
            "episode_range": f"{min(episodes.keys())}-{max(episodes.keys())}" if episodes else "none",
        },
        "episode_summary": {
            ep_num: {
                "number": ep["number"],
                "title": ep["title"],
                "panel_size": ep["panel_size"],
                "format": ep["format"],
                "pqs": ep["pqs"],
                "tier": ep["tier"],
                "engagement": ep["engagement"],
                "content": ep["content"],
                "delivery": ep["delivery"],
                "structure": ep["structure"],
                "duration_min": ep["duration_min"],
            }
            for ep_num, ep in sorted(episodes.items())
        },
        "analysis": {
            "panel_size_effect": panel_size,
            "format_effect": format_analysis,
            "recommendations": recommendations,
        },
    }

    return report


def generate_html_report(report: dict, episodes: dict) -> str:
    """Generate an HTML report with CSS charts."""

    config = _load_config()
    podcast_name = config.get("podcast_name", "Podcast")
    accent_color = config.get("branding", {}).get("accent_color", "#132257")

    def bar(value, max_val=100, color="#4CAF50"):
        width = min(value / max_val * 100, 100)
        return f'<div class="bar-container"><div class="bar" style="width:{width}%;background:{color}">{value}</div></div>'

    def color_for_score(score):
        if score >= 75:
            return "#4CAF50"
        elif score >= 65:
            return "#FF9800"
        else:
            return "#f44336"

    episode_rows = ""
    for ep_num in sorted(episodes.keys()):
        ep = episodes[ep_num]
        episode_rows += f"""
        <tr class="{'highlight-best' if ep['pqs'] >= 78 else 'highlight-worst' if ep['pqs'] < 60 else ''}">
            <td><strong>{ep['number']}</strong></td>
            <td>{ep['title'][:40]}</td>
            <td>{ep['panel_size']}</td>
            <td>{ep['format']}</td>
            <td>{bar(ep['pqs'], color=color_for_score(ep['pqs']))}</td>
            <td>{bar(ep['engagement'], color=color_for_score(ep['engagement']))}</td>
            <td>{bar(ep['content'], color=color_for_score(ep['content']))}</td>
            <td>{round(ep['duration_min'], 1)} min</td>
        </tr>"""

    panel_size_data = report["analysis"]["panel_size_effect"]
    panel_rows = ""
    for size in sorted(panel_size_data.keys()):
        d = panel_size_data[size]
        panel_rows += f"""
        <tr class="{'highlight-best' if d['avg_pqs'] >= 78 else ''}">
            <td><strong>{d['label']}</strong></td>
            <td>{d['episode_count']}</td>
            <td>{bar(d['avg_pqs'], color=color_for_score(d['avg_pqs']))}</td>
            <td>{bar(d['avg_engagement'], color=color_for_score(d['avg_engagement']))}</td>
            <td>{bar(d['avg_content'], color=color_for_score(d['avg_content']))}</td>
            <td>{bar(d['avg_delivery'], color=color_for_score(d['avg_delivery']))}</td>
        </tr>"""

    # Recommendations
    recs = report["analysis"]["recommendations"]
    rec_html = ""
    for rec in recs:
        rec_html += f"""
        <div class="recommendation-card">
            <h3>{rec['category'].replace('_', ' ').title()}</h3>
            <p>{rec['recommendation']}</p>
            <span class="confidence-badge">{rec.get('confidence', 'moderate')} confidence ({rec.get('data_points', '?')} episodes)</span>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Panel Chemistry Analysis - {podcast_name}</title>
<style>
    :root {{
        --accent: {accent_color};
        --green: #4CAF50;
        --orange: #FF9800;
        --red: #f44336;
        --gold: #e2b04a;
        --light: #f5f5f5;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: var(--light);
        color: #333;
        line-height: 1.6;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    header {{
        background: linear-gradient(135deg, var(--accent), #16213e);
        color: white;
        padding: 40px 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        text-align: center;
    }}
    header h1 {{ font-size: 2em; margin-bottom: 8px; }}
    header .meta {{ margin-top: 15px; font-size: 0.9em; opacity: 0.7; }}
    section {{
        background: white;
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }}
    section h2 {{
        color: var(--accent);
        border-bottom: 3px solid var(--gold);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
    th {{ background: var(--accent); color: white; padding: 12px 8px; text-align: left; }}
    td {{ padding: 10px 8px; border-bottom: 1px solid #eee; }}
    tr:hover {{ background: #f8f9fa; }}
    .highlight-best {{ background: #e8f5e9 !important; }}
    .highlight-worst {{ background: #ffebee !important; }}
    .bar-container {{
        background: #e0e0e0; border-radius: 4px;
        height: 24px; min-width: 80px;
    }}
    .bar {{
        height: 100%; border-radius: 4px;
        display: flex; align-items: center; justify-content: flex-end;
        padding: 0 6px; color: white; font-size: 0.8em; font-weight: 600;
        min-width: 30px;
    }}
    .recommendation-card {{
        background: #f8f9fa;
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    }}
    .recommendation-card h3 {{ color: var(--accent); margin-bottom: 8px; }}
    .confidence-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-top: 8px;
        background: #e3f2fd;
        color: #1565c0;
    }}
    footer {{
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 0.85em;
    }}
</style>
</head>
<body>
<div class="container">

<header>
    <h1>Panel Chemistry Analysis</h1>
    <div>{podcast_name}</div>
    <div class="meta">
        PQS v3.0 Framework | {len(episodes)} Episodes Analyzed | Generated {datetime.now().strftime('%Y-%m-%d')}
    </div>
</header>

<section>
    <h2>Episode Overview</h2>
    <table>
        <thead>
            <tr>
                <th>#</th><th>Title</th><th>Panel Size</th><th>Format</th>
                <th>PQS</th><th>Engagement</th><th>Content</th><th>Duration</th>
            </tr>
        </thead>
        <tbody>{episode_rows}</tbody>
    </table>
</section>

<section>
    <h2>Panel Size Effect</h2>
    <table>
        <thead>
            <tr>
                <th>Configuration</th><th>Episodes</th>
                <th>Avg PQS</th><th>Avg Engagement</th><th>Avg Content</th><th>Avg Delivery</th>
            </tr>
        </thead>
        <tbody>{panel_rows}</tbody>
    </table>
</section>

<section>
    <h2>Recommendations</h2>
    {rec_html}
</section>

<footer>
    Panel Chemistry Analysis - {podcast_name} Intelligence System<br/>
    PQS v3.0 Framework | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</footer>

</div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Panel Chemistry Analysis for podcast episodes"
    )
    parser.add_argument(
        "--reports-dir", type=str, default=None,
        help="Reports directory (default: PROJECT_ROOT/reports)"
    )

    args = parser.parse_args()
    reports_dir = Path(args.reports_dir) if args.reports_dir else REPORTS_DIR

    config = _load_config()
    podcast_name = config.get("podcast_name", "Podcast")

    print(f"Panel Chemistry Analysis - {podcast_name}")
    print("=" * 55)
    print()

    # Step 1: Discover episodes
    print("Step 1: Discovering episode data...")
    episodes = discover_episodes(reports_dir)
    if not episodes:
        print("  No episodes with PQS data found.")
        print(f"  Looked in: {reports_dir}/episode_*/pqs_v3_scores.json")
        sys.exit(1)
    print(f"  Found {len(episodes)} episodes: {sorted(episodes.keys())}")
    print()

    # Step 2: Build analysis
    print("Step 2: Computing analysis...")
    report = build_analysis_report(episodes)
    print("  Analysis complete")
    print()

    # Step 3: Save JSON report
    json_path = reports_dir / "panel_chemistry_analysis.json"
    print(f"Step 3: Saving JSON report to {json_path}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  JSON report saved ({json_path.stat().st_size} bytes)")
    print()

    # Step 4: Generate and save HTML report
    html_path = reports_dir / "panel_chemistry_analysis.html"
    print(f"Step 4: Generating HTML report to {html_path}")
    html = generate_html_report(report, episodes)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report saved ({html_path.stat().st_size} bytes)")
    print()

    # Step 5: Print summary
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    panel_data = report["analysis"]["panel_size_effect"]
    for size in sorted(panel_data.keys(), key=lambda s: panel_data[s]["avg_pqs"], reverse=True):
        d = panel_data[size]
        print(f"  {d['label']}: PQS {d['avg_pqs']}, Engagement {d['avg_engagement']}, Content {d['avg_content']}")

    print("\nDone.")


if __name__ == "__main__":
    main()

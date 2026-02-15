"""
Command-line interface for Podcast Intelligence System.

Usage:
    podcast-intel ingest          # Fetch RSS and download new episodes
    podcast-intel transcribe 42   # Transcribe episode 42
    podcast-intel analyze 42      # Run full analysis on episode 42
    podcast-intel report 42       # Generate reports for episode 42
    podcast-intel mock            # Generate mock data for testing
    podcast-intel watch           # Check RSS for new episodes (one-shot)
    podcast-intel watch --auto-analyze   # Check + trigger analysis pipeline
    podcast-intel watch --dry-run        # Preview without side effects
    podcast-intel watch --output-json    # JSON output for CI integration
    podcast-intel events check           # Check for recent community events
    podcast-intel events upcoming        # Show upcoming community events
    podcast-intel events briefing        # Generate briefing for latest event
"""

import argparse
import sys

from podcast_intel.config import get_config, load_podcast_yaml


def cmd_ingest(args):
    """Fetch RSS feed and download new episodes."""
    config = get_config()
    if not config.rss_url:
        print("ERROR: No RSS URL configured.")
        print("Set PODCAST_INTEL_RSS_URL or add rss_url to podcast.yaml")
        sys.exit(1)

    from podcast_intel.ingestion.rss_parser import parse_rss_feed
    episodes = parse_rss_feed(config.rss_url)
    print(f"Found {len(episodes)} episodes in RSS feed")


def cmd_transcribe(args):
    """Transcribe an episode."""
    print(f"Transcribing episode {args.episode}...")
    print("(Not yet implemented -- use tools/run_episode_analysis.py)")


def cmd_analyze(args):
    """Run analysis on an episode."""
    print(f"Analyzing episode {args.episode}...")
    print("(Not yet implemented -- use tools/run_episode_analysis.py)")


def cmd_report(args):
    """Generate reports for an episode."""
    print(f"Generating report for episode {args.episode}...")
    print("(Not yet implemented -- use tools/generate_one_pager.py)")


def cmd_mock(args):
    """Generate mock data for testing."""
    from podcast_intel.ingestion.mock_ingest import generate_mock_episodes, main as mock_main
    mock_main()


def cmd_watch(args):
    """Check RSS feed for new episodes and optionally trigger analysis."""
    config = get_config()

    from podcast_intel.triggers.rss_watcher import run_watch

    result = run_watch(config=config, once=True, dry_run=args.dry_run)

    # JSON output mode (for CI/automation)
    if args.output_json:
        print(result.to_json())
        if result.errors:
            sys.exit(1)
        sys.exit(0)

    # Human-readable output
    if result.errors:
        for err in result.errors:
            print(f"ERROR: {err}")
        sys.exit(1)

    if result.has_new_episodes:
        print(f"Found {len(result.episodes)} new episode(s):")
        for ep in result.episodes:
            duration_info = f", duration={ep.duration}" if ep.duration else ""
            print(f"  - {ep.title} (guid={ep.guid[:40]}){duration_info}")

        if args.dry_run:
            print("\n[dry-run] No actions taken.")
        elif args.auto_analyze:
            print("\nTriggering analysis pipeline...")
            # Import and run ingest for each new episode
            from podcast_intel.ingestion.rss_parser import parse_rss_feed
            for ep in result.episodes:
                print(f"  Ingesting: {ep.title}")
            print("(Full auto-analyze pipeline not yet wired -- use tools/run_episode_analysis.py)")
    else:
        print("No new episodes found.")
        if result.feed_title:
            print(f"Feed: {result.feed_title} ({result.total_feed_episodes} episodes)")
        print(f"Known episodes: {result.known_guid_count}")


# ---------------------------------------------------------------------------
#  Events subcommands
# ---------------------------------------------------------------------------

def _load_events_config():
    """
    Load community_events trigger configuration from podcast.yaml.

    Returns:
        Dictionary with the community_events trigger config

    Raises:
        SystemExit: If no community_events config is found
    """
    yaml_config = load_podcast_yaml()
    triggers = yaml_config.get("triggers", {})
    ce_config = triggers.get("community_events", {})

    if not ce_config:
        print("ERROR: No community_events trigger configured in podcast.yaml")
        print("Add a triggers.community_events section to your podcast.yaml")
        sys.exit(1)

    if not ce_config.get("enabled", True):
        print("WARNING: community_events trigger is disabled in podcast.yaml")

    return ce_config, yaml_config


def cmd_events_check(args):
    """Check for recent community events."""
    from podcast_intel.triggers.community_events import check_community_events

    ce_config, yaml_config = _load_events_config()
    result = check_community_events(ce_config)

    # JSON output mode (for CI/automation)
    if args.output_json:
        print(result.to_json())
        if result.errors:
            sys.exit(1)
        sys.exit(0)

    # Human-readable output
    if result.errors:
        for err in result.errors:
            print(f"ERROR: {err}")
        if not result.events:
            sys.exit(1)

    if result.has_events:
        print(f"Found {len(result.events)} event(s) via '{result.provider_name}':")
        for event in result.events:
            from podcast_intel.triggers.providers import get_provider
            provider_config = ce_config.get("provider_config", {})
            try:
                provider = get_provider(result.provider_name, provider_config)
                print(f"  - {provider.format_event(event)}")
            except Exception:
                print(f"  - [{event.status}] {event.summary}")
    else:
        print(f"No events found (provider: {result.provider_name}).")

    print(f"Checked at: {result.checked_at}")


def cmd_events_upcoming(args):
    """Show upcoming community events."""
    from podcast_intel.triggers.community_events import check_upcoming_events

    ce_config, yaml_config = _load_events_config()
    result = check_upcoming_events(ce_config)

    # JSON output mode
    if args.output_json:
        print(result.to_json())
        if result.errors:
            sys.exit(1)
        sys.exit(0)

    # Human-readable output
    if result.errors:
        for err in result.errors:
            print(f"ERROR: {err}")
        if not result.events:
            sys.exit(1)

    if result.has_events:
        print(f"Upcoming events ({result.provider_name}):")
        for event in result.events:
            date_str = event.date[:10] if event.date else "TBD"
            teams_str = " vs ".join(event.teams) if event.teams else event.summary
            comp_str = f" ({event.competition})" if event.competition else ""
            print(f"  - [{date_str}] {teams_str}{comp_str}")
    else:
        print(f"No upcoming events found (provider: {result.provider_name}).")


def cmd_events_briefing(args):
    """Generate briefing for a community event."""
    from podcast_intel.triggers.community_events import check_community_events
    from podcast_intel.triggers.briefing_generator import generate_briefing

    ce_config, yaml_config = _load_events_config()

    # Get events
    result = check_community_events(ce_config)

    if result.errors:
        for err in result.errors:
            print(f"WARNING: {err}")

    if not result.events:
        print("No events found to generate briefing for.")
        sys.exit(1)

    # Find target event
    target_event = None
    if args.event_id:
        for event in result.events:
            if event.event_id == args.event_id:
                target_event = event
                break
        if target_event is None:
            print(f"ERROR: Event with ID '{args.event_id}' not found.")
            print("Available event IDs:")
            for event in result.events:
                print(f"  - {event.event_id}: {event.summary}")
            sys.exit(1)
    else:
        # Use most recent finished event, or first event
        finished = [e for e in result.events if e.status == "FINISHED"]
        target_event = finished[0] if finished else result.events[0]

    print(f"Generating briefing for: {target_event.summary}")

    # Determine formats and output dir
    on_event = ce_config.get("on_event", {})
    briefing_config = on_event.get("briefing", {})
    formats = briefing_config.get("formats", ["html"])
    output_dir = briefing_config.get("output_dir", "reports/briefings")

    # Generate briefing
    file_paths = generate_briefing(
        event=target_event,
        config=yaml_config,
        formats=formats,
        output_dir=output_dir,
    )

    print(f"\nGenerated {len(file_paths)} file(s):")
    for fmt, path in file_paths.items():
        print(f"  - {fmt}: {path}")


def main():
    parser = argparse.ArgumentParser(
        prog="podcast-intel",
        description="Podcast Intelligence System -- analyze and improve your podcast",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    sub_ingest = subparsers.add_parser("ingest", help="Fetch RSS and download new episodes")
    sub_ingest.set_defaults(func=cmd_ingest)

    # transcribe
    sub_transcribe = subparsers.add_parser("transcribe", help="Transcribe an episode")
    sub_transcribe.add_argument("episode", type=int, help="Episode number")
    sub_transcribe.set_defaults(func=cmd_transcribe)

    # analyze
    sub_analyze = subparsers.add_parser("analyze", help="Run analysis on an episode")
    sub_analyze.add_argument("episode", type=int, help="Episode number")
    sub_analyze.set_defaults(func=cmd_analyze)

    # report
    sub_report = subparsers.add_parser("report", help="Generate reports for an episode")
    sub_report.add_argument("episode", type=int, help="Episode number")
    sub_report.set_defaults(func=cmd_report)

    # mock
    sub_mock = subparsers.add_parser("mock", help="Generate mock data for testing")
    sub_mock.set_defaults(func=cmd_mock)

    # watch
    sub_watch = subparsers.add_parser(
        "watch",
        help="Check RSS feed for new episodes",
    )
    sub_watch.add_argument(
        "--auto-analyze",
        action="store_true",
        default=False,
        help="Automatically trigger analysis pipeline for new episodes",
    )
    sub_watch.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would happen without taking action",
    )
    sub_watch.add_argument(
        "--output-json",
        action="store_true",
        default=False,
        help="Output result as JSON (for CI/automation)",
    )
    sub_watch.set_defaults(func=cmd_watch)

    # events -- subcommand group
    sub_events = subparsers.add_parser(
        "events",
        help="Community events: check, upcoming, briefing",
    )
    events_subparsers = sub_events.add_subparsers(
        dest="events_command",
        help="Events subcommands",
    )

    # events check
    sub_events_check = events_subparsers.add_parser(
        "check",
        help="Check for recent community events",
    )
    sub_events_check.add_argument(
        "--output-json",
        action="store_true",
        default=False,
        help="Output result as JSON (for CI/automation)",
    )
    sub_events_check.set_defaults(func=cmd_events_check)

    # events upcoming
    sub_events_upcoming = events_subparsers.add_parser(
        "upcoming",
        help="Show upcoming community events",
    )
    sub_events_upcoming.add_argument(
        "--output-json",
        action="store_true",
        default=False,
        help="Output result as JSON (for CI/automation)",
    )
    sub_events_upcoming.set_defaults(func=cmd_events_upcoming)

    # events briefing
    sub_events_briefing = events_subparsers.add_parser(
        "briefing",
        help="Generate briefing for a community event",
    )
    sub_events_briefing.add_argument(
        "--event-id",
        default=None,
        help="Specific event ID to generate briefing for (default: most recent)",
    )
    sub_events_briefing.set_defaults(func=cmd_events_briefing)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Handle events subcommand group
    if args.command == "events":
        if not hasattr(args, "events_command") or args.events_command is None:
            sub_events.print_help()
            sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

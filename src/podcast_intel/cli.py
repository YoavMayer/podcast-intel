"""
Command-line interface for Podcast Intelligence System.

Usage:
    podcast-intel ingest          # Fetch RSS and download new episodes
    podcast-intel transcribe 42   # Transcribe episode 42
    podcast-intel analyze 42      # Run full analysis on episode 42
    podcast-intel report 42       # Generate reports for episode 42
    podcast-intel mock            # Generate mock data for testing
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

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

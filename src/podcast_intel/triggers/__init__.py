"""
Trigger system for automated podcast monitoring and pipeline execution.

Provides configurable triggers that watch for new episodes via RSS feeds
and can automatically kick off ingestion, transcription, analysis, and
reporting pipelines.

Supported triggers:
- RSS watcher: Polls an RSS feed for new episodes by comparing GUIDs
  against known episodes in the database or JSON registry.

Configuration via podcast.yaml:
    triggers:
      rss_watch:
        enabled: true
        schedule_cron: "*/15 * * * *"
        on_new_episode:
          pipeline: [ingest, transcribe, analyze, report]
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TriggerState:
    """
    Represents the current state of a trigger.

    Tracks whether a trigger is enabled, when it last ran, how many
    times it has fired, and any accumulated errors.

    Attributes:
        name: Trigger identifier (e.g., "rss_watch")
        enabled: Whether the trigger is active
        last_run: Timestamp of most recent execution
        run_count: Total number of times this trigger has fired
        last_error: Most recent error message, if any
        metadata: Arbitrary key-value pairs for trigger-specific state
    """

    name: str
    enabled: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_run(self, error: Optional[str] = None) -> None:
        """
        Record a trigger execution.

        Args:
            error: Error message if the run failed, None for success
        """
        self.last_run = datetime.now()
        self.run_count += 1
        self.last_error = error


@dataclass
class TriggerConfig:
    """
    Configuration for a single trigger, parsed from podcast.yaml.

    Attributes:
        name: Trigger identifier
        enabled: Whether the trigger should be active
        schedule_cron: Cron expression for scheduling (e.g., "*/15 * * * *")
        pipeline: List of pipeline steps to run on trigger fire
        options: Additional trigger-specific options
    """

    name: str
    enabled: bool = True
    schedule_cron: str = "*/15 * * * *"
    pipeline: List[str] = field(default_factory=lambda: ["ingest"])
    options: Dict[str, Any] = field(default_factory=dict)


def load_triggers(config: Dict[str, Any]) -> List[TriggerConfig]:
    """
    Load trigger configurations from podcast.yaml config dict.

    Parses the ``triggers`` section of the configuration and returns
    a list of TriggerConfig objects for each defined trigger.

    Args:
        config: Parsed podcast.yaml configuration dictionary

    Returns:
        List of TriggerConfig objects for all defined triggers

    Example:
        >>> yaml_config = load_podcast_yaml()
        >>> triggers = load_triggers(yaml_config)
        >>> for t in triggers:
        ...     print(f"{t.name}: enabled={t.enabled}")
    """
    triggers_section = config.get("triggers", {})
    trigger_configs: List[TriggerConfig] = []

    for name, settings in triggers_section.items():
        if not isinstance(settings, dict):
            continue

        enabled = settings.get("enabled", True)
        schedule_cron = settings.get("schedule_cron", "*/15 * * * *")

        # Extract pipeline from on_new_episode or top-level pipeline
        on_new = settings.get("on_new_episode", {})
        if isinstance(on_new, dict):
            pipeline = on_new.get("pipeline", ["ingest"])
        else:
            pipeline = settings.get("pipeline", ["ingest"])

        # Collect remaining options
        reserved_keys = {"enabled", "schedule_cron", "on_new_episode", "pipeline"}
        options = {k: v for k, v in settings.items() if k not in reserved_keys}

        trigger_configs.append(
            TriggerConfig(
                name=name,
                enabled=enabled,
                schedule_cron=schedule_cron,
                pipeline=pipeline,
                options=options,
            )
        )

    return trigger_configs


__all__ = [
    "TriggerState",
    "TriggerConfig",
    "load_triggers",
]

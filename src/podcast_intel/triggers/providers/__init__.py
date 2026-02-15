"""
Provider registry for community event sources.

Maps provider names (from podcast.yaml) to their implementation classes.
Providers are lazily imported to avoid pulling in unnecessary dependencies.

Supported providers:
- ``football`` -- Football-data.org API for match results and fixtures

Example:
    >>> from podcast_intel.triggers.providers import get_provider
    >>> provider = get_provider("football", {"api_key_env": "FOOTBALL_DATA_API_KEY"})
    >>> events = provider.fetch_recent_events()
"""

from typing import Any, Dict

from podcast_intel.triggers.community_events import CommunityEventProvider


# ---------------------------------------------------------------------------
#  Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: Dict[str, str] = {
    "football": "podcast_intel.triggers.providers.football.FootballProvider",
}


def get_provider(
    name: str,
    config: Dict[str, Any],
) -> CommunityEventProvider:
    """
    Get an instantiated provider by name.

    Looks up the provider class in the registry, imports it lazily,
    and returns an initialized instance configured with the given
    provider_config dict from podcast.yaml.

    Args:
        name: Provider name as configured in podcast.yaml (e.g., "football")
        config: Provider-specific configuration dictionary

    Returns:
        An initialized CommunityEventProvider instance

    Raises:
        ValueError: If the provider name is not found in the registry

    Example:
        >>> provider = get_provider("football", {
        ...     "api_key_env": "FOOTBALL_DATA_API_KEY",
        ...     "team_id": 73,
        ... })
    """
    if name not in _PROVIDER_REGISTRY:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown community event provider '{name}'. "
            f"Available providers: {available}"
        )

    class_path = _PROVIDER_REGISTRY[name]
    module_path, class_name = class_path.rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)

    return provider_class(config)


def list_providers() -> Dict[str, str]:
    """
    List all registered provider names and their class paths.

    Returns:
        Dictionary mapping provider name to fully qualified class path
    """
    return dict(_PROVIDER_REGISTRY)


__all__ = [
    "get_provider",
    "list_providers",
]

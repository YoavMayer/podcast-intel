"""
Language presets for the Podcast Intelligence System.

A preset supplies the per-LANGUAGE defaults -- NLP model IDs and the filler-word
lexicon -- for a language code. Presets are shipped inside the package and are
never user-authored; the per-SHOW file is ``podcast.yaml``.

Preset files live beside this module as ``{language}.yaml`` and are selected by
``podcast.language`` in ``podcast.yaml`` (see :func:`podcast_intel.config.get_config`).

Example:
    >>> from podcast_intel.presets import AVAILABLE_PRESETS, load_preset
    >>> sorted(AVAILABLE_PRESETS)
    ['en', 'he']
    >>> load_preset("he")["models"]["transcription"]
    'ivrit-ai/whisper-large-v3-turbo'
"""

from functools import cache, lru_cache
from importlib import resources
from typing import Any

import yaml

#: Language code -> preset filename, resolved relative to this package.
AVAILABLE_PRESETS: dict[str, str] = {
    "en": "english.yaml",
    "he": "hebrew.yaml",
}


def _normalize(code: str) -> str:
    """Lowercase a language code and drop any region suffix ('he-IL' -> 'he')."""
    return str(code or "").strip().lower().replace("_", "-").split("-")[0]


@cache
def _read_preset(filename: str) -> str:
    """Read a preset file out of the installed package (works inside a wheel)."""
    return resources.files(__package__).joinpath(filename).read_text(encoding="utf-8")


def has_preset(code: str) -> bool:
    """Return True if a preset ships for this language code."""
    return _normalize(code) in AVAILABLE_PRESETS


def load_preset(code: str) -> dict[str, Any]:
    """
    Load the language preset for an ISO 639-1 code.

    Args:
        code: Language code, e.g. ``"en"`` or ``"he"``. Case-insensitive; a
            region suffix is ignored (``"he-IL"`` resolves to ``"he"``).

    Returns:
        The parsed preset mapping. Top-level keys are ``language``, ``models``
        (``transcription`` / ``ner`` / ``sentiment``) and ``filler_words``.

    Raises:
        ValueError: If no preset ships for that code. Use :func:`has_preset` to
            test first, or :func:`podcast_intel.config.get_config`, which falls
            back to the built-in defaults instead of raising.

    Example:
        >>> load_preset("he")["language"]
        'he'
    """
    key = _normalize(code)
    if key not in AVAILABLE_PRESETS:
        available = ", ".join(sorted(AVAILABLE_PRESETS))
        raise ValueError(
            f"No language preset for '{code}'. Available presets: {available}"
        )

    data = yaml.safe_load(_read_preset(AVAILABLE_PRESETS[key])) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Preset '{AVAILABLE_PRESETS[key]}' is not a YAML mapping")
    return data


__all__ = ["AVAILABLE_PRESETS", "load_preset", "has_preset"]

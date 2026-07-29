"""
Configuration management for the Podcast Intelligence System.

Provides centralized configuration using Pydantic for validation and
environment variable support.

``get_config()`` layers four sources, lowest precedence first:

1. Built-in defaults (the field defaults on :class:`Config`)
2. The language preset for ``podcast.language`` (``presets/{lang}.yaml``)
3. ``podcast.yaml`` in the current directory or a parent
4. Environment variables (``PODCAST_INTEL_*``) and ``.env``
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from podcast_intel.presets import has_preset, load_preset

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default data paths relative to project root
DB_PATH = PROJECT_ROOT / "data" / "db" / "podcast_intel.db"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"


def load_podcast_yaml(search_dir: Path | None = None) -> dict:
    """
    Load podcast.yaml configuration file.

    Searches from ``search_dir`` -- or, by default, the CURRENT WORKING
    DIRECTORY -- walking up to 3 parent directories. ``podcast.yaml`` is the
    user's per-show file and lives in the user's project, which is not where the
    package is installed; defaulting to ``PROJECT_ROOT`` would look inside
    site-packages for anyone who pip-installed podcast-intel. PROJECT_ROOT is
    still tried as a last resort so a source checkout keeps working when the
    process is started from elsewhere.

    Args:
        search_dir: Directory to start searching from

    Returns:
        Dictionary with podcast.yaml contents, or empty dict if not found
    """
    starts = [search_dir] if search_dir is not None else [Path.cwd(), PROJECT_ROOT]
    for start in starts:
        for parent in [start] + list(start.parents)[:3]:
            candidate = parent / "podcast.yaml"
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
    return {}


class Config(BaseSettings):
    """
    Application configuration with environment variable support.

    Instantiating ``Config()`` directly reads only environment variables, ``.env``
    and the field defaults, in that order of precedence. Use :func:`get_config` to
    also layer in the language preset and ``podcast.yaml``.

    Example:
        export PODCAST_INTEL_RSS_URL="https://example.com/feed.rss"
        export PODCAST_INTEL_DB_PATH="/custom/path/db.sqlite"
    """

    model_config = SettingsConfigDict(
        env_prefix="PODCAST_INTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
    )

    # Podcast identity
    language: str = Field(
        default="en",
        description="Podcast language (ISO 639-1 code, e.g. 'en', 'he')"
    )

    # RSS Feed
    rss_url: str = Field(
        default="",
        description="RSS feed URL for the podcast"
    )

    # Storage paths
    db_path: Path = Field(
        default=DB_PATH,
        description="Path to SQLite database file"
    )
    audio_dir: Path = Field(
        default=AUDIO_DIR,
        description="Directory for downloaded audio files"
    )
    embeddings_dir: Path = Field(
        default=EMBEDDINGS_DIR,
        description="Directory for vector store embeddings"
    )

    # Transcription settings
    transcription_model: str = Field(
        default="openai/whisper-large-v3-turbo",
        description="Whisper model to use for transcription"
    )
    transcription_device: str = Field(
        default="cuda",
        description="Device for transcription (cuda/cpu)"
    )
    transcription_compute_type: str = Field(
        default="float16",
        description="Compute type for faster-whisper"
    )

    # Diarization settings
    diarization_enabled: bool = Field(
        default=True,
        description="Enable speaker diarization"
    )
    huggingface_token: str | None = Field(
        default=None,
        description=(
            "Hugging Face token. NOT needed by the packaged diarizer "
            "(MFCC + spectral clustering); only tools/diarize_episode.py uses it."
        )
    )

    # Analysis settings.
    # NOTE: no packaged code reads the model IDs below -- the NER and sentiment
    # stages are planned, not shipped. Setting them changes nothing today.
    ner_model: str = Field(
        default="dslim/bert-base-NER",
        description="Named Entity Recognition model (reserved; NER is not implemented)"
    )
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="Sentiment model (reserved; sentiment analysis is not implemented)"
    )

    # Filler lexicon, normally resolved from the language preset.
    # Empty means "fall back to the built-in English list in
    # analysis/filler_detector.py".
    filler_words: list[str] = Field(
        default_factory=list,
        description="Filler words for the active language (resolved from the preset)"
    )

    # Search settings (reserved; semantic search is not implemented)
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Embedding model (reserved; semantic search is not implemented)"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Reranker model (reserved; semantic search is not implemented)"
    )

    # LLM settings (reserved; no packaged code calls an LLM)
    llm_provider: str = Field(
        default="openai",
        description="LLM provider (reserved; no packaged code calls an LLM)"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name (reserved; no packaged code calls an LLM)"
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for LLM provider (reserved; no packaged code calls an LLM)"
    )

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> bool:
        """Validate that all critical paths are accessible."""
        try:
            self.ensure_directories()
            return True
        except (PermissionError, OSError) as e:
            print(f"Error validating paths: {e}")
            return False


#: podcast.yaml path -> Config field name, for the keys Config actually models.
#: Dotted paths are walked through nested mappings.
_YAML_TO_FIELD: dict[str, str] = {
    "podcast.language": "language",
    "podcast.rss_url": "rss_url",
    "models.transcription": "transcription_model",
    "models.ner": "ner_model",
    "models.sentiment": "sentiment_model",
    "models.embedding": "embedding_model",
    "models.reranker": "reranker_model",
    "transcription.device": "transcription_device",
    "transcription.compute_type": "transcription_compute_type",
    "transcription.diarization_enabled": "diarization_enabled",
    "analysis.filler_words": "filler_words",
    "paths.db_path": "db_path",
    "paths.audio_dir": "audio_dir",
    "paths.embeddings_dir": "embeddings_dir",
}

#: Preset key -> Config field name.
_PRESET_TO_FIELD: dict[str, str] = {
    "language": "language",
    "models.transcription": "transcription_model",
    "models.ner": "ner_model",
    "models.sentiment": "sentiment_model",
    "filler_words": "filler_words",
}


def _dig(data: dict, dotted: str) -> Any:
    """Walk a dotted path through nested mappings; return None if absent."""
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _project(data: dict, mapping: dict[str, str]) -> dict[str, Any]:
    """Project a nested config mapping onto flat Config field names."""
    out: dict[str, Any] = {}
    for dotted, field_name in mapping.items():
        value = _dig(data, dotted)
        if value is not None:
            out[field_name] = value
    return out


def resolve_language(
    podcast_yaml: dict | None = None,
    env_language: str | None = None,
) -> str:
    """
    Decide which language preset to load.

    Environment (``PODCAST_INTEL_LANGUAGE``) wins over ``podcast.yaml``, which
    wins over the ``language`` field default.

    Args:
        podcast_yaml: Parsed podcast.yaml, or None to skip that layer
        env_language: Language already resolved from the environment, if any

    Returns:
        An ISO 639-1 language code
    """
    if env_language:
        return env_language
    yaml_language = _dig(podcast_yaml or {}, "podcast.language")
    if isinstance(yaml_language, str) and yaml_language:
        return yaml_language
    default = Config.model_fields["language"].default
    return str(default)


def get_config(search_dir: Path | None = None) -> Config:
    """
    Get the application configuration instance.

    Merges four layers, lowest precedence first:

    1. Built-in defaults (the field defaults on :class:`Config`)
    2. The language preset for the resolved language (``presets/{lang}.yaml``)
    3. ``podcast.yaml`` (see :func:`load_podcast_yaml`)
    4. Environment variables (``PODCAST_INTEL_*``) and ``.env``

    A language with no shipped preset is not an error: layer 2 is skipped and
    the built-in defaults stand.

    Args:
        search_dir: Directory to start the podcast.yaml search from

    Returns:
        Config: Application configuration

    Example:
        >>> config = get_config()          # doctest: +SKIP
        >>> config.transcription_model     # doctest: +SKIP
        'ivrit-ai/whisper-large-v3-turbo'
    """
    # Layer 4 first: pydantic-settings records which fields the environment
    # (or .env) actually set, and those must not be overwritten by layers 2-3.
    config = Config()
    env_fields = set(config.model_fields_set)

    podcast_yaml = load_podcast_yaml(search_dir)

    language = resolve_language(
        podcast_yaml,
        env_language=config.language if "language" in env_fields else None,
    )

    overrides: dict[str, Any] = {}
    if has_preset(language):
        overrides.update(_project(load_preset(language), _PRESET_TO_FIELD))
    overrides.update(_project(podcast_yaml, _YAML_TO_FIELD))

    for field_name, value in overrides.items():
        if field_name in env_fields:
            continue  # environment wins
        setattr(config, field_name, value)  # validate_assignment coerces

    config.ensure_directories()
    return config

"""
Configuration management for the Podcast Intelligence System.

Provides centralized configuration using Pydantic for validation and
environment variable support. Supports podcast.yaml for per-project settings.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default data paths relative to project root
DB_PATH = PROJECT_ROOT / "data" / "db" / "podcast_intel.db"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"


def load_podcast_yaml(search_dir: Optional[Path] = None) -> dict:
    """
    Load podcast.yaml configuration file.

    Searches for podcast.yaml starting from search_dir (or PROJECT_ROOT)
    and walking up to 3 parent directories.

    Args:
        search_dir: Directory to start searching from

    Returns:
        Dictionary with podcast.yaml contents, or empty dict if not found
    """
    start = search_dir or PROJECT_ROOT
    for parent in [start] + list(start.parents)[:3]:
        candidate = parent / "podcast.yaml"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


class Config(BaseSettings):
    """
    Application configuration with environment variable support.

    Configuration can be provided via:
    1. Environment variables (prefixed with PODCAST_INTEL_)
    2. .env file
    3. podcast.yaml
    4. Default values

    Example:
        export PODCAST_INTEL_RSS_URL="https://example.com/feed.rss"
        export PODCAST_INTEL_DB_PATH="/custom/path/db.sqlite"
    """

    model_config = SettingsConfigDict(
        env_prefix="PODCAST_INTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
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
    huggingface_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token for pyannote-audio"
    )

    # Analysis settings
    ner_model: str = Field(
        default="dslim/bert-base-NER",
        description="Named Entity Recognition model"
    )
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="Sentiment analysis model"
    )

    # Search settings
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Embedding model for semantic search"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Reranker model for search results"
    )

    # LLM settings for coaching/topics
    llm_provider: str = Field(
        default="openai",
        description="LLM provider (openai/anthropic/local)"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name"
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for LLM provider"
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


def get_config() -> Config:
    """
    Get the application configuration instance.

    Merges settings from environment variables, .env file,
    and podcast.yaml (if present).

    Returns:
        Config: Application configuration
    """
    config = Config()
    config.ensure_directories()
    return config

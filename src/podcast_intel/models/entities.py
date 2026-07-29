"""
Pydantic data models for episodes, segments, entities.

Defines type-safe data models with validation for all major entities
in the system. These models provide serialization, validation, and
type hints for database records and API responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class EpisodeType(str, Enum):
    """Episode type enumeration."""
    FULL = "full"
    TRAILER = "trailer"
    BONUS = "bonus"


class TranscriptionStatus(str, Enum):
    """Transcription processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EntityType(str, Enum):
    """Entity type enumeration."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    OTHER = "other"


class Language(str, Enum):
    """Language enumeration."""
    HEBREW = "he"
    ENGLISH = "en"
    MIXED = "mixed"


class Episode(BaseModel):
    """
    Episode data model.

    Represents a podcast episode with metadata from RSS feed
    and processing status.
    """
    id: int | None = None
    guid: str
    title: str
    description: str | None = None
    pub_date: datetime
    audio_url: str
    audio_path: str | None = None
    duration_seconds: int | None = None
    file_size_bytes: int | None = None
    episode_type: EpisodeType = EpisodeType.FULL
    transcription_status: TranscriptionStatus = TranscriptionStatus.PENDING
    pqs_score: float | None = Field(None, ge=0.0, le=100.0)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class Speaker(BaseModel):
    """
    Speaker/panelist data model.

    Represents an identified speaker across episodes.
    """
    id: int | None = None
    name: str
    name_localized: str | None = None
    voice_embedding: bytes | None = None
    is_host: bool = False
    created_at: datetime | None = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class Segment(BaseModel):
    """
    Transcript segment data model.

    Represents a single diarized segment with speaker attribution,
    timestamps, and transcript text.
    """
    id: int | None = None
    episode_id: int
    speaker_id: int | None = None
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    text: str
    word_count: int = Field(ge=0, default=0)
    language: Language = Language.ENGLISH
    sentiment_score: float | None = Field(None, ge=-1.0, le=1.0)
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    created_at: datetime | None = None

    @field_validator("end_time")
    @classmethod
    def validate_time_range(cls, v: float, info: ValidationInfo) -> float:
        """Validate end_time > start_time."""
        if "start_time" in info.data and v <= info.data["start_time"]:
            raise ValueError("end_time must be greater than start_time")
        return v

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class Entity(BaseModel):
    """
    Named entity data model.

    Represents a canonical entity (person, organization, etc.) with
    optional multilingual names and external identifiers.
    """
    id: int | None = None
    canonical_name: str
    name_localized: str | None = None
    entity_type: EntityType
    external_id: str | None = None
    metadata_json: dict[str, Any] | None = None
    created_at: datetime | None = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class EntityMention(BaseModel):
    """
    Entity mention in a segment.

    Links an entity to a specific segment with context.
    """
    id: int | None = None
    entity_id: int
    segment_id: int
    episode_id: int
    mention_text: str
    start_offset: int | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime | None = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class Metric(BaseModel):
    """
    Computed metric for episode or speaker.

    Stores analysis results like WPM, talk-time, filler rates, etc.
    """
    id: int | None = None
    episode_id: int
    speaker_id: int | None = None
    metric_name: str
    metric_value: float
    metric_unit: str | None = None
    computed_at: datetime | None = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class SilenceEvent(BaseModel):
    """
    Dead air or significant silence event.

    Represents gaps in speech for pacing analysis.
    """
    id: int | None = None
    episode_id: int
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    duration: float = Field(gt=0.0)
    event_type: str = "dead_air"
    preceding_speaker_id: int | None = None
    following_speaker_id: int | None = None
    created_at: datetime | None = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class CoachingNote(BaseModel):
    """
    LLM-generated coaching feedback for a speaker.

    Contains strengths, improvement areas, and trend observations.
    """
    id: int | None = None
    episode_id: int
    speaker_id: int
    strengths: list[str]
    improvements: list[str]
    trends: dict[str, Any] | None = None
    generated_by: str = "gpt-4o-mini"
    generated_at: datetime | None = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True

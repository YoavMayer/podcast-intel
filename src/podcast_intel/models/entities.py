"""
Pydantic data models for episodes, segments, entities.

Defines type-safe data models with validation for all major entities
in the system. These models provide serialization, validation, and
type hints for database records and API responses.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator


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
    PLAYER = "player"
    CLUB = "club"
    COMPETITION = "competition"
    MANAGER = "manager"
    VENUE = "venue"
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
    id: Optional[int] = None
    guid: str
    title: str
    description: Optional[str] = None
    pub_date: datetime
    audio_url: str
    audio_path: Optional[str] = None
    duration_seconds: Optional[int] = None
    file_size_bytes: Optional[int] = None
    episode_type: EpisodeType = EpisodeType.FULL
    transcription_status: TranscriptionStatus = TranscriptionStatus.PENDING
    pqs_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class Speaker(BaseModel):
    """
    Speaker/panelist data model.

    Represents an identified speaker across episodes.
    """
    id: Optional[int] = None
    name: str
    name_localized: Optional[str] = None
    voice_embedding: Optional[bytes] = None
    is_host: bool = False
    created_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class Segment(BaseModel):
    """
    Transcript segment data model.

    Represents a single diarized segment with speaker attribution,
    timestamps, and transcript text.
    """
    id: Optional[int] = None
    episode_id: int
    speaker_id: Optional[int] = None
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    text: str
    word_count: int = Field(ge=0, default=0)
    language: Language = Language.ENGLISH
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None

    @field_validator("end_time")
    @classmethod
    def validate_time_range(cls, v: float, info) -> float:
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

    Represents a canonical entity (player, club, etc.) with
    optional multilingual names and external identifiers.
    """
    id: Optional[int] = None
    canonical_name: str
    name_localized: Optional[str] = None
    entity_type: EntityType
    external_id: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class EntityMention(BaseModel):
    """
    Entity mention in a segment.

    Links an entity to a specific segment with context.
    """
    id: Optional[int] = None
    entity_id: int
    segment_id: int
    episode_id: int
    mention_text: str
    start_offset: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class Metric(BaseModel):
    """
    Computed metric for episode or speaker.

    Stores analysis results like WPM, talk-time, filler rates, etc.
    """
    id: Optional[int] = None
    episode_id: int
    speaker_id: Optional[int] = None
    metric_name: str
    metric_value: float
    metric_unit: Optional[str] = None
    computed_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class SilenceEvent(BaseModel):
    """
    Dead air or significant silence event.

    Represents gaps in speech for pacing analysis.
    """
    id: Optional[int] = None
    episode_id: int
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    duration: float = Field(gt=0.0)
    event_type: str = "dead_air"
    preceding_speaker_id: Optional[int] = None
    following_speaker_id: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class CoachingNote(BaseModel):
    """
    LLM-generated coaching feedback for a speaker.

    Contains strengths, improvement areas, and trend observations.
    """
    id: Optional[int] = None
    episode_id: int
    speaker_id: int
    strengths: List[str]
    improvements: List[str]
    trends: Optional[Dict[str, Any]] = None
    generated_by: str = "gpt-4o-mini"
    generated_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True

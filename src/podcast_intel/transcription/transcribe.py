"""
Abstract transcription interface.

Defines the interface for transcription implementations, allowing
pluggable backends (Whisper, cloud APIs, mocks) with consistent API.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from podcast_intel.presets import DEFAULT_LANGUAGE


class TranscriptionResult:
    """
    Container for transcription results.

    Attributes:
        segments: List of transcript segments with timestamps
        language: Detected language code
        duration: Audio duration in seconds
        diarization: Optional speaker diarization results
    """

    def __init__(
        self,
        segments: list[dict[str, Any]],
        language: str = DEFAULT_LANGUAGE,
        duration: float = 0.0,
        diarization: list[dict[str, Any]] | None = None
    ):
        self.segments = segments
        self.language = language
        self.duration = duration
        self.diarization = diarization or []


class TranscriptionInterface(ABC):
    """
    Abstract base class for transcription implementations.

    All transcription backends must implement this interface to ensure
    consistent behavior across the system.
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: str = DEFAULT_LANGUAGE,
        diarize: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio file with optional speaker diarization.

        Args:
            audio_path: Path to audio file
            language: ISO 639-1 code. Defaults to presets.DEFAULT_LANGUAGE,
                which is the language the rest of the config resolves from.
            diarize: Whether to perform speaker diarization

        Returns:
            TranscriptionResult with segments and metadata
        """
        pass

    @abstractmethod
    def get_word_timestamps(self, audio_path: Path) -> list[dict[str, Any]]:
        """
        Get word-level timestamps for audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of word timestamp dictionaries
        """
        pass

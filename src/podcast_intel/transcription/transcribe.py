"""
Abstract transcription interface.

Defines the interface for transcription implementations, allowing
pluggable backends (Whisper, cloud APIs, mocks) with consistent API.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional


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
        segments: List[Dict[str, Any]],
        language: str = "en",
        duration: float = 0.0,
        diarization: Optional[List[Dict[str, Any]]] = None
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
        language: str = "en",
        diarize: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio file with optional speaker diarization.

        Args:
            audio_path: Path to audio file
            language: Language code (default: "en" for English)
            diarize: Whether to perform speaker diarization

        Returns:
            TranscriptionResult with segments and metadata
        """
        pass

    @abstractmethod
    def get_word_timestamps(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Get word-level timestamps for audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of word timestamp dictionaries
        """
        pass

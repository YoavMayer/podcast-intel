"""
Transcription module for speech-to-text and speaker diarization.

Provides both production (faster-whisper) and mock implementations
for transcription and speaker separation.
"""

from podcast_intel.transcription.transcribe import TranscriptionInterface
from podcast_intel.transcription.whisper_transcriber import WhisperTranscriber
from podcast_intel.transcription.mock_transcribe import MockTranscriber

__all__ = ["TranscriptionInterface", "WhisperTranscriber", "MockTranscriber"]

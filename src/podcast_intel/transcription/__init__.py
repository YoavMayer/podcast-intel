"""
Transcription module for speech-to-text and speaker diarization.

Provides both production (faster-whisper) and mock implementations
for transcription and speaker separation.

``WhisperTranscriber`` needs the optional ``[transcription]`` extra
(``faster-whisper``, ``torch``). It is therefore imported LAZILY: importing
``podcast_intel.transcription`` on a core install must not raise
``ModuleNotFoundError``. The heavy dependency is only pulled in when the name is
actually touched::

    from podcast_intel.transcription import WhisperTranscriber   # imports faster_whisper
    from podcast_intel.transcription import MockTranscriber      # core deps only

The same rule applies to ``podcast_intel.transcription.diarize``, which needs
``librosa`` / ``soundfile`` / ``scikit-learn`` / ``numpy`` from the same extra and
is deliberately not re-exported here.
"""

from typing import TYPE_CHECKING, Any

from podcast_intel.transcription.mock_transcribe import MockTranscriber
from podcast_intel.transcription.transcribe import TranscriptionInterface

if TYPE_CHECKING:  # pragma: no cover - for type checkers only, never at runtime
    from podcast_intel.transcription.whisper_transcriber import WhisperTranscriber

_LAZY_ATTRS = {
    "WhisperTranscriber": "podcast_intel.transcription.whisper_transcriber",
}

__all__ = ["TranscriptionInterface", "WhisperTranscriber", "MockTranscriber"]


def __getattr__(name: str) -> Any:
    """Import optional-extra members on first access (PEP 562)."""
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list:
    return sorted(__all__)

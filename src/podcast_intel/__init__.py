"""
Podcast Intelligence System

An open-source framework for analyzing and improving podcasts.
Provides RSS ingestion, transcription with speaker diarization, delivery and
filler analysis, and PQS v3 quality scoring.
"""

__version__ = "0.2.0"
__author__ = "Podcast Intel Team"

from podcast_intel.config import Config

__all__ = ["Config", "__version__"]

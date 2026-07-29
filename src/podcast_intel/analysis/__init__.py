"""
Analysis module for NLP, metrics computation, and quality scoring.

Provides comprehensive analysis capabilities including NER, sentiment analysis,
filler detection, silence analysis, and overall quality scoring.
"""

from podcast_intel.analysis.filler_detector import detect_fillers
from podcast_intel.analysis.metrics import compute_episode_metrics
from podcast_intel.analysis.scorer import (
    PROFILE_VERSION,
    ProfileVersionError,
    check_comparable,
    compute_pqs,
    is_comparable,
)

__all__ = [
    "PROFILE_VERSION",
    "ProfileVersionError",
    "check_comparable",
    "compute_episode_metrics",
    "compute_pqs",
    "detect_fillers",
    "is_comparable",
]

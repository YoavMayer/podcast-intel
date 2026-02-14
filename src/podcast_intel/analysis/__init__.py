"""
Analysis module for NLP, metrics computation, and quality scoring.

Provides comprehensive analysis capabilities including NER, sentiment analysis,
filler detection, silence analysis, and overall quality scoring.
"""

from podcast_intel.analysis.metrics import compute_episode_metrics
from podcast_intel.analysis.filler_detector import detect_fillers
from podcast_intel.analysis.scorer import compute_pqs

__all__ = ["compute_episode_metrics", "detect_fillers", "compute_pqs"]

"""
Coaching module for LLM-based feedback generation.

Provides automated coaching notes for speakers based on metrics
and interruption analysis for panel dynamics.
"""

from podcast_intel.coaching.coach import generate_coaching_notes
from podcast_intel.coaching.interruptions import analyze_interruptions

__all__ = ["generate_coaching_notes", "analyze_interruptions"]

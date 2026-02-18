"""
Filler word detection.

Detects and counts filler words (e.g., "um", "like", "you know") in
transcript segments using a curated dictionary and regex patterns.

Tracks filler frequency and rate per speaker for coaching insights.
The filler word list is configurable via language presets (see presets/).
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict


# Default English filler words and discourse markers.
# For other languages, load filler lists from the preset YAML.
DEFAULT_FILLERS = [
    "um", "uh", "uh huh",
    "like",
    "you know",
    "I mean",
    "basically",
    "right?",
    "so",
    "well",
    "okay", "ok",
    "actually",
    "literally",
    "kind of", "kinda",
    "sort of", "sorta",
    "honestly",
    "I guess",
    "whatever",
]

# Hebrew filler words (mirrored from presets/hebrew.yaml).
HEBREW_FILLERS = [
    "\u05d0\u05de\u05de",       # ammm
    "\u05d0\u05d4\u05d4\u05d4", # ahhh
    "\u05db\u05d0\u05d9\u05dc\u05d5", # ke'ilu
    "\u05e0\u05d5",             # nu
    "\u05d9\u05e2\u05e0\u05d9", # ya'ani
    "\u05d1\u05d2\u05d3\u05d5\u05dc", # begadol
    "\u05d1\u05e7\u05d9\u05e6\u05d5\u05e8", # bekitzur
    "\u05e0\u05db\u05d5\u05df?", # nakhon?
    "\u05d0\u05d6",             # az
    "\u05d8\u05d5\u05d1",       # tov
    "\u05d0\u05d5\u05e7\u05d9\u05d9", # okay
    "\u05e9\u05de\u05e2",       # shma
    "\u05d1\u05e2\u05e6\u05dd", # be'etzem
    "\u05ea\u05e8\u05d0\u05d4", # tir'eh
    "\u05e4\u05e9\u05d5\u05d8", # pashut
]


def get_default_fillers(language: str = "en") -> List[str]:
    """Return the default filler word list for a given language.

    Args:
        language: ISO language code ("en" for English, "he" for Hebrew).

    Returns:
        List of filler word strings.
    """
    if language == "he":
        return list(HEBREW_FILLERS)
    # Default to English for any unrecognised language code.
    return list(DEFAULT_FILLERS)


def build_filler_pattern(filler_words: List[str]) -> re.Pattern:
    """Build a compiled regex pattern for filler detection.

    Multi-word fillers are sorted longest-first so that greedy alternation
    matches the longest candidate before falling back to shorter ones.
    Word boundaries (``\\b``) are used on both sides to avoid partial
    matches inside longer words.

    Args:
        filler_words: List of filler word strings to match.

    Returns:
        Compiled case-insensitive regex pattern.
    """
    # Sort by length descending so multi-word fillers match first.
    sorted_fillers = sorted(filler_words, key=len, reverse=True)
    # Escape each filler for regex safety, then join with alternation.
    escaped = [re.escape(f) for f in sorted_fillers]
    # Use word boundaries to avoid matching inside other words.
    combined = "|".join(escaped)
    pattern = re.compile(rf"\b(?:{combined})\b", re.IGNORECASE)
    return pattern


def detect_fillers_in_text(
    text: str,
    filler_words: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Find all filler word occurrences in *text*.

    Args:
        text: The transcript text to scan.
        filler_words: Custom filler list. Defaults to English fillers.

    Returns:
        List of dicts with keys ``word``, ``start_pos``, ``end_pos``.
    """
    if not text:
        return []

    if filler_words is None:
        filler_words = get_default_fillers("en")

    pattern = build_filler_pattern(filler_words)
    results: List[Dict[str, Any]] = []
    for match in pattern.finditer(text):
        results.append({
            "word": match.group().lower(),
            "start_pos": match.start(),
            "end_pos": match.end(),
        })
    return results


def count_fillers_in_text(
    text: str,
    filler_words: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Count each filler word's occurrences in *text*.

    Args:
        text: Transcript text.
        filler_words: Custom filler list. Defaults to English fillers.

    Returns:
        Dictionary mapping each found filler word (lower-cased) to its count.
    """
    hits = detect_fillers_in_text(text, filler_words)
    counts: Dict[str, int] = defaultdict(int)
    for hit in hits:
        counts[hit["word"]] += 1
    return dict(counts)


def compute_filler_rate(filler_count: int, duration_seconds: float) -> float:
    """Compute filler rate as fillers per minute.

    Args:
        filler_count: Total number of filler words detected.
        duration_seconds: Duration of the speech in seconds.

    Returns:
        Fillers per minute. Returns 0.0 when duration is zero or negative.
    """
    if duration_seconds <= 0:
        return 0.0
    return filler_count / (duration_seconds / 60.0)


def detect_fillers(
    segments: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Detect filler words across all segments and aggregate per speaker.

    Args:
        segments: List of transcript segments. Each segment must have
            ``speaker`` (str), ``text`` (str), ``start`` (float), and
            ``end`` (float) keys.

    Returns:
        Dictionary mapping speaker name to a dict with:
        - ``total_fillers``: total filler count
        - ``filler_counts``: per-filler-word counts
        - ``filler_rate``: fillers per minute
        - ``occurrences``: list of individual filler hits

    Example:
        >>> fillers = detect_fillers(segments)
        >>> print(fillers["Host"]["total_fillers"])
        >>> print(fillers["Host"]["filler_rate"])
    """
    speaker_data: Dict[str, Dict[str, Any]] = {}

    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        text = seg.get("text", "")
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        if speaker not in speaker_data:
            speaker_data[speaker] = {
                "total_fillers": 0,
                "filler_counts": defaultdict(int),
                "filler_rate": 0.0,
                "occurrences": [],
                "_total_duration": 0.0,
            }

        hits = detect_fillers_in_text(text)
        for hit in hits:
            speaker_data[speaker]["total_fillers"] += 1
            speaker_data[speaker]["filler_counts"][hit["word"]] += 1
            speaker_data[speaker]["occurrences"].append({
                "word": hit["word"],
                "start_pos": hit["start_pos"],
                "end_pos": hit["end_pos"],
                "segment_start": start,
            })

        speaker_data[speaker]["_total_duration"] += max(0.0, end - start)

    # Compute per-speaker filler rates and clean up internal fields.
    for speaker, data in speaker_data.items():
        duration = data.pop("_total_duration")
        data["filler_rate"] = compute_filler_rate(data["total_fillers"], duration)
        data["filler_counts"] = dict(data["filler_counts"])

    return speaker_data


def extract_filler_positions(text: str) -> List[Tuple[str, int]]:
    """Extract filler words with their character positions.

    This mirrors the behaviour of the reference implementation in
    ``mock_transcribe._find_filler_words_in_text`` but uses the regex-based
    approach for consistency.

    Args:
        text: Transcript text.

    Returns:
        List of ``(filler_text, position)`` tuples.
    """
    hits = detect_fillers_in_text(text)
    return [(h["word"], h["start_pos"]) for h in hits]

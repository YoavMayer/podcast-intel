"""
Episode and speaker metric computation.

Computes core metrics including:
- Episode duration
- Talk-time per speaker (absolute and percentage)
- Speaking pace (words per minute)
- Talk-time balance (Gini coefficient)
- Word count statistics
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict


def compute_episode_metrics(
    segments: List[Dict[str, Any]],
    episode_duration: float,
) -> Dict[str, Any]:
    """Compute all episode-level metrics from transcript segments.

    Args:
        segments: List of transcript segments. Each segment has keys
            ``speaker`` (str), ``text`` (str), ``start`` (float),
            ``end`` (float).
        episode_duration: Total episode duration in seconds.

    Returns:
        Dictionary with the following keys:
        - ``episode_duration``: supplied duration
        - ``total_talk_time``: sum of all segment durations
        - ``silence_time``: episode_duration minus total_talk_time
        - ``talk_time_per_speaker``: {speaker: seconds}
        - ``talk_time_pct``: {speaker: percentage}
        - ``speaking_pace``: {speaker: wpm}
        - ``word_counts``: {speaker: int}
        - ``total_words``: total word count
        - ``talk_time_balance``: Gini coefficient

    Example:
        >>> metrics = compute_episode_metrics(segments, 3600)
        >>> print(metrics["total_talk_time"])
    """
    if not segments:
        return {
            "episode_duration": episode_duration,
            "total_talk_time": 0.0,
            "silence_time": episode_duration,
            "talk_time_per_speaker": {},
            "talk_time_pct": {},
            "speaking_pace": {},
            "word_counts": {},
            "total_words": 0,
            "talk_time_balance": 0.0,
        }

    talk_times = compute_talk_time(segments)
    total_talk = sum(talk_times.values())
    word_counts = compute_word_counts(segments)
    pace = compute_speaking_pace(segments)
    balance = compute_talk_time_balance(talk_times)

    # Talk-time percentages.
    talk_pct: Dict[str, float] = {}
    if total_talk > 0:
        for speaker, secs in talk_times.items():
            talk_pct[speaker] = (secs / total_talk) * 100.0
    else:
        for speaker in talk_times:
            talk_pct[speaker] = 0.0

    return {
        "episode_duration": episode_duration,
        "total_talk_time": total_talk,
        "silence_time": max(0.0, episode_duration - total_talk),
        "talk_time_per_speaker": talk_times,
        "talk_time_pct": talk_pct,
        "speaking_pace": pace,
        "word_counts": word_counts,
        "total_words": sum(word_counts.values()),
        "talk_time_balance": balance,
    }


def compute_talk_time(segments: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute total talk time per speaker in seconds.

    Args:
        segments: List of transcript segments.

    Returns:
        Dictionary mapping speaker name to total seconds spoken.

    Example:
        >>> talk_time = compute_talk_time(segments)
        >>> print(talk_time["Host"])
    """
    times: Dict[str, float] = defaultdict(float)
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        duration = max(0.0, end - start)
        times[speaker] += duration
    return dict(times)


def compute_speaking_pace(segments: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute average words per minute (WPM) per speaker.

    For each segment the WPM is calculated as
    ``word_count / (duration_seconds / 60)``.  The per-speaker result is
    the weighted average across all of that speaker's segments.

    Args:
        segments: List of transcript segments.

    Returns:
        Dictionary mapping speaker name to average WPM.

    Example:
        >>> wpm = compute_speaking_pace(segments)
        >>> print(wpm["Host"])
    """
    speaker_words: Dict[str, int] = defaultdict(int)
    speaker_duration: Dict[str, float] = defaultdict(float)

    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        text = seg.get("text", "")
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        duration = max(0.0, end - start)

        words = len(text.split()) if text.strip() else 0
        speaker_words[speaker] += words
        speaker_duration[speaker] += duration

    pace: Dict[str, float] = {}
    for speaker in speaker_words:
        dur = speaker_duration[speaker]
        if dur > 0:
            pace[speaker] = speaker_words[speaker] / (dur / 60.0)
        else:
            pace[speaker] = 0.0

    return pace


def compute_talk_time_balance(talk_times: Dict[str, float]) -> float:
    """Compute Gini coefficient of talk-time distribution.

    A value of 0 indicates perfectly equal talk time across speakers;
    a value approaching 1 indicates one speaker dominates.

    Uses the formula:
        Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    where x_i are the sorted talk-time values and i is 1-indexed.

    Args:
        talk_times: Mapping of speaker name to total seconds spoken.

    Returns:
        Gini coefficient in the range [0, 1].  Returns 0.0 for
        zero or single-speaker cases.
    """
    values = sorted(talk_times.values())
    n = len(values)
    if n <= 1:
        return 0.0

    total = sum(values)
    if total == 0:
        return 0.0

    # i is 1-indexed
    weighted_sum = sum(i * x for i, x in enumerate(values, 1))
    gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n
    # Clamp to [0, 1] to handle floating-point edge cases.
    return max(0.0, min(1.0, gini))


def compute_word_counts(segments: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute total word count per speaker.

    Args:
        segments: List of transcript segments.

    Returns:
        Dictionary mapping speaker name to word count.
    """
    counts: Dict[str, int] = defaultdict(int)
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        text = seg.get("text", "")
        words = len(text.split()) if text.strip() else 0
        counts[speaker] += words
    return dict(counts)

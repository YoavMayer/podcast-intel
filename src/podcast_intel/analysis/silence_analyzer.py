"""
Dead air, micro-pause, and pacing analysis.

Analyzes silence patterns in episodes to detect:
- Dead air: Gaps >2 seconds between segments
- Micro-pauses: Natural pauses 0.3-2.0s between words
- Pacing patterns: Monotony risk and hesitation spikes

Critical metric for podcast quality scoring (Section B, Metric #4).
"""

from typing import List, Dict, Any, Optional
import math


def detect_silence_gaps(
    segments: List[Dict[str, Any]],
    min_gap: float = 0.5,
) -> List[Dict[str, Any]]:
    """Find gaps between consecutive segments that are >= *min_gap* seconds.

    Segments are assumed to be sorted by start time.

    Args:
        segments: Sorted list of transcript segments. Each segment has keys
            ``start`` (float), ``end`` (float), ``speaker`` (str), ``text`` (str).
        min_gap: Minimum gap duration in seconds to report.

    Returns:
        List of gap dicts with keys ``start``, ``end``, ``duration``,
        ``before_speaker``, ``after_speaker``.
    """
    if len(segments) < 2:
        return []

    gaps: List[Dict[str, Any]] = []
    for i in range(len(segments) - 1):
        gap_start = segments[i].get("end", 0.0)
        gap_end = segments[i + 1].get("start", 0.0)
        duration = gap_end - gap_start

        if duration >= min_gap:
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "duration": duration,
                "before_speaker": segments[i].get("speaker", "unknown"),
                "after_speaker": segments[i + 1].get("speaker", "unknown"),
            })

    return gaps


def detect_dead_air(
    segments: List[Dict[str, Any]],
    threshold: float = 5.0,
) -> List[Dict[str, Any]]:
    """Detect dead-air gaps (silence >= *threshold* seconds).

    This is a convenience wrapper around :func:`detect_silence_gaps`.

    Args:
        segments: Sorted list of segments.
        threshold: Minimum gap duration to classify as dead air (seconds).

    Returns:
        List of dead-air instances with ``start``, ``end``, ``duration``,
        ``before_speaker``, ``after_speaker``.
    """
    return detect_silence_gaps(segments, min_gap=threshold)


def analyze_micro_pauses(
    segments: List[Dict[str, Any]],
    min_pause: float = 0.3,
    max_pause: float = 2.0,
) -> List[Dict[str, Any]]:
    """Find gaps in the micro-pause range between segments.

    Micro-pauses are gaps with duration satisfying
    ``min_pause <= duration < max_pause`` (if *max_pause* is provided) or
    ``min_pause <= duration <= max_pause``.

    Args:
        segments: Sorted list of transcript segments.
        min_pause: Minimum pause duration in seconds.
        max_pause: Maximum pause duration in seconds.

    Returns:
        List of micro-pause dicts with ``start``, ``end``, ``duration``,
        ``before_speaker``, ``after_speaker``.
    """
    # Get all gaps that are at least min_pause long, then filter by max.
    all_gaps = detect_silence_gaps(segments, min_gap=min_pause)
    return [g for g in all_gaps if g["duration"] <= max_pause]


def compute_silence_density(
    segments: List[Dict[str, Any]],
    episode_duration: float,
) -> float:
    """Compute total silence time as a fraction of episode duration.

    Silence is defined as the time not covered by any segment.

    Args:
        segments: List of transcript segments.
        episode_duration: Total episode duration in seconds.

    Returns:
        Value between 0.0 and 1.0. Returns 0.0 when episode_duration
        is zero or negative.
    """
    if episode_duration <= 0:
        return 0.0

    total_talk = sum(
        max(0.0, seg.get("end", 0.0) - seg.get("start", 0.0))
        for seg in segments
    )
    silence = max(0.0, episode_duration - total_talk)
    return min(1.0, silence / episode_duration)


def compute_silence_stats(
    segments: List[Dict[str, Any]],
    episode_duration: float,
) -> Dict[str, Any]:
    """Return comprehensive silence statistics for an episode.

    Args:
        segments: Sorted list of transcript segments.
        episode_duration: Total episode duration in seconds.

    Returns:
        Dictionary with keys:
        - ``total_silence_seconds``: total silence in the episode
        - ``silence_density``: fraction of episode that is silence
        - ``gap_count``: number of gaps >= 0.5 s
        - ``avg_gap_duration``: mean gap duration (0.0 if no gaps)
        - ``max_gap_duration``: longest gap duration (0.0 if no gaps)
        - ``dead_air_count``: number of gaps >= 5.0 s
        - ``micro_pause_count``: number of gaps in [0.3, 2.0] s
    """
    density = compute_silence_density(segments, episode_duration)
    total_talk = sum(
        max(0.0, seg.get("end", 0.0) - seg.get("start", 0.0))
        for seg in segments
    )
    total_silence = max(0.0, episode_duration - total_talk) if episode_duration > 0 else 0.0

    gaps = detect_silence_gaps(segments, min_gap=0.5)
    dead_air = detect_dead_air(segments, threshold=5.0)
    micro = analyze_micro_pauses(segments, min_pause=0.3, max_pause=2.0)

    gap_durations = [g["duration"] for g in gaps]
    avg_gap = sum(gap_durations) / len(gap_durations) if gap_durations else 0.0
    max_gap = max(gap_durations) if gap_durations else 0.0

    return {
        "total_silence_seconds": total_silence,
        "silence_density": density,
        "gap_count": len(gaps),
        "avg_gap_duration": avg_gap,
        "max_gap_duration": max_gap,
        "dead_air_count": len(dead_air),
        "micro_pause_count": len(micro),
    }


def detect_speaker_gaps(
    segments: List[Dict[str, Any]],
    min_gap: float = 1.0,
) -> List[Dict[str, Any]]:
    """Find gaps that occur between *different* speakers.

    These transitions can indicate hesitation, topic shifts, or
    interviewer/interviewee turn-taking pauses.

    Args:
        segments: Sorted list of transcript segments.
        min_gap: Minimum gap duration in seconds.

    Returns:
        List of gap dicts (same format as :func:`detect_silence_gaps`)
        filtered to only include cross-speaker transitions.
    """
    all_gaps = detect_silence_gaps(segments, min_gap=min_gap)
    return [
        g for g in all_gaps
        if g["before_speaker"] != g["after_speaker"]
    ]


# ---- Legacy / advanced helpers kept for backward compatibility ----


def analyze_silence_and_pacing(
    segments: List[Dict[str, Any]],
    word_timestamps: Dict[int, List[Dict[str, Any]]],
    episode_duration: float,
) -> Dict[str, Any]:
    """Comprehensive silence and pacing analysis.

    Combines dead-air detection, micro-pause analysis per speaker,
    and pacing alerts (monotony / hesitation).

    Args:
        segments: List of diarized segments.
        word_timestamps: Word-level timestamps per speaker.
        episode_duration: Total episode duration in seconds.

    Returns:
        Dictionary with ``dead_air``, ``micro_pauses``, and ``pacing_alerts``.
    """
    dead = detect_dead_air(segments, threshold=2.0)
    dead_total = sum(d["duration"] for d in dead)

    # Per-speaker micro-pause analysis using word timestamps.
    micro_by_speaker: Dict[int, Dict[str, float]] = {}
    for spk_id, words in word_timestamps.items():
        pauses = _word_level_pauses(words, min_pause=0.3, max_pause=2.0)
        if pauses:
            avg_dur = sum(pauses) / len(pauses)
        else:
            avg_dur = 0.0
        micro_by_speaker[spk_id] = {
            "count": len(pauses),
            "avg_pause_duration": avg_dur,
        }

    # Pacing alerts.
    pacing_alerts: List[Dict[str, Any]] = []
    for spk_id, words in word_timestamps.items():
        pauses = _word_level_pauses(words, min_pause=0.3, max_pause=2.0)
        if len(pauses) >= 5:
            mono = detect_monotony_risk(pauses)
            if mono is not None:
                mono["speaker_id"] = spk_id
                pacing_alerts.append(mono)

    return {
        "dead_air": {
            "instances": dead,
            "count": len(dead),
            "total_seconds": dead_total,
        },
        "micro_pauses": micro_by_speaker,
        "pacing_alerts": pacing_alerts,
    }


def detect_monotony_risk(
    pause_durations: List[float],
    cv_threshold: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """Detect monotonous pacing from pause regularity.

    Low coefficient of variation indicates robotic, overly regular delivery.

    Args:
        pause_durations: List of pause durations in seconds.
        cv_threshold: CV threshold below which monotony is flagged.

    Returns:
        Alert dictionary if monotony detected, ``None`` otherwise.
    """
    if len(pause_durations) < 3:
        return None

    cv = compute_coefficient_of_variation(pause_durations)
    if cv < cv_threshold:
        return {
            "type": "monotony",
            "cv": cv,
            "threshold": cv_threshold,
            "message": f"Pacing is unusually regular (CV={cv:.3f} < {cv_threshold}). "
                       "Consider varying delivery rhythm.",
        }
    return None


def detect_hesitation_spikes(
    segments: List[Dict[str, Any]],
    word_timestamps: List[Dict[str, Any]],
    window_size: float = 30.0,
) -> List[Dict[str, Any]]:
    """Detect hesitation spikes in speech.

    Identifies windows where mid-sentence pause frequency exceeds 2x the
    speaker's average.

    Args:
        segments: Speaker segments.
        word_timestamps: Word-level timestamps.
        window_size: Sliding window size in seconds.

    Returns:
        List of hesitation spike alerts.
    """
    if not word_timestamps:
        return []

    pauses = _word_level_pauses(word_timestamps, min_pause=0.3, max_pause=2.0)
    if not pauses:
        return []

    # Get timestamps of pauses for windowed analysis.
    pause_times: List[float] = []
    for i in range(len(word_timestamps) - 1):
        gap = word_timestamps[i + 1].get("start", 0.0) - word_timestamps[i].get("end", 0.0)
        if 0.3 <= gap <= 2.0:
            pause_times.append(word_timestamps[i].get("end", 0.0))

    if not pause_times:
        return []

    total_duration = pause_times[-1] - pause_times[0] if len(pause_times) > 1 else 0.0
    if total_duration <= 0:
        return []

    avg_rate = len(pause_times) / (total_duration / window_size) if total_duration > 0 else 0.0

    spikes: List[Dict[str, Any]] = []
    start = pause_times[0]
    end_time = pause_times[-1]
    window_start = start

    while window_start + window_size <= end_time:
        window_end = window_start + window_size
        count_in_window = sum(
            1 for t in pause_times if window_start <= t < window_end
        )
        if avg_rate > 0 and count_in_window > 2 * avg_rate:
            spikes.append({
                "type": "hesitation_spike",
                "window_start": window_start,
                "window_end": window_end,
                "pause_count": count_in_window,
                "avg_rate": avg_rate,
            })
        window_start += window_size / 2  # 50% overlap

    return spikes


def compute_coefficient_of_variation(values: List[float]) -> float:
    """Compute coefficient of variation (CV = std_dev / mean).

    Args:
        values: List of numeric values.

    Returns:
        Coefficient of variation. Returns 0.0 for empty lists or
        when the mean is zero.
    """
    if not values:
        return 0.0

    n = len(values)
    mean = sum(values) / n
    if mean == 0:
        return 0.0

    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)
    return std_dev / mean


# ---- Internal helpers ----


def _word_level_pauses(
    words: List[Dict[str, Any]],
    min_pause: float = 0.3,
    max_pause: float = 2.0,
) -> List[float]:
    """Extract inter-word pause durations within a range.

    Args:
        words: Word-level timestamp dicts with ``start`` and ``end``.
        min_pause: Minimum pause to include.
        max_pause: Maximum pause to include.

    Returns:
        List of pause durations.
    """
    pauses: List[float] = []
    for i in range(len(words) - 1):
        gap = words[i + 1].get("start", 0.0) - words[i].get("end", 0.0)
        if min_pause <= gap <= max_pause:
            pauses.append(gap)
    return pauses

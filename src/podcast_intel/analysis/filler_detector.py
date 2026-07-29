"""
Filler word detection.

Detects and counts filler words (e.g. "um", "like", "you know", or the Hebrew
"כאילו", "יעני") in transcript segments using a regex built from a language's
filler lexicon.

The lexicon is NOT hard-coded here. It comes from the language presets
(``presets/{lang}.yaml``), so a filler list is written down exactly once and the
same words drive both ``Config.filler_words`` and this module's defaults. Pass
``language=`` to pick one, or ``filler_words=`` to supply your own.

Tracks filler frequency and rate per speaker for coaching insights.
"""

import re
from collections import defaultdict
from typing import Any

from podcast_intel.presets import DEFAULT_LANGUAGE, has_preset, load_preset


def get_default_fillers(language: str | None = None) -> list[str]:
    """Return the filler lexicon for a language, from its preset.

    Args:
        language: ISO 639-1 code, e.g. ``"he"`` or ``"en"``. ``None`` -- and any
            language with no shipped preset -- uses
            :data:`~podcast_intel.presets.DEFAULT_LANGUAGE`.

    Returns:
        List of filler word strings. A fresh list each call, so callers may
        mutate it.

    Example:
        >>> "כאילו" in get_default_fillers("he")
        True
        >>> "um" in get_default_fillers("en")
        True
    """
    code = language if language and has_preset(language) else DEFAULT_LANGUAGE
    words = load_preset(code).get("filler_words") or []
    return [str(word) for word in words]


def _is_word_char(char: str) -> bool:
    """True if ``char`` is a regex word character (Unicode-aware, so Hebrew counts)."""
    return bool(char) and (char.isalnum() or char == "_")


def _bounded(filler: str) -> str:
    """Escape one filler and add ``\\b`` only on the ends where it can match.

    ``\\b`` sits between a word character and a non-word character. Wrapping the
    whole alternation in ``\\b...\\b`` therefore made every filler that starts or
    ends in punctuation unmatchable: after the ``?`` of ``"right?"`` (or of the
    Hebrew ``"נכון?"``) the next character is a space, so both sides are
    non-word and the boundary can never hold. Those fillers were silently dead.

    Args:
        filler: A raw filler string.

    Returns:
        A regex fragment matching that filler with boundaries where they apply.
    """
    prefix = r"\b" if _is_word_char(filler[:1]) else ""
    suffix = r"\b" if _is_word_char(filler[-1:]) else ""
    return f"{prefix}{re.escape(filler)}{suffix}"


def build_filler_pattern(filler_words: list[str]) -> re.Pattern:
    """Build a compiled regex pattern for filler detection.

    Multi-word fillers are sorted longest-first so that greedy alternation
    matches the longest candidate before falling back to shorter ones.
    Word boundaries are applied per filler rather than around the whole
    alternation -- see :func:`_bounded` for why that distinction is load-bearing.

    Args:
        filler_words: List of filler word strings to match.

    Returns:
        Compiled case-insensitive regex pattern. Matches nothing if the list is
        empty.
    """
    # Sort by length descending so multi-word fillers match first.
    sorted_fillers = sorted(filler_words, key=len, reverse=True)
    if not sorted_fillers:
        # An empty alternation would compile to a pattern matching everywhere.
        return re.compile(r"(?!)")
    combined = "|".join(_bounded(f) for f in sorted_fillers)
    return re.compile(f"(?:{combined})", re.IGNORECASE)


def detect_fillers_in_text(
    text: str,
    filler_words: list[str] | None = None,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Find all filler word occurrences in *text*.

    Args:
        text: The transcript text to scan.
        filler_words: Custom filler list. When omitted, the lexicon for
            *language* is used.
        language: ISO 639-1 code selecting the preset lexicon. Defaults to
            :data:`~podcast_intel.presets.DEFAULT_LANGUAGE`.

    Returns:
        List of dicts with keys ``word``, ``start_pos``, ``end_pos``.
    """
    if not text:
        return []

    if filler_words is None:
        filler_words = get_default_fillers(language)

    pattern = build_filler_pattern(filler_words)
    results: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        results.append({
            "word": match.group().lower(),
            "start_pos": match.start(),
            "end_pos": match.end(),
        })
    return results


def count_fillers_in_text(
    text: str,
    filler_words: list[str] | None = None,
    language: str | None = None,
) -> dict[str, int]:
    """Count each filler word's occurrences in *text*.

    Args:
        text: Transcript text.
        filler_words: Custom filler list. When omitted, the lexicon for
            *language* is used.
        language: ISO 639-1 code selecting the preset lexicon.

    Returns:
        Dictionary mapping each found filler word (lower-cased) to its count.
    """
    hits = detect_fillers_in_text(text, filler_words, language)
    counts: dict[str, int] = defaultdict(int)
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
    segments: list[dict[str, Any]],
    filler_words: list[str] | None = None,
    language: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Detect filler words across all segments and aggregate per speaker.

    Args:
        segments: List of transcript segments. Each segment must have
            ``speaker`` (str), ``text`` (str), ``start`` (float), and
            ``end`` (float) keys.
        filler_words: Custom filler list, e.g. ``get_config().filler_words``.
            When omitted, the lexicon for *language* is used.
        language: ISO 639-1 code selecting the preset lexicon. Defaults to
            :data:`~podcast_intel.presets.DEFAULT_LANGUAGE`. Before this
            parameter existed the function always scanned with the English
            list, so a Hebrew episode scored zero fillers.

    Returns:
        Dictionary mapping speaker name to a dict with:
        - ``total_fillers``: total filler count
        - ``filler_counts``: per-filler-word counts
        - ``filler_rate``: fillers per minute
        - ``occurrences``: list of individual filler hits

    Example:
        >>> fillers = detect_fillers(segments, language="he")  # doctest: +SKIP
        >>> print(fillers["Host"]["total_fillers"])            # doctest: +SKIP
        >>> print(fillers["Host"]["filler_rate"])              # doctest: +SKIP
    """
    if filler_words is None:
        filler_words = get_default_fillers(language)

    speaker_data: dict[str, dict[str, Any]] = {}

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

        hits = detect_fillers_in_text(text, filler_words)
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


def extract_filler_positions(
    text: str,
    language: str | None = None,
) -> list[tuple[str, int]]:
    """Extract filler words with their character positions.

    This mirrors the behaviour of the reference implementation in
    ``mock_transcribe._find_filler_words_in_text`` but uses the regex-based
    approach for consistency.

    Args:
        text: Transcript text.
        language: ISO 639-1 code selecting the preset lexicon.

    Returns:
        List of ``(filler_text, position)`` tuples.
    """
    hits = detect_fillers_in_text(text, language=language)
    return [(h["word"], h["start_pos"]) for h in hits]

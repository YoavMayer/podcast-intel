"""
Tests for analysis/filler_detector.py.

Two things are pinned here:

* the lexicon comes from the language presets and ``detect_fillers`` can be
  pointed at a language -- before this it always scanned with the English list,
  so a Hebrew episode reported zero fillers;
* the ``\\b`` regression: wrapping the whole alternation in ``\\b...\\b`` made
  every filler ending in punctuation unmatchable, silently.
"""

import pytest

from podcast_intel.analysis.filler_detector import (
    build_filler_pattern,
    count_fillers_in_text,
    detect_fillers,
    detect_fillers_in_text,
    extract_filler_positions,
    get_default_fillers,
)
from podcast_intel.presets import DEFAULT_LANGUAGE, load_preset

HEBREW_YES = "נכון?"        # "right?" -- the Hebrew filler that ends in '?'
HEBREW_LIKE = "כאילו"       # "ke'ilu" -- the most common Hebrew filler


# ------------------------------------------------------------------ #
# 1. The lexicon is the preset's, not a hand-copied mirror
# ------------------------------------------------------------------ #

class TestDefaultFillers:
    @pytest.mark.parametrize("code", ["he", "en"])
    def test_lexicon_is_exactly_the_preset_list(self, code: str):
        assert get_default_fillers(code) == load_preset(code)["filler_words"]

    def test_no_language_means_the_default_language(self):
        assert get_default_fillers() == get_default_fillers(DEFAULT_LANGUAGE)
        assert HEBREW_LIKE in get_default_fillers()

    def test_unknown_language_falls_back_to_the_default_language(self):
        assert get_default_fillers("es") == get_default_fillers(DEFAULT_LANGUAGE)

    def test_the_two_lexicons_are_disjoint(self):
        assert set(get_default_fillers("he")).isdisjoint(get_default_fillers("en"))

    def test_the_returned_list_is_the_callers_to_mutate(self):
        first = get_default_fillers("he")
        first.append("not-a-filler")
        assert "not-a-filler" not in get_default_fillers("he")


# ------------------------------------------------------------------ #
# 2. The \b regression: fillers ending in punctuation
# ------------------------------------------------------------------ #

class TestTrailingBoundaryRegression:
    """``\\b`` only holds between a word and a non-word character.

    ``\\b(?:right\\?|um|so)\\b`` can therefore never match ``right?``: the
    character after ``?`` is a space, so both sides of the closing boundary are
    non-word. Measured before the fix, ``["right?", "um", "so"]`` over
    ``"so right? um ok"`` returned only ``['so', 'um']``.
    """

    def test_english_filler_ending_in_punctuation_is_found(self):
        found = detect_fillers_in_text("so right? um ok", ["right?", "um", "so"])
        assert [hit["word"] for hit in found] == ["so", "right?", "um"]

    def test_hebrew_filler_ending_in_punctuation_is_found(self):
        text = f"אז {HEBREW_YES} {HEBREW_LIKE} בסדר"
        found = detect_fillers_in_text(text, language="he")
        assert HEBREW_YES in [hit["word"] for hit in found]

    def test_the_hebrew_preset_really_ships_that_filler(self):
        """Guards against the test passing because the word quietly left the list."""
        assert HEBREW_YES in load_preset("he")["filler_words"]
        assert HEBREW_YES.endswith("?")

    def test_offsets_still_point_at_the_match(self):
        text = "so right? um ok"
        for hit in detect_fillers_in_text(text, ["right?", "um", "so"]):
            assert text[hit["start_pos"]:hit["end_pos"]].lower() == hit["word"]

    def test_a_filler_that_starts_with_punctuation_also_matches(self):
        assert count_fillers_in_text("well ...like it is", ["...like"]) == {"...like": 1}


# ------------------------------------------------------------------ #
# 3. Boundaries that MUST still hold
# ------------------------------------------------------------------ #

class TestWordBoundariesStillApply:
    def test_no_match_inside_a_longer_word(self):
        assert detect_fillers_in_text("umbrella soccer likeness", ["um", "so", "like"]) == []

    def test_hebrew_filler_does_not_match_inside_a_longer_word(self):
        assert detect_fillers_in_text(f"{HEBREW_LIKE}נו", [HEBREW_LIKE]) == []

    def test_longest_multi_word_filler_wins(self):
        found = detect_fillers_in_text("you know what", ["you know", "you"])
        assert [hit["word"] for hit in found] == ["you know"]

    def test_matching_is_case_insensitive(self):
        assert count_fillers_in_text("Um, UM, um", ["um"]) == {"um": 3}

    def test_an_empty_lexicon_matches_nothing(self):
        assert build_filler_pattern([]).search("um like you know so") is None
        assert detect_fillers_in_text("um like you know", []) == []

    def test_empty_text_is_not_an_error(self):
        assert detect_fillers_in_text("", ["um"]) == []


# ------------------------------------------------------------------ #
# 4. detect_fillers() can be pointed at a language
# ------------------------------------------------------------------ #

HEBREW_SEGMENTS = [
    {"speaker": "מנחה", "text": f"{HEBREW_LIKE} זה מה שאמרתי", "start": 0.0, "end": 30.0},
    {"speaker": "מנחה", "text": f"אז {HEBREW_YES} נמשיך", "start": 30.0, "end": 60.0},
    {"speaker": "אורח", "text": "בלי מילות מילוי כאן", "start": 60.0, "end": 90.0},
]

ENGLISH_SEGMENTS = [
    {"speaker": "Host", "text": "um so this is basically it", "start": 0.0, "end": 60.0},
]


class TestDetectFillersLanguage:
    def test_hebrew_segments_are_scanned_with_the_hebrew_lexicon(self):
        result = detect_fillers(HEBREW_SEGMENTS, language="he")
        assert result["מנחה"]["total_fillers"] >= 3
        assert HEBREW_LIKE in result["מנחה"]["filler_counts"]
        assert HEBREW_YES in result["מנחה"]["filler_counts"]
        assert result["אורח"]["total_fillers"] == 0

    def test_hebrew_segments_score_zero_under_the_english_lexicon(self):
        """The defect this parameter fixes: the wrong list finds nothing."""
        result = detect_fillers(HEBREW_SEGMENTS, language="en")
        assert sum(s["total_fillers"] for s in result.values()) == 0

    def test_the_default_language_is_used_when_none_is_given(self):
        assert detect_fillers(HEBREW_SEGMENTS) == detect_fillers(
            HEBREW_SEGMENTS, language=DEFAULT_LANGUAGE
        )

    def test_an_explicit_lexicon_overrides_the_language(self):
        result = detect_fillers(ENGLISH_SEGMENTS, filler_words=["basically"], language="he")
        assert result["Host"]["filler_counts"] == {"basically": 1}

    def test_filler_rate_is_per_minute_of_that_speaker(self):
        result = detect_fillers(ENGLISH_SEGMENTS, language="en")
        # 3 fillers ("um", "so", "basically") over exactly one minute
        assert result["Host"]["total_fillers"] == 3
        assert result["Host"]["filler_rate"] == pytest.approx(3.0)

    def test_extract_filler_positions_takes_a_language_too(self):
        positions = extract_filler_positions(HEBREW_SEGMENTS[0]["text"], language="he")
        assert positions == [(HEBREW_LIKE, 0)]

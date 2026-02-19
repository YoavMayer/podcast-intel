"""Tests for PQS v3 scoring engine (scorer.py)."""

import pytest

from podcast_intel.analysis.scorer import (
    _piecewise_linear,
    _clamp,
    _score,
    DOMAIN_WEIGHTS,
    AUDIO_WEIGHTS,
    DELIVERY_WEIGHTS,
    STRUCTURE_WEIGHTS,
    CONTENT_WEIGHTS,
    ENGAGEMENT_WEIGHTS,
    score_loudness_compliance,
    score_signal_to_noise,
    score_clipping_index,
    score_loudness_range,
    score_duration_fit,
    score_filler_rate,
    score_filler_trajectory,
    score_monotony_risk,
    score_opinion_fact_ratio,
    score_conversational_energy,
    score_q4_sustain,
    score_dropoff_risk_index,
    compute_audio_domain,
    compute_delivery_domain,
    compute_structure_domain,
    compute_content_domain,
    compute_engagement_domain,
    compute_pqs,
    compute_pqs_from_domain_scores,
    compute_domain_from_sub_scores,
    normalize_metric,
    compute_audio_quality_score,
    compute_content_depth_score,
    compute_structure_score,
    compute_delivery_score,
)


# ------------------------------------------------------------------ #
# 1. Piecewise-linear interpolation
# ------------------------------------------------------------------ #

class TestPiecewiseLinear:
    BP = [(0.0, 0), (5.0, 50), (10.0, 100)]

    def test_exact_breakpoints(self):
        assert _piecewise_linear(0.0, self.BP) == 0.0
        assert _piecewise_linear(5.0, self.BP) == 50.0
        assert _piecewise_linear(10.0, self.BP) == 100.0

    def test_between_breakpoints(self):
        assert _piecewise_linear(2.5, self.BP) == 25.0
        assert _piecewise_linear(7.5, self.BP) == 75.0

    def test_below_minimum_clamps_to_first_score(self):
        assert _piecewise_linear(-10.0, self.BP) == 0.0

    def test_above_maximum_clamps_to_last_score(self):
        assert _piecewise_linear(999.0, self.BP) == 100.0

    def test_single_segment(self):
        bp = [(0.0, 0), (10.0, 100)]
        assert _piecewise_linear(5.0, bp) == 50.0

    def test_flat_segment(self):
        """Two breakpoints with same x should return y1."""
        bp = [(0.0, 0), (5.0, 50), (5.0, 80), (10.0, 100)]
        assert _piecewise_linear(5.0, bp) == 50.0  # hits first match

    def test_non_monotonic_scores(self):
        """Inverted-U shape (score rises then falls)."""
        bp = [(0.0, 0), (5.0, 100), (10.0, 0)]
        assert _piecewise_linear(2.5, bp) == 50.0
        assert _piecewise_linear(7.5, bp) == 50.0


# ------------------------------------------------------------------ #
# 2. Clamp utility
# ------------------------------------------------------------------ #

class TestClamp:
    def test_within_range(self):
        assert _clamp(50.0) == 50.0

    def test_below_range(self):
        assert _clamp(-5.0) == 0.0

    def test_above_range(self):
        assert _clamp(120.0) == 100.0

    def test_custom_bounds(self):
        assert _clamp(5.0, 10.0, 20.0) == 10.0
        assert _clamp(25.0, 10.0, 20.0) == 20.0


# ------------------------------------------------------------------ #
# 3. _score convenience wrapper
# ------------------------------------------------------------------ #

class TestScoreWrapper:
    def test_result_is_rounded(self):
        result = _score(2.5, [(0.0, 0), (10.0, 100)])
        assert result == 25.0

    def test_result_clamped_to_0_100(self):
        # breakpoints that could yield > 100
        bp = [(0.0, 0), (5.0, 110)]
        result = _score(5.0, bp)
        assert result <= 100.0


# ------------------------------------------------------------------ #
# 4. Weight sanity checks
# ------------------------------------------------------------------ #

ALL_WEIGHTS = {
    "DOMAIN": DOMAIN_WEIGHTS,
    "AUDIO": AUDIO_WEIGHTS,
    "DELIVERY": DELIVERY_WEIGHTS,
    "STRUCTURE": STRUCTURE_WEIGHTS,
    "CONTENT": CONTENT_WEIGHTS,
    "ENGAGEMENT": ENGAGEMENT_WEIGHTS,
}


@pytest.mark.parametrize("name,weights", list(ALL_WEIGHTS.items()))
def test_weights_sum_to_one(name, weights):
    assert abs(sum(weights.values()) - 1.0) < 1e-9, f"{name} weights sum to {sum(weights.values())}"


def test_domain_weights_have_five_entries():
    assert len(DOMAIN_WEIGHTS) == 5


# ------------------------------------------------------------------ #
# 5. Representative sub-metric scoring functions
# ------------------------------------------------------------------ #

class TestAudioSubMetrics:
    def test_loudness_compliance_perfect(self):
        """Exactly -16 LUFS = 0 deviation => 100."""
        assert score_loudness_compliance(-16.0) == 100.0

    def test_loudness_compliance_far_off(self):
        """10+ LUFS deviation => 0."""
        assert score_loudness_compliance(-26.0) == 0.0

    def test_snr_low(self):
        assert score_signal_to_noise(0.0) == 5.0

    def test_snr_high(self):
        assert score_signal_to_noise(25.0) == 100.0

    def test_clipping_zero(self):
        """Special-case: 0 clipping => perfect."""
        assert score_clipping_index(0.0) == 100.0

    def test_clipping_severe(self):
        assert score_clipping_index(1.0) == 0.0

    def test_loudness_range_sweet_spot(self):
        """7 dB LRA = 100."""
        assert score_loudness_range(7.0) == 100.0

    def test_loudness_range_extremes(self):
        assert score_loudness_range(0.0) == 0.0
        assert score_loudness_range(15.0) == 0.0


class TestDeliverySubMetrics:
    def test_filler_rate_clean(self):
        assert score_filler_rate(0.0) == 100.0

    def test_filler_rate_terrible(self):
        assert score_filler_rate(8.0) == 0.0

    def test_filler_trajectory_improving(self):
        assert score_filler_trajectory(-10.0) == 100.0

    def test_filler_trajectory_worsening(self):
        assert score_filler_trajectory(10.0) == 0.0

    def test_monotony_risk_sweet_spot(self):
        assert score_monotony_risk(0.22) == 100.0

    def test_monotony_risk_flat(self):
        assert score_monotony_risk(0.0) == 10.0


class TestStructureSubMetrics:
    def test_duration_fit_sweet_spot(self):
        assert score_duration_fit(60.0) == 100.0

    def test_duration_fit_too_short(self):
        assert score_duration_fit(30.0) == 0.0

    def test_duration_fit_too_long(self):
        assert score_duration_fit(80.0) == 0.0


class TestContentSubMetrics:
    def test_opinion_fact_ratio_balanced(self):
        assert score_opinion_fact_ratio(1.0) == 100.0

    def test_opinion_fact_ratio_extreme(self):
        assert score_opinion_fact_ratio(3.0) == 10.0


class TestEngagementSubMetrics:
    def test_conversational_energy_zero(self):
        assert score_conversational_energy(0.0) == 0.0

    def test_conversational_energy_high(self):
        assert score_conversational_energy(6.0) == 100.0

    def test_q4_sustain_good(self):
        """Sustain index <= 0.85 is ideal (100)."""
        assert score_q4_sustain(0.5) == 100.0
        assert score_q4_sustain(0.85) == 100.0

    def test_dropoff_risk_low(self):
        assert score_dropoff_risk_index(0.0) == 100.0

    def test_dropoff_risk_high(self):
        assert score_dropoff_risk_index(10.0) == 0.0


# ------------------------------------------------------------------ #
# 6. All scores are clamped 0-100
# ------------------------------------------------------------------ #

def test_scores_always_in_range():
    """Spot-check that a few scoring functions never exceed bounds."""
    for val in [-100, -10, 0, 5, 50, 100, 200]:
        assert 0 <= score_loudness_compliance(val) <= 100
        assert 0 <= score_signal_to_noise(val) <= 100
        assert 0 <= score_filler_rate(val) <= 100
        assert 0 <= score_duration_fit(val) <= 100
        assert 0 <= score_opinion_fact_ratio(val) <= 100
        assert 0 <= score_conversational_energy(val) <= 100


# ------------------------------------------------------------------ #
# 7. Domain compute functions return correct structure
# ------------------------------------------------------------------ #

SAMPLE_AUDIO = {"lufs": -16.0, "snr_db": 20.0, "clip_pct": 0.0, "lra_db": 7.0, "speech_band_pct": 75.0}
SAMPLE_DELIVERY = {
    "filler_rate": 3.0, "core_filler_rate": 0.5, "silence_density": 1.5,
    "dead_air_count": 0.0, "wpm_consistency": 5.0, "filler_trajectory": -3.5,
    "monotony_risk": 0.22, "interruption_rate": 5.0, "hesitation_rate": 2.0,
}
SAMPLE_STRUCTURE = {
    "duration_fit": 60.0, "segment_balance": 0.02, "fragmentation": 1.0,
    "monologue_depth": 8.0, "opening_tightness": 1.0, "transition_quality": 7.0,
    "closing_quality": 5.0, "intro_structure": 4.0,
}
SAMPLE_CONTENT = {
    "analytical_depth_ratio": 10.0, "content_words_per_minute": 120.0,
    "topic_coverage_breadth": 0.90, "discussion_density": 2.2,
    "domain_entity_density": 5.0, "match_reference_density": 3.0,
    "tactical_depth_density": 2.5, "opinion_fact_ratio": 1.0,
}
SAMPLE_ENGAGEMENT = {
    "conversational_energy": 5.0, "debate_indicator": 4.5,
    "q4_sustain": 0.85, "crosstalk_ratio": 15.0, "fatigue_signal": -0.5,
    "sentiment_flow_volatility": 0.7, "hook_score": 3.0,
    "memorable_moment_density": 5.0, "dropoff_risk_index": 3.0,
}


@pytest.mark.parametrize("compute_fn,metrics,weight_dict", [
    (compute_audio_domain, SAMPLE_AUDIO, AUDIO_WEIGHTS),
    (compute_delivery_domain, SAMPLE_DELIVERY, DELIVERY_WEIGHTS),
    (compute_structure_domain, SAMPLE_STRUCTURE, STRUCTURE_WEIGHTS),
    (compute_content_domain, SAMPLE_CONTENT, CONTENT_WEIGHTS),
    (compute_engagement_domain, SAMPLE_ENGAGEMENT, ENGAGEMENT_WEIGHTS),
])
def test_domain_compute_returns_correct_structure(compute_fn, metrics, weight_dict):
    result = compute_fn(metrics)
    assert "sub_scores" in result
    assert "weights" in result
    assert "domain_score" in result
    assert set(result["sub_scores"].keys()) == set(weight_dict.keys())
    assert result["weights"] is weight_dict
    assert 0 <= result["domain_score"] <= 100


# ------------------------------------------------------------------ #
# 8. Full compute_pqs integration
# ------------------------------------------------------------------ #

def test_compute_pqs_integration():
    result = compute_pqs(
        SAMPLE_AUDIO, SAMPLE_DELIVERY, SAMPLE_STRUCTURE,
        SAMPLE_CONTENT, SAMPLE_ENGAGEMENT,
    )
    assert "pqs_v3" in result
    assert "domains" in result
    assert "domain_weights" in result
    assert "formula" in result
    assert 0 <= result["pqs_v3"] <= 100
    assert set(result["domains"].keys()) == {"audio", "delivery", "structure", "content", "engagement"}

    # Verify the composite is correct weighted sum of domain scores
    expected = sum(
        result["domains"][d]["domain_score"] * DOMAIN_WEIGHTS[d]
        for d in DOMAIN_WEIGHTS
    )
    assert result["pqs_v3"] == round(expected, 2)


# ------------------------------------------------------------------ #
# 9. compute_pqs_from_domain_scores matches docstring example
# ------------------------------------------------------------------ #

def test_compute_pqs_from_domain_scores_docstring():
    result = compute_pqs_from_domain_scores(52.95, 79.10, 74.12, 68.96, 56.27)
    assert result == 68.39


def test_compute_pqs_from_domain_scores_all_zeros():
    assert compute_pqs_from_domain_scores(0, 0, 0, 0, 0) == 0.0


def test_compute_pqs_from_domain_scores_all_100():
    assert compute_pqs_from_domain_scores(100, 100, 100, 100, 100) == 100.0


# ------------------------------------------------------------------ #
# 10. compute_domain_from_sub_scores
# ------------------------------------------------------------------ #

def test_compute_domain_from_sub_scores():
    sub_scores = {"a": 80.0, "b": 60.0}
    weights = {"a": 0.6, "b": 0.4}
    # 80*0.6 + 60*0.4 = 48 + 24 = 72
    assert compute_domain_from_sub_scores(sub_scores, weights) == 72.0


# ------------------------------------------------------------------ #
# 11. normalize_metric
# ------------------------------------------------------------------ #

class TestNormalizeMetric:
    def test_normal_midpoint(self):
        assert normalize_metric(50.0, 0.0, 100.0) == 50.0

    def test_normal_at_min(self):
        assert normalize_metric(0.0, 0.0, 100.0) == 0.0

    def test_normal_at_max(self):
        assert normalize_metric(100.0, 0.0, 100.0) == 100.0

    def test_below_min_clamps(self):
        assert normalize_metric(-10.0, 0.0, 100.0) == 0.0

    def test_above_max_clamps(self):
        assert normalize_metric(200.0, 0.0, 100.0) == 100.0

    def test_inverse_false(self):
        assert normalize_metric(75.0, 0.0, 100.0, inverse=False) == 75.0

    def test_inverse_true(self):
        assert normalize_metric(75.0, 0.0, 100.0, inverse=True) == 25.0

    def test_inverse_at_min(self):
        assert normalize_metric(0.0, 0.0, 100.0, inverse=True) == 100.0

    def test_equal_range_returns_50(self):
        assert normalize_metric(42.0, 10.0, 10.0) == 50.0


# ------------------------------------------------------------------ #
# 12. Backward-compatible wrapper functions
# ------------------------------------------------------------------ #

def test_compute_audio_quality_score_wrapper():
    full = compute_audio_domain(SAMPLE_AUDIO)
    assert compute_audio_quality_score(SAMPLE_AUDIO) == full["domain_score"]


def test_compute_content_depth_score_wrapper():
    full = compute_content_domain(SAMPLE_CONTENT)
    assert compute_content_depth_score(SAMPLE_CONTENT) == full["domain_score"]


def test_compute_structure_score_wrapper():
    full = compute_structure_domain(SAMPLE_STRUCTURE)
    assert compute_structure_score(SAMPLE_STRUCTURE) == full["domain_score"]


def test_compute_delivery_score_wrapper():
    full = compute_delivery_domain(SAMPLE_DELIVERY)
    assert compute_delivery_score(SAMPLE_DELIVERY) == full["domain_score"]

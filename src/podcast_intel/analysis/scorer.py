"""
Podcast Quality Score v3.0 (PQS v3).

Computes the composite Podcast Quality Score from 0-100 based on
5 weighted domains and 39 sub-metrics:

  PQS_v3 = 0.10 * Audio + 0.25 * Delivery + 0.20 * Structure
         + 0.25 * Content + 0.20 * Engagement

Framework reference: reports/pqs_v3_framework.md
Predecessor: PQS v2.1 (4 domains, 21 sub-metrics)
"""

from typing import Dict, Any, List, Tuple


# ============================================================
# Utility
# ============================================================

def _piecewise_linear(value: float, breakpoints: List[Tuple[float, float]]) -> float:
    """Interpolate a score from piecewise-linear breakpoints.

    Args:
        value: Raw metric value.
        breakpoints: [(threshold, score), ...] sorted by threshold ascending.

    Returns:
        Interpolated score.
    """
    if value <= breakpoints[0][0]:
        return float(breakpoints[0][1])
    if value >= breakpoints[-1][0]:
        return float(breakpoints[-1][1])
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= value <= x1:
            if x1 == x0:
                return float(y1)
            t = (value - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return float(breakpoints[-1][1])


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _score(value: float, breakpoints: List[Tuple[float, float]]) -> float:
    """Convenience: piecewise-linear then clamp to [0, 100]."""
    return round(_clamp(_piecewise_linear(value, breakpoints)), 2)


# ============================================================
# Domain weights (v3)
# ============================================================

DOMAIN_WEIGHTS = {
    "audio": 0.10,
    "delivery": 0.25,
    "structure": 0.20,
    "content": 0.25,
    "engagement": 0.20,
}


# ============================================================
# 1. Audio Quality  (10%)  --  5 sub-metrics, ALL new in v3
# ============================================================

AUDIO_WEIGHTS = {
    "loudness_compliance": 0.30,
    "signal_to_noise": 0.25,
    "clipping_index": 0.15,
    "loudness_range": 0.15,
    "spectral_balance": 0.15,
}


def score_loudness_compliance(lufs: float) -> float:
    """Deviation from -16 LUFS target (Spotify/Apple standard)."""
    deviation = abs(lufs - (-16.0))
    return _score(deviation, [
        (0.0, 100), (1.0, 100), (3.0, 80), (6.0, 50), (10.0, 0),
    ])


def score_signal_to_noise(snr_db: float) -> float:
    """Speech clarity vs noise floor (dB)."""
    return _score(snr_db, [
        (0.0, 5), (0.5, 5), (2.0, 15), (6.0, 50), (15.0, 80), (25.0, 100),
    ])


def score_clipping_index(clip_pct: float) -> float:
    """Percentage of samples exceeding 0.99 peak amplitude."""
    if clip_pct == 0.0:
        return 100.0
    return _score(clip_pct, [
        (0.0, 100), (0.01, 90), (0.1, 60), (1.0, 0),
    ])


def score_loudness_range(lra_db: float) -> float:
    """Loudness Range (LRA). Inverted-U centered at 7.0 dB."""
    return _score(lra_db, [
        (0.0, 0), (3.0, 0), (5.0, 75), (7.0, 100), (10.0, 44), (15.0, 0),
    ])


def score_spectral_balance(speech_band_pct: float) -> float:
    """Energy concentration in 250-4000 Hz speech band (%)."""
    return _score(speech_band_pct, [
        (0.0, 0), (40.0, 0), (55.0, 50), (65.0, 80), (75.0, 100), (100.0, 100),
    ])


def compute_audio_domain(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Compute Audio Quality domain from raw measurements.

    Expected keys: lufs, snr_db, clip_pct, lra_db, speech_band_pct
    """
    scoring = {
        "loudness_compliance": score_loudness_compliance(metrics["lufs"]),
        "signal_to_noise": score_signal_to_noise(metrics["snr_db"]),
        "clipping_index": score_clipping_index(metrics["clip_pct"]),
        "loudness_range": score_loudness_range(metrics["lra_db"]),
        "spectral_balance": score_spectral_balance(metrics["speech_band_pct"]),
    }
    total = sum(scoring[k] * AUDIO_WEIGHTS[k] for k in AUDIO_WEIGHTS)
    return {"sub_scores": scoring, "weights": AUDIO_WEIGHTS,
            "domain_score": round(total, 2)}


# ============================================================
# 2. Delivery & Dynamics  (25%)  --  9 sub-metrics
# ============================================================

DELIVERY_WEIGHTS = {
    "filler_rate": 0.20,
    "core_filler_rate": 0.12,
    "silence_density": 0.12,
    "dead_air_count": 0.08,
    "wpm_consistency": 0.12,
    "filler_trajectory": 0.16,
    "monotony_risk": 0.10,
    "interruption_rate": 0.05,
    "hesitation_rate": 0.05,
}


def score_filler_rate(filler_pct: float) -> float:
    """Filler words as % of total words."""
    return _score(filler_pct, [
        (0.0, 100), (2.5, 100), (4.0, 80), (6.0, 50), (8.0, 0),
    ])


def score_core_filler_rate(core_pct: float) -> float:
    """Core hesitation fillers as %."""
    return _score(core_pct, [
        (0.0, 100), (0.5, 100), (1.0, 85), (2.0, 60), (3.0, 0),
    ])


def score_silence_density(silence_pct: float) -> float:
    """Non-speech time as % of episode duration."""
    return _score(silence_pct, [
        (0.0, 100), (1.5, 100), (2.5, 80), (4.0, 50), (6.0, 0),
    ])


def score_dead_air_count(events_per_hour: float) -> float:
    """Silences >5 s per hour of content."""
    return _score(events_per_hour, [
        (0.0, 100), (0.5, 95), (1.0, 80), (2.0, 60), (3.0, 35), (4.0, 0),
    ])


def score_wpm_consistency(cv_pct: float) -> float:
    """WPM coefficient of variation (%). Inverted-U at 5 %."""
    return _score(cv_pct, [
        (0.0, 30), (2.0, 60), (3.0, 80), (5.0, 100),
        (8.0, 80), (12.0, 40), (20.0, 0),
    ])


def score_filler_trajectory(change_pct: float) -> float:
    """Quarter-over-quarter filler-rate change (%). Negative = improving."""
    return _score(change_pct, [
        (-10.0, 100), (-3.5, 100), (-1.0, 92), (0.0, 82),
        (5.0, 45), (7.0, 15), (10.0, 0),
    ])


def score_monotony_risk(cv: float) -> float:
    """Pace variability CV (segment-level WPM). Inverted-U at 0.22."""
    return _score(cv, [
        (0.0, 10), (0.08, 10), (0.15, 80), (0.22, 100),
        (0.30, 80), (0.40, 40), (0.50, 0),
    ])


def score_interruption_rate(per_hour: float) -> float:
    """Cross-speaker interruptions per hour. Inverted-U at 5/h."""
    return _score(per_hour, [
        (0.0, 40), (2.0, 70), (5.0, 100), (8.0, 70), (15.0, 30), (20.0, 0),
    ])


def score_hesitation_rate(per_hour: float) -> float:
    """Within-speaker hesitations per hour. Lower is better."""
    return _score(per_hour, [
        (0.0, 100), (2.0, 100), (5.0, 80), (10.0, 50), (20.0, 0),
    ])


def compute_delivery_domain(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Compute Delivery & Dynamics domain from raw measurements."""
    fns = {
        "filler_rate": score_filler_rate,
        "core_filler_rate": score_core_filler_rate,
        "silence_density": score_silence_density,
        "dead_air_count": score_dead_air_count,
        "wpm_consistency": score_wpm_consistency,
        "filler_trajectory": score_filler_trajectory,
        "monotony_risk": score_monotony_risk,
        "interruption_rate": score_interruption_rate,
        "hesitation_rate": score_hesitation_rate,
    }
    scoring = {k: fn(metrics[k]) for k, fn in fns.items()}
    total = sum(scoring[k] * DELIVERY_WEIGHTS[k] for k in DELIVERY_WEIGHTS)
    return {"sub_scores": scoring, "weights": DELIVERY_WEIGHTS,
            "domain_score": round(total, 2)}


# ============================================================
# 3. Structure & Flow  (20%)  --  8 sub-metrics
# ============================================================

STRUCTURE_WEIGHTS = {
    "duration_fit": 0.1875,
    "segment_balance": 0.1125,
    "fragmentation": 0.15,
    "monologue_depth": 0.1125,
    "opening_tightness": 0.1875,
    "transition_quality": 0.10,
    "closing_quality": 0.07,
    "intro_structure": 0.08,
}


def score_duration_fit(minutes: float) -> float:
    """Episode duration. Sweet spot: 55-65 min."""
    return _score(minutes, [
        (0.0, 0), (30.0, 0), (45.0, 50), (55.0, 100), (65.0, 100),
        (70.0, 60), (80.0, 0),
    ])


def score_segment_balance(cv: float) -> float:
    """Speaker talk-time balance (CV). Lower = more balanced."""
    return _score(cv, [
        (0.0, 100), (0.02, 100), (0.04, 88), (0.06, 70),
        (0.08, 58), (0.12, 30), (0.20, 0),
    ])


def score_fragmentation(pct: float) -> float:
    """Fragmentation index (%). Lower = better coherence."""
    return _score(pct, [
        (0.0, 100), (1.0, 95), (2.0, 88), (3.0, 75),
        (5.0, 50), (7.0, 25), (10.0, 0),
    ])


def score_monologue_depth(pct: float) -> float:
    """Extended monologue share (%). Inverted-U at 8 %."""
    return _score(pct, [
        (0.0, 0), (2.0, 30), (4.0, 65), (6.0, 85),
        (8.0, 100), (12.0, 80), (15.0, 50), (20.0, 0),
    ])


def score_opening_tightness(ratio: float) -> float:
    """Q1 silence ratio vs rest. 1.0 = perfect parity."""
    return _score(ratio, [
        (0.5, 100), (1.0, 100), (1.1, 90), (1.15, 85),
        (1.25, 70), (1.4, 60), (1.5, 50), (2.0, 0),
    ])


def score_transition_quality(raw_score: float) -> float:
    """Transition quality (0-10 scale from structure gap analysis)."""
    return _score(raw_score, [
        (0.0, 0), (1.0, 15), (3.0, 50), (5.0, 75), (7.0, 100), (10.0, 100),
    ])


def score_closing_quality(elements: float) -> float:
    """Closing elements present (0-5: summary, callback, goodbye, preview, CTA)."""
    return _score(elements, [
        (0.0, 0), (1.0, 25), (2.0, 50), (3.0, 75), (4.0, 90), (5.0, 100),
    ])


def score_intro_structure(raw_score: float) -> float:
    """Intro structure score (0-5 scale)."""
    return _score(raw_score, [
        (0.0, 10), (1.0, 10), (2.0, 55), (3.0, 80), (4.0, 100), (5.0, 100),
    ])


def compute_structure_domain(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Compute Structure & Flow domain from raw measurements."""
    fns = {
        "duration_fit": score_duration_fit,
        "segment_balance": score_segment_balance,
        "fragmentation": score_fragmentation,
        "monologue_depth": score_monologue_depth,
        "opening_tightness": score_opening_tightness,
        "transition_quality": score_transition_quality,
        "closing_quality": score_closing_quality,
        "intro_structure": score_intro_structure,
    }
    scoring = {k: fn(metrics[k]) for k, fn in fns.items()}
    total = sum(scoring[k] * STRUCTURE_WEIGHTS[k] for k in STRUCTURE_WEIGHTS)
    return {"sub_scores": scoring, "weights": STRUCTURE_WEIGHTS,
            "domain_score": round(total, 2)}


# ============================================================
# 4. Content Depth  (25%)  --  8 sub-metrics
# ============================================================

CONTENT_WEIGHTS = {
    "analytical_depth_ratio": 0.21875,
    "content_words_per_minute": 0.175,
    "topic_coverage_breadth": 0.13125,
    "discussion_density": 0.175,
    "domain_entity_density": 0.08,
    "match_reference_density": 0.07,
    "tactical_depth_density": 0.08,
    "opinion_fact_ratio": 0.07,
}


def score_analytical_depth_ratio(pct: float) -> float:
    """Analytical segments as % of total. Higher = deeper analysis."""
    return _score(pct, [
        (0.0, 0), (3.0, 20), (5.0, 48), (8.0, 82),
        (10.0, 95), (12.0, 100),
    ])


def score_content_words_per_minute(cwpm: float) -> float:
    """Content words per minute (excluding fillers). Higher = denser."""
    return _score(cwpm, [
        (80.0, 0), (90.0, 20), (100.0, 44), (110.0, 82),
        (120.0, 100), (130.0, 100),
    ])


def score_topic_coverage_breadth(entropy: float) -> float:
    """Topic diversity (Shannon entropy). Higher = broader coverage."""
    return _score(entropy, [
        (0.0, 0), (0.4, 0), (0.6, 30), (0.75, 60),
        (0.80, 75), (0.85, 85), (0.90, 95), (0.95, 100),
    ])


def score_discussion_density(words_per_sec: float) -> float:
    """Discussion density (words per second). Higher = more substantial."""
    return _score(words_per_sec, [
        (1.2, 0), (1.5, 20), (1.8, 50), (2.0, 75),
        (2.2, 95), (2.4, 100),
    ])


def score_domain_entity_density(per_kw: float) -> float:
    """Domain-related entity mentions per 1000 words."""
    return _score(per_kw, [
        (0.0, 10), (1.5, 10), (3.0, 55), (5.0, 80),
        (8.0, 100),
    ])


def score_match_reference_density(per_kw: float) -> float:
    """Match event references per 1000 words."""
    return _score(per_kw, [
        (0.0, 0), (0.5, 0), (1.5, 50), (3.0, 80), (5.0, 100),
    ])


def score_tactical_depth_density(per_kw: float) -> float:
    """Tactical concept mentions per 1000 words."""
    return _score(per_kw, [
        (0.0, 10), (0.5, 10), (1.5, 55), (2.5, 80), (4.0, 100),
    ])


def score_opinion_fact_ratio(ratio: float) -> float:
    """Opinion-to-fact ratio. Inverted-U at 1.0 (balanced)."""
    return _score(ratio, [
        (0.0, 10), (0.3, 10), (0.5, 40), (0.8, 80), (1.0, 100),
        (1.3, 80), (2.0, 40), (3.0, 10),
    ])


def compute_content_domain(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Compute Content Depth domain from raw measurements.

    Expected keys: analytical_depth_ratio, content_words_per_minute,
    topic_coverage_breadth, discussion_density, domain_entity_density,
    match_reference_density, tactical_depth_density, opinion_fact_ratio
    """
    fns = {
        "analytical_depth_ratio": score_analytical_depth_ratio,
        "content_words_per_minute": score_content_words_per_minute,
        "topic_coverage_breadth": score_topic_coverage_breadth,
        "discussion_density": score_discussion_density,
        "domain_entity_density": score_domain_entity_density,
        "match_reference_density": score_match_reference_density,
        "tactical_depth_density": score_tactical_depth_density,
        "opinion_fact_ratio": score_opinion_fact_ratio,
    }
    scoring = {k: fn(metrics[k]) for k, fn in fns.items()}
    total = sum(scoring[k] * CONTENT_WEIGHTS[k] for k in CONTENT_WEIGHTS)
    return {"sub_scores": scoring, "weights": CONTENT_WEIGHTS,
            "domain_score": round(total, 2)}


# ============================================================
# 5. Engagement Proxies  (20%)  --  9 sub-metrics
# ============================================================

ENGAGEMENT_WEIGHTS = {
    "conversational_energy": 0.14,
    "debate_indicator": 0.105,
    "q4_sustain": 0.14,
    "crosstalk_ratio": 0.14,
    "fatigue_signal": 0.175,
    "sentiment_flow_volatility": 0.08,
    "hook_score": 0.07,
    "memorable_moment_density": 0.08,
    "dropoff_risk_index": 0.07,
}


def score_conversational_energy(segs_per_min: float) -> float:
    """Speaker turns per minute. Higher = more interactive."""
    return _score(segs_per_min, [
        (0.0, 0), (2.0, 0), (3.0, 20), (4.0, 48),
        (4.5, 65), (5.0, 80), (6.0, 100),
    ])


def score_debate_indicator(ratio: float) -> float:
    """Debate balance ratio. Inverted-U at ~4.5."""
    return _score(ratio, [
        (0.0, 0), (1.0, 30), (2.0, 60), (3.0, 80),
        (4.5, 100), (6.0, 70), (8.0, 40), (10.0, 0),
    ])


def score_q4_sustain(index: float) -> float:
    """Q4 quality relative to episode average. 1.0 = parity."""
    return _score(index, [
        (0.5, 100), (0.85, 100), (0.95, 90), (1.0, 60),
        (1.05, 50), (1.10, 40), (1.15, 30), (1.25, 10), (1.50, 0),
    ])


def score_crosstalk_ratio(pct: float) -> float:
    """Overlapping speech as % of total. Higher = more interactive."""
    return _score(pct, [
        (0.0, 0), (3.0, 10), (5.0, 30), (8.0, 50),
        (10.0, 60), (15.0, 85), (20.0, 97), (25.0, 100),
    ])


def score_fatigue_signal(change_rate: float) -> float:
    """Filler-rate change per quarter. Negative = improving."""
    return _score(change_rate, [
        (-1.0, 100), (-0.5, 95), (-0.3, 85), (-0.1, 60),
        (0.0, 50), (0.2, 30), (0.5, 10), (0.7, 0),
    ])


def score_sentiment_flow_volatility(volatility: float) -> float:
    """Emotional dynamics index. Inverted-U at 0.7."""
    return _score(volatility, [
        (0.0, 10), (0.2, 10), (0.3, 40), (0.5, 70), (0.7, 100),
        (0.9, 70), (1.2, 40), (1.5, 10),
    ])


def score_hook_score(hooks: float) -> float:
    """Opening hooks detected in first 5 min. Step function."""
    return _score(hooks, [
        (0.0, 10), (1.0, 40), (2.0, 70), (3.0, 85), (4.0, 100),
    ])


def score_memorable_moment_density(per_10min: float) -> float:
    """High-energy moments per 10-minute block."""
    return _score(per_10min, [
        (0.0, 10), (1.0, 10), (3.0, 55), (5.0, 80), (8.0, 100),
    ])


def score_dropoff_risk_index(index: float) -> float:
    """Listener drop-off risk. Lower = better retention."""
    return _score(index, [
        (0.0, 100), (3.0, 100), (5.0, 75), (7.0, 50), (10.0, 0),
    ])


def compute_engagement_domain(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Compute Engagement Proxies domain from raw measurements."""
    fns = {
        "conversational_energy": score_conversational_energy,
        "debate_indicator": score_debate_indicator,
        "q4_sustain": score_q4_sustain,
        "crosstalk_ratio": score_crosstalk_ratio,
        "fatigue_signal": score_fatigue_signal,
        "sentiment_flow_volatility": score_sentiment_flow_volatility,
        "hook_score": score_hook_score,
        "memorable_moment_density": score_memorable_moment_density,
        "dropoff_risk_index": score_dropoff_risk_index,
    }
    scoring = {k: fn(metrics[k]) for k, fn in fns.items()}
    total = sum(scoring[k] * ENGAGEMENT_WEIGHTS[k] for k in ENGAGEMENT_WEIGHTS)
    return {"sub_scores": scoring, "weights": ENGAGEMENT_WEIGHTS,
            "domain_score": round(total, 2)}


# ============================================================
# Composite scoring
# ============================================================

def compute_pqs(
    audio_metrics: Dict[str, float],
    delivery_metrics: Dict[str, float],
    structure_metrics: Dict[str, float],
    content_metrics: Dict[str, float],
    engagement_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Compute PQS v3 composite from raw metric values.

    Each dict maps sub-metric keys to raw values. See individual
    ``compute_*_domain`` functions for required keys.

    Returns:
        Dict with ``pqs_v3`` composite, per-domain breakdowns, and formula.
    """
    domains = {
        "audio": compute_audio_domain(audio_metrics),
        "delivery": compute_delivery_domain(delivery_metrics),
        "structure": compute_structure_domain(structure_metrics),
        "content": compute_content_domain(content_metrics),
        "engagement": compute_engagement_domain(engagement_metrics),
    }
    composite = sum(
        domains[d]["domain_score"] * DOMAIN_WEIGHTS[d] for d in DOMAIN_WEIGHTS
    )
    return {
        "pqs_v3": round(composite, 2),
        "domains": domains,
        "domain_weights": DOMAIN_WEIGHTS,
        "formula": (
            "0.10*Audio + 0.25*Delivery + 0.20*Structure"
            " + 0.25*Content + 0.20*Engagement"
        ),
    }


def compute_pqs_from_domain_scores(
    audio: float,
    delivery: float,
    structure: float,
    content: float,
    engagement: float,
) -> float:
    """Compute PQS v3 composite from pre-computed domain scores (0-100).

    >>> compute_pqs_from_domain_scores(52.95, 79.10, 74.12, 68.96, 56.27)
    68.39
    """
    return round(
        0.10 * audio
        + 0.25 * delivery
        + 0.20 * structure
        + 0.25 * content
        + 0.20 * engagement,
        2,
    )


def compute_domain_from_sub_scores(
    sub_scores: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Weighted sum of pre-computed sub-metric scores."""
    return round(sum(sub_scores[k] * weights[k] for k in weights), 2)


# ============================================================
# Backward-compatible helpers
# ============================================================

def compute_audio_quality_score(metrics: Dict[str, float]) -> float:
    """Compute Audio Quality domain score (0-100). Wrapper."""
    return compute_audio_domain(metrics)["domain_score"]


def compute_content_depth_score(metrics: Dict[str, float]) -> float:
    """Compute Content Depth domain score (0-100). Wrapper."""
    return compute_content_domain(metrics)["domain_score"]


def compute_structure_score(metrics: Dict[str, float]) -> float:
    """Compute Structure & Flow domain score (0-100). Wrapper."""
    return compute_structure_domain(metrics)["domain_score"]


def compute_delivery_score(metrics: Dict[str, float]) -> float:
    """Compute Delivery & Dynamics domain score (0-100). Wrapper."""
    return compute_delivery_domain(metrics)["domain_score"]


def normalize_metric(
    value: float,
    target_min: float,
    target_max: float,
    inverse: bool = False,
) -> float:
    """Normalize a raw value to 0-100 given a target range.

    Args:
        value: Raw metric value.
        target_min: Low end of target range (maps to 0 or 100).
        target_max: High end of target range (maps to 100 or 0).
        inverse: If True, higher values get lower scores.

    Returns:
        Score between 0 and 100.
    """
    if target_max == target_min:
        return 50.0
    ratio = (value - target_min) / (target_max - target_min)
    ratio = max(0.0, min(1.0, ratio))
    if inverse:
        ratio = 1.0 - ratio
    return round(ratio * 100, 2)

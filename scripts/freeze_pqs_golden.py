#!/usr/bin/env python3
"""
Freeze the PQS golden fixture for the current scoring profile.

The golden pins what ``compute_pqs`` returns for a fixed set of synthetic
inputs, so any change to a weight or a breakpoint table shows up as a diff
instead of silently rescoring every historical artifact.

Run it ONLY when a profile change is intended, and bump
``scorer.PROFILE_VERSION`` in the same commit:

    PYTHONPATH=src python scripts/freeze_pqs_golden.py

It refuses to overwrite a golden that still carries the current
PROFILE_VERSION, so an accidental run cannot launder an unintended change.

The inputs below are synthetic and domain-neutral on purpose. The golden is
tracked in a public repo, so it must never carry real episode measurements.

Superseded goldens are kept beside the current one as
``tests/fixtures/pqs_golden_<version>.json``. They are what makes a profile
boundary auditable after the fact: ``tests/test_pqs_profile_version.py`` diffs
the current golden against the previous one to prove the change stayed inside
the domain it claimed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from podcast_intel.analysis.scorer import PROFILE_VERSION, compute_pqs

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pqs_golden.json"

# Three points on the curve: a strong episode, a weak one, and the clamp edges.
CASES: dict[str, dict[str, dict[str, float]]] = {
    "strong": {
        "audio": {
            "lufs": -16.0, "snr_db": 20.0, "clip_pct": 0.0,
            "lra_db": 7.0, "speech_band_pct": 75.0,
        },
        "delivery": {
            "filler_rate": 3.0, "core_filler_rate": 0.5, "silence_density": 1.5,
            "dead_air_count": 0.0, "wpm_consistency": 5.0,
            "filler_trajectory": -3.5, "monotony_risk": 0.22,
            "interruption_rate": 5.0, "hesitation_rate": 2.0,
        },
        "structure": {
            "duration_fit": 60.0, "segment_balance": 0.02, "fragmentation": 1.0,
            "monologue_depth": 8.0, "opening_tightness": 1.0,
            "transition_quality": 7.0, "closing_quality": 5.0,
            "intro_structure": 4.0,
        },
        "content": {
            "analytical_depth_ratio": 10.0, "content_words_per_minute": 120.0,
            "topic_coverage_breadth": 0.90, "discussion_density": 2.2,
            "domain_entity_density": 5.0, "opinion_fact_ratio": 1.0,
        },
        "engagement": {
            "conversational_energy": 5.0, "debate_indicator": 4.5,
            "q4_sustain": 0.85, "crosstalk_ratio": 15.0, "fatigue_signal": -0.5,
            "sentiment_flow_volatility": 0.7, "hook_score": 3.0,
            "memorable_moment_density": 5.0, "dropoff_risk_index": 3.0,
        },
    },
    "weak": {
        "audio": {
            "lufs": -23.5, "snr_db": 4.0, "clip_pct": 0.35,
            "lra_db": 12.5, "speech_band_pct": 47.0,
        },
        "delivery": {
            "filler_rate": 6.8, "core_filler_rate": 2.4, "silence_density": 4.7,
            "dead_air_count": 3.1, "wpm_consistency": 14.0,
            "filler_trajectory": 6.0, "monotony_risk": 0.44,
            "interruption_rate": 17.0, "hesitation_rate": 13.5,
        },
        "structure": {
            "duration_fit": 38.0, "segment_balance": 0.13,
            "fragmentation": 6.4, "monologue_depth": 17.0,
            "opening_tightness": 1.45, "transition_quality": 2.0,
            "closing_quality": 1.0, "intro_structure": 1.5,
        },
        "content": {
            "analytical_depth_ratio": 3.5, "content_words_per_minute": 92.0,
            "topic_coverage_breadth": 0.55, "discussion_density": 1.45,
            "domain_entity_density": 1.2, "opinion_fact_ratio": 2.4,
        },
        "engagement": {
            "conversational_energy": 2.5, "debate_indicator": 8.5,
            "q4_sustain": 1.22, "crosstalk_ratio": 2.0, "fatigue_signal": 0.45,
            "sentiment_flow_volatility": 1.35, "hook_score": 0.0,
            "memorable_moment_density": 0.5, "dropoff_risk_index": 8.5,
        },
    },
    "edges": {
        "audio": {
            "lufs": -40.0, "snr_db": 0.0, "clip_pct": 5.0,
            "lra_db": 0.0, "speech_band_pct": 100.0,
        },
        "delivery": {
            "filler_rate": 0.0, "core_filler_rate": 9.0, "silence_density": 0.0,
            "dead_air_count": 25.0, "wpm_consistency": 0.0,
            "filler_trajectory": -50.0, "monotony_risk": 0.0,
            "interruption_rate": 40.0, "hesitation_rate": 0.0,
        },
        "structure": {
            "duration_fit": 0.0, "segment_balance": 0.0, "fragmentation": 25.0,
            "monologue_depth": 0.0, "opening_tightness": 5.0,
            "transition_quality": 10.0, "closing_quality": 0.0,
            "intro_structure": 5.0,
        },
        "content": {
            "analytical_depth_ratio": 0.0, "content_words_per_minute": 500.0,
            "topic_coverage_breadth": 0.0, "discussion_density": 9.9,
            "domain_entity_density": 0.0, "opinion_fact_ratio": 9.9,
        },
        "engagement": {
            "conversational_energy": 0.0, "debate_indicator": 25.0,
            "q4_sustain": 0.0, "crosstalk_ratio": 99.0, "fatigue_signal": -9.0,
            "sentiment_flow_volatility": 0.0, "hook_score": 9.0,
            "memorable_moment_density": 0.0, "dropoff_risk_index": 25.0,
        },
    },
}


def build_golden() -> dict[str, object]:
    """Compute the golden payload for every case under the current profile."""
    cases = {}
    for name, inputs in CASES.items():
        cases[name] = {
            "inputs": inputs,
            "expected": compute_pqs(
                inputs["audio"],
                inputs["delivery"],
                inputs["structure"],
                inputs["content"],
                inputs["engagement"],
            ),
        }
    return {
        "profile_version": PROFILE_VERSION,
        "note": (
            "Frozen output of compute_pqs() for the profile named above. "
            "Regenerate with scripts/freeze_pqs_golden.py ONLY alongside an "
            "intended PROFILE_VERSION bump. Inputs are synthetic."
        ),
        "cases": cases,
    }


def main() -> int:
    if GOLDEN_PATH.exists():
        existing = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        if existing.get("profile_version") == PROFILE_VERSION:
            print(
                f"refusing to overwrite: {GOLDEN_PATH.name} already pins profile "
                f"{PROFILE_VERSION!r}. Bump scorer.PROFILE_VERSION first.",
                file=sys.stderr,
            )
            return 1

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_golden()
    GOLDEN_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"froze {len(CASES)} case(s) at profile {PROFILE_VERSION} -> {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

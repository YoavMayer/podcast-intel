"""
Guards that make a PQS scoring-profile change impossible to land silently.

Removing a content sub-metric renormalises the remaining content weights, which
shifts every composite score. Historical scoring artifacts stay readable but
stop being comparable, and nothing about the numbers themselves says so -- a
rescale looks exactly like a quality regression.

Three guards close that gap:

* the golden pins what ``compute_pqs`` returns today, so a weight or breakpoint
  change surfaces as a fixture diff rather than a quiet rescore;
* ``check_comparable`` flags any artifact carrying a different profile version,
  or none at all;
* the blast-radius tests assert the four NON-content domain blocks are
  byte-identical to the golden, proving a content-domain change stayed in the
  content domain.

By design, a real profile change turns the golden tests RED. That is the alarm,
not a flake. Clearing it takes a deliberate PROFILE_VERSION bump plus
``scripts/freeze_pqs_golden.py`` -- the blast-radius tests must stay green
throughout, because they are what proves the change was confined.

That happened once already: profile 3.0.0 -> 3.1.0 removed the two
sport-specific content sub-metrics. The superseded golden is kept as
``fixtures/pqs_golden_3.0.0.json`` and section 5 diffs the two, so the boundary
stays auditable after the current golden has been regenerated.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from podcast_intel.analysis import scorer
from podcast_intel.analysis.scorer import (
    CONTENT_WEIGHTS,
    DOMAIN_WEIGHTS,
    NON_CONTENT_DOMAINS,
    PROFILE_VERSION,
    PROFILE_VERSION_KEY,
    ProfileVersionError,
    artifact_profile_version,
    check_comparable,
    compute_pqs,
    is_comparable,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_PATH = FIXTURES / "pqs_golden.json"

#: The superseded golden, kept so the 3.0.0 -> 3.1.0 boundary stays auditable.
PREVIOUS_GOLDEN_PATH = FIXTURES / "pqs_golden_3.0.0.json"

#: The two sport-specific content sub-metrics profile 3.1.0 removed.
REMOVED_CONTENT_SUB_METRICS = ("match_reference_density", "tactical_depth_density")


def _load_golden(path: Path = GOLDEN_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


GOLDEN = _load_golden()
PREVIOUS_GOLDEN = _load_golden(PREVIOUS_GOLDEN_PATH)
CASE_NAMES = sorted(GOLDEN["cases"])


def _recompute(case_name: str) -> dict[str, Any]:
    """Score a golden case's inputs under the current profile."""
    inputs = GOLDEN["cases"][case_name]["inputs"]
    return compute_pqs(
        inputs["audio"],
        inputs["delivery"],
        inputs["structure"],
        inputs["content"],
        inputs["engagement"],
    )


def _canonical(value: Any) -> str:
    """Serialize deterministically, so equality is genuinely byte-level."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


# ------------------------------------------------------------------ #
# 1. The version is real and reaches the caller
# ------------------------------------------------------------------ #

def test_compute_pqs_stamps_the_profile_version():
    """A score is useless for comparison unless it says which profile made it."""
    result = _recompute("strong")
    assert result[PROFILE_VERSION_KEY] == PROFILE_VERSION


def test_profile_version_is_a_non_empty_string():
    assert isinstance(PROFILE_VERSION, str) and PROFILE_VERSION.strip()


def test_every_domain_is_either_content_or_protected():
    """No domain may sit outside the content/non-content split unnoticed."""
    scored = set(_recompute("strong")["domains"])
    assert scored == {*NON_CONTENT_DOMAINS, "content"}
    assert set(DOMAIN_WEIGHTS) == scored


# ------------------------------------------------------------------ #
# 2. The golden: current scores are frozen
# ------------------------------------------------------------------ #

def test_golden_pins_the_current_profile_version():
    """A golden left behind at an older profile would vacuously pass below."""
    assert GOLDEN["profile_version"] == PROFILE_VERSION, (
        "tests/fixtures/pqs_golden.json pins profile "
        f"{GOLDEN['profile_version']!r} but the code is at {PROFILE_VERSION!r}. "
        "Regenerate it with scripts/freeze_pqs_golden.py."
    )


def test_golden_covers_more_than_one_case():
    assert len(CASE_NAMES) >= 3


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_golden_scores_are_unchanged(case_name: str):
    """Any weight or breakpoint edit lands here first, as a diff."""
    expected = GOLDEN["cases"][case_name]["expected"]
    assert _canonical(_recompute(case_name)) == _canonical(expected), (
        f"case {case_name!r} no longer scores as frozen. If the change was "
        "intended, bump scorer.PROFILE_VERSION and rerun "
        "scripts/freeze_pqs_golden.py."
    )


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_golden_composite_matches_its_domain_scores(case_name: str):
    """The frozen composite is internally consistent, not just a copied number."""
    expected = GOLDEN["cases"][case_name]["expected"]
    recomputed = sum(
        expected["domains"][d]["domain_score"] * DOMAIN_WEIGHTS[d]
        for d in DOMAIN_WEIGHTS
    )
    assert expected["pqs_v3"] == round(recomputed, 2)


# ------------------------------------------------------------------ #
# 3. Old artifacts are FLAGGED, never silently compared
# ------------------------------------------------------------------ #

def test_artifact_from_an_older_profile_is_flagged():
    old = {PROFILE_VERSION_KEY: "2.9.0", "pqs_v3": 71.4}
    assert artifact_profile_version(old) == "2.9.0"
    assert not is_comparable(old)
    with pytest.raises(ProfileVersionError):
        check_comparable(old)


def test_legacy_nested_profile_block_is_flagged():
    """The shape real historical pqs_recomputed.json artifacts actually carry."""
    legacy = {"profile": {"name": "example_show", "version": "3.2"}, "pqs_v3": 77.7}
    assert artifact_profile_version(legacy) == "3.2"
    assert not is_comparable(legacy)
    with pytest.raises(ProfileVersionError):
        check_comparable(legacy)


def test_unversioned_artifact_is_flagged_not_assumed_current():
    """Unknown must never degrade to 'current' -- that is the silent failure."""
    unversioned = {"pqs_v3": 68.4, "domains": {}}
    assert artifact_profile_version(unversioned) is None
    assert not is_comparable(unversioned)
    with pytest.raises(ProfileVersionError, match="no scoring-profile version"):
        check_comparable(unversioned)


def test_flag_message_names_both_profiles_and_the_source():
    """An alarm that does not say what to do gets ignored."""
    with pytest.raises(ProfileVersionError) as excinfo:
        check_comparable({PROFILE_VERSION_KEY: "2.9.0"}, source="reports/ep_190.json")
    message = str(excinfo.value)
    assert "reports/ep_190.json" in message
    assert "2.9.0" in message
    assert PROFILE_VERSION in message
    assert "not comparable" in message


def test_a_freshly_computed_score_passes_the_gate():
    """The guard must not flag the very scores it is meant to protect."""
    result = _recompute("weak")
    assert is_comparable(result)
    check_comparable(result)


def test_the_golden_itself_passes_the_gate():
    for case_name in CASE_NAMES:
        check_comparable(GOLDEN["cases"][case_name]["expected"])


@pytest.mark.parametrize("bad", [{"profile": {}}, {"profile": None}, {PROFILE_VERSION_KEY: ""}])
def test_malformed_version_blocks_are_flagged(bad: dict[str, Any]):
    assert artifact_profile_version(bad) is None
    with pytest.raises(ProfileVersionError):
        check_comparable(bad)


# ------------------------------------------------------------------ #
# 4. Blast radius: a content change must not move any other domain
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("case_name", CASE_NAMES)
@pytest.mark.parametrize("domain", NON_CONTENT_DOMAINS)
def test_non_content_domain_is_byte_identical_to_the_golden(case_name: str, domain: str):
    """These four must survive the content-domain change untouched.

    Byte-level on the whole domain block: sub-scores, sub-metric weights and the
    domain score. Any of them moving means a supposedly content-only edit
    reached further than the content domain.
    """
    expected = GOLDEN["cases"][case_name]["expected"]["domains"][domain]
    actual = _recompute(case_name)["domains"][domain]
    assert _canonical(actual) == _canonical(expected), (
        f"{domain!r} changed in case {case_name!r}. A content-domain change must "
        "not move any other domain; this is outside the intended blast radius."
    )


def test_non_content_domains_are_exactly_the_four_untouched_ones():
    assert set(NON_CONTENT_DOMAINS) == {"audio", "delivery", "structure", "engagement"}
    assert "content" not in NON_CONTENT_DOMAINS


@pytest.mark.parametrize("domain", NON_CONTENT_DOMAINS)
def test_non_content_domain_weight_is_unchanged(domain: str):
    """Renormalising sub-metric weights must not leak into the domain weights."""
    frozen = GOLDEN["cases"][CASE_NAMES[0]]["expected"]["domain_weights"]
    assert DOMAIN_WEIGHTS[domain] == frozen[domain]


def test_content_domain_weight_is_unchanged():
    """Content sub-weights renormalise to 1.0; the DOMAIN weight stays 0.25."""
    frozen = GOLDEN["cases"][CASE_NAMES[0]]["expected"]["domain_weights"]
    assert DOMAIN_WEIGHTS["content"] == frozen["content"] == 0.25


# ------------------------------------------------------------------ #
# 5. The 3.0.0 -> 3.1.0 boundary itself, checked against the OLD golden
# ------------------------------------------------------------------ #
#
# Section 4 compares the code against the CURRENT golden, so once the golden is
# regenerated it can only prove that today's code is self-consistent. These
# tests diff the two frozen goldens instead, so the claim "the de-sport change
# touched nothing but the content domain" stays falsifiable forever.

PREVIOUS_PROFILE_VERSION = "3.0.0"

#: Composite scores measured either side of the boundary, per golden case.
#: These are the numbers the CHANGELOG's "not comparable" warning is about.
BOUNDARY_COMPOSITES = {
    "strong": (95.78, 96.34),
    "weak": (25.40, 26.04),
    "edges": (43.77, 43.43),
}


def test_the_previous_golden_is_present_and_pins_the_previous_profile():
    assert PREVIOUS_GOLDEN["profile_version"] == PREVIOUS_PROFILE_VERSION
    assert PREVIOUS_GOLDEN["profile_version"] != PROFILE_VERSION
    assert sorted(PREVIOUS_GOLDEN["cases"]) == CASE_NAMES


@pytest.mark.parametrize("case_name", CASE_NAMES)
@pytest.mark.parametrize("domain", NON_CONTENT_DOMAINS)
def test_boundary_left_non_content_domains_byte_identical(case_name: str, domain: str):
    """Across the profile boundary, four of the five domains never moved."""
    before = PREVIOUS_GOLDEN["cases"][case_name]["expected"]["domains"][domain]
    after = GOLDEN["cases"][case_name]["expected"]["domains"][domain]
    assert _canonical(after) == _canonical(before), (
        f"{domain!r} moved between {PREVIOUS_PROFILE_VERSION} and {PROFILE_VERSION} "
        f"in case {case_name!r}; the de-sport change was supposed to be confined "
        "to the content domain."
    )


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_boundary_kept_every_surviving_content_sub_score(case_name: str):
    """Only the WEIGHTS changed: each surviving sub-metric's raw score is identical.

    That is what makes the composite shift a rescale rather than a re-measurement.
    """
    before = PREVIOUS_GOLDEN["cases"][case_name]["expected"]["domains"]["content"]
    after = GOLDEN["cases"][case_name]["expected"]["domains"]["content"]
    for metric in CONTENT_WEIGHTS:
        assert after["sub_scores"][metric] == before["sub_scores"][metric], metric


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_boundary_dropped_exactly_the_two_sport_sub_metrics(case_name: str):
    before = PREVIOUS_GOLDEN["cases"][case_name]["expected"]["domains"]["content"]
    after = GOLDEN["cases"][case_name]["expected"]["domains"]["content"]
    assert set(before["sub_scores"]) - set(after["sub_scores"]) == set(
        REMOVED_CONTENT_SUB_METRICS
    )
    assert set(after["sub_scores"]) - set(before["sub_scores"]) == set()


def test_content_weights_are_the_previous_ones_renormalised():
    """Six weights, each old/0.85 to 4 dp -- relative proportions untouched.

    Naive rounding sums to 1.0001, so one weight carries a 1e-4 residue. The
    tolerance below is that residue and nothing more: a weight that was actually
    re-tuned rather than rescaled would miss it.
    """
    before = PREVIOUS_GOLDEN["cases"][CASE_NAMES[0]]["expected"]["domains"]["content"]
    old_weights = before["weights"]
    removed_mass = sum(old_weights[m] for m in REMOVED_CONTENT_SUB_METRICS)
    assert removed_mass == pytest.approx(0.15)

    surviving_mass = 1.0 - removed_mass
    for metric, weight in CONTENT_WEIGHTS.items():
        assert weight == pytest.approx(
            old_weights[metric] / surviving_mass, abs=1e-4
        ), metric
    assert sum(CONTENT_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)


def test_the_rounding_residue_lands_on_the_smallest_weight():
    """Exactly one weight absorbs it, and it is the least influential one."""
    before = PREVIOUS_GOLDEN["cases"][CASE_NAMES[0]]["expected"]["domains"]["content"]
    surviving_mass = 1.0 - sum(
        before["weights"][m] for m in REMOVED_CONTENT_SUB_METRICS
    )
    naive = {
        metric: round(before["weights"][metric] / surviving_mass, 4)
        for metric in CONTENT_WEIGHTS
    }
    adjusted = [m for m, w in CONTENT_WEIGHTS.items() if w != naive[m]]

    assert adjusted == [min(CONTENT_WEIGHTS, key=lambda m: naive[m])]
    assert round(sum(naive.values()), 4) == 1.0001


def test_content_weights_hold_no_sport_specific_metric():
    assert len(CONTENT_WEIGHTS) == 6
    assert set(CONTENT_WEIGHTS).isdisjoint(REMOVED_CONTENT_SUB_METRICS)
    for metric in REMOVED_CONTENT_SUB_METRICS:
        assert not hasattr(scorer, f"score_{metric}")


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_boundary_composites_are_the_measured_ones(case_name: str):
    """The exact before/after the CHANGELOG entry warns about."""
    before, after = BOUNDARY_COMPOSITES[case_name]
    assert PREVIOUS_GOLDEN["cases"][case_name]["expected"]["pqs_v3"] == before
    assert GOLDEN["cases"][case_name]["expected"]["pqs_v3"] == after


def test_the_previous_golden_is_flagged_as_not_comparable():
    """An artifact scored under 3.0.0 must never be compared to a 3.1.0 score."""
    for case_name in CASE_NAMES:
        artifact = PREVIOUS_GOLDEN["cases"][case_name]["expected"]
        assert not is_comparable(artifact)
        with pytest.raises(ProfileVersionError, match=PREVIOUS_PROFILE_VERSION):
            check_comparable(artifact, source=PREVIOUS_GOLDEN_PATH.name)

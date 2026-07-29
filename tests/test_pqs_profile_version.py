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
"""

import json
from pathlib import Path
from typing import Any

import pytest

from podcast_intel.analysis.scorer import (
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

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "pqs_golden.json"


def _load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


GOLDEN = _load_golden()
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

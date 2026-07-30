"""
The two-language acceptance test.

The charter says this framework serves many podcasts, mainly in Hebrew. That
claim is only real if a stranger can point ``podcast.language`` at a language
and have the whole stack follow. So the pipeline is run twice over the mock
corpus -- once with ``language: en``, once with ``language: he`` -- and the two
runs are compared.

What a run must supply is exactly four keys in ``podcast.yaml``:
``podcast.name``, ``podcast.language``, ``podcast.rss_url`` and
``speakers.default``. Nothing about models, and nothing about filler words.

Asserted, per the acceptance criteria:

1. the resolved transcription model DIFFERS between the two runs;
2. the Hebrew filler lexicon is the active one on the Hebrew run, and it
   actually detects Hebrew fillers;
3. the briefing renders ``dir="rtl"`` for Hebrew and ``dir="ltr"`` for English.
"""

import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from podcast_intel.analysis.filler_detector import detect_fillers, get_default_fillers
from podcast_intel.config import get_config, load_podcast_yaml
from podcast_intel.ingestion.mock_ingest import generate_mock_episodes
from podcast_intel.models.database import Database
from podcast_intel.transcription.mock_transcribe import generate_mock_transcription
from podcast_intel.triggers.briefing_generator import generate_briefing
from podcast_intel.triggers.community_events import CommunityEvent

HEBREW_TRANSCRIPTION = "ivrit-ai/whisper-large-v3-turbo"
ENGLISH_TRANSCRIPTION = "openai/whisper-large-v3-turbo"

#: A Hebrew utterance with three fillers: "כאילו", "אז" and "נכון?".
HEBREW_UTTERANCE = "כאילו אז נכון? זה בדיוק העניין"

PODCASTS = {
    "he": """
        podcast:
          name: "פודקאסט לדוגמה"
          language: "he"
          rss_url: "https://feeds.example.com/he/rss"
        speakers:
          default: ["מנחה", "אורח"]
        """,
    "en": """
        podcast:
          name: "Example Podcast"
          language: "en"
          rss_url: "https://feeds.example.com/en/rss"
        speakers:
          default: ["Host", "Guest"]
        """,
}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path: Path):
    """No PODCAST_INTEL_* env var may decide the outcome of an acceptance test."""
    import os

    for key in list(os.environ):
        if key.startswith("PODCAST_INTEL_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def run_pipeline(root: Path, language: str) -> dict[str, Any]:
    """Run the mock pipeline for one language and report what it resolved.

    Only ``podcast.yaml`` differs between the two calls.
    """
    project = root / language
    project.mkdir(parents=True, exist_ok=True)
    (project / "podcast.yaml").write_text(
        textwrap.dedent(PODCASTS[language]), encoding="utf-8"
    )

    config = get_config(search_dir=project)
    podcast_yaml = load_podcast_yaml(project)

    # The mock corpus: episodes + one transcription, in this run's own database.
    db = Database(project / "mock.db")
    db.initialize()
    episode_ids = generate_mock_episodes(db, count=2)
    generate_mock_transcription(db, episode_ids[0])
    with db.get_connection() as conn:
        rows = db.get_segments_by_episode(conn, episode_ids[0])
    segments = [
        {
            "speaker": str(row["speaker_id"]),
            "text": row["text"],
            "start": row["start_time"],
            "end": row["end_time"],
        }
        for row in rows
    ]

    briefing = generate_briefing(
        CommunityEvent(
            event_id="acceptance-1",
            event_type="episode",
            status="SCHEDULED",
            date="2026-08-01T20:00:00Z",
            summary="Acceptance run",
        ),
        podcast_yaml,
        formats=["html"],
    )

    return {
        "config": config,
        "segments": segments,
        "fillers_on_corpus": detect_fillers(segments, filler_words=config.filler_words),
        "fillers_on_hebrew_line": detect_fillers(
            [{"speaker": "s", "text": HEBREW_UTTERANCE, "start": 0.0, "end": 60.0}],
            filler_words=config.filler_words,
        ),
        "html": briefing["html"],
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, dict[str, Any]]:
    root = tmp_path_factory.mktemp("acceptance")
    return {language: run_pipeline(root, language) for language in PODCASTS}


# ------------------------------------------------------------------ #
# 0. The contract: four keys, nothing about models
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("language", sorted(PODCASTS))
def test_a_stranger_supplies_only_four_keys(language: str):
    data = yaml.safe_load(textwrap.dedent(PODCASTS[language]))
    assert set(data) == {"podcast", "speakers"}
    assert set(data["podcast"]) == {"name", "language", "rss_url"}
    assert data["speakers"]["default"]
    assert "models" not in data
    assert "filler_words" not in yaml.safe_dump(data)


# ------------------------------------------------------------------ #
# 1. The resolved transcription model differs
# ------------------------------------------------------------------ #

def test_the_resolved_transcription_model_differs(runs):
    he = runs["he"]["config"].transcription_model
    en = runs["en"]["config"].transcription_model
    assert he == HEBREW_TRANSCRIPTION
    assert en == ENGLISH_TRANSCRIPTION
    assert he != en


def test_the_whole_nlp_stack_follows_the_language(runs):
    """Not just transcription -- NER and sentiment move too."""
    he, en = runs["he"]["config"], runs["en"]["config"]
    assert he.ner_model == "dicta-il/dictabert-ner"
    assert he.sentiment_model == "avichr/heBERT_sentiment_analysis"
    assert en.ner_model != he.ner_model
    assert en.sentiment_model != he.sentiment_model


# ------------------------------------------------------------------ #
# 2. The Hebrew filler lexicon is the active one
# ------------------------------------------------------------------ #

def test_the_hebrew_run_activates_the_hebrew_lexicon(runs):
    active = runs["he"]["config"].filler_words
    assert active == get_default_fillers("he")
    assert "כאילו" in active
    assert set(active).isdisjoint(runs["en"]["config"].filler_words)


def test_the_hebrew_lexicon_actually_detects_hebrew_fillers(runs):
    """A configured list that finds nothing would be a paper claim."""
    detected = runs["he"]["fillers_on_hebrew_line"]["s"]
    assert detected["total_fillers"] == 3
    assert set(detected["filler_counts"]) == {"כאילו", "אז", "נכון?"}

    # The same line under the English run finds nothing -- the lists really differ.
    assert runs["en"]["fillers_on_hebrew_line"]["s"]["total_fillers"] == 0


def test_the_english_run_still_scores_the_english_mock_corpus(runs):
    """The mock corpus is English, so only the English run should light up.

    This is the honest reading of a Hebrew-first framework shipping an English
    mock corpus: the Hebrew run finding zero fillers here is correct behaviour,
    not a broken lexicon -- proved by the Hebrew line above.
    """
    en_total = sum(s["total_fillers"] for s in runs["en"]["fillers_on_corpus"].values())
    he_total = sum(s["total_fillers"] for s in runs["he"]["fillers_on_corpus"].values())
    assert runs["en"]["segments"], "the mock corpus produced no segments"
    assert en_total > 0
    assert he_total == 0


# ------------------------------------------------------------------ #
# 3. The briefing renders RTL for Hebrew
# ------------------------------------------------------------------ #

def test_the_hebrew_briefing_renders_rtl(runs):
    html = runs["he"]["html"]
    assert 'dir="rtl"' in html
    assert 'lang="he"' in html
    assert "direction: rtl;" in html


def test_the_english_briefing_renders_ltr(runs):
    html = runs["en"]["html"]
    assert 'dir="ltr"' in html
    assert 'lang="en"' in html
    assert 'dir="rtl"' not in html


def test_an_unconfigured_briefing_is_rtl_hebrew_by_default(runs):
    """Hebrew-first means the DEFAULT is RTL, not just the configured case."""
    html = generate_briefing(
        CommunityEvent(
            event_id="acceptance-2",
            event_type="episode",
            status="SCHEDULED",
            date="2026-08-01T20:00:00Z",
            summary="No podcast.yaml at all",
        ),
        {},
        formats=["html"],
    )["html"]
    assert 'dir="rtl"' in html
    assert 'lang="he"' in html
    assert "Heebo" in html

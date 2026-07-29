"""
Tests for the language-preset mechanism and the four-layer config merge.

Covers:
- ``podcast_intel.presets.AVAILABLE_PRESETS`` / ``load_preset()``
- ``get_config()`` layering: defaults < preset < podcast.yaml < environment
- The acceptance claim in the README: ``language: he`` resolves the Hebrew stack
  (ivrit-ai transcription + the Hebrew filler lexicon) and ``language: en`` does not
"""

import textwrap
from pathlib import Path
from typing import Any

import pytest

from podcast_intel.config import Config, get_config, load_podcast_yaml, resolve_language
from podcast_intel.presets import (
    AVAILABLE_PRESETS,
    DEFAULT_LANGUAGE,
    has_preset,
    load_preset,
    preset_value,
)

HEBREW_TRANSCRIPTION = "ivrit-ai/whisper-large-v3-turbo"
ENGLISH_TRANSCRIPTION = "openai/whisper-large-v3-turbo"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path: Path):
    """Strip PODCAST_INTEL_* from the environment and keep .env out of reach."""
    import os

    for key in list(os.environ):
        if key.startswith("PODCAST_INTEL_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def write_podcast_yaml(directory: Path, body: str) -> Path:
    """Write a podcast.yaml into a fresh nested directory and return that dir."""
    project = directory / "project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "podcast.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    return project


# ---------------------------------------------------------------------------
#  presets/__init__.py
# ---------------------------------------------------------------------------


class TestPresetApi:
    """The API CONTRIBUTING.md documents."""

    def test_available_presets_are_en_and_he(self):
        assert AVAILABLE_PRESETS == {"en": "english.yaml", "he": "hebrew.yaml"}

    def test_every_declared_preset_actually_loads(self):
        for code in AVAILABLE_PRESETS:
            preset = load_preset(code)
            assert preset["language"] == code
            assert preset["models"]["transcription"]
            assert len(preset["filler_words"]) >= 10

    def test_hebrew_preset_names_the_hebrew_stack(self):
        preset = load_preset("he")
        assert preset["models"]["transcription"] == HEBREW_TRANSCRIPTION
        assert preset["models"]["ner"] == "dicta-il/dictabert-ner"
        assert preset["models"]["sentiment"] == "avichr/heBERT_sentiment_analysis"
        assert "כאילו" in preset["filler_words"]

    def test_english_preset_is_not_the_hebrew_stack(self):
        preset = load_preset("en")
        assert preset["models"]["transcription"] == ENGLISH_TRANSCRIPTION
        assert "כאילו" not in preset["filler_words"]
        assert "um" in preset["filler_words"]

    def test_code_is_normalized(self):
        assert load_preset("HE") == load_preset("he")
        assert load_preset("he-IL") == load_preset("he")
        assert has_preset("EN") is True

    def test_unknown_code_raises_with_the_available_list(self):
        with pytest.raises(ValueError) as exc:
            load_preset("es")
        assert "es" in str(exc.value)
        assert "en, he" in str(exc.value)
        assert has_preset("es") is False

    def test_preset_is_readable_through_importlib_resources(self):
        """Guards the packaging path -- not a filesystem-relative open()."""
        from importlib import resources

        for filename in AVAILABLE_PRESETS.values():
            handle = resources.files("podcast_intel.presets").joinpath(filename)
            assert handle.is_file()


# ---------------------------------------------------------------------------
#  get_config() layering
# ---------------------------------------------------------------------------


class TestHebrewFirstDefaults:
    """The built-in defaults ARE the default-language preset, not a copy of it."""

    def test_default_language_is_hebrew(self):
        assert DEFAULT_LANGUAGE == "he"
        assert DEFAULT_LANGUAGE in AVAILABLE_PRESETS
        assert Config.model_fields["language"].default == DEFAULT_LANGUAGE

    @pytest.mark.parametrize(
        "field_name,preset_key",
        [
            ("transcription_model", "models.transcription"),
            ("ner_model", "models.ner"),
            ("sentiment_model", "models.sentiment"),
        ],
    )
    def test_model_defaults_come_from_the_default_preset(
        self, field_name: str, preset_key: str
    ):
        assert Config.model_fields[field_name].default == preset_value(preset_key)

    def test_bare_config_is_already_the_hebrew_stack(self):
        """Config() with no podcast.yaml and no env still resolves Hebrew."""
        config = Config()
        assert config.language == "he"
        assert config.transcription_model == HEBREW_TRANSCRIPTION
        assert config.transcription_model != ENGLISH_TRANSCRIPTION

    def test_config_module_hardcodes_no_english_model_id(self):
        """No English default may survive in the code as a second source of truth.

        The Hebrew IDs are allowed to appear in a docstring example; an English
        one appearing at all would mean a hardcoded default was left behind.
        """
        import podcast_intel.config as config_module

        source = Path(config_module.__file__).read_text(encoding="utf-8")
        for model_id in load_preset("en")["models"].values():
            assert model_id not in source, model_id

    def test_preset_value_reads_the_default_language(self):
        assert preset_value("models.transcription") == HEBREW_TRANSCRIPTION
        assert preset_value("models.transcription", "en") == ENGLISH_TRANSCRIPTION
        assert preset_value("models.nope") is None


class TestLanguageResolvesTheStack:
    """The acceptance claim: `language: he` selects the Hebrew stack."""

    def test_hebrew_podcast_yaml_resolves_the_hebrew_stack(self, tmp_path: Path):
        project = write_podcast_yaml(
            tmp_path,
            """
            podcast:
              name: "פודקאסט לדוגמה"
              language: "he"
              rss_url: "https://feeds.example.com/he/rss"
            """,
        )
        config = get_config(search_dir=project)

        assert config.language == "he"
        assert config.transcription_model == HEBREW_TRANSCRIPTION
        assert config.ner_model == "dicta-il/dictabert-ner"
        assert config.sentiment_model == "avichr/heBERT_sentiment_analysis"
        assert "כאילו" in config.filler_words
        assert "יעני" in config.filler_words

    def test_english_podcast_yaml_does_not_resolve_the_hebrew_stack(
        self, tmp_path: Path
    ):
        project = write_podcast_yaml(
            tmp_path,
            """
            podcast:
              name: "Tech Talk Weekly"
              language: "en"
              rss_url: "https://feeds.example.com/en/rss"
            """,
        )
        config = get_config(search_dir=project)

        assert config.language == "en"
        assert config.transcription_model == ENGLISH_TRANSCRIPTION
        assert config.transcription_model != HEBREW_TRANSCRIPTION
        assert config.ner_model != "dicta-il/dictabert-ner"
        assert "כאילו" not in config.filler_words
        assert "um" in config.filler_words

    def test_the_two_languages_differ(self, tmp_path: Path):
        he = get_config(
            search_dir=write_podcast_yaml(
                tmp_path / "a", 'podcast:\n  language: "he"\n'
            )
        )
        en = get_config(
            search_dir=write_podcast_yaml(
                tmp_path / "b", 'podcast:\n  language: "en"\n'
            )
        )
        assert he.transcription_model != en.transcription_model
        assert set(he.filler_words).isdisjoint(en.filler_words)


class TestPrecedence:
    """defaults < preset < podcast.yaml < environment."""

    def test_podcast_yaml_overrides_the_preset(self, tmp_path: Path):
        project = write_podcast_yaml(
            tmp_path,
            """
            podcast:
              language: "he"
            models:
              transcription: "openai/whisper-large-v3"
            """,
        )
        config = get_config(search_dir=project)

        assert config.transcription_model == "openai/whisper-large-v3"
        # untouched keys still come from the Hebrew preset
        assert config.sentiment_model == "avichr/heBERT_sentiment_analysis"

    def test_environment_overrides_podcast_yaml(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("PODCAST_INTEL_NER_MODEL", "custom/ner-model")
        project = write_podcast_yaml(
            tmp_path,
            """
            podcast:
              language: "he"
            models:
              ner: "yaml/ner-model"
            """,
        )
        config = get_config(search_dir=project)

        assert config.ner_model == "custom/ner-model"
        assert config.transcription_model == HEBREW_TRANSCRIPTION

    def test_environment_language_overrides_podcast_yaml_language(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setenv("PODCAST_INTEL_LANGUAGE", "he")
        project = write_podcast_yaml(tmp_path, 'podcast:\n  language: "en"\n')
        config = get_config(search_dir=project)

        assert config.language == "he"
        assert config.transcription_model == HEBREW_TRANSCRIPTION

    def test_defaults_stand_when_nothing_is_configured(self, tmp_path: Path):
        """Unconfigured means Hebrew: the charter default, not an English one."""
        empty = tmp_path / "empty" / "nested"
        empty.mkdir(parents=True)
        config = get_config(search_dir=empty)

        assert config.language == Config.model_fields["language"].default
        assert config.language == DEFAULT_LANGUAGE == "he"
        assert config.transcription_model == HEBREW_TRANSCRIPTION
        assert config.ner_model == "dicta-il/dictabert-ner"
        assert config.sentiment_model == "avichr/heBERT_sentiment_analysis"

    def test_unknown_language_is_not_an_error_and_keeps_defaults(self, tmp_path: Path):
        project = write_podcast_yaml(tmp_path, 'podcast:\n  language: "es"\n')
        config = get_config(search_dir=project)

        assert config.language == "es"
        # No Spanish preset ships, so layer 2 is skipped and the built-in
        # defaults -- which are the DEFAULT_LANGUAGE preset's -- stand.
        assert config.transcription_model == HEBREW_TRANSCRIPTION
        assert config.filler_words == []

    def test_rss_url_comes_from_podcast_yaml(self, tmp_path: Path):
        project = write_podcast_yaml(
            tmp_path,
            """
            podcast:
              language: "he"
              rss_url: "https://feeds.example.com/mypodcast.rss"
            """,
        )
        assert get_config(search_dir=project).rss_url == (
            "https://feeds.example.com/mypodcast.rss"
        )


class TestPodcastYamlDiscovery:
    """podcast.yaml lives in the USER's project, not next to the installed package."""

    def test_found_in_the_current_working_directory(self, tmp_path: Path, monkeypatch):
        project = write_podcast_yaml(tmp_path, 'podcast:\n  language: "he"\n')
        monkeypatch.chdir(project)

        # no search_dir -> must still find the cwd's podcast.yaml
        assert get_config().transcription_model == HEBREW_TRANSCRIPTION

    def test_found_in_a_parent_of_the_current_working_directory(
        self, tmp_path: Path, monkeypatch
    ):
        project = write_podcast_yaml(tmp_path, 'podcast:\n  language: "he"\n')
        deep = project / "a" / "b"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)

        assert load_podcast_yaml()["podcast"]["language"] == "he"


class TestResolveLanguage:
    def test_env_wins(self):
        assert resolve_language({"podcast": {"language": "en"}}, "he") == "he"

    def test_yaml_wins_over_default(self):
        assert resolve_language({"podcast": {"language": "he"}}, None) == "he"

    def test_default_when_nothing_set(self):
        assert resolve_language({}, None) == Config.model_fields["language"].default


# ---------------------------------------------------------------------------
#  The shipped quickstart example must stay loadable
# ---------------------------------------------------------------------------


def test_quickstart_example_is_tracked_and_parses():
    """examples/quickstart/podcast.yaml is published and is valid podcast.yaml."""
    repo_root = Path(__file__).resolve().parents[1]
    quickstart = repo_root / "examples" / "quickstart" / "podcast.yaml"
    assert quickstart.is_file()

    data: dict[str, Any] = load_podcast_yaml(quickstart.parent)
    assert data["podcast"]["language"] in AVAILABLE_PRESETS
    assert data["podcast"]["name"]
    assert data["speakers"]["default"]

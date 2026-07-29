# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Language presets are now a working mechanism, not a promise.**
  `podcast_intel.presets` ships `AVAILABLE_PRESETS`, `load_preset(code)` and
  `has_preset(code)`; presets are read out of the installed package with
  `importlib.resources`, so they work from a wheel.
- **`get_config()` performs the four-layer merge the docs already described:**
  `defaults < language preset < podcast.yaml < environment/.env`. Setting
  `podcast.language: "he"` now actually resolves `ivrit-ai/whisper-large-v3-turbo`,
  `dicta-il/dictabert-ner`, `avichr/heBERT_sentiment_analysis` and the Hebrew
  filler lexicon. A language with no shipped preset falls back to the defaults
  rather than raising.
- `Config.filler_words`, resolved from the active language preset.
- `[tool.setuptools.package-data]` for `presets/*.yaml` -- before this the preset
  YAMLs were in **no** built wheel or sdist.
- `ingestion/downloader.py` implemented (streaming download to a `.part`
  sidecar, MP3 header validation, `HEAD`-based size lookup). It was three
  placeholder bodies returning `None` against `-> bool` signatures while
  `ingestion/__init__.py` exported `download_episode`.
- New tests: `tests/test_presets.py` (preset API, four-layer precedence,
  he-vs-en acceptance), `tests/test_lazy_imports.py` (no `__init__` may import an
  optional extra), `tests/test_downloader.py`, `tests/test_fixtures.py`.
- All six `tests/conftest.py` fixtures implemented. They were placeholder bodies
  returning `None` against declared return types, while CONTRIBUTING.md told
  contributors to "use fixtures from `tests/conftest.py`".

### Fixed

- **`import podcast_intel.transcription` no longer fails on a core install.**
  `transcription/__init__.py` imported `whisper_transcriber` -- and therefore
  `faster_whisper`, a `[transcription]`-extra dependency -- at module level.
  `WhisperTranscriber` is now resolved lazily via PEP 562 `__getattr__`. This is
  also what made `tests/test_mock_system.py` uncollectable.
- **`load_podcast_yaml()` searched the installed package directory, not the
  user's project.** It now searches the current working directory first
  (`PROJECT_ROOT` remains a last-resort fallback), which is what the
  documentation always claimed.
- `tools/analyze_panel_chemistry.py` called `sys.exit()` without importing `sys`
  (`NameError` on the "no episodes found" path).
- `examples/quickstart/podcast.yaml` was matched by `.gitignore`'s bare
  `podcast.yaml` rule and was absent from the published repo, although README,
  CONTRIBUTING and `examples/README.md` all point at it.

### Changed

- CI is green again: the `[tool.ruff]` config moved to the post-0.2
  `[tool.ruff.lint]` layout, all 431 ruff findings in `src/`+`tests/` are fixed
  (annotations modernised to PEP 585/604, dead locals removed), and `mypy
  --strict`-style `disallow_untyped_defs` passes with 0 errors.
- Documentation honesty pass: removed the "Specialization System"
  (`speakers.yaml` / `entities.yaml` / `scoring_weights.yaml`) sections from
  README, CONTRIBUTING, `docs/CONFIGURATION.md` and `examples/README.md` -- no
  packaged code reads those filenames; removed the Spanish preset from the
  "Available Presets" table (it does not ship); replaced the `ValidationError`
  section documenting validators that do not exist with a statement of what is
  and is not validated; replaced `docs/ARCHITECTURE.md`'s call to the
  nonexistent `merge_configs()`; corrected the `FillerDetector` class example in
  CONTRIBUTING (the module exposes functions, not a class); removed
  `podcast-intel init` / `podcast-intel analyze` from `examples/README.md`
  (neither is a CLI verb).
- `tests/test_mock_system.py::test_language_check_constraint` asserted a
  `he`/`en`/`mixed` allowlist that the schema deliberately dropped in 0.2.0; it
  now asserts the real length-based constraint, in line with a framework that
  serves many podcasts in many languages.

## [0.2.0] - 2026-02-18

### Added

- Filler word detection with regex-based matching for English and Hebrew
- Silence analysis: gap detection, micro-pause, dead-air, and silence density metrics
- Episode metrics: talk time, speaking pace, word counts, talk-time balance (Gini)
- Community infrastructure: CODE_OF_CONDUCT.md, SECURITY.md, CONTRIBUTING.md
- GitHub issue templates (bug report, feature request) and PR template
- Honest feature status matrix in README
- "Try It in 60 Seconds" quick start section in README

### Changed

- EntityType enum generalized from sports-specific (PLAYER, CLUB, COMPETITION) to domain-agnostic (PERSON, ORGANIZATION, LOCATION, EVENT, OTHER)
- Database language constraint relaxed from hard-coded allowlist to length-based validation (`CHECK (length(language) BETWEEN 2 AND 10)`)
- NER pipeline: renamed `football_dict_path` to `custom_entity_dict_path`
- Mock data titles and descriptions generalized to non-sports-specific content
- PQS_FRAMEWORK.md fully aligned with scorer.py (all 5 domains, 39 sub-metrics match code)
- README updated to reflect actual CLI commands and feature status
- Removed non-existent `podcast-intel init` command from documentation
- Removed undocumented Spanish language preset claim

### Fixed

- Documentation-code drift: PQS_FRAMEWORK.md sub-metric names, weights, and input keys now match scorer.py exactly

## [0.1.0] - 2026-02-18

### Added

- PQS v3 scoring engine with 5 domains and 39 sub-metrics
- Whisper-based transcription with PyAnnote speaker diarization
- BERT NER and RoBERTa sentiment analysis pipelines
- SQLite database with 10-table schema and Pydantic models
- RSS feed ingestion and episode metadata extraction
- Mock data generation system for testing
- CLI commands: `ingest`, `mock`, `watch`, `events`
- RSS watcher with automation triggers
- Community events integration with provider system
- English and Hebrew language presets
- HTML report generation via `tools/` scripts

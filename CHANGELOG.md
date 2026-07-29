# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-29

### Changed

- **BREAKING -- `CommunityEvent` is no longer shaped like a football fixture.**
  The generic event dataclass carried `teams`, `score` and `competition` as
  first-class fields, so every community event in the base -- a meetup, a
  release, a council vote -- was modelled as a match. The fields are now
  `participants`, `result` and `context`.

  The old names still work for one release: they are accepted as constructor
  keywords and readable and writable as attributes, each raising a
  `DeprecationWarning` that names its replacement. `DEPRECATED_FIELD_ALIASES`
  declares the mapping. **The serialized form has no alias** -- `to_dict()` and
  `to_json()` emit `participants`/`result`/`context` only, so anything reading
  a stored event payload by key must be updated now. The aliases are scheduled
  for removal one release after 0.4.0.
- **The briefing generator is domain-neutral.** It reads only the generic
  fields, and `headline_slots()` lays an event out from participant count
  alone: no participants renders the summary, one renders without inventing an
  opponent, two keep the familiar three-slot layout, three or more list in the
  lead. Two participants with no result get a neutral separator rather than
  "vs". The fixture-shaped CSS hooks are renamed with them
  (`score-display` -> `headline`, `team` -> `participant`, `score` -> `result`,
  `teams-row` -> `participants-row`, `competition` -> `context`), and saved
  filenames are `<date>_<participant>_<participant>` instead of `_vs_`.
- **Football talking points moved behind a provider hook.** The match-shaped
  templates (player ratings, formations, predicted lineups, season standings)
  were emitted by `briefing_generator` for every event of every podcast. They
  now live in `FootballProvider.talking_points()`, reached through the new
  optional `CommunityEventProvider.talking_points()` hook. `generate_briefing()`
  takes a `provider=` argument and the events CLI passes the configured one; a
  podcast that does not load the football provider never sees football
  vocabulary. Without a provider -- or if a provider returns nothing or raises
  -- the briefing falls back to neutral prompts built from `participants`,
  `result`, `context` and `status`. The football provider itself is unchanged
  in behaviour and still resolves from `podcast.yaml` as `provider: "football"`.
- **The mock corpus is a neutral panel discussion.** `mock_transcribe.py`
  shipped an all-football template corpus that named real clubs (Arsenal,
  Chelsea, Liverpool, Manchester City) under a "Generic entities" header. It is
  what the demo prints, so it was the first thing a stranger read. The
  templates are now about a podcast and its subject matter, and the fill-in
  entities are invented. `MATCH_ANALYSIS_TEMPLATES`,
  `PLAYER_DISCUSSION_TEMPLATES`, `TACTICAL_ANALYSIS_TEMPLATES` and
  `TRANSFER_TALK_TEMPLATES` are renamed `TOPIC_ANALYSIS_TEMPLATES`,
  `PERSON_DISCUSSION_TEMPLATES`, `DEEP_DIVE_TEMPLATES` and
  `NEWS_TALK_TEMPLATES`; `PLAYERS`/`CLUBS`/`FORMATIONS`/`POSITIONS`/`STATS`/
  `STAT_VALUES` become `PEOPLE`/`ORGANIZATIONS`/`APPROACHES`/`ROLES`/`METRICS`/
  `METRIC_VALUES`, and `TOPICS` replaces the inline concept list. `ENGLISH_TERMS`
  keeps its name and now holds the corpus's own vocabulary.

### Added

- `CommunityEventProvider.talking_points()` -- optional, non-abstract hook
  returning domain-specific discussion prompts. Returning an empty list (the
  base behaviour) selects the neutral prompts, so existing providers need no
  change.
- `briefing_generator.headline_slots()` and `NEUTRAL_SEPARATOR`, the public
  layout primitive described above.
- `tests/test_briefing_generator.py` (30 tests) covering the neutral renderer,
  the provider hook and its failure modes, and the filename shape.
- Tests pinning that football stays a working plug-in after the generic layer
  stopped assuming it, and that the mock corpus contains no sport vocabulary
  in its templates, its entities or a generated transcript.

## [0.3.0] - 2026-07-29

### Removed

- **BREAKING -- the two sport-specific PQS content sub-metrics.**
  `match_reference_density` (weight 0.07) and `tactical_depth_density` (0.08)
  are gone from `CONTENT_WEIGHTS`, from `compute_content_domain()`'s function
  map and from `docs/PQS_FRAMEWORK.md`, together with
  `score_match_reference_density()` and `score_tactical_depth_density()`. The
  remaining six weights are renormalised from 0.85 to 1.0. Content Depth is now
  6 sub-metrics and PQS v3.1 is 37, not 39.

  **Content and composite scores are not comparable across 0.2.x -> 0.3.0.**
  Two sport-specific metrics were removed and the remaining six renormalised, so
  a score computed under 0.2.x and one computed under 0.3.0 measure different
  things even when the underlying episode is identical. Do not plot them on one
  trend line; rescore the source data instead.

  `scorer.PROFILE_VERSION` moves `3.0.0` -> `3.1.0` and `compute_pqs()` stamps
  it into every result, so `check_comparable()` raises on a 3.0.0 artifact
  rather than reporting the rescale as a regression. Measured effect on the
  golden cases: strong 95.78 -> 96.34, weak 25.40 -> 26.04, edges 43.77 ->
  43.43; the `audio`, `delivery`, `structure` and `engagement` blocks are
  byte-identical. `compute_content_domain()` ignores the two removed keys, so
  a caller still passing them gets a 3.1.0 score instead of a crash.
- `analysis.filler_detector.HEBREW_FILLERS` and `DEFAULT_FILLERS`. They were
  hand-copied mirrors of `presets/hebrew.yaml` and `presets/english.yaml`; the
  module now reads the presets. `get_default_fillers()` keeps its name and
  returns the same Hebrew list.

### Changed

- **Defaults are Hebrew-first.** `Config.language` defaults to `"he"`, and the
  `transcription_model` / `ner_model` / `sentiment_model` defaults are read from
  `presets/hebrew.yaml` through `presets.DEFAULT_LANGUAGE` rather than being
  hardcoded a second time -- an unconfigured install resolves
  `ivrit-ai/whisper-large-v3-turbo`, `dicta-il/dictabert-ner` and
  `avichr/heBERT_sentiment_analysis`. An English show sets
  `podcast.language: "en"` and gets the English preset. `transcribe.py` and
  `whisper_transcriber.py` follow the same default.
- **Briefings render RTL Hebrew by default** (`briefing_generator`): `direction`
  `ltr` -> `rtl`, `language` `en` -> `he`, and the font stack now leads with
  Heebo. `direction` is derived from `podcast.language` unless podcast.yaml
  states it explicitly, so an English show no longer had to know about a
  separate key to stop getting RTL markup.
- `detect_fillers()`, `detect_fillers_in_text()`, `count_fillers_in_text()` and
  `extract_filler_positions()` take `language` (and `detect_fillers` also
  `filler_words`). Previously `detect_fillers()` always scanned with the English
  list, so a Hebrew episode reported zero fillers.
- PQS golden fixtures: `tests/fixtures/pqs_golden.json` is regenerated at
  profile 3.1.0 and the superseded one is kept as
  `tests/fixtures/pqs_golden_3.0.0.json`, so the blast-radius proof survives the
  regeneration instead of becoming self-referential.
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

### Fixed

- **`build_filler_pattern()` could never match a filler ending in punctuation.**
  The whole alternation was wrapped in `\b...\b`, and a word boundary cannot
  hold between `?` and a following space. Measured: `["right?", "um", "so"]`
  over `"so right? um ok"` returned `['so', 'um']`. Hebrew `"נכון?"`
  (`presets/hebrew.yaml`) was dead the same way. Boundaries are now applied per
  filler, only on the ends where they can apply.
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

### Added

- `presets.DEFAULT_LANGUAGE` (`"he"`) and `presets.preset_value()` -- the single
  place the default language and its defaults are declared.
- `briefing_generator.text_direction()`, mapping a language code to `rtl`/`ltr`.
- `tests/test_language_acceptance.py` -- the two-language acceptance test: the
  mock pipeline is run over `language: en` and `language: he` from a four-key
  `podcast.yaml`, asserting the resolved transcription model differs, the Hebrew
  filler lexicon is active and actually detects Hebrew fillers, and the briefing
  renders `dir="rtl"` for Hebrew and `dir="ltr"` for English.
- `tests/test_filler_detector.py` -- lexicon-from-preset, the `\b` regression
  and the boundaries that must still hold.
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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

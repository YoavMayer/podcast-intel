# Configuration Reference

This document provides a complete reference for configuring podcast-intel.

## Table of Contents

- [podcast.yaml Schema](#podcastyaml-schema)
- [Environment Variables](#environment-variables)
- [Language Presets](#language-presets)
- [Configuration Precedence](#configuration-precedence)

## podcast.yaml Schema

The `podcast.yaml` file is the primary configuration file for your podcast project.

### Complete Example

```yaml
podcast:
  name: "My Podcast"
  language: "en"
  rss_url: "https://example.com/feed.rss"

speakers:
  default:
    - "Alice Chen"
    - "Bob Martinez"
    - "Carol Smith"
  episodes:
    42:
      - "Alice Chen"
      - "Guest Speaker"

models:
  transcription: "openai/whisper-large-v3-turbo"
  ner: "dslim/bert-base-NER"
  sentiment: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  embedding: "BAAI/bge-m3"
  reranker: "BAAI/bge-reranker-v2-m3"

branding:
  show_name: "MY PODCAST"
  primary_color: "#2563eb"
  footer_text: "My Podcast Intelligence"

analysis:
  episode_dir_prefix: "episode"
  episodes_json: "episodes.json"

scoring:
  domain_weights:
    audio: 0.10
    delivery: 0.25
    structure: 0.20
    content: 0.25
    engagement: 0.20
```

### Field Reference

#### `podcast` (required)

Core podcast metadata.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | **required** | Podcast name |
| `language` | string | `"en"` | ISO 639-1 language code (e.g., `"en"`, `"he"`, `"es"`) |
| `rss_url` | string | `""` | RSS feed URL |
| `description` | string | `""` | Podcast description |

#### `speakers` (required)

Speaker configuration with per-episode overrides.

**Format:**

```yaml
speakers:
  default:
    - "Speaker 1 Name"
    - "Speaker 2 Name"
  episodes:
    42:
      - "Speaker 1 Name"
      - "Guest Name"
    43:
      - "Speaker 1 Name"
      - "Different Guest"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default` | list[string] | **required** | Default speaker names for all episodes |
| `episodes` | dict[int, list[string]] | `{}` | Per-episode speaker overrides |

#### `models` (optional)

NLP model configuration. If not specified, uses language preset defaults.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transcription` | string | `"openai/whisper-large-v3-turbo"` | Whisper model for transcription |
| `ner` | string | `"dslim/bert-base-NER"` | *Reserved* -- NER is not implemented |
| `sentiment` | string | `"cardiffnlp/twitter-roberta-base-sentiment-latest"` | *Reserved* -- sentiment analysis is not implemented |
| `embedding` | string | `"BAAI/bge-m3"` | *Reserved* -- semantic search is not implemented |
| `reranker` | string | `"BAAI/bge-reranker-v2-m3"` | *Reserved* -- semantic search is not implemented |

> **Reserved keys are accepted and stored, but nothing reads them.** Only
> `transcription` affects a packaged code path today.

**Supported Transcription Models:**

- `openai/whisper-large-v3-turbo` (default, English/multilingual)
- `openai/whisper-large-v3` (higher quality, slower)
- `openai/whisper-medium` (faster, lower quality)
- `ivrit-ai/whisper-large-v3-turbo` (Hebrew-optimized)

#### `branding` (optional)

Customization for HTML reports.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `show_name` | string | Uppercase `podcast.name` | Show name displayed in reports |
| `primary_color` | string | `"#2563eb"` | Primary accent color (hex) |
| `footer_text` | string | `"{podcast.name} Intelligence"` | Footer text in reports |

#### `analysis` (optional)

Analysis pipeline configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `episode_dir_prefix` | string | `"episode"` | Directory prefix for episode folders (e.g., `episode_42/`) |
| `episodes_json` | string | `"episodes.json"` | Filename for episodes metadata |

#### `scoring` (optional)

Custom PQS scoring weights.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `domain_weights` | dict | See below | Domain weights (must sum to 1.0) |

**Default Domain Weights:**

```yaml
scoring:
  domain_weights:
    audio: 0.10       # 10%
    delivery: 0.25    # 25%
    structure: 0.20   # 20%
    content: 0.25     # 25%
    engagement: 0.20  # 20%
```

## Environment Variables

All configuration can be overridden with environment variables prefixed with `PODCAST_INTEL_`.

### Core Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PODCAST_INTEL_LANGUAGE` | string | `"en"` | Podcast language |
| `PODCAST_INTEL_RSS_URL` | string | `""` | RSS feed URL |

### Paths

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PODCAST_INTEL_DB_PATH` | path | `./data/db/podcast_intel.db` | SQLite database path |
| `PODCAST_INTEL_AUDIO_DIR` | path | `./data/audio` | Audio files directory |
| `PODCAST_INTEL_EMBEDDINGS_DIR` | path | `./data/embeddings` | Vector embeddings directory |

### Transcription

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PODCAST_INTEL_TRANSCRIPTION_MODEL` | string | `"openai/whisper-large-v3-turbo"` | Whisper model |
| `PODCAST_INTEL_TRANSCRIPTION_DEVICE` | string | `"cuda"` | Device (`cuda` or `cpu`) |
| `PODCAST_INTEL_TRANSCRIPTION_COMPUTE_TYPE` | string | `"float16"` | Compute type for faster-whisper |
| `PODCAST_INTEL_DIARIZATION_ENABLED` | bool | `true` | Enable speaker diarization |
| `PODCAST_INTEL_HUGGINGFACE_TOKEN` | string | `""` | Hugging Face token (required for diarization) |

### Analysis

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
All four are *reserved*: they are parsed into `Config`, but no packaged code reads them.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PODCAST_INTEL_NER_MODEL` | string | `"dslim/bert-base-NER"` | Reserved -- NER not implemented |
| `PODCAST_INTEL_SENTIMENT_MODEL` | string | `"cardiffnlp/twitter-roberta-base-sentiment-latest"` | Reserved -- sentiment not implemented |
| `PODCAST_INTEL_EMBEDDING_MODEL` | string | `"BAAI/bge-m3"` | Reserved -- search not implemented |
| `PODCAST_INTEL_RERANKER_MODEL` | string | `"BAAI/bge-reranker-v2-m3"` | Reserved -- search not implemented |

### LLM

Reserved. No packaged code calls an LLM.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PODCAST_INTEL_LLM_PROVIDER` | string | `"openai"` | Reserved |
| `PODCAST_INTEL_LLM_MODEL` | string | `"gpt-4o-mini"` | Reserved |
| `PODCAST_INTEL_LLM_API_KEY` | string | `""` | Reserved |

### Example .env File

```bash
# Basic config
PODCAST_INTEL_LANGUAGE=en
PODCAST_INTEL_RSS_URL=https://feeds.example.com/mypodcast.rss

# Optional: only tools/diarize_episode.py needs this.
# The packaged diarizer runs on CPU without a token.
PODCAST_INTEL_HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx

# Optional: Custom paths
PODCAST_INTEL_DB_PATH=/custom/path/podcast.db
PODCAST_INTEL_AUDIO_DIR=/mnt/storage/audio

# Optional: Use CPU instead of GPU
PODCAST_INTEL_TRANSCRIPTION_DEVICE=cpu
```

## Language Presets

Language presets provide pre-configured models and filler words for different languages.

### Available Presets

| Language | Code | Transcription | NER | Sentiment |
|----------|------|---------------|-----|-----------|
| English | `en` | whisper-large-v3-turbo | bert-base-NER | roberta-sentiment |
| Hebrew | `he` | ivrit-ai/whisper-large-v3-turbo | dictabert-ner | heBERT |

### Using a Preset

Simply set the language in `podcast.yaml`:

```yaml
podcast:
  language: "he"  # Automatically uses Hebrew preset
```

### Overriding Preset Models

You can override individual models while keeping other preset defaults:

```yaml
podcast:
  language: "he"

models:
  transcription: "openai/whisper-large-v3"  # Override transcription model
  # NER and sentiment still use Hebrew preset defaults
```

### Preset File Format

Preset files live in `src/podcast_intel/presets/{language}.yaml`. Two ship with the
package -- `english.yaml` and `hebrew.yaml`. The format is:

```yaml
# src/podcast_intel/presets/hebrew.yaml
language: "he"

models:
  transcription: "ivrit-ai/whisper-large-v3-turbo"
  ner: "dicta-il/dictabert-ner"
  sentiment: "avichr/heBERT_sentiment_analysis"

filler_words:
  - "אממ"
  - "כאילו"
  - "יעני"
```

See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-language-presets) for how to add one.

## Configuration Precedence

`get_config()` merges four layers, highest precedence first:

1. **Environment variables** (e.g., `PODCAST_INTEL_LANGUAGE=he`) and the **`.env`** file
2. **podcast.yaml** in the current directory or a parent
3. **Language preset** for the resolved `podcast.language` (`presets/{lang}.yaml`)
4. **Built-in defaults** (the field defaults on `Config`)

The language itself is resolved with the same precedence: `PODCAST_INTEL_LANGUAGE`
beats `podcast.language`, which beats the `"en"` default.

### Which keys participate

Only keys that `Config` models are merged. Everything else in `podcast.yaml`
(`podcast.name`, `speakers`, `branding`, `scoring`, `triggers`) is read directly
by the callers that need it -- `cli.py` and `triggers/rss_watcher.py` -- via
`load_podcast_yaml()`.

| podcast.yaml key | Config field |
|---|---|
| `podcast.language` | `language` |
| `podcast.rss_url` | `rss_url` |
| `models.transcription` | `transcription_model` |
| `models.ner` | `ner_model` |
| `models.sentiment` | `sentiment_model` |
| `models.embedding` | `embedding_model` |
| `models.reranker` | `reranker_model` |
| `transcription.device` | `transcription_device` |
| `transcription.compute_type` | `transcription_compute_type` |
| `transcription.diarization_enabled` | `diarization_enabled` |
| `analysis.filler_words` | `filler_words` |
| `paths.db_path` / `paths.audio_dir` / `paths.embeddings_dir` | the matching path fields |

A preset contributes `language`, the three `models.*` IDs and `filler_words`.

### Example

Given this setup:

```yaml
# podcast.yaml
podcast:
  language: "he"

models:
  transcription: "openai/whisper-large-v3"
```

And this environment variable:

```bash
export PODCAST_INTEL_NER_MODEL="custom/ner-model"
```

The final configuration is:

- `transcription_model`: `"openai/whisper-large-v3"` (from podcast.yaml)
- `ner_model`: `"custom/ner-model"` (from the environment)
- `sentiment_model`: `"avichr/heBERT_sentiment_analysis"` (from the Hebrew preset)
- `filler_words`: the Hebrew lexicon (from the Hebrew preset)
- `embedding_model`: `"BAAI/bge-m3"` (built-in default -- no preset supplies it)

This behaviour is covered by `tests/test_presets.py`.

## Validation

`podcast.yaml` is parsed with `yaml.safe_load` and is **not** schema-validated today:
unknown keys are ignored and malformed values are not rejected. The only validation
that runs is Pydantic's on the `Config` object itself (types of environment
variables and `.env` entries), which raises a `pydantic.ValidationError` on, for
example, a non-boolean `PODCAST_INTEL_DIARIZATION_ENABLED`.

Consequences worth knowing:

- An unknown `podcast.language` (anything outside the shipped presets) is **not** an
  error. No preset is applied and the built-in defaults stand.
- A typo in a key name silently does nothing.
- `branding.primary_color` and `scoring.domain_weights` are not checked.

A `podcast.yaml` schema validator is **planned, not shipped**. Do not rely on
configuration errors being reported.

## Best Practices

1. **Use language presets** - They're optimized for each language
2. **Override sparingly** - Only override models if you have a specific reason
3. **Version control** - Commit `podcast.yaml` to track configuration changes
4. **Use .env for secrets** - Keep API keys and tokens in `.env` (not in git)
5. **Document customizations** - Add comments explaining why you changed defaults

## Examples

See the [examples/](../examples/) directory for complete configuration examples:

- `examples/quickstart/podcast.yaml` - Basic English tech podcast

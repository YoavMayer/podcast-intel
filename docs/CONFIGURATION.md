# Configuration Reference

This document provides a complete reference for configuring podcast-intel.

## Table of Contents

- [podcast.yaml Schema](#podcastyaml-schema)
- [Environment Variables](#environment-variables)
- [Language Presets](#language-presets)
- [Specialization Files](#specialization-files)
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
| `ner` | string | `"dslim/bert-base-NER"` | Named Entity Recognition model |
| `sentiment` | string | `"cardiffnlp/twitter-roberta-base-sentiment-latest"` | Sentiment analysis model |
| `embedding` | string | `"BAAI/bge-m3"` | Embedding model for semantic search |
| `reranker` | string | `"BAAI/bge-reranker-v2-m3"` | Reranker model for search results |

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
| `PODCAST_INTEL_NER_MODEL` | string | `"dslim/bert-base-NER"` | NER model |
| `PODCAST_INTEL_SENTIMENT_MODEL` | string | `"cardiffnlp/twitter-roberta-base-sentiment-latest"` | Sentiment model |
| `PODCAST_INTEL_EMBEDDING_MODEL` | string | `"BAAI/bge-m3"` | Embedding model |
| `PODCAST_INTEL_RERANKER_MODEL` | string | `"BAAI/bge-reranker-v2-m3"` | Reranker model |

### LLM (for coaching)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PODCAST_INTEL_LLM_PROVIDER` | string | `"openai"` | LLM provider (`openai`, `anthropic`, `local`) |
| `PODCAST_INTEL_LLM_MODEL` | string | `"gpt-4o-mini"` | LLM model name |
| `PODCAST_INTEL_LLM_API_KEY` | string | `""` | API key for LLM provider |

### Example .env File

```bash
# Basic config
PODCAST_INTEL_LANGUAGE=en
PODCAST_INTEL_RSS_URL=https://feeds.example.com/mypodcast.rss

# Required for diarization
PODCAST_INTEL_HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx

# Optional: Custom paths
PODCAST_INTEL_DB_PATH=/custom/path/podcast.db
PODCAST_INTEL_AUDIO_DIR=/mnt/storage/audio

# Optional: Use CPU instead of GPU
PODCAST_INTEL_TRANSCRIPTION_DEVICE=cpu

# Optional: LLM for coaching
PODCAST_INTEL_LLM_PROVIDER=openai
PODCAST_INTEL_LLM_API_KEY=sk-xxxxxxxxxxxxx
```

## Language Presets

Language presets provide pre-configured models and filler words for different languages.

### Available Presets

| Language | Code | Transcription | NER | Sentiment |
|----------|------|---------------|-----|-----------|
| English | `en` | whisper-large-v3-turbo | bert-base-NER | roberta-sentiment |
| Hebrew | `he` | ivrit-ai/whisper-large-v3-turbo | dictabert-ner | heBERT |
| Spanish | `es` | whisper-large-v3-turbo | bert-spanish-ner | beto-sentiment |

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

Preset files are located in `src/podcast_intel/presets/{language}.yaml`:

```yaml
# src/podcast_intel/presets/spanish.yaml
language: "es"

models:
  transcription: "openai/whisper-large-v3-turbo"
  ner: "mrm8488/bert-spanish-cased-finetuned-ner"
  sentiment: "finiteautomata/beto-sentiment-analysis"

filler_words:
  - "eh"
  - "este"
  - "bueno"
  - "pues"
  - "o sea"
  - "entonces"
  - "digamos"
```

## Specialization Files

For advanced use cases, you can create additional YAML files for domain-specific configuration.

### speakers.yaml

Detailed speaker profiles with metadata:

```yaml
speakers:
  - id: host
    name: "Alice Chen"
    role: host
    bio: "Tech journalist with 10 years experience"
    expertise:
      - "artificial intelligence"
      - "software engineering"
      - "startups"
    social:
      twitter: "@alicechen"
      linkedin: "alice-chen"

  - id: cohost
    name: "Bob Martinez"
    role: co-host
    bio: "Former CTO turned podcaster"
    expertise:
      - "cloud computing"
      - "devops"
      - "leadership"
```

### entities.yaml

Domain-specific entity lists for improved NER:

```yaml
entity_categories:
  companies:
    - "Google"
    - "Apple"
    - "Microsoft"
    - "Amazon"
    - "Meta"

  technologies:
    - "Python"
    - "JavaScript"
    - "React"
    - "TensorFlow"
    - "Kubernetes"

  people:
    - "Sam Altman"
    - "Satya Nadella"
    - "Sundar Pichai"

  products:
    - "ChatGPT"
    - "GitHub Copilot"
    - "Claude"
```

### scoring_weights.yaml

Custom PQS weights for different podcast types:

```yaml
# For interview podcasts
domain_weights:
  audio: 0.15      # Higher audio quality expectations
  delivery: 0.20
  structure: 0.15
  content: 0.35    # Focus on content quality
  engagement: 0.15

# For news/commentary podcasts
# domain_weights:
#   audio: 0.10
#   delivery: 0.30  # Clear delivery is critical
#   structure: 0.25 # Well-organized segments
#   content: 0.25
#   engagement: 0.10
```

## Configuration Precedence

Configuration is merged in the following order (highest precedence first):

1. **Environment variables** (e.g., `PODCAST_INTEL_LANGUAGE=he`)
2. **podcast.yaml** in current directory or parent directories
3. **Language preset** (based on `language` setting)
4. **Built-in defaults**

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

The final configuration will be:

- `transcription`: `"openai/whisper-large-v3"` (from podcast.yaml)
- `ner`: `"custom/ner-model"` (from environment variable)
- `sentiment`: `"avichr/heBERT_sentiment_analysis"` (from Hebrew preset)
- `embedding`: `"BAAI/bge-m3"` (from built-in defaults)

## Validation

podcast-intel validates configuration at startup using Pydantic. Common validation errors:

### Invalid Language Code

```
ValidationError: language must be a 2-letter ISO 639-1 code (e.g., 'en', 'he')
```

**Fix:** Use a valid language code like `"en"`, `"he"`, or `"es"`.

### Missing Required Fields

```
ValidationError: podcast.name is required
```

**Fix:** Add the `name` field to your `podcast.yaml`:

```yaml
podcast:
  name: "My Podcast"
```

### Invalid Color Format

```
ValidationError: primary_color must be a hex color (e.g., '#ff0000')
```

**Fix:** Use hex format for colors:

```yaml
branding:
  primary_color: "#2563eb"  # Correct
  # primary_color: "blue"   # Wrong
```

### Domain Weights Don't Sum to 1.0

```
ValidationError: scoring.domain_weights must sum to 1.0
```

**Fix:** Ensure weights add up to exactly 1.0:

```yaml
scoring:
  domain_weights:
    audio: 0.10
    delivery: 0.25
    structure: 0.20
    content: 0.25
    engagement: 0.20  # Total = 1.0
```

## Best Practices

1. **Use language presets** - They're optimized for each language
2. **Override sparingly** - Only override models if you have a specific reason
3. **Version control** - Commit `podcast.yaml` to track configuration changes
4. **Use .env for secrets** - Keep API keys and tokens in `.env` (not in git)
5. **Document customizations** - Add comments explaining why you changed defaults

## Examples

See the [examples/](../examples/) directory for complete configuration examples:

- `examples/quickstart/podcast.yaml` - Basic English tech podcast
- `examples/hebrew-sports/podcast.yaml` - Hebrew sports podcast with specialization
- `examples/interview-show/podcast.yaml` - Interview podcast with custom weights

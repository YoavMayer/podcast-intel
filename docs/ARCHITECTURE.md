# Architecture Overview

This document provides a technical overview of the podcast-intel system architecture.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         podcast-intel                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Ingestion  │  │Transcription │  │   Analysis   │          │
│  │              │  │              │  │              │          │
│  │ RSS Parser   │→ │   Whisper    │→ │ Filler Det.  │          │
│  │ Downloader   │  │ MFCC +       │  │ Silence      │          │
│  │              │  │ Spectral     │  │ Metrics      │          │
│  │              │  │ Clustering   │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                  │                  │
│         ↓                  ↓                  ↓                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │              SQLite Database (Episodes)           │          │
│  │  - Metadata  - Transcripts  - Analysis Results   │          │
│  └──────────────────────────────────────────────────┘          │
│         │                                                        │
│         ↓                                                        │
│  ┌──────────────┐                                               │
│  │   Scoring    │                                               │
│  │              │                                               │
│  │  PQS v3      │                                               │
│  │              │                                               │
│  └──────────────┘                                               │
│         │                                                        │
│         ↓                                                        │
│  ┌──────────────────────────────────────────────────┐          │
│  │           HTML Reports & JSON Exports             │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Module Overview

### 1. Ingestion (`podcast_intel/ingestion/`)

Fetches podcast episodes from RSS feeds and downloads audio files.

**Key Components:**

- `rss_parser.py` - Parses RSS/Atom feeds using feedparser
- `downloader.py` - Downloads MP3 files with progress tracking
- `mock_ingest.py` - Generates mock episodes for testing

**Data Flow:**

```
RSS Feed URL → parse_rss_feed() → Episode metadata
Episode URL → download_audio() → MP3 file in data/audio/
```

**Database Tables:**
- `episodes` - Episode metadata (guid, title, pub_date, audio_url)

### 2. Transcription (`podcast_intel/transcription/`)

Converts audio to timestamped text with speaker labels.

**Key Components:**

- `whisper_transcriber.py` - Whisper-based speech-to-text
- `diarize.py` - Speaker diarization: MFCC features + spectral clustering (`librosa`,
  `soundfile`, `scikit-learn`, CPU-only; no PyAnnote and no Hugging Face token)
- `transcribe.py` - Main transcription interface

**Pipeline:**

```
MP3 file → Whisper → Word-level timestamps
         ↓
    MFCC + spectral clustering → Speaker segments (SPEAKER_00, SPEAKER_01)
         ↓
    Merge → Transcript with speaker labels
```

> A PyAnnote-based diarizer exists in `tools/diarize_episode.py`. It is a standalone
> script, is **not** part of the installed package, and requires its own dependencies.

**Output Format:**

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "speaker": "Alice",
      "text": "Welcome to the show!",
      "words": [
        {"start": 0.0, "end": 0.4, "word": "Welcome", "confidence": 0.98}
      ]
    }
  ]
}
```

**Database Tables:**
- `segments` - Diarized transcript segments with speaker labels and timestamps
- `speakers` - Identified speakers with optional voice embeddings

### 3. Analysis (`podcast_intel/analysis/`)

Analysis primitives for extracting insights from a transcript.

**Key Components:**

- `filler_detector.py` - Filler word detection with regex
- `silence_analyzer.py` - Pause and silence detection
- `metrics.py` - Delivery metrics (pace, energy, pitch)
- `scorer.py` - PQS v3 scoring engine

**Pipeline:**

```
Transcript → Filler Detector → Filler word locations and counts
          ↓
  Silence Analyzer → Pause durations and frequencies
          ↓
      Metrics → Speaking pace, vocal energy, pitch
          ↓
      Scorer → PQS v3 score (0-100)
```

> **Not implemented.** NER and sentiment analysis are *planned*, not shipped. The
> `entities` / `entity_mentions` tables exist in the schema and the `ner` / `sentiment`
> model IDs exist in the config, but no packaged code populates them. Configuring a
> model ID does not make the stage run.

**Database Tables:**
- `metrics` - Episode and speaker-level metrics
- `silence_events` - Detected silence gaps and dead air
- `entities` + `entity_mentions` - reserved for the planned NER stage; unpopulated today

### 4. Not implemented: coaching and semantic search

Earlier revisions of this document described a `podcast_intel/coaching/` package
(LLM-powered per-speaker feedback) and a `podcast_intel/search/` package (embeddings +
ChromaDB + reranking). **Neither ships.** Both were placeholder modules whose every
function returned `None`, so they were removed rather than left importable. They remain
on the roadmap; `git log` retains the removed skeletons.

The `coaching_notes` and `embeddings` database tables are still in the schema and are
unpopulated by any packaged code.

### 5. Models (`podcast_intel/models/`)

Data models and database layer.

**Key Components:**

- `schema.py` - SQLite schema definition and initialization
- `database.py` - SQLite access layer with context-managed connections
- `entities.py` - Pydantic data models and entity type definitions

**Database Tables (10 total):**

| Table | Purpose |
|-------|---------|
| `episodes` | Episode metadata (guid, title, pub_date, audio_url, pqs_score) |
| `speakers` | Identified panelists/hosts with optional voice embeddings |
| `segments` | Diarized transcript segments (one per speaker turn) |
| `entities` | Canonical entity records (person, organization, location, event) |
| `entity_mentions` | Links entities to specific segments |
| `metrics` | Computed metrics per episode/speaker (WPM, talk time, etc.) |
| `silence_events` | Dead air and significant silence gaps |
| `embeddings` | Segment vector embeddings for semantic search |
| `coaching_notes` | LLM-generated per-speaker feedback |
| `topic_inventory` | Extracted discussion topics |

See `src/podcast_intel/models/schema.py` for the full schema definition.

## Configuration System

Configuration is managed through a hierarchical system:

```
Environment Variables (highest priority)
         ↓
    podcast.yaml
         ↓
  Language Preset
         ↓
  Built-in Defaults (lowest priority)
```

**Configuration Flow:**

```python
from podcast_intel.config import get_config

# One call performs all four layers:
#   defaults < presets/{language}.yaml < podcast.yaml < environment/.env
config = get_config()
config.transcription_model   # e.g. 'ivrit-ai/whisper-large-v3-turbo' under language: he
```

Internally (`config.py`):

```python
config     = Config()                       # environment + .env + defaults
language   = resolve_language(podcast_yaml, env_language)
overrides  = project(load_preset(language)) # skipped when no preset ships
overrides |= project(load_podcast_yaml())   # podcast.yaml beats the preset
# fields the environment set are never overwritten
```

Only the keys `Config` models are merged; the rest of `podcast.yaml`
(`speakers`, `branding`, `scoring`, `triggers`) is read directly by `cli.py` and
`triggers/rss_watcher.py` through `load_podcast_yaml()`. See
[CONFIGURATION.md](CONFIGURATION.md#configuration-precedence) for the full key map.

## CLI Architecture

The CLI (`podcast_intel/cli.py`) provides a simple interface to the system:

```
podcast-intel
├── ingest        # Fetch RSS and download episodes
├── mock          # Generate mock data for testing
├── watch         # Check RSS for new episodes (automation-friendly)
├── events        # Community events (check / upcoming / briefing)
├── transcribe    # Transcribe an episode (planned -- use tools/ for now)
├── analyze       # Full analysis pipeline (planned -- use tools/ for now)
└── report        # Generate HTML reports (planned -- use tools/ for now)
```

**Command Flow:**

```
User runs: python tools/run_episode_analysis.py 42

1. Load configuration (podcast.yaml + .env)
2. Check if audio exists, download if needed
3. Transcribe with Whisper + diarization
4. Run analysis pipeline (fillers, silence, metrics)
5. Compute PQS scores
6. Save to database
7. Generate HTML report
```

## Tools Directory

Standalone scripts for specific analysis tasks:

- `run_episode_analysis.py` - Full analysis pipeline for one episode
- `generate_one_pager.py` - Create executive summary HTML report
- `analyze_panel_chemistry.py` - Speaker interaction analysis
- `diarize_episode.py` - Speaker diarization with PyAnnote (script-only; PyAnnote is not
  a package dependency -- install `pyannote-audio` yourself to use this one)
- `merge_diarization.py` - Merge transcript with diarization
- `text_based_diarization.py` - Fallback diarization using text patterns

**Why tools/ instead of CLI?**

Tools provide more flexibility for:
- Custom pipelines
- Batch processing
- Integration with other systems
- Development and debugging

## Performance Considerations

### Transcription

- **GPU acceleration**: Use CUDA for 10-20x speedup
- **Model size**: Large models are slower but more accurate
- **Batch processing**: Process multiple episodes in parallel

```python
# CPU: ~10x real-time (1 hour audio = 10 hours processing)
# GPU: ~0.5-1x real-time (1 hour audio = 30-60 min processing)
```

### Diarization

- **Packaged diarizer** (MFCC + spectral clustering): CPU-only, no GPU memory needed
- **`tools/diarize_episode.py`** (PyAnnote, optional): 4-8 GB GPU memory, ~2-3x real-time on GPU

### NLP Models

- **Model loading**: Cache models to avoid reloading
- **Batch inference**: Process segments in batches

```python
# Load model once
model = load_model("dslim/bert-base-NER")

# Process in batches
for batch in chunks(segments, batch_size=32):
    entities = model.predict(batch)
```

### Database

- SQLite is sufficient for < 10,000 episodes
- For larger deployments, consider PostgreSQL
- Use indexes on frequently queried fields

```sql
CREATE INDEX idx_episodes_pub_date ON episodes(pub_date);
CREATE INDEX idx_segments_episode ON segments(episode_id);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_metrics_episode ON metrics(episode_id);
```

## Extension Points

### Adding New Analysis Modules

1. Create a new file in `podcast_intel/analysis/`
2. Implement the analysis function
3. Register in the analysis pipeline

```python
# podcast_intel/analysis/my_analyzer.py
def analyze_custom_metric(transcript):
    # Your analysis logic
    return metric_value

# Register in pipeline
from podcast_intel.analysis.my_analyzer import analyze_custom_metric
pipeline.register('custom_metric', analyze_custom_metric)
```

### Adding New Language Presets

1. Create `src/podcast_intel/presets/{language}.yaml`
2. Specify models and filler words
3. Test with sample audio

### Adding New Scoring Metrics

1. Add scoring function in `podcast_intel/analysis/scorer.py`
2. Update domain weights if needed
3. Add tests for the new metric

## Testing Strategy

```
tests/
├── conftest.py              # Fixtures and test configuration
├── test_mock_system.py      # Integration tests with mock data (87 tests)
├── test_scorer.py           # PQS v3 scoring engine tests
└── ...                      # Additional test modules
```

**Test Levels:**

1. **Unit tests**: Individual functions and classes
2. **Integration tests**: Full pipeline with mock data
3. **End-to-end tests**: Real audio files (in CI)

## Deployment

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,all]"
```

### Production

```bash
# Install without dev dependencies (not yet published to PyPI)
pip install "podcast-intel[all] @ git+https://github.com/YoavMayer/podcast-intel.git"

# Set up environment
export PODCAST_INTEL_DB_PATH=/data/podcast.db
export PODCAST_INTEL_AUDIO_DIR=/data/audio
export PODCAST_INTEL_HUGGINGFACE_TOKEN=hf_xxx

# Run analysis (via the standalone tools/ scripts)
python tools/run_episode_analysis.py 42
```

### Docker (Future)

```dockerfile
FROM python:3.10-slim
RUN pip install "podcast-intel[all] @ git+https://github.com/YoavMayer/podcast-intel.git"
CMD ["podcast-intel", "watch"]
```

## Dependencies

**Core:**
- `feedparser` - RSS parsing
- `requests` - HTTP downloads
- `pydantic` - Configuration validation
- `PyYAML` - YAML parsing

**Transcription (`[transcription]` extra):**
- `faster-whisper` - Optimized Whisper implementation
- `torch` - PyTorch backend
- `librosa`, `soundfile`, `scikit-learn`, `numpy` - MFCC + spectral-clustering diarization

**Analysis (`[analysis]` extra):**
- `transformers` - Hugging Face models
- `numpy` - Numerical computation
- `scipy` - Scientific computation

**Development:**
- `pytest` - Testing framework
- `ruff` - Linting and formatting
- `mypy` - Type checking

## Future Enhancements

1. **Web UI**: Browse episodes and view reports in browser
2. **Real-time analysis**: Live transcription during recording
3. **Multi-podcast support**: Compare across different podcasts
4. **Export integrations**: Push to Spotify, Apple Podcasts
5. **Advanced coaching**: Personalized improvement plans
6. **Collaborative features**: Team feedback and annotations

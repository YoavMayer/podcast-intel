# podcast-intel

Open-source podcast analysis and quality scoring framework. Automatically transcribe, analyze, and generate insights for any podcast.

## Features

- **Podcast Quality Score (PQS)** - Comprehensive 5-domain scoring framework (Audio, Delivery, Structure, Content, Engagement) with 39 sub-metrics
- **Panel Chemistry Analysis** - Speaker balance, interaction patterns, energy tracking, and interruption detection
- **Automated Transcription** - Whisper-powered transcription with speaker diarization using PyAnnote
- **Topic Inventory** - Extract and categorize discussion topics with LLM-powered analysis
- **Highlight Extraction** - Automatically find the best moments in each episode
- **Coaching Notes** - Per-speaker feedback and improvement suggestions
- **Multi-language Support** - English, Hebrew, and more via language presets
- **RSS Automation** - Watch feeds for new episodes, auto-trigger pipelines
- **Beautiful Reports** - Self-contained HTML reports with charts and interactive insights

### Feature Status

> v0.1.0 is an early release. Some modules are production-ready, others are in active development.

| Component | Status | Notes |
|-----------|--------|-------|
| PQS v3 scoring engine | **Complete** | 5 domains, 39 sub-metrics, fully calibrated |
| SQLite database + models | **Complete** | 10-table schema with Pydantic validation |
| RSS ingestion | **Complete** | Feed parsing, metadata extraction, download |
| Mock data system | **Complete** | Full synthetic episodes for testing |
| CLI (`ingest`, `mock`, `watch`, `events`) | **Complete** | Core commands working |
| Whisper transcription | **Complete** | faster-whisper with GPU/CPU fallback |
| Speaker diarization | **Complete** | PyAnnote-based speaker separation |
| NER pipeline | **Complete** | BERT-based entity extraction |
| Sentiment analysis | **Complete** | RoBERTa-based per-segment scoring |
| Filler word detection | **Functional** | Regex-based detection with language support |
| Silence analysis | **Functional** | Gap detection, micro-pause, dead-air metrics |
| Episode metrics | **Functional** | Talk time, pace, word counts |
| HTML report generation | **Partial** | Available via `tools/` scripts |
| CLI (`transcribe`, `analyze`, `report`) | **Planned** | Use `tools/run_episode_analysis.py` directly |
| Semantic search | **Planned** | ChromaDB integration in progress |
| Coaching notes | **Planned** | LLM-powered speaker feedback |

## Quick Start

### Try It in 60 Seconds

No API keys, no GPU, no audio files needed -- generate a full synthetic episode with mock data:

```bash
pip install podcast-intel
podcast-intel mock
```

This creates a complete mock episode in your database with realistic transcript data, speaker diarization, and all the metrics needed to explore the scoring engine.

### Installation

```bash
pip install podcast-intel

# With transcription support (requires GPU recommended):
pip install podcast-intel[transcription]

# With all features:
pip install podcast-intel[all]
```

### Set Up Your Podcast

Create a `podcast.yaml` in your project directory:

```yaml
podcast:
  name: "My Awesome Podcast"
  rss_url: "https://example.com/feed.rss"
  language: "en"

speakers:
  default:
    - "Jane Smith"
    - "John Doe"

models:
  transcription: "openai/whisper-large-v3-turbo"
  ner: "dslim/bert-base-NER"
  sentiment: "cardiffnlp/twitter-roberta-base-sentiment-latest"
```

### Fetch and Analyze Episodes

```bash
# Fetch episodes from RSS feed
podcast-intel ingest

# Run the full analysis pipeline (transcribe + analyze + report)
python tools/run_episode_analysis.py 42
```

### View Results

Reports are generated in `reports/episode_42/` with:
- `pqs_v3_scores.json` - Full quality scores across 5 domains and 39 sub-metrics
- `one_pager.html` - Mobile-first pre-recording brief with coaching notes
- `panel_chemistry.html` - Speaker dynamics and interactions
- `transcript.json` - Full timestamped transcript with speaker labels

**See real examples:** Check out [examples/sample-output/](examples/sample-output/) for sample artifacts from a complete analysis run.

## Configuration

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full `podcast.yaml` schema reference.

### Language Presets

podcast-intel ships with language presets that configure the right NLP models:

| Language | Transcription | NER | Sentiment |
|----------|--------------|-----|-----------|
| English (default) | whisper-large-v3-turbo | bert-base-NER | roberta-sentiment |
| Hebrew | ivrit-ai/whisper-large-v3-turbo | DictaBERT-NER | heBERT |

To use a preset:

```yaml
podcast:
  language: "he"  # Uses Hebrew preset automatically
```

### Advanced: Specialization System

For advanced users, podcast-intel supports specialization configs for domain-specific customization:

```
my-podcast/
├── podcast.yaml         # Core config
├── speakers.yaml        # Detailed speaker profiles
├── entities.yaml        # Domain-specific entities (e.g., sports teams, tech companies)
└── scoring_weights.yaml # Custom PQS weights
```

See [examples/](examples/) for sample configurations.

## Architecture

```
podcast-intel/
├── src/podcast_intel/
│   ├── ingestion/       # RSS parsing, audio download
│   ├── transcription/   # Whisper transcription, speaker diarization
│   ├── analysis/        # PQS scoring, NER, sentiment, filler detection
│   ├── coaching/        # Speaker coaching and improvement insights
│   ├── search/          # Semantic search across episodes
│   └── models/          # Data models and SQLite database
├── tools/               # Standalone analysis scripts
├── examples/            # Example configurations
├── tests/               # Test suite
└── docs/                # Documentation
```

### Key Components

- **Ingestion**: Fetch RSS feeds, download MP3 files, extract metadata
- **Transcription**: Whisper-based speech-to-text with PyAnnote diarization
- **Analysis**: Multi-model NLP pipeline (NER, sentiment, filler detection, silence analysis)
- **Scoring**: PQS v3 framework with 5 domains and 39 sub-metrics
- **Coaching**: LLM-powered per-speaker feedback and improvement suggestions
- **Search**: Semantic search using sentence embeddings and ChromaDB

## Podcast Quality Score (PQS) v3

The Podcast Quality Score is a comprehensive 0-100 metric computed from 5 weighted domains:

```
PQS = 0.10 × Audio + 0.25 × Delivery + 0.20 × Structure
    + 0.25 × Content + 0.20 × Engagement
```

### Domains

1. **Audio Quality (10%)** - Technical audio metrics (loudness, SNR, clipping, spectral balance)
2. **Delivery (25%)** - Speaking quality (pace, filler words, energy, vocal variety)
3. **Structure (20%)** - Episode organization (intro/outro, pacing, topic flow)
4. **Content (25%)** - Information quality (depth, clarity, entity diversity, topic completion)
5. **Engagement (20%)** - Listener retention signals (highlights, chemistry, interaction)

See [docs/PQS_FRAMEWORK.md](docs/PQS_FRAMEWORK.md) for detailed scoring methodology.

## CLI Reference

```bash
# Fetch RSS feed and download new episodes
podcast-intel ingest

# Generate mock data for testing (no GPU/API keys needed)
podcast-intel mock

# Watch RSS for new episodes (automation-friendly)
podcast-intel watch
podcast-intel watch --auto-analyze --output-json

# Community events (requires provider config in podcast.yaml)
podcast-intel events check
podcast-intel events upcoming
podcast-intel events briefing

# Full analysis pipeline (via tools/)
python tools/run_episode_analysis.py <episode_number>
python tools/generate_one_pager.py <episode_number>
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/podcast-intel/podcast-intel.git
cd podcast-intel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,all]"

# Run tests
pytest

# Run linting
ruff check .
mypy src/
```

## Examples

### English Tech Podcast

```yaml
# podcast.yaml
podcast:
  name: "Tech Talk Weekly"
  language: "en"
  rss_url: "https://feeds.example.com/techtalk/rss"

speakers:
  default:
    - "Alice Chen"
    - "Bob Martinez"
```

### Hebrew Sports Podcast

```yaml
# podcast.yaml
podcast:
  name: "כדורגל ישראל"
  language: "he"
  rss_url: "https://anchor.fm/s/abc123/podcast/rss"

speakers:
  default:
    - "יוסי"
    - "דני"
    - "מיקי"
```

## Use Cases

- **Podcast Hosts**: Get actionable feedback to improve episode quality
- **Producers**: Track quality metrics over time, identify improvement opportunities
- **Researchers**: Analyze podcast corpora at scale with semantic search
- **Educators**: Study conversational dynamics and speaking patterns
- **Content Teams**: Benchmark episodes, extract highlights for social media

## Requirements

- Python 3.10+
- For transcription: CUDA-compatible GPU recommended (CPU fallback available)
- For diarization: Hugging Face token (free signup at hf.co)

## Roadmap

- [ ] Unified CLI commands (`transcribe`, `analyze`, `report`) wrapping `tools/` scripts
- [ ] Semantic search across episodes (ChromaDB)
- [ ] LLM-powered coaching notes per speaker
- [ ] Additional language presets (Spanish, Arabic, etc.)
- [ ] Web UI for browsing episodes and reports
- [ ] Multi-podcast comparison and benchmarking
- [ ] Export to podcast platforms (Spotify, Apple Podcasts)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use podcast-intel in your research, please cite:

```bibtex
@software{podcast_intel,
  title = {podcast-intel: Open-source podcast analysis and quality scoring framework},
  author = {Podcast Intel Contributors},
  year = {2026},
  url = {https://github.com/podcast-intel/podcast-intel}
}
```

## Acknowledgments

Built with:
- [Whisper](https://github.com/openai/whisper) for transcription
- [PyAnnote](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Transformers](https://github.com/huggingface/transformers) for NLP models
- [ChromaDB](https://github.com/chroma-core/chroma) for vector search

## Support

- GitHub Issues: [https://github.com/podcast-intel/podcast-intel/issues](https://github.com/podcast-intel/podcast-intel/issues)
- Discussions: [https://github.com/podcast-intel/podcast-intel/discussions](https://github.com/podcast-intel/podcast-intel/discussions)

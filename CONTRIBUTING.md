# Contributing to podcast-intel

Thank you for your interest in contributing to podcast-intel! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Adding Language Presets](#adding-language-presets)
- [Creating Specializations](#creating-specializations)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Assume good intentions

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) CUDA-compatible GPU for transcription

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/podcast-intel/podcast-intel.git
cd podcast-intel

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,all]"
```

### Environment Variables

Create a `.env` file in the project root for local development:

```bash
# Optional: Override default paths
PODCAST_INTEL_DB_PATH=/custom/path/podcast_intel.db
PODCAST_INTEL_AUDIO_DIR=/custom/path/audio

# Required for diarization
PODCAST_INTEL_HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx

# Optional: LLM API keys for coaching features
PODCAST_INTEL_LLM_PROVIDER=openai
PODCAST_INTEL_LLM_API_KEY=sk-xxxxxxxxxxxxx
```

### Verify Installation

```bash
# Run tests to ensure everything is working
pytest

# Try the CLI
podcast-intel --help
```

## Project Structure

```
podcast-intel/
├── src/podcast_intel/          # Main source code
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── config.py               # Configuration management
│   ├── analysis/               # Analysis pipeline
│   │   ├── filler_detector.py  # Filler word detection
│   │   ├── metrics.py          # Metric computation
│   │   ├── ner_pipeline.py     # Named entity recognition
│   │   ├── scorer.py           # PQS v3 scoring
│   │   ├── sentiment.py        # Sentiment analysis
│   │   └── silence_analyzer.py # Silence detection
│   ├── coaching/               # Speaker coaching
│   │   ├── coach.py            # Coaching engine
│   │   └── interruptions.py    # Interruption detection
│   ├── ingestion/              # Data ingestion
│   │   ├── downloader.py       # Audio downloader
│   │   ├── rss_parser.py       # RSS feed parser
│   │   └── mock_ingest.py      # Mock data generator
│   ├── models/                 # Data models
│   │   ├── database.py         # SQLite database
│   │   ├── entities.py         # Entity models
│   │   └── schema.py           # Pydantic schemas
│   ├── presets/                # Language presets
│   │   ├── english.yaml
│   │   └── hebrew.yaml
│   ├── search/                 # Semantic search
│   │   ├── embedder.py         # Embedding generation
│   │   ├── query.py            # Query interface
│   │   └── vector_store.py     # ChromaDB integration
│   └── transcription/          # Transcription pipeline
│       ├── diarize.py          # Speaker diarization
│       ├── transcribe.py       # Transcription interface
│       ├── whisper_transcriber.py  # Whisper implementation
│       └── mock_transcribe.py  # Mock transcriber
├── tools/                      # Standalone analysis tools
│   ├── run_episode_analysis.py
│   ├── generate_one_pager.py
│   ├── analyze_panel_chemistry.py
│   ├── diarize_episode.py
│   ├── merge_diarization.py
│   └── text_based_diarization.py
├── tests/                      # Test suite
│   ├── conftest.py             # Pytest fixtures
│   └── test_mock_system.py     # Integration tests
├── examples/                   # Example configurations
│   └── quickstart/
│       └── podcast.yaml
├── docs/                       # Documentation
│   ├── CONFIGURATION.md
│   ├── PQS_FRAMEWORK.md
│   └── ARCHITECTURE.md
├── pyproject.toml              # Project metadata and dependencies
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## Running Tests

We use pytest for testing.

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=podcast_intel --cov-report=html

# Run specific test file
pytest tests/test_mock_system.py

# Run tests matching a pattern
pytest -k "test_transcription"

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures from `tests/conftest.py`

Example test:

```python
def test_filler_detection():
    from podcast_intel.analysis.filler_detector import FillerDetector

    detector = FillerDetector(language="en")
    text = "Um, like, I think, you know, it's basically great."

    fillers = detector.detect(text)
    assert len(fillers) > 0
    assert any(f.word == "um" for f in fillers)
```

## Code Style

We use `ruff` for linting and formatting, and `mypy` for type checking.

### Linting

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Type Checking

```bash
# Run mypy
mypy src/

# Check specific file
mypy src/podcast_intel/analysis/scorer.py
```

### Style Guidelines

- **Line length**: Maximum 100 characters
- **Imports**: Organize with `ruff` (stdlib, third-party, local)
- **Type hints**: Use type hints for all public functions
- **Docstrings**: Use Google-style docstrings

Example:

```python
def compute_pqs_score(
    audio: float,
    delivery: float,
    structure: float,
    content: float,
    engagement: float
) -> float:
    """Compute the Podcast Quality Score.

    Args:
        audio: Audio domain score (0-100)
        delivery: Delivery domain score (0-100)
        structure: Structure domain score (0-100)
        content: Content domain score (0-100)
        engagement: Engagement domain score (0-100)

    Returns:
        Overall PQS score (0-100)
    """
    return (
        0.10 * audio +
        0.25 * delivery +
        0.20 * structure +
        0.25 * content +
        0.20 * engagement
    )
```

## Pull Request Process

1. **Fork the repository** and create a new branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Run the test suite** and ensure all tests pass:
   ```bash
   pytest
   ruff check .
   mypy src/
   ```

5. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add support for Spanish language preset"
   ```

6. **Push to your fork** and create a pull request:
   ```bash
   git push origin feature/my-new-feature
   ```

7. **Write a clear PR description** including:
   - What problem does this solve?
   - How does it work?
   - Any breaking changes?
   - Screenshots (if UI changes)

8. **Wait for review** - maintainers will review and may request changes

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Update documentation for new features
- Add entries to CHANGELOG.md (if applicable)
- Ensure CI passes (tests, linting, type checking)

## Adding Language Presets

Language presets configure NLP models and filler words for different languages.

### Create a New Preset

1. Create `src/podcast_intel/presets/{language}.yaml`:

```yaml
# Spanish language preset
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

2. Update `src/podcast_intel/presets/__init__.py`:

```python
AVAILABLE_PRESETS = {
    "en": "english.yaml",
    "he": "hebrew.yaml",
    "es": "spanish.yaml",  # Add your preset
}
```

3. Add tests in `tests/test_presets.py`:

```python
def test_spanish_preset():
    preset = load_preset("es")
    assert preset["language"] == "es"
    assert "eh" in preset["filler_words"]
```

4. Update documentation in `docs/CONFIGURATION.md`

### Guidelines for Language Presets

- Use models from Hugging Face Hub when possible
- Include 10-20 common filler words
- Test transcription quality on sample audio
- Document any special requirements (e.g., RTL text)

## Creating Specializations

Specializations are domain-specific configurations for podcasts (e.g., sports, tech, finance).

### Example: Sports Podcast Specialization

```
examples/sports-podcast/
├── podcast.yaml           # Base config
├── speakers.yaml          # Speaker profiles
├── entities.yaml          # Sports-specific entities
└── scoring_weights.yaml   # Custom PQS weights
```

**speakers.yaml:**
```yaml
speakers:
  - id: host
    name: "John Smith"
    role: host
    expertise: ["soccer", "basketball"]

  - id: analyst
    name: "Maria Garcia"
    role: analyst
    expertise: ["statistics", "tactics"]
```

**entities.yaml:**
```yaml
entity_categories:
  teams:
    - "Manchester United"
    - "Real Madrid"
    - "Barcelona"

  players:
    - "Lionel Messi"
    - "Cristiano Ronaldo"

  leagues:
    - "Premier League"
    - "La Liga"
```

**scoring_weights.yaml:**
```yaml
# Custom PQS weights for sports podcasts
domain_weights:
  audio: 0.10
  delivery: 0.20
  structure: 0.20
  content: 0.30      # Higher weight on content for sports analysis
  engagement: 0.20
```

### Testing Specializations

Create example podcasts in `examples/` and ensure they work end-to-end.

## Release Process

Maintainers follow this process for releases:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create a git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. Build and publish to PyPI:
   ```bash
   python -m build
   twine upload dist/*
   ```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Join our community chat (link TBD)

Thank you for contributing to podcast-intel!

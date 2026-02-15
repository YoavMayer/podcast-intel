# GitHub Actions Workflows

This directory contains the CI/CD and automation workflows for the podcast-intel project.

## Workflows Overview

| Workflow | File | Schedule | Purpose |
|----------|------|----------|---------|
| **CI** | `workflows/ci.yml` | Push / PR to `main` | Lint, type check, test across Python 3.10--3.12 |
| **Episode Watch** | `workflows/episode-watch.yml` | Every 15 min | Poll RSS feed for new episodes and run analysis |
| **Community Events** | `workflows/community-events.yml` | Daily 06:00 UTC | Check for community events and generate briefings |

All three workflows are **reusable** via `workflow_call`, so private or downstream
repositories can include them without duplicating YAML.

---

## Required GitHub Secrets

Configure these in **Settings > Secrets and variables > Actions**:

| Secret | Used by | Required | Description |
|--------|---------|----------|-------------|
| `PODCAST_INTEL_RSS_URL` | Episode Watch | No | RSS feed URL. Falls back to `podcast.yaml` `podcast.rss_url` if not set. |
| `FOOTBALL_DATA_API_KEY` | Community Events | No | API key for [football-data.org](https://www.football-data.org/). Required only if using the `football-data` provider. |

> **Note:** Secrets are optional in the OSS repo because all podcast-specific
> configuration lives in `podcast.yaml`. Secrets are primarily useful for
> private forks that want to keep API keys out of committed files.

---

## Workflow Details

### CI (`ci.yml`)

Runs automatically on every push to `main` and on pull requests.

**What it does:**

1. Checks out the repository
2. Sets up Python (matrix: 3.10, 3.11, 3.12)
3. Installs `podcast-intel[dev]` (includes ruff, mypy, pytest)
4. Runs `ruff check` and `ruff format --check` for linting
5. Runs `mypy` for type checking (if `[tool.mypy]` is configured in `pyproject.toml`)
6. Runs `pytest` with coverage

**Reusable usage:**

```yaml
# In your repo's .github/workflows/ci.yml
jobs:
  ci:
    uses: podcast-intel/podcast-intel/.github/workflows/ci.yml@main
```

### Episode Watch (`episode-watch.yml`)

Polls the podcast RSS feed every 15 minutes for new episodes.

**What it does:**

1. Runs `podcast-intel watch --once --output-json` to check for new episodes
2. If new episodes are found, runs the analysis pipeline
3. Uploads watch results and analysis reports as artifacts
4. Creates a job summary with episode details

**Manual trigger:**

Go to **Actions > Episode Watch > Run workflow** and choose:
- `dry_run`: Preview only, no analysis
- `auto_analyze`: Whether to run the analysis pipeline

**Reusable usage:**

```yaml
# In your private repo
jobs:
  watch:
    uses: podcast-intel/podcast-intel/.github/workflows/episode-watch.yml@main
    secrets:
      PODCAST_INTEL_RSS_URL: ${{ secrets.MY_RSS_URL }}
```

### Community Events (`community-events.yml`)

Checks for community events daily and generates briefings.

**What it does:**

1. Runs `podcast-intel events check --output-json` to find recent events
2. If events are found, generates briefings in the configured formats
3. Uploads briefings as artifacts
4. Optionally commits briefings to `reports/briefings/`
5. Creates a job summary with event details

**Manual trigger:**

Go to **Actions > Community Events > Run workflow** and choose:
- `briefing_formats`: Output formats (e.g., `html,whatsapp,markdown`)
- `auto_commit`: Whether to commit briefings to the repo

**Reusable usage:**

```yaml
# In your private repo
jobs:
  events:
    uses: podcast-intel/podcast-intel/.github/workflows/community-events.yml@main
    secrets:
      FOOTBALL_DATA_API_KEY: ${{ secrets.MY_FOOTBALL_KEY }}
    with:
      auto_commit: true
```

---

## Configuration via `podcast.yaml`

All podcast-specific values live in `podcast.yaml` at the project root.
Workflows read this file at runtime -- no hardcoded values in the YAML.

### Example `podcast.yaml` with triggers configuration

```yaml
podcast:
  name: "My Podcast"
  language: "en"
  rss_url: "https://example.com/feed.rss"

speakers:
  default:
    - "Host"
    - "Co-host"

models:
  transcription: "openai/whisper-large-v3-turbo"
  ner: "dslim/bert-base-NER"
  sentiment: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  embedding: "BAAI/bge-m3"

branding:
  show_name: "MY PODCAST"
  primary_color: "#2563eb"
  footer_text: "Podcast Intelligence"

analysis:
  episode_dir_prefix: "episode"
  episodes_json: "episodes.json"

# ---------------------------------------------------------------
# Triggers -- configure automation behavior
# ---------------------------------------------------------------
triggers:
  # RSS watcher settings (used by episode-watch.yml)
  rss_watch:
    enabled: true
    # How many recent GUIDs to remember for deduplication
    known_guids_limit: 500

  # Community events settings (used by community-events.yml)
  community_events:
    enabled: true
    provider: "football-data"     # Provider name (see providers/)
    provider_config:
      team_id: 73                 # football-data.org team ID
      competition_ids:            # Competitions to monitor
        - "PL"
        - "CL"
        - "FAC"
    on_event:
      briefing:
        formats:
          - "html"
          - "whatsapp"
        output_dir: "reports/briefings"
```

### Enabling / Disabling Triggers

Each trigger section has an `enabled` flag:

```yaml
triggers:
  rss_watch:
    enabled: false    # Disable RSS polling
  community_events:
    enabled: true     # Keep events active
```

When `enabled: false`, the CLI commands will print a warning and skip
processing. The GitHub Actions cron still fires but exits early.

To completely disable a scheduled workflow, you can also disable it from the
GitHub Actions UI: go to **Actions > [Workflow Name] > ... > Disable workflow**.

---

## Artifacts

Both the Episode Watch and Community Events workflows upload artifacts:

| Artifact | Workflow | Retention | Contents |
|----------|----------|-----------|----------|
| `watch-result` | Episode Watch | 30 days | JSON output from RSS watcher |
| `episode-reports` | Episode Watch | 90 days | Generated analysis reports |
| `events-result` | Community Events | 30 days | JSON output from events check |
| `event-briefings` | Community Events | 90 days | Generated briefing files |

Download artifacts from the **Actions** tab after each workflow run.

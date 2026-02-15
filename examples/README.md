# podcast-intel Examples

This directory contains example configurations for different types of podcasts.

## Available Examples

### Sample Output

See what podcast-intel produces after analyzing an episode.

**Path:** `examples/sample-output/`

**Files:**
- `pqs_v3_scores.json` -- Full PQS v3 quality scores with 5 domains, 39 sub-metrics, speaker breakdown, and coaching notes
- `pre_recording_brief_ep43.html` -- Mobile-first HTML brief that panelists review in 2 minutes before recording
- `pipeline_results.json` -- Pipeline execution summary showing each step (RSS, download, transcribe, diarize, analyze, report)
- `transcript_snippet.json` -- Sample transcript with word-level timestamps and speaker attribution

**Open the brief in your browser:**

```bash
open examples/sample-output/pre_recording_brief_ep43.html
```

### Quickstart

A basic English tech podcast configuration to get you started.

**Path:** `examples/quickstart/podcast.yaml`

**Use case:** General-purpose podcast, tech/business discussions

**Features:**
- English language preset
- Standard speakers configuration
- Default models and branding
- Ready to use out-of-the-box

**To use:**

```bash
# Copy to your project
cp examples/quickstart/podcast.yaml /path/to/your/podcast/

# Edit with your details
vim podcast.yaml

# Start analyzing
podcast-intel ingest
podcast-intel analyze 1
```

## Creating Your Own Configuration

### 1. Start with a template

```bash
# Copy the quickstart example
cp examples/quickstart/podcast.yaml ./

# Or use the CLI to generate a template
podcast-intel init
```

### 2. Customize for your podcast

Edit `podcast.yaml` with your podcast details:

```yaml
podcast:
  name: "Your Podcast Name"
  rss_url: "https://your-podcast-feed.com/rss"
  language: "en"  # or "he", "es", etc.

speakers:
  default:
    - "Your Name"
    - "Co-host Name"
```

### 3. Add custom branding (optional)

```yaml
branding:
  show_name: "YOUR PODCAST"
  primary_color: "#FF6B35"  # Your brand color
  footer_text: "Your Podcast Intelligence"
```

### 4. Adjust scoring weights (optional)

For interview podcasts, emphasize content:

```yaml
scoring:
  domain_weights:
    audio: 0.15
    delivery: 0.20
    structure: 0.15
    content: 0.35  # Higher weight on guest insights
    engagement: 0.15
```

For conversational/entertainment podcasts, emphasize engagement:

```yaml
scoring:
  domain_weights:
    audio: 0.10
    delivery: 0.20
    structure: 0.15
    content: 0.20
    engagement: 0.35  # Higher weight on chemistry
```

## Advanced: Specialization Configs

For domain-specific podcasts (sports, finance, etc.), create additional config files:

### Example: Sports Podcast

```
my-sports-podcast/
├── podcast.yaml         # Base configuration
├── speakers.yaml        # Detailed speaker profiles
├── entities.yaml        # Sports-specific entities
└── scoring_weights.yaml # Custom PQS weights
```

**speakers.yaml:**

```yaml
speakers:
  - id: host
    name: "John Smith"
    role: host
    expertise:
      - "soccer"
      - "basketball"
    social:
      twitter: "@johnsmith"

  - id: analyst
    name: "Maria Garcia"
    role: analyst
    expertise:
      - "statistics"
      - "tactics"
```

**entities.yaml:**

```yaml
entity_categories:
  teams:
    - "Manchester United"
    - "Real Madrid"
    - "Barcelona"
    - "Bayern Munich"

  players:
    - "Lionel Messi"
    - "Cristiano Ronaldo"
    - "Kylian Mbappé"

  leagues:
    - "Premier League"
    - "La Liga"
    - "Bundesliga"
    - "Serie A"

  competitions:
    - "Champions League"
    - "World Cup"
    - "Europa League"
```

**scoring_weights.yaml:**

```yaml
# Custom weights for sports analysis podcasts
domain_weights:
  audio: 0.10
  delivery: 0.20
  structure: 0.20
  content: 0.30      # Emphasize analysis quality
  engagement: 0.20

# Optionally, customize sub-metric weights within domains
content_weights:
  topic_depth: 0.25         # Deep tactical analysis
  entity_diversity: 0.20    # Mention many teams/players
  domain_relevance: 0.20    # Sports-specific terms
  information_density: 0.15
  # ... other sub-metrics
```

## Language-Specific Examples

### Hebrew Podcast

```yaml
podcast:
  name: "הפודקאסט שלי"
  language: "he"
  rss_url: "https://anchor.fm/s/abc123/podcast/rss"

speakers:
  default:
    - "יוסי כהן"
    - "דני לוי"
    - "מיקי אברמוב"

models:
  # Uses Hebrew preset automatically
  # Override if needed:
  transcription: "ivrit-ai/whisper-large-v3-turbo"
  ner: "dicta-il/dictabert-ner"
  sentiment: "avichr/heBERT_sentiment_analysis"

branding:
  show_name: "הפודקאסט שלי"
  primary_color: "#0066cc"
  footer_text: "מערכת ניתוח הפודקאסט"
```

### Spanish Podcast

```yaml
podcast:
  name: "Mi Podcast"
  language: "es"
  rss_url: "https://feeds.example.com/mipodcast.rss"

speakers:
  default:
    - "María González"
    - "Carlos Rodríguez"

branding:
  show_name: "MI PODCAST"
  primary_color: "#FF4136"
  footer_text: "Inteligencia del Podcast"
```

## Multi-Episode Speaker Overrides

If you have different guests each episode:

```yaml
speakers:
  default:
    - "Host Name"
    - "Co-host Name"

  # Per-episode overrides
  episodes:
    1:
      - "Host Name"
      - "Guest: Jane Doe"
    2:
      - "Host Name"
      - "Guest: John Smith"
    3:
      - "Host Name"
      - "Co-host Name"  # Back to regular co-host
```

## Testing Your Configuration

After creating your configuration:

```bash
# Generate mock data to test
podcast-intel mock

# Analyze a real episode
podcast-intel analyze 1

# Check the generated report
open reports/episode_1/one_pager.html
```

## Need Help?

- See [docs/CONFIGURATION.md](../docs/CONFIGURATION.md) for full config reference
- See [docs/PQS_FRAMEWORK.md](../docs/PQS_FRAMEWORK.md) for scoring details
- Open an issue on GitHub for questions or bugs

## Contributing Examples

Have a great configuration for a specific podcast type? Please contribute it!

1. Create a new directory: `examples/your-podcast-type/`
2. Add your `podcast.yaml` and any specialization files
3. Add a README explaining the configuration
4. Submit a pull request

Examples we'd love to see:

- News/commentary podcasts
- Educational/teaching podcasts
- True crime podcasts
- Comedy podcasts
- Interview shows
- Panel discussions
- Solo podcasts

Thank you for using podcast-intel!

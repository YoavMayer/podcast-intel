# Podcast Quality Score (PQS) v3.0 Framework

The Podcast Quality Score (PQS) is a comprehensive 0-100 metric that evaluates podcast episodes across multiple dimensions of quality.

## Overview

```
PQS = 0.10 × Audio + 0.25 × Delivery + 0.20 × Structure
    + 0.25 × Content + 0.20 × Engagement
```

### Version History

- **PQS v1.0**: Basic metrics (audio, speaking pace)
- **PQS v2.1**: Expanded to 4 domains, 21 sub-metrics
- **PQS v3.0**: Current version - 5 domains, 39 sub-metrics

## The 5 Domains

| Domain | Weight | Focus | Sub-metrics |
|--------|--------|-------|-------------|
| **Audio Quality** | 10% | Technical audio production | 5 |
| **Delivery** | 25% | Speaking quality and clarity | 9 |
| **Structure** | 20% | Episode organization | 8 |
| **Content** | 25% | Information quality | 10 |
| **Engagement** | 20% | Listener retention signals | 7 |

## 1. Audio Quality (10%)

Technical audio production metrics.

### Sub-metrics (5 total)

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| **Loudness Compliance** | 30% | Deviation from -16 LUFS (Spotify/Apple standard) | ±1 dB |
| **Signal-to-Noise Ratio** | 25% | Speech clarity vs background noise | > 20 dB |
| **Clipping Index** | 15% | Percentage of samples exceeding 0.99 peak | 0% |
| **Loudness Range** | 15% | Dynamic range consistency (LU) | 6-12 LU |
| **Spectral Balance** | 15% | Frequency distribution quality | Balanced |

### Scoring Function

```python
def score_audio_quality(metrics):
    loudness = score_loudness_compliance(metrics['lufs'])
    snr = score_signal_to_noise(metrics['snr_db'])
    clipping = score_clipping_index(metrics['clip_pct'])
    dynamic = score_loudness_range(metrics['loudness_range_lu'])
    spectral = score_spectral_balance(metrics['spectral_balance'])

    return (
        0.30 * loudness +
        0.25 * snr +
        0.15 * clipping +
        0.15 * dynamic +
        0.15 * spectral
    )
```

### Interpretation

- **90-100**: Professional studio quality
- **75-89**: Excellent home studio
- **60-74**: Good quality, minor issues
- **40-59**: Acceptable, noticeable issues
- **0-39**: Poor quality, needs improvement

## 2. Delivery (25%)

Speaking quality, pace, and vocal characteristics.

### Sub-metrics (9 total)

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| **Speaking Pace** | 20% | Words per minute | 140-180 wpm |
| **Filler Word Rate** | 18% | Filler words per 100 words | < 3 per 100 |
| **Vocal Energy** | 15% | Average RMS energy (dB) | -25 to -20 dB |
| **Pitch Variety** | 12% | Fundamental frequency standard deviation | > 20 Hz |
| **Pause Quality** | 12% | Natural pause rate and duration | 8-15% silence |
| **Articulation Clarity** | 10% | Confidence scores from transcription | > 0.85 |
| **Vocal Consistency** | 8% | Energy variance across speakers | < 5 dB |
| **Interruption Rate** | 3% | Interruptions per minute | < 2 per min |
| **Overlap Percentage** | 2% | Speech overlap percentage | < 5% |

### Scoring Function

```python
def score_delivery(metrics):
    pace = score_speaking_pace(metrics['wpm'])
    fillers = score_filler_rate(metrics['filler_per_100'])
    energy = score_vocal_energy(metrics['avg_rms_db'])
    pitch = score_pitch_variety(metrics['f0_std_hz'])
    pauses = score_pause_quality(metrics['silence_pct'])
    clarity = score_articulation(metrics['avg_confidence'])
    consistency = score_vocal_consistency(metrics['energy_variance'])
    interruptions = score_interruptions(metrics['interruptions_per_min'])
    overlap = score_overlap(metrics['overlap_pct'])

    return (
        0.20 * pace +
        0.18 * fillers +
        0.15 * energy +
        0.12 * pitch +
        0.12 * pauses +
        0.10 * clarity +
        0.08 * consistency +
        0.03 * interruptions +
        0.02 * overlap
    )
```

### Interpretation

- **90-100**: Exceptional speaking quality, engaging and clear
- **75-89**: Professional delivery, minimal issues
- **60-74**: Good delivery, some room for improvement
- **40-59**: Acceptable, noticeable filler words or pacing issues
- **0-39**: Poor delivery, needs significant improvement

## 3. Structure (20%)

Episode organization and pacing.

### Sub-metrics (8 total)

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| **Intro Quality** | 20% | Intro length and energy | 30-90 sec |
| **Outro Quality** | 15% | Outro presence and length | 15-60 sec |
| **Segment Balance** | 18% | Even distribution of topics | Even splits |
| **Topic Transitions** | 15% | Smooth transitions between topics | Clear |
| **Pacing Consistency** | 12% | Variance in speaking pace | < 20 wpm |
| **Time Structure** | 10% | Episode length appropriateness | 30-60 min |
| **Content Density** | 8% | Information per minute | Balanced |
| **Narrative Flow** | 2% | Logical progression | Coherent |

### Scoring Function

```python
def score_structure(metrics):
    intro = score_intro_quality(metrics['intro_seconds'], metrics['intro_energy'])
    outro = score_outro_quality(metrics['outro_seconds'])
    segments = score_segment_balance(metrics['segment_durations'])
    transitions = score_transitions(metrics['transition_quality'])
    pacing = score_pacing_consistency(metrics['wpm_variance'])
    timing = score_time_structure(metrics['duration_minutes'])
    density = score_content_density(metrics['info_per_minute'])
    flow = score_narrative_flow(metrics['flow_score'])

    return (
        0.20 * intro +
        0.15 * outro +
        0.18 * segments +
        0.15 * transitions +
        0.12 * pacing +
        0.10 * timing +
        0.08 * density +
        0.02 * flow
    )
```

### Interpretation

- **90-100**: Excellently structured, professional flow
- **75-89**: Well-organized, clear segments
- **60-74**: Good structure, minor pacing issues
- **40-59**: Acceptable, could use better organization
- **0-39**: Poor structure, disorganized

## 4. Content (25%)

Information quality, depth, and relevance.

### Sub-metrics (10 total)

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| **Topic Depth** | 20% | Average discussion time per topic | 5-15 min |
| **Entity Diversity** | 18% | Unique entities mentioned | > 20 |
| **Domain Relevance** | 15% | Domain-specific entity density | > 10% |
| **Information Density** | 12% | Entities per 100 words | 3-8 |
| **Topic Completion** | 10% | Percentage of topics fully covered | > 80% |
| **Factual Accuracy** | 10% | Verifiable claims vs total claims | > 90% |
| **Expert Language** | 8% | Technical term usage | Appropriate |
| **Citation Quality** | 4% | Sources and references | Present |
| **Insight Originality** | 2% | Novel insights vs common knowledge | Some |
| **Takeaway Clarity** | 1% | Clear actionable takeaways | 3-5 |

### Scoring Function

```python
def score_content(metrics):
    depth = score_topic_depth(metrics['avg_topic_duration_min'])
    diversity = score_entity_diversity(metrics['unique_entities'])
    relevance = score_domain_relevance(metrics['domain_entity_pct'])
    density = score_info_density(metrics['entities_per_100_words'])
    completion = score_topic_completion(metrics['completed_topics_pct'])
    accuracy = score_factual_accuracy(metrics['verifiable_claims_pct'])
    expertise = score_expert_language(metrics['technical_term_rate'])
    citations = score_citations(metrics['citation_count'])
    originality = score_originality(metrics['novel_insight_pct'])
    takeaways = score_takeaways(metrics['takeaway_count'])

    return (
        0.20 * depth +
        0.18 * diversity +
        0.15 * relevance +
        0.12 * density +
        0.10 * completion +
        0.10 * accuracy +
        0.08 * expertise +
        0.04 * citations +
        0.02 * originality +
        0.01 * takeaways
    )
```

### Interpretation

- **90-100**: Exceptional content, deep insights, highly informative
- **75-89**: Strong content, good depth and relevance
- **60-74**: Good content, some superficiality
- **40-59**: Acceptable, lacks depth or focus
- **0-39**: Poor content, shallow or off-topic

## 5. Engagement (20%)

Listener retention and interaction signals.

### Sub-metrics (7 total)

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| **Highlight Moments** | 25% | Number of engaging highlights | 5-10 |
| **Panel Chemistry** | 22% | Speaker interaction quality | Balanced |
| **Sentiment Variance** | 18% | Emotional range and dynamics | Varied |
| **Energy Peaks** | 15% | Moments of high energy | 3-5 |
| **Question Engagement** | 10% | Questions asked and answered | > 5 |
| **Humor Indicators** | 8% | Laughter and light moments | Present |
| **Audience Hooks** | 2% | Compelling openings/cliffhangers | Present |

### Scoring Function

```python
def score_engagement(metrics):
    highlights = score_highlights(metrics['highlight_count'])
    chemistry = score_chemistry(metrics['balance_score'])
    sentiment = score_sentiment_variance(metrics['sentiment_std'])
    energy = score_energy_peaks(metrics['energy_peak_count'])
    questions = score_questions(metrics['question_count'])
    humor = score_humor(metrics['laughter_count'])
    hooks = score_hooks(metrics['hook_score'])

    return (
        0.25 * highlights +
        0.22 * chemistry +
        0.18 * sentiment +
        0.15 * energy +
        0.10 * questions +
        0.08 * humor +
        0.02 * hooks
    )
```

### Interpretation

- **90-100**: Highly engaging, compelling content
- **75-89**: Engaging, good dynamics
- **60-74**: Moderately engaging
- **40-59**: Some engaging moments, inconsistent
- **0-39**: Low engagement, monotonous

## Overall PQS Interpretation

| Score Range | Grade | Description | Action Items |
|-------------|-------|-------------|--------------|
| **90-100** | A+ | Exceptional | Maintain quality, use as reference |
| **85-89** | A | Excellent | Minor polish, ready to publish |
| **75-84** | B+ | Very Good | Small improvements, strong episode |
| **65-74** | B | Good | Address specific weak domains |
| **55-64** | C+ | Acceptable | Needs improvement in multiple areas |
| **45-54** | C | Below Average | Significant work needed |
| **0-44** | D/F | Poor | Major overhaul required |

## Customizing Weights

You can customize domain weights in `podcast.yaml` for different podcast types:

### Interview Podcast

```yaml
scoring:
  domain_weights:
    audio: 0.15      # Higher audio standards
    delivery: 0.20
    structure: 0.15
    content: 0.35    # Focus on guest insights
    engagement: 0.15
```

### News/Commentary Podcast

```yaml
scoring:
  domain_weights:
    audio: 0.10
    delivery: 0.30   # Clear delivery is critical
    structure: 0.25  # Well-organized segments
    content: 0.25
    engagement: 0.10
```

### Conversational/Entertainment Podcast

```yaml
scoring:
  domain_weights:
    audio: 0.10
    delivery: 0.20
    structure: 0.15
    content: 0.20
    engagement: 0.35  # Chemistry and entertainment
```

## Score Calculation Example

Given these domain scores:

- Audio: 85
- Delivery: 78
- Structure: 82
- Content: 90
- Engagement: 75

```python
PQS = (0.10 × 85) + (0.25 × 78) + (0.20 × 82) + (0.25 × 90) + (0.20 × 75)
    = 8.5 + 19.5 + 16.4 + 22.5 + 15.0
    = 81.9
```

**Result**: PQS = 81.9 (B+ / Very Good)

## Using PQS for Improvement

### Step 1: Identify Weak Domains

Look at individual domain scores to find areas needing improvement.

Example:
- Audio: 85 ✓
- Delivery: 65 ⚠️ (needs work)
- Structure: 82 ✓
- Content: 90 ✓
- Engagement: 75 ✓

### Step 2: Drill Down to Sub-metrics

Within the weak domain (Delivery), check sub-metrics:

- Speaking Pace: 85 ✓
- Filler Word Rate: 45 ⚠️ (too many fillers)
- Vocal Energy: 72 ✓
- Pitch Variety: 55 ⚠️ (monotone)

### Step 3: Take Action

Based on sub-metrics:
1. **Reduce filler words**: Practice awareness, use pauses instead
2. **Increase pitch variety**: Vary emphasis, use inflection

### Step 4: Track Progress

Compare PQS across episodes to measure improvement:

```
Episode 40: PQS = 72.3
Episode 41: PQS = 75.8 (+3.5)
Episode 42: PQS = 78.2 (+2.4)
Episode 43: PQS = 81.9 (+3.7)
```

## Best Practices

1. **Don't obsess over perfect scores** - PQS > 80 is excellent
2. **Focus on trends** - Consistent improvement matters more than individual scores
3. **Customize weights** - Adjust for your podcast type and audience
4. **Combine with qualitative feedback** - PQS complements, doesn't replace, listener feedback
5. **Use coaching insights** - Read the per-speaker coaching notes for specific guidance

## Technical Notes

### Piecewise Linear Scoring

Most sub-metrics use piecewise linear interpolation between breakpoints:

```python
def _piecewise_linear(value, breakpoints):
    # Example: filler_word_rate
    # breakpoints = [(0, 100), (2, 90), (5, 60), (10, 20), (15, 0)]
    # 0 fillers = 100, 2 = 90, 5 = 60, 10 = 20, 15+ = 0
    # Linear interpolation between points
```

This allows for:
- Non-linear relationships (diminishing returns)
- Clear target ranges
- Graceful degradation

### Missing Metrics

If a metric can't be computed (e.g., no diarization data), it's excluded and weights are redistributed proportionally.

```python
if not has_diarization:
    # Exclude speaker-dependent metrics
    # Redistribute weights to remaining metrics
```

## References

- **Loudness Standards**: ITU-R BS.1770-4, EBU R 128
- **Transcription**: OpenAI Whisper, PyAnnote Audio
- **NER**: BERT-based Named Entity Recognition
- **Sentiment**: RoBERTa-based Sentiment Analysis

## Version Changelog

### v3.0 (Current)
- Added Audio Quality domain (5 new metrics)
- Expanded Content domain (4 new metrics)
- Refined Engagement domain (3 new metrics)
- Total: 39 sub-metrics across 5 domains

### v2.1
- 4 domains, 21 sub-metrics
- Focus on delivery and content

### v1.0
- Basic metrics: audio quality, speaking pace
- Simple weighted average

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
| **Delivery & Dynamics** | 25% | Speaking quality and fluency | 9 |
| **Structure & Flow** | 20% | Episode organization | 8 |
| **Content Depth** | 25% | Information quality and density | 8 |
| **Engagement Proxies** | 20% | Listener retention signals | 9 |

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
def compute_audio_domain(metrics):
    # Expected keys: lufs, snr_db, clip_pct, lra_db, speech_band_pct
    loudness = score_loudness_compliance(metrics['lufs'])
    snr = score_signal_to_noise(metrics['snr_db'])
    clipping = score_clipping_index(metrics['clip_pct'])
    lra = score_loudness_range(metrics['lra_db'])
    spectral = score_spectral_balance(metrics['speech_band_pct'])

    return (
        0.30 * loudness +
        0.25 * snr +
        0.15 * clipping +
        0.15 * lra +
        0.15 * spectral
    )
```

### Interpretation

- **90-100**: Professional studio quality
- **75-89**: Excellent home studio
- **60-74**: Good quality, minor issues
- **40-59**: Acceptable, noticeable issues
- **0-39**: Poor quality, needs improvement

## 2. Delivery & Dynamics (25%)

Speaking fluency, filler management, and pacing dynamics.

### Sub-metrics (9 total)

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| **Filler Rate** | 20% | Filler words as % of total words | < 2.5% |
| **Filler Trajectory** | 16% | Quarter-over-quarter filler-rate change (negative = improving) | < 0% |
| **Core Filler Rate** | 12% | Core hesitation fillers (um, uh) as % | < 0.5% |
| **Silence Density** | 12% | Non-speech time as % of episode duration | < 1.5% |
| **WPM Consistency** | 12% | Words-per-minute coefficient of variation (%) | ~5% (inverted-U) |
| **Monotony Risk** | 10% | Segment-level WPM variability (CV) | ~0.22 (inverted-U) |
| **Dead Air Count** | 8% | Silences >5 s per hour of content | < 0.5/hr |
| **Interruption Rate** | 5% | Cross-speaker interruptions per hour | ~5/hr (inverted-U) |
| **Hesitation Rate** | 5% | Within-speaker hesitations per hour | < 2/hr |

### Scoring Function

```python
def compute_delivery_domain(metrics):
    # Expected keys: filler_rate, core_filler_rate, silence_density,
    #   dead_air_count, wpm_consistency, filler_trajectory,
    #   monotony_risk, interruption_rate, hesitation_rate
    return (
        0.20 * score_filler_rate(metrics['filler_rate']) +
        0.16 * score_filler_trajectory(metrics['filler_trajectory']) +
        0.12 * score_core_filler_rate(metrics['core_filler_rate']) +
        0.12 * score_silence_density(metrics['silence_density']) +
        0.12 * score_wpm_consistency(metrics['wpm_consistency']) +
        0.10 * score_monotony_risk(metrics['monotony_risk']) +
        0.08 * score_dead_air_count(metrics['dead_air_count']) +
        0.05 * score_interruption_rate(metrics['interruption_rate']) +
        0.05 * score_hesitation_rate(metrics['hesitation_rate'])
    )
```

### Interpretation

- **90-100**: Exceptional fluency, minimal fillers, dynamic pacing
- **75-89**: Professional delivery, occasional fillers
- **60-74**: Good delivery, some filler/pacing issues
- **40-59**: Acceptable, noticeable filler words or dead air
- **0-39**: Poor delivery, needs significant improvement

## 3. Structure & Flow (20%)

Episode organization, pacing, and structural elements.

### Sub-metrics (8 total)

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| **Duration Fit** | 18.75% | Episode duration sweet spot | 55-65 min |
| **Opening Tightness** | 18.75% | Q1 silence ratio vs rest of episode (1.0 = parity) | <= 1.0 |
| **Fragmentation** | 15% | Fragmentation index (%), lower = better coherence | < 1% |
| **Segment Balance** | 11.25% | Speaker talk-time balance (CV), lower = more balanced | < 0.02 |
| **Monologue Depth** | 11.25% | Extended monologue share (%) | ~8% (inverted-U) |
| **Transition Quality** | 10% | Transition quality score (0-10 scale) | >= 7 |
| **Intro Structure** | 8% | Intro structure score (0-5 scale) | >= 4 |
| **Closing Quality** | 7% | Closing elements present (0-5: summary, callback, goodbye, preview, CTA) | >= 4 |

### Scoring Function

```python
def compute_structure_domain(metrics):
    # Expected keys: duration_fit, segment_balance, fragmentation,
    #   monologue_depth, opening_tightness, transition_quality,
    #   closing_quality, intro_structure
    return (
        0.1875 * score_duration_fit(metrics['duration_fit']) +
        0.1875 * score_opening_tightness(metrics['opening_tightness']) +
        0.15   * score_fragmentation(metrics['fragmentation']) +
        0.1125 * score_segment_balance(metrics['segment_balance']) +
        0.1125 * score_monologue_depth(metrics['monologue_depth']) +
        0.10   * score_transition_quality(metrics['transition_quality']) +
        0.08   * score_intro_structure(metrics['intro_structure']) +
        0.07   * score_closing_quality(metrics['closing_quality'])
    )
```

### Interpretation

- **90-100**: Excellently structured, professional flow
- **75-89**: Well-organized, clear segments
- **60-74**: Good structure, minor pacing issues
- **40-59**: Acceptable, could use better organization
- **0-39**: Poor structure, disorganized

## 4. Content Depth (25%)

Information quality, analytical depth, and topic coverage.

### Sub-metrics (8 total)

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| **Analytical Depth Ratio** | 21.875% | Analytical segments as % of total | >= 12% |
| **Content Words per Minute** | 17.5% | Content words per minute (excluding fillers) | >= 120 cwpm |
| **Discussion Density** | 17.5% | Words per second of discussion | >= 2.2 w/s |
| **Topic Coverage Breadth** | 13.125% | Topic diversity (Shannon entropy) | >= 0.90 |
| **Domain Entity Density** | 8% | Domain-related entity mentions per 1000 words | >= 5/kw |
| **Tactical Depth Density** | 8% | Tactical concept mentions per 1000 words | >= 2.5/kw |
| **Match Reference Density** | 7% | Event references per 1000 words | >= 3/kw |
| **Opinion-Fact Ratio** | 7% | Opinion-to-fact ratio (balanced = best) | ~1.0 (inverted-U) |

### Scoring Function

```python
def compute_content_domain(metrics):
    # Expected keys: analytical_depth_ratio, content_words_per_minute,
    #   topic_coverage_breadth, discussion_density, domain_entity_density,
    #   match_reference_density, tactical_depth_density, opinion_fact_ratio
    return (
        0.21875 * score_analytical_depth_ratio(metrics['analytical_depth_ratio']) +
        0.175   * score_content_words_per_minute(metrics['content_words_per_minute']) +
        0.175   * score_discussion_density(metrics['discussion_density']) +
        0.13125 * score_topic_coverage_breadth(metrics['topic_coverage_breadth']) +
        0.08    * score_domain_entity_density(metrics['domain_entity_density']) +
        0.08    * score_tactical_depth_density(metrics['tactical_depth_density']) +
        0.07    * score_match_reference_density(metrics['match_reference_density']) +
        0.07    * score_opinion_fact_ratio(metrics['opinion_fact_ratio'])
    )
```

### Interpretation

- **90-100**: Exceptional content, deep analysis, dense and varied
- **75-89**: Strong content, good depth and coverage
- **60-74**: Good content, some areas lack depth
- **40-59**: Acceptable, shallow or narrow focus
- **0-39**: Poor content, low information density

## 5. Engagement Proxies (20%)

Listener retention signals and conversational dynamics.

### Sub-metrics (9 total)

| Metric | Weight | Description | Target |
|--------|--------|-------------|--------|
| **Fatigue Signal** | 17.5% | Filler-rate change per quarter (negative = improving) | < -0.3 |
| **Conversational Energy** | 14% | Speaker turns per minute | >= 6/min |
| **Q4 Sustain** | 14% | Q4 quality relative to episode average (1.0 = parity) | <= 0.85 |
| **Crosstalk Ratio** | 14% | Overlapping speech as % of total | >= 20% |
| **Debate Indicator** | 10.5% | Debate balance ratio | ~4.5 (inverted-U) |
| **Sentiment Flow Volatility** | 8% | Emotional dynamics index | ~0.7 (inverted-U) |
| **Memorable Moment Density** | 8% | High-energy moments per 10-minute block | >= 5 |
| **Hook Score** | 7% | Opening hooks detected in first 5 min | >= 3 |
| **Drop-off Risk Index** | 7% | Listener drop-off risk (lower = better retention) | < 3 |

### Scoring Function

```python
def compute_engagement_domain(metrics):
    # Expected keys: conversational_energy, debate_indicator, q4_sustain,
    #   crosstalk_ratio, fatigue_signal, sentiment_flow_volatility,
    #   hook_score, memorable_moment_density, dropoff_risk_index
    return (
        0.175  * score_fatigue_signal(metrics['fatigue_signal']) +
        0.14   * score_conversational_energy(metrics['conversational_energy']) +
        0.14   * score_q4_sustain(metrics['q4_sustain']) +
        0.14   * score_crosstalk_ratio(metrics['crosstalk_ratio']) +
        0.105  * score_debate_indicator(metrics['debate_indicator']) +
        0.08   * score_sentiment_flow_volatility(metrics['sentiment_flow_volatility']) +
        0.08   * score_memorable_moment_density(metrics['memorable_moment_density']) +
        0.07   * score_hook_score(metrics['hook_score']) +
        0.07   * score_dropoff_risk_index(metrics['dropoff_risk_index'])
    )
```

### Interpretation

- **90-100**: Highly engaging, strong dynamics, sustained energy
- **75-89**: Engaging, good conversational flow
- **60-74**: Moderately engaging
- **40-59**: Some engaging moments, inconsistent energy
- **0-39**: Low engagement, monotonous or fatigued

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

- Filler Rate: 45 ⚠️ (too many fillers)
- Filler Trajectory: 30 ⚠️ (getting worse each quarter)
- WPM Consistency: 85 ✓
- Monotony Risk: 55 ⚠️ (low pacing variation)
- Silence Density: 90 ✓

### Step 3: Take Action

Based on sub-metrics:
1. **Reduce filler words**: Practice awareness, use deliberate pauses instead
2. **Improve filler trajectory**: Track filler rate per quarter, aim for downward trend
3. **Vary pacing**: Use intentional speed changes for emphasis

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
- Added Audio Quality domain (5 sub-metrics)
- Delivery: 9 sub-metrics focused on fluency and filler management
- Structure: 8 sub-metrics for episode flow and organization
- Content: 8 sub-metrics for depth, density, and coverage
- Engagement: 9 sub-metrics for retention and dynamics
- Total: 39 sub-metrics across 5 domains

### v2.1
- 4 domains, 21 sub-metrics
- Focus on delivery and content

### v1.0
- Basic metrics: audio quality, speaking pace
- Simple weighted average

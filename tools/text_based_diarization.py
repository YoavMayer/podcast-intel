#!/usr/bin/env python3
"""
Text-based heuristic speaker identification for podcast transcripts.

When audio-based diarization is not possible (no audio files, no GPU),
this script uses transcript text patterns to identify speakers:

  1. Introduction detection: "I'm [name]", "Let's start with [name]"
  2. Name-calling patterns: names preceding responses
  3. Response patterns: after someone is called, the next segment is theirs
  4. Turn-taking heuristics: long gaps suggest speaker change,
     very short segments in sequence suggest same speaker
  5. Host detection: segments with "welcome", intro formulas

This is an imperfect heuristic approach but provides meaningful labels
even without audio features. It works best when configured with
the known speaker names for the podcast.

Usage:
    python tools/text_based_diarization.py --episode 200
    python tools/text_based_diarization.py --all
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Speaker knowledge base (populated from config) ───────────────────────

KNOWN_SPEAKERS: dict[str, str] = {}
"""Mapping of short names to full names, e.g. {"alice": "Alice Smith"}."""

# Episode-specific speaker configurations (populated at runtime)
EPISODE_CONFIG: dict[int, dict] = {}
"""Per-episode configuration: speakers, host, n_speakers, notes."""

# ── Pattern definitions (English) ─────────────────────────────────────────

# Host introduction patterns
HOST_INTRO_PATTERNS = [
    r"welcome.*podcast",
    r"welcome back",
    r"hello.*everyone",
    r"hi.*everyone",
    r"good (morning|afternoon|evening)",
    r"this is .* podcast",
]

# Question patterns (host asking panelists)
QUESTION_PATTERNS = [
    r"what do you think",
    r"what'?s your (take|opinion|view)",
    r"how do you see",
    r"let'?s hear from",
    r"let'?s start with",
    r"what have you got",
    r"tell us",
    r"go ahead",
]

# Agreement/response patterns (suggest responding to someone else)
RESPONSE_PATTERNS = [
    r"^I agree",
    r"^exactly",
    r"^right",
    r"^but ",
    r"^no, ",
    r"^yes, ",
    r"^look,",
    r"^listen,",
    r"^well,",
]

# Common English filler words
FILLER_WORDS = [
    "um", "uh", "uhh", "umm", "ummm",
    "like", "you know", "I mean", "basically",
    "actually", "literally", "right", "so",
    "well", "okay", "ok",
]


def _load_config() -> dict:
    """Load podcast.yaml configuration if available."""
    config_path = PROJECT_ROOT / "podcast.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
    return {}


def _init_speaker_knowledge(config: dict):
    """Initialize speaker knowledge base from config."""
    global KNOWN_SPEAKERS
    speakers = config.get("speakers", [])
    for s in speakers:
        if isinstance(s, dict):
            name = s.get("name", "")
            short = s.get("short_name", name.split()[0] if name else "")
        else:
            name = str(s)
            short = name.split()[0] if name else ""
        if short and name:
            KNOWN_SPEAKERS[short.lower()] = name


# ═══════════════════════════════════════════════════════════════════════════
#  Pattern detection engine
# ═══════════════════════════════════════════════════════════════════════════

class TextDiarizer:
    """Heuristic speaker identification from transcript text."""

    def __init__(self, episode_num: int, segments: list[dict]):
        self.episode_num = episode_num
        self.segments = segments
        self.config = EPISODE_CONFIG.get(episode_num, {
            "speakers": list(KNOWN_SPEAKERS.values()) if KNOWN_SPEAKERS else ["Host", "Speaker 2", "Speaker 3"],
            "host": list(KNOWN_SPEAKERS.values())[0] if KNOWN_SPEAKERS else "Host",
            "n_speakers": 3,
        })
        self.labels: list[Optional[str]] = [None] * len(segments)
        self.confidence: list[float] = [0.0] * len(segments)
        self.evidence: list[list[str]] = [[] for _ in range(len(segments))]

        # Pre-compute gaps between segments
        self.gaps: list[float] = [0.0]
        for i in range(1, len(segments)):
            gap = segments[i]["start"] - segments[i - 1]["end"]
            self.gaps.append(gap)

    def run(self) -> list[dict]:
        """Run all heuristic passes and return enriched segments."""
        print(f"  Running text-based diarization for episode {self.episode_num}")
        print(f"  Config: {self.config.get('notes', 'default')}")
        print(f"  Segments: {len(self.segments)}")

        # Pass 1: Solo episode shortcut (highest priority)
        if self.config.get("n_speakers", 0) == 1:
            self._pass_solo_episode()
            return self._build_output()

        # Pass 2: Direct identification (strongest signal)
        self._pass_introduction_detection()

        # Pass 3: Self-identification patterns
        self._pass_self_identification()

        # Pass 4: Name-calling -> next segment assignment
        self._pass_name_calling()

        # Pass 5: Question-response patterns
        self._pass_question_response()

        # Pass 6: Gap-based turn detection (large gaps = likely speaker change)
        self._pass_gap_based_turns()

        # Pass 7: Propagate with boundaries -- only within "turn blocks"
        self._pass_block_propagation()

        # Pass 8: Round-robin for multi-speaker episodes
        self._pass_round_robin_fill()

        # Build enriched output
        return self._build_output()

    def _pass_introduction_detection(self):
        """Detect host introductions in opening segments."""
        for i, seg in enumerate(self.segments[:30]):  # Check first 30 segments
            text = seg["text"]
            text_lower = text.lower()
            for pattern in HOST_INTRO_PATTERNS:
                if re.search(pattern, text_lower):
                    host = self.config.get("host")
                    if host:
                        self._assign(i, host, 0.95, f"host_intro: {pattern}")
                    break

            # Check for panel introductions: "Let's start with [name]"
            for name_short, name_full in KNOWN_SPEAKERS.items():
                if name_full in self.config.get("speakers", []):
                    intro_pattern = rf"(start with|welcome|joining us|also|with us).*{re.escape(name_short)}"
                    if re.search(intro_pattern, text_lower):
                        host = self.config.get("host")
                        if host:
                            self._assign(i, host, 0.9, f"introducing_{name_short}")

    def _pass_self_identification(self):
        """Detect segments where speaker identifies themselves."""
        for i, seg in enumerate(self.segments):
            text_lower = seg["text"].lower()
            for name_short, name_full in KNOWN_SPEAKERS.items():
                if name_full in self.config.get("speakers", []):
                    patterns = [
                        rf"i'?m {re.escape(name_short)}",
                        rf"this is {re.escape(name_short)}",
                        rf"my name is {re.escape(name_short)}",
                        re.escape(name_full.lower()),
                    ]
                    for pattern in patterns:
                        if re.search(pattern, text_lower):
                            self._assign(i, name_full, 0.9, f"self_id: {pattern}")
                            break

    def _pass_name_calling(self):
        """When someone calls a name, the NEXT segment is likely that person."""
        for i, seg in enumerate(self.segments[:-1]):
            text_lower = seg["text"].lower()
            for name_short, name_full in KNOWN_SPEAKERS.items():
                if name_full not in self.config.get("speakers", []):
                    continue

                # Check if this segment ends with or contains a name call
                name_patterns = [
                    # Name at end: "what do you think, Bob?"
                    rf"{re.escape(name_short)}\s*[,?\.\!]?\s*$",
                    # Name as first word in a short segment (calling out)
                    rf"^{re.escape(name_short)}\s*[,?\.\!]?\s*$",
                    # "Let's hear from X" or "go ahead X"
                    rf"(hear from|go ahead|over to)\s+{re.escape(name_short)}",
                    # "X, what do you..."
                    rf"^{re.escape(name_short)},\s+(what|how|tell|do you)",
                ]
                for np_pattern in name_patterns:
                    if re.search(np_pattern, text_lower):
                        # Next segment: the named person responds
                        self._assign(i + 1, name_full, 0.8, f"called_by_name: {name_short}")
                        # Also assign next few segments (person often speaks for a bit)
                        for j in range(i + 2, min(i + 5, len(self.segments))):
                            if self.gaps[j] < 1.5:
                                self._assign(j, name_full, 0.55, f"continuing_after_call: {name_short}")
                            else:
                                break

                        # Caller is likely the host or another speaker
                        if self.labels[i] is None and self.config.get("host"):
                            self._assign(i, self.config["host"], 0.6, "likely_caller")
                        break

    def _pass_question_response(self):
        """Detect question-response patterns for turn assignment."""
        for i, seg in enumerate(self.segments[:-1]):
            text_lower = seg["text"].lower()
            for pattern in QUESTION_PATTERNS:
                if re.search(pattern, text_lower):
                    host = self.config.get("host")
                    if host and self.labels[i] is None:
                        self._assign(i, host, 0.45, f"asking_question: {pattern}")
                    break

    def _pass_solo_episode(self):
        """For solo episodes, assign all segments to the single speaker."""
        speaker = self.config.get("speakers", ["Host"])[0]
        for i in range(len(self.segments)):
            self._assign(i, speaker, 1.0, "solo_episode")

    def _pass_gap_based_turns(self):
        """
        Use inter-segment gaps to detect likely turn changes.

        Large gaps (> 1.5s) suggest a different person is about to speak.
        """
        all_gaps = [g for g in self.gaps[1:] if g >= 0]
        if not all_gaps:
            return
        median_gap = sorted(all_gaps)[len(all_gaps) // 2]
        turn_threshold = max(1.0, median_gap * 2.5)

        # Identify turn boundaries
        turn_starts = [0]
        for i in range(1, len(self.segments)):
            if self.gaps[i] > turn_threshold:
                turn_starts.append(i)

        n_turns = len(turn_starts)
        print(f"  Gap-based turns: {n_turns} turns detected "
              f"(threshold={turn_threshold:.2f}s, median_gap={median_gap:.2f}s)")

        # For each turn block, propagate the label of the highest-confidence segment
        for t_idx, t_start in enumerate(turn_starts):
            t_end = turn_starts[t_idx + 1] if t_idx + 1 < n_turns else len(self.segments)

            block_label = None
            block_conf = 0.0
            for j in range(t_start, t_end):
                if self.labels[j] is not None and self.confidence[j] > block_conf:
                    block_label = self.labels[j]
                    block_conf = self.confidence[j]

            if block_label and block_conf >= 0.35:
                for j in range(t_start, t_end):
                    if self.labels[j] is None:
                        self._assign(j, block_label, 0.3, f"turn_block_{t_idx}")

    def _pass_block_propagation(self):
        """
        Propagate labels within contiguous blocks (no large gaps).
        """
        max_gap = 1.5  # seconds

        blocks: list[list[int]] = []
        current_block: list[int] = [0]
        for i in range(1, len(self.segments)):
            if self.gaps[i] < max_gap:
                current_block.append(i)
            else:
                blocks.append(current_block)
                current_block = [i]
        blocks.append(current_block)

        for block in blocks:
            best_label = None
            best_conf = 0.0
            for idx in block:
                if self.labels[idx] is not None and self.confidence[idx] > best_conf:
                    best_label = self.labels[idx]
                    best_conf = self.confidence[idx]

            if best_label and best_conf >= 0.3:
                prop_conf = min(best_conf * 0.6, 0.25)
                for idx in block:
                    if self.labels[idx] is None:
                        self._assign(idx, best_label, prop_conf,
                                     f"block_propagation (source_conf={best_conf:.2f})")

    def _pass_round_robin_fill(self):
        """
        Fill remaining unlabeled segments using round-robin among speakers.
        """
        speakers = self.config.get("speakers", ["Host"])
        n_speakers = len(speakers)
        if n_speakers <= 1:
            fill_speaker = speakers[0] if speakers else "UNKNOWN"
            for i in range(len(self.segments)):
                if self.labels[i] is None:
                    self._assign(i, fill_speaker, 0.05, "fill_single")
            return

        host = self.config.get("host")
        non_host = [s for s in speakers if s != host]

        unlabeled = [i for i in range(len(self.segments)) if self.labels[i] is None]
        if not unlabeled:
            return

        print(f"  Round-robin fill: {len(unlabeled)} remaining unlabeled segments")

        # Group unlabeled into contiguous runs
        runs: list[list[int]] = []
        current_run: list[int] = [unlabeled[0]]
        for i in range(1, len(unlabeled)):
            if unlabeled[i] == unlabeled[i - 1] + 1:
                current_run.append(unlabeled[i])
            else:
                runs.append(current_run)
                current_run = [unlabeled[i]]
        runs.append(current_run)

        speaker_cycle_idx = 0
        for run in runs:
            first_idx = run[0]
            last_idx = run[-1]

            before_speaker = None
            for j in range(first_idx - 1, max(first_idx - 5, -1), -1):
                if self.labels[j] is not None:
                    before_speaker = self.labels[j]
                    break

            after_speaker = None
            for j in range(last_idx + 1, min(last_idx + 5, len(self.segments))):
                if self.labels[j] is not None:
                    after_speaker = self.labels[j]
                    break

            if before_speaker and after_speaker and before_speaker == after_speaker:
                fill = before_speaker
            elif before_speaker:
                gap = self.gaps[first_idx] if first_idx > 0 else 0
                if gap > 1.0 and non_host:
                    fill = non_host[speaker_cycle_idx % len(non_host)]
                    speaker_cycle_idx += 1
                else:
                    fill = before_speaker
            elif after_speaker:
                fill = after_speaker
            else:
                fill = speakers[speaker_cycle_idx % n_speakers]
                speaker_cycle_idx += 1

            for idx in run:
                self._assign(idx, fill, 0.05, "round_robin_fill")

    def _assign(self, idx: int, speaker: str, confidence: float, reason: str):
        """Assign a speaker label, only if new confidence is higher."""
        if self.confidence[idx] < confidence:
            self.labels[idx] = speaker
            self.confidence[idx] = confidence
        self.evidence[idx].append(f"{reason} [{confidence:.2f}]")

    def _build_output(self) -> list[dict]:
        """Build enriched segments with speaker labels."""
        enriched = []
        for i, seg in enumerate(self.segments):
            enriched_seg = dict(seg)
            enriched_seg["speaker_id"] = self.labels[i] or "UNKNOWN"
            enriched_seg["speaker_confidence"] = round(self.confidence[i], 2)
            enriched_seg["speaker_evidence"] = self.evidence[i]
            enriched.append(enriched_seg)
        return enriched


# ═══════════════════════════════════════════════════════════════════════════
#  Per-speaker metrics
# ═══════════════════════════════════════════════════════════════════════════

def count_fillers(text: str) -> int:
    """Count filler words in text."""
    count = 0
    text_lower = text.lower()
    for word in text_lower.split():
        clean = re.sub(r"[,.\?!;:\-]", "", word)
        if clean in [f.lower() for f in FILLER_WORDS]:
            count += 1
    return count


def compute_speaker_metrics(enriched_segments: list[dict]) -> dict[str, dict[str, Any]]:
    """Compute per-speaker metrics from enriched transcript."""
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "segment_count": 0,
        "word_count": 0,
        "total_duration_s": 0.0,
        "filler_count": 0,
    })

    for seg in enriched_segments:
        spk = seg.get("speaker_id", "UNKNOWN")
        text = seg.get("text", "")
        words = text.split()
        duration = seg["end"] - seg["start"]

        stats[spk]["segment_count"] += 1
        stats[spk]["word_count"] += len(words)
        stats[spk]["total_duration_s"] += duration
        stats[spk]["filler_count"] += count_fillers(text)

    result = {}
    for spk, s in stats.items():
        duration_min = s["total_duration_s"] / 60.0
        wpm = s["word_count"] / duration_min if duration_min > 0 else 0
        filler_rate = (s["filler_count"] / s["word_count"] * 100) if s["word_count"] > 0 else 0
        avg_seg_len = s["total_duration_s"] / s["segment_count"] if s["segment_count"] > 0 else 0
        avg_seg_words = s["word_count"] / s["segment_count"] if s["segment_count"] > 0 else 0

        result[spk] = {
            "display_name": spk,
            "segment_count": s["segment_count"],
            "word_count": s["word_count"],
            "total_duration_s": round(s["total_duration_s"], 1),
            "total_duration_min": round(duration_min, 1),
            "wpm": round(wpm, 1),
            "filler_count": s["filler_count"],
            "filler_rate_pct": round(filler_rate, 2),
            "avg_segment_length_s": round(avg_seg_len, 2),
            "avg_segment_words": round(avg_seg_words, 1),
        }

    return dict(result)


def compute_label_stats(enriched: list[dict]) -> dict:
    """Compute statistics about the labeling quality."""
    total = len(enriched)
    by_speaker = defaultdict(int)
    by_confidence = {"high": 0, "medium": 0, "low": 0, "none": 0}

    for seg in enriched:
        spk = seg.get("speaker_id", "UNKNOWN")
        conf = seg.get("speaker_confidence", 0)
        by_speaker[spk] += 1
        if conf >= 0.7:
            by_confidence["high"] += 1
        elif conf >= 0.3:
            by_confidence["medium"] += 1
        elif conf > 0:
            by_confidence["low"] += 1
        else:
            by_confidence["none"] += 1

    return {
        "total_segments": total,
        "labeled_segments": total - by_speaker.get("UNKNOWN", 0),
        "segments_by_speaker": dict(by_speaker),
        "confidence_distribution": by_confidence,
        "high_confidence_pct": round(by_confidence["high"] / total * 100, 1) if total > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def process_episode(episode_num: int) -> dict[str, Any]:
    """Run text-based diarization for one episode and save results."""
    print(f"\n{'='*70}")
    print(f"  TEXT-BASED DIARIZATION: Episode {episode_num}")
    print(f"{'='*70}")

    # Load transcript
    transcript_path = REPORTS_DIR / f"episode_{episode_num}" / "transcript.json"
    if not transcript_path.exists():
        print(f"  ERROR: Transcript not found: {transcript_path}")
        return {"error": f"transcript_not_found: {transcript_path}"}

    segments = json.loads(transcript_path.read_text(encoding="utf-8"))
    print(f"  Loaded {len(segments)} segments")

    # Run diarization
    diarizer = TextDiarizer(episode_num, segments)
    enriched = diarizer.run()

    # Compute metrics
    metrics = compute_speaker_metrics(enriched)
    label_stats = compute_label_stats(enriched)

    # Print summary
    print(f"\n  Label statistics:")
    print(f"    Total segments:   {label_stats['total_segments']}")
    print(f"    Labeled:          {label_stats['labeled_segments']}")
    print(f"    High confidence:  {label_stats['high_confidence_pct']}%")
    print(f"\n  Segments by speaker:")
    for spk, count in sorted(label_stats["segments_by_speaker"].items(),
                              key=lambda x: x[1], reverse=True):
        pct = count / label_stats["total_segments"] * 100
        print(f"    {spk}: {count} ({pct:.1f}%)")

    print(f"\n  Speaker metrics:")
    for spk, m in sorted(metrics.items(), key=lambda x: x[1]["word_count"], reverse=True):
        print(f"    {spk}:")
        print(f"      Words: {m['word_count']}, Duration: {m['total_duration_min']:.1f} min")
        print(f"      WPM: {m['wpm']:.0f}, Filler rate: {m['filler_rate_pct']:.1f}%")
        print(f"      Avg segment: {m['avg_segment_length_s']:.1f}s, {m['avg_segment_words']:.0f} words")

    # Save outputs
    output_dir = REPORTS_DIR / f"episode_{episode_num}" / "diarization"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enriched transcript (with evidence stripped for cleaner output)
    enriched_clean = []
    for seg in enriched:
        clean_seg = {k: v for k, v in seg.items() if k != "speaker_evidence"}
        enriched_clean.append(clean_seg)

    enriched_path = output_dir / "enriched_transcript.json"
    enriched_path.write_text(
        json.dumps(enriched_clean, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Saved: {enriched_path}")

    # Full enriched with evidence (for debugging)
    debug_path = output_dir / "enriched_transcript_debug.json"
    debug_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Diarization segments (for merge_diarization.py compatibility)
    diarization_segs = [
        {"start": s["start"], "end": s["end"], "speaker": s["speaker_id"]}
        for s in enriched
    ]
    diarization_path = output_dir / "diarization_segments.json"
    diarization_path.write_text(
        json.dumps(diarization_segs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Speaker metrics
    metrics_path = output_dir / "speaker_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved: {metrics_path}")

    # Label statistics
    stats_path = output_dir / "label_statistics.json"
    stats_path.write_text(
        json.dumps(label_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    summary = {
        "episode": episode_num,
        "method": "text_heuristic",
        "config": EPISODE_CONFIG.get(episode_num, {}),
        "total_segments": len(enriched),
        "label_statistics": label_stats,
        "speaker_metrics": metrics,
    }

    summary_path = output_dir / "diarization_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved: {summary_path}")

    # Print first 10 segments as sample
    print(f"\n  Sample (first 10 segments):")
    for seg in enriched[:10]:
        spk = seg["speaker_id"]
        conf = seg.get("speaker_confidence", 0)
        text_preview = seg["text"][:60]
        print(f"    [{seg['start']:6.1f}s] [{conf:.2f}] {spk}: {text_preview}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Text-based heuristic speaker identification"
    )
    parser.add_argument(
        "--episode", "-e", type=int,
        help="Episode number to process"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all episodes with transcripts"
    )

    args = parser.parse_args()

    # Load config and initialize speaker knowledge
    config = _load_config()
    _init_speaker_knowledge(config)

    if args.all:
        # Find all episodes with transcripts
        episodes = []
        for ep_dir in sorted(REPORTS_DIR.glob("episode_*")):
            if (ep_dir / "transcript.json").exists():
                try:
                    ep_num = int(ep_dir.name.replace("episode_", ""))
                    episodes.append(ep_num)
                except ValueError:
                    pass
        if not episodes:
            print("No episodes with transcripts found.")
            sys.exit(1)
    elif args.episode:
        episodes = [args.episode]
    else:
        parser.print_help()
        print("\nError: specify --episode or --all")
        sys.exit(1)

    results = []
    for ep in episodes:
        result = process_episode(ep)
        results.append(result)

    print("\n" + "=" * 70)
    print("  ALL EPISODES COMPLETE")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  Episode {r.get('episode', '?')}: ERROR - {r['error']}")
        else:
            ep = r["episode"]
            stats = r["label_statistics"]
            print(f"  Episode {ep}: {stats['total_segments']} segments, "
                  f"{stats['labeled_segments']} labeled, "
                  f"{stats['high_confidence_pct']}% high confidence")


if __name__ == "__main__":
    main()

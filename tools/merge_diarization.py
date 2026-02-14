#!/usr/bin/env python3
"""
Merge diarization output with Whisper transcript segments.

This script takes a diarization output (list of speaker turns with timestamps)
and a Whisper transcript (list of segments with text and word-level timestamps),
then merges them by maximum timestamp overlap.

Usage:
    python tools/merge_diarization.py \
        --transcript reports/episode_200/transcript.json \
        --diarization reports/episode_200/diarization/diarization_segments.json \
        --output reports/episode_200/diarization/enriched_transcript.json

    python tools/merge_diarization.py --episode 200

Input formats:
    transcript.json:  [{"start": float, "end": float, "text": str, "words": [...]}]
    diarization.json: [{"start": float, "end": float, "speaker": str}]

Output format:
    enriched_transcript.json: same as transcript but with "speaker_id" field added
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def compute_overlap(seg_start: float, seg_end: float,
                    d_start: float, d_end: float) -> float:
    """Compute temporal overlap between two intervals in seconds."""
    overlap_start = max(seg_start, d_start)
    overlap_end = min(seg_end, d_end)
    return max(0.0, overlap_end - overlap_start)


def merge_segments(
    transcript_segments: list[dict],
    diarization_segments: list[dict],
    overlap_threshold: float = 0.0,
) -> list[dict]:
    """
    Merge diarization labels into transcript segments by timestamp overlap.

    For each transcript segment, finds the diarization segment with the
    largest temporal overlap and assigns that speaker label. If the best
    overlap is below overlap_threshold, marks as UNKNOWN.

    Supports two merge strategies:
    1. Segment-level: assigns one speaker per transcript segment
    2. Word-level: if transcript has word timestamps, assigns at word level
       then picks the majority speaker for the segment

    Args:
        transcript_segments: Whisper transcript segments
        diarization_segments: Speaker turn segments from diarization
        overlap_threshold: Minimum overlap (seconds) to assign a speaker

    Returns:
        Enriched transcript segments with 'speaker_id' field
    """
    # Sort diarization segments by start time for efficient lookup
    d_sorted = sorted(diarization_segments, key=lambda x: x["start"])

    enriched = []
    for seg in transcript_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Strategy 1: Simple segment-level overlap
        best_speaker = "UNKNOWN"
        best_overlap = overlap_threshold

        for d_seg in d_sorted:
            # Early termination: if diarization segment starts after our end
            if d_seg["start"] > seg_end:
                break
            # Skip segments that end before ours starts
            if d_seg["end"] < seg_start:
                continue

            overlap = compute_overlap(seg_start, seg_end, d_seg["start"], d_seg["end"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]

        # Strategy 2: Word-level voting (if words have timestamps)
        if "words" in seg and seg["words"] and best_speaker == "UNKNOWN":
            speaker_votes: dict[str, float] = {}
            for word in seg["words"]:
                w_start = word.get("start", seg_start)
                w_end = word.get("end", seg_end)
                for d_seg in d_sorted:
                    if d_seg["start"] > w_end:
                        break
                    if d_seg["end"] < w_start:
                        continue
                    overlap = compute_overlap(w_start, w_end, d_seg["start"], d_seg["end"])
                    if overlap > 0:
                        spk = d_seg["speaker"]
                        speaker_votes[spk] = speaker_votes.get(spk, 0) + overlap

            if speaker_votes:
                best_speaker = max(speaker_votes, key=speaker_votes.get)

        enriched_seg = dict(seg)
        enriched_seg["speaker_id"] = best_speaker
        enriched.append(enriched_seg)

    return enriched


def apply_speaker_name_map(
    enriched_segments: list[dict],
    name_map: dict[str, str],
) -> list[dict]:
    """
    Replace generic speaker IDs (SPEAKER_A) with real names.

    Args:
        enriched_segments: Segments with speaker_id field
        name_map: Mapping from speaker_id to display name

    Returns:
        Updated segments (modifies in place and returns)
    """
    for seg in enriched_segments:
        spk = seg.get("speaker_id", "")
        if spk in name_map:
            seg["speaker_name"] = name_map[spk]
    return enriched_segments


def compute_merge_quality(enriched_segments: list[dict]) -> dict[str, Any]:
    """
    Compute quality metrics for the merge result.

    Returns:
        Dict with quality indicators
    """
    total = len(enriched_segments)
    unknown = sum(1 for s in enriched_segments if s.get("speaker_id") == "UNKNOWN")
    speakers = set(s.get("speaker_id", "UNKNOWN") for s in enriched_segments)

    # Check for rapid speaker switches (possible errors)
    rapid_switches = 0
    for i in range(1, len(enriched_segments)):
        prev_spk = enriched_segments[i - 1].get("speaker_id")
        curr_spk = enriched_segments[i].get("speaker_id")
        gap = enriched_segments[i]["start"] - enriched_segments[i - 1]["end"]
        if prev_spk != curr_spk and gap < 0.5:
            rapid_switches += 1

    return {
        "total_segments": total,
        "unknown_segments": unknown,
        "unknown_rate_pct": round(unknown / total * 100, 1) if total > 0 else 0,
        "unique_speakers": len(speakers),
        "speakers": sorted(speakers),
        "rapid_switches": rapid_switches,
        "rapid_switch_rate_pct": round(rapid_switches / total * 100, 1) if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge diarization labels with Whisper transcript"
    )
    parser.add_argument(
        "--episode", "-e", type=int,
        help="Episode number (auto-resolves paths)"
    )
    parser.add_argument(
        "--transcript", "-t", type=str,
        help="Path to Whisper transcript JSON"
    )
    parser.add_argument(
        "--diarization", "-d", type=str,
        help="Path to diarization segments JSON"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output path for enriched transcript"
    )
    parser.add_argument(
        "--name-map", type=str,
        help="Path to speaker name mapping JSON"
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=0.0,
        help="Minimum overlap (seconds) to assign speaker (default: 0.0)"
    )

    args = parser.parse_args()

    # Auto-resolve paths from episode number
    if args.episode:
        ep_dir = REPORTS_DIR / f"episode_{args.episode}"
        dia_dir = ep_dir / "diarization"
        transcript_path = ep_dir / "transcript.json"
        diarization_path = dia_dir / "diarization_segments.json"
        output_path = dia_dir / "enriched_transcript.json"
        name_map_path = dia_dir / "speaker_name_map.json"
    else:
        if not args.transcript or not args.diarization:
            parser.print_help()
            print("\nError: specify --episode or both --transcript and --diarization")
            sys.exit(1)
        transcript_path = Path(args.transcript)
        diarization_path = Path(args.diarization)
        output_path = Path(args.output) if args.output else diarization_path.parent / "enriched_transcript.json"
        name_map_path = Path(args.name_map) if args.name_map else None

    # Load inputs
    print(f"Loading transcript: {transcript_path}")
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    print(f"  {len(transcript)} segments")

    print(f"Loading diarization: {diarization_path}")
    diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    print(f"  {len(diarization)} speaker turns")

    # Merge
    print("Merging...")
    enriched = merge_segments(transcript, diarization, args.overlap_threshold)

    # Apply name map if available
    if args.episode and name_map_path and name_map_path.exists():
        name_map = json.loads(name_map_path.read_text(encoding="utf-8"))
        enriched = apply_speaker_name_map(enriched, name_map)
        print(f"  Applied name map: {name_map}")
    elif args.name_map:
        name_map = json.loads(Path(args.name_map).read_text(encoding="utf-8"))
        enriched = apply_speaker_name_map(enriched, name_map)
        print(f"  Applied name map: {name_map}")

    # Quality check
    quality = compute_merge_quality(enriched)
    print(f"\nMerge quality:")
    print(f"  Total segments: {quality['total_segments']}")
    print(f"  Unknown segments: {quality['unknown_segments']} ({quality['unknown_rate_pct']}%)")
    print(f"  Unique speakers: {quality['unique_speakers']}")
    print(f"  Rapid switches: {quality['rapid_switches']} ({quality['rapid_switch_rate_pct']}%)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved enriched transcript: {output_path}")

    # Save quality metrics
    quality_path = output_path.parent / "merge_quality.json"
    quality_path.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved merge quality: {quality_path}")


if __name__ == "__main__":
    main()

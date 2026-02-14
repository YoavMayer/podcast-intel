#!/usr/bin/env python3
"""
Speaker diarization pipeline for podcast episodes.

This script:
  1. Downloads episode audio from the URL in the episodes registry
  2. Runs speaker diarization (pyannote-audio if available, else MFCC+clustering)
  3. Merges diarization labels with existing Whisper transcript segments
  4. Outputs enriched transcript with speaker labels and per-speaker metrics

Requirements:
  Core (always needed):
    - librosa>=0.11.0
    - scikit-learn>=1.0
    - numpy
    - scipy
    - requests

  Optional (better quality):
    - pyannote-audio>=3.1.0  (requires HuggingFace token + GPU recommended)
    - torch>=2.0.0

  If pyannote is not available, falls back to MFCC spectral-clustering approach.

Usage:
    python tools/diarize_episode.py --episode 200 --n-speakers 4
    python tools/diarize_episode.py --episode 202 --n-speakers 1
    python tools/diarize_episode.py --episode 199 --n-speakers 2
    python tools/diarize_episode.py --all

Environment variables:
    HF_TOKEN        - HuggingFace token for pyannote (optional)
    PYANNOTE_CACHE  - Cache dir for pyannote models (optional)
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
EPISODES_JSON = DATA_DIR / "episodes.json"

# ── Known speakers per episode (populated at runtime or via config) ───────
EPISODE_SPEAKERS: dict[int, list[str]] = {}

# ── Pyannote availability check ───────────────────────────────────────────
PYANNOTE_AVAILABLE = False
try:
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline

    PYANNOTE_AVAILABLE = True
except ImportError:
    pass


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


# ═══════════════════════════════════════════════════════════════════════════
#  Audio download
# ═══════════════════════════════════════════════════════════════════════════

def download_audio(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download audio file with progress reporting. Returns path to file."""
    if dest.exists():
        print(f"  Audio already exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading audio from {url[:80]}...")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (chunk_size * 128) == 0:
                pct = downloaded / total * 100
                print(f"    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)")

    print(f"  Download complete: {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def get_episode_audio_url(episode_num: int) -> Optional[str]:
    """Look up audio URL for an episode number from the episodes JSON."""
    if not EPISODES_JSON.exists():
        return None
    episodes = json.loads(EPISODES_JSON.read_text(encoding="utf-8"))
    for ep in episodes:
        ep_num = ep.get("epNum")
        if ep_num == episode_num:
            return ep["audioUrl"]
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Pyannote diarization (GPU/HF token required)
# ═══════════════════════════════════════════════════════════════════════════

def diarize_with_pyannote(
    audio_path: Path, n_speakers: int
) -> list[dict[str, Any]]:
    """
    Run pyannote-audio speaker diarization.

    Requires:
      - pip install pyannote-audio torch
      - HuggingFace token with access to pyannote/speaker-diarization-3.1
      - GPU strongly recommended (CPU works but is very slow)

    Returns list of {"start": float, "end": float, "speaker": str}
    """
    if not PYANNOTE_AVAILABLE:
        raise RuntimeError(
            "pyannote-audio is not installed. Install with:\n"
            "  pip install pyannote-audio torch\n"
            "You also need a HuggingFace token with access to:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "Set the HF_TOKEN environment variable."
        )

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable not set. "
            "Get a token at https://huggingface.co/settings/tokens "
            "and accept the pyannote model terms."
        )

    print("  Loading pyannote pipeline...")
    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("  Using GPU for diarization")
    else:
        print("  WARNING: No GPU detected. Pyannote on CPU will be very slow.")

    print(f"  Running diarization (n_speakers={n_speakers})...")
    diarization = pipeline(
        str(audio_path),
        num_speakers=n_speakers,
    )

    # Convert to list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    print(f"  Pyannote returned {len(segments)} speaker turns")
    return segments


# ═══════════════════════════════════════════════════════════════════════════
#  MFCC + Spectral Clustering diarization (CPU fallback)
# ═══════════════════════════════════════════════════════════════════════════

def extract_segment_embedding(
    y: np.ndarray, sr: int, start: float, end: float
) -> Optional[np.ndarray]:
    """Extract a fixed-size audio embedding for a time segment using MFCCs."""
    import librosa

    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = y[start_sample:end_sample]

    if len(segment) < int(sr * 0.3):
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # MFCCs (20 coefficients)
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Statistics: mean + std for each coefficient
        features = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1),
            mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1),
        ])

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)

        # Pitch (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            segment, sr=sr, fmin=60, fmax=400
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        f0_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 150.0
        f0_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 30.0

        extra = np.array([
            float(spectral_centroid.mean()), float(spectral_centroid.std()),
            float(spectral_bandwidth.mean()), float(spectral_bandwidth.std()),
            f0_mean, f0_std,
        ])

    return np.concatenate([features, extra])


def diarize_with_clustering(
    audio_path: Path,
    transcript_segments: list[dict],
    n_speakers: int,
) -> list[dict[str, Any]]:
    """
    CPU-only diarization using MFCC features + spectral clustering.

    Extracts audio embeddings for each transcript segment, then clusters
    them into n_speakers groups using agglomerative clustering with
    cosine distance.

    Returns list of {"start": float, "end": float, "speaker": str}
    """
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    print(f"  Loading audio: {audio_path}")
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    duration_s = len(y) / sr
    print(f"  Audio loaded: {duration_s:.1f}s at {sr}Hz")

    # Extract embeddings for each transcript segment
    embeddings = []
    valid_indices = []

    print(f"  Extracting embeddings for {len(transcript_segments)} segments...")
    for i, seg in enumerate(transcript_segments):
        emb = extract_segment_embedding(y, sr, seg["start"], seg["end"])
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(i)

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(transcript_segments)}] "
                  f"({len(embeddings)} valid embeddings)")

    print(f"  Valid embeddings: {len(embeddings)} / {len(transcript_segments)}")

    if len(embeddings) < n_speakers:
        print("  ERROR: Not enough valid segments for clustering")
        return []

    # Normalize features
    X = np.array(embeddings)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle edge case: 1 speaker
    if n_speakers == 1:
        labels = np.zeros(len(embeddings), dtype=int)
        sil_score = 1.0
    else:
        # Agglomerative clustering with cosine distance
        clustering = AgglomerativeClustering(
            n_clusters=n_speakers,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(X_scaled)

        if len(set(labels)) > 1:
            sil_score = silhouette_score(X_scaled, labels, metric="cosine")
        else:
            sil_score = 0.0

    print(f"  Silhouette score: {sil_score:.3f} (>0.3 = good separation)")

    # Build result segments with speaker labels
    # Sort clusters by total duration (most talkative first = Speaker_A)
    cluster_durations: dict[int, float] = {}
    for idx, seg_i in enumerate(valid_indices):
        seg = transcript_segments[seg_i]
        label = int(labels[idx])
        cluster_durations[label] = cluster_durations.get(label, 0) + (seg["end"] - seg["start"])

    sorted_clusters = sorted(cluster_durations.items(), key=lambda x: x[1], reverse=True)
    speaker_names = [f"SPEAKER_{chr(65 + rank)}" for rank in range(len(sorted_clusters))]
    cluster_to_speaker = {
        cl: speaker_names[rank] for rank, (cl, _) in enumerate(sorted_clusters)
    }

    # Build output
    diarization_segments = []
    for idx, seg_i in enumerate(valid_indices):
        seg = transcript_segments[seg_i]
        label = int(labels[idx])
        diarization_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": cluster_to_speaker[label],
        })

    # For segments that were skipped (too short), interpolate from neighbors
    all_seg_speakers = {}
    for idx, seg_i in enumerate(valid_indices):
        all_seg_speakers[seg_i] = cluster_to_speaker[int(labels[idx])]

    for i in range(len(transcript_segments)):
        if i not in all_seg_speakers:
            # Find nearest valid segment
            best_dist = float("inf")
            best_speaker = speaker_names[0]
            for vi in valid_indices:
                dist = abs(vi - i)
                if dist < best_dist:
                    best_dist = dist
                    best_speaker = all_seg_speakers[vi]
            diarization_segments.append({
                "start": transcript_segments[i]["start"],
                "end": transcript_segments[i]["end"],
                "speaker": best_speaker,
            })

    # Sort by start time
    diarization_segments.sort(key=lambda x: x["start"])

    print(f"  Clustering complete: {len(diarization_segments)} segments, "
          f"{len(set(cluster_to_speaker.values()))} speakers")

    # Print cluster stats
    for cl, dur in sorted_clusters:
        spk = cluster_to_speaker[cl]
        pct = dur / sum(d for _, d in sorted_clusters) * 100
        print(f"    {spk}: {dur/60:.1f} min ({pct:.0f}%)")

    return diarization_segments


# ═══════════════════════════════════════════════════════════════════════════
#  Merge diarization with transcript
# ═══════════════════════════════════════════════════════════════════════════

def merge_diarization_with_transcript(
    transcript_segments: list[dict],
    diarization_segments: list[dict],
) -> list[dict]:
    """
    Merge diarization labels into transcript segments by timestamp overlap.

    For each transcript segment, finds the diarization segment with maximum
    temporal overlap and assigns that speaker label.

    Returns enriched transcript segments with added 'speaker_id' field.
    """
    enriched = []

    for seg in transcript_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Find best overlapping diarization segment
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for d_seg in diarization_segments:
            overlap_start = max(seg_start, d_seg["start"])
            overlap_end = min(seg_end, d_seg["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]

        enriched_seg = dict(seg)
        enriched_seg["speaker_id"] = best_speaker
        enriched.append(enriched_seg)

    return enriched


# ═══════════════════════════════════════════════════════════════════════════
#  Per-speaker metrics
# ═══════════════════════════════════════════════════════════════════════════

# Common English filler words
FILLER_WORDS = [
    "um", "uh", "uhh", "umm", "ummm",
    "like", "you know", "I mean", "basically",
    "actually", "literally", "right", "so",
    "well", "okay", "ok",
]


def count_fillers(text: str) -> int:
    """Count filler word occurrences in text."""
    count = 0
    text_lower = text.lower()
    words = text_lower.split()
    for word in words:
        clean = re.sub(r"[,.\?!;:\-]", "", word)
        if clean in [f.lower() for f in FILLER_WORDS]:
            count += 1
    return count


def compute_speaker_metrics(
    enriched_segments: list[dict],
    speaker_name_map: Optional[dict[str, str]] = None,
) -> dict[str, dict[str, Any]]:
    """
    Compute per-speaker metrics from enriched transcript segments.

    Returns dict of speaker_id -> {
        "display_name": str,
        "segment_count": int,
        "word_count": int,
        "total_duration_s": float,
        "wpm": float,
        "filler_count": int,
        "filler_rate_pct": float,
        "avg_segment_length_s": float,
        "avg_segment_words": float,
    }
    """
    from collections import defaultdict

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

    speaker_name_map = speaker_name_map or {}

    result = {}
    for spk, s in stats.items():
        duration_min = s["total_duration_s"] / 60.0
        wpm = s["word_count"] / duration_min if duration_min > 0 else 0
        filler_rate = (s["filler_count"] / s["word_count"] * 100) if s["word_count"] > 0 else 0
        avg_seg_len = s["total_duration_s"] / s["segment_count"] if s["segment_count"] > 0 else 0
        avg_seg_words = s["word_count"] / s["segment_count"] if s["segment_count"] > 0 else 0

        result[spk] = {
            "display_name": speaker_name_map.get(spk, spk),
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


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_diarization_pipeline(
    episode_num: int,
    n_speakers: int,
    use_pyannote: bool = False,
    keep_audio: bool = False,
) -> dict[str, Any]:
    """
    Run the full diarization pipeline for one episode.

    Steps:
      1. Load existing transcript
      2. Download audio
      3. Run diarization (pyannote or clustering)
      4. Merge labels with transcript
      5. Compute per-speaker metrics
      6. Save outputs

    Returns summary dict.
    """
    print(f"\n{'='*70}")
    print(f"  DIARIZATION PIPELINE: Episode {episode_num}")
    print(f"  Speakers: {n_speakers}")
    print(f"  Method: {'pyannote' if use_pyannote else 'MFCC+clustering'}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # ── 1. Load transcript ─────────────────────────────────────────────
    transcript_path = REPORTS_DIR / f"episode_{episode_num}" / "transcript.json"
    if not transcript_path.exists():
        print(f"  ERROR: Transcript not found: {transcript_path}")
        return {"error": f"transcript_not_found: {transcript_path}"}

    transcript_segments = json.loads(transcript_path.read_text(encoding="utf-8"))
    print(f"  Loaded transcript: {len(transcript_segments)} segments")

    # ── 2. Download audio ──────────────────────────────────────────────
    audio_url = get_episode_audio_url(episode_num)
    if not audio_url:
        print(f"  ERROR: No audio URL found for episode {episode_num}")
        return {"error": f"no_audio_url for episode {episode_num}"}

    audio_dir = DATA_DIR / "audio"
    audio_path = audio_dir / f"episode_{episode_num}.mp3"
    download_audio(audio_url, audio_path)

    # ── 3. Run diarization ─────────────────────────────────────────────
    if use_pyannote and PYANNOTE_AVAILABLE:
        diarization_segments = diarize_with_pyannote(audio_path, n_speakers)
    else:
        if use_pyannote and not PYANNOTE_AVAILABLE:
            print("  WARNING: pyannote requested but not available. "
                  "Falling back to MFCC+clustering.")
        diarization_segments = diarize_with_clustering(
            audio_path, transcript_segments, n_speakers
        )

    if not diarization_segments:
        print("  ERROR: Diarization produced no segments")
        return {"error": "diarization_failed"}

    # ── 4. Merge with transcript ───────────────────────────────────────
    enriched = merge_diarization_with_transcript(
        transcript_segments, diarization_segments
    )
    print(f"\n  Merged: {len(enriched)} enriched segments")

    # ── 5. Map speaker names ───────────────────────────────────────────
    known_speakers = EPISODE_SPEAKERS.get(episode_num, [])
    speaker_name_map = {}
    if known_speakers:
        # Heuristic: map by talk-time rank to known speaker list
        unique_speakers = sorted(set(s["speaker_id"] for s in enriched))
        spk_durations = {}
        for seg in enriched:
            spk = seg["speaker_id"]
            spk_durations[spk] = spk_durations.get(spk, 0) + (seg["end"] - seg["start"])
        sorted_speakers = sorted(spk_durations.items(), key=lambda x: x[1], reverse=True)

        for i, (spk_id, _) in enumerate(sorted_speakers):
            if i < len(known_speakers):
                speaker_name_map[spk_id] = known_speakers[i]
            else:
                speaker_name_map[spk_id] = spk_id

        print(f"  Speaker name mapping:")
        for spk_id, name in speaker_name_map.items():
            dur = spk_durations.get(spk_id, 0)
            print(f"    {spk_id} -> {name} ({dur/60:.1f} min)")

    # ── 6. Compute metrics ─────────────────────────────────────────────
    metrics = compute_speaker_metrics(enriched, speaker_name_map)

    # ── 7. Save outputs ───────────────────────────────────────────────
    output_dir = REPORTS_DIR / f"episode_{episode_num}" / "diarization"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enriched transcript
    enriched_path = output_dir / "enriched_transcript.json"
    enriched_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Saved enriched transcript: {enriched_path}")

    # Raw diarization segments
    diarization_path = output_dir / "diarization_segments.json"
    diarization_path.write_text(
        json.dumps(diarization_segments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Speaker metrics
    metrics_path = output_dir / "speaker_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved speaker metrics: {metrics_path}")

    # Speaker name mapping
    mapping_path = output_dir / "speaker_name_map.json"
    mapping_path.write_text(
        json.dumps(speaker_name_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    elapsed = time.time() - start_time
    summary = {
        "episode": episode_num,
        "n_speakers": n_speakers,
        "method": "pyannote" if (use_pyannote and PYANNOTE_AVAILABLE) else "mfcc_clustering",
        "total_segments": len(enriched),
        "speaker_name_map": speaker_name_map,
        "speaker_metrics": metrics,
        "processing_time_s": round(elapsed, 1),
    }

    summary_path = output_dir / "diarization_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved summary: {summary_path}")

    # ── 8. Cleanup audio if requested ──────────────────────────────────
    if not keep_audio and audio_path.exists():
        audio_path.unlink()
        print(f"  Cleaned up audio: {audio_path}")

    print(f"\n  Pipeline complete in {elapsed:.0f}s")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Speaker diarization pipeline for podcast episodes"
    )
    parser.add_argument(
        "--episode", "-e", type=int,
        help="Episode number to diarize (e.g., 200)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all episodes with existing transcripts"
    )
    parser.add_argument(
        "--n-speakers", "-n", type=int, default=None,
        help="Number of speakers (default: auto from config or 3)"
    )
    parser.add_argument(
        "--speakers", "-s", type=str, default="",
        help="Comma-separated speaker names for this episode"
    )
    parser.add_argument(
        "--pyannote", action="store_true",
        help="Use pyannote-audio (requires HF_TOKEN, GPU recommended)"
    )
    parser.add_argument(
        "--keep-audio", action="store_true",
        help="Keep downloaded audio files (default: delete after processing)"
    )

    args = parser.parse_args()

    # Load config
    config = _load_config()

    # Register speakers from CLI args
    if args.speakers:
        speaker_list = [s.strip() for s in args.speakers.split(",") if s.strip()]
    else:
        cfg_speakers = config.get("speakers", [])
        speaker_list = [s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in cfg_speakers]

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
        if speaker_list:
            EPISODE_SPEAKERS[ep] = speaker_list

        n_speakers = args.n_speakers
        if n_speakers is None:
            n_speakers = len(EPISODE_SPEAKERS.get(ep, speaker_list or ["Speaker 1", "Speaker 2", "Speaker 3"]))

        result = run_diarization_pipeline(
            episode_num=ep,
            n_speakers=n_speakers,
            use_pyannote=args.pyannote,
            keep_audio=args.keep_audio,
        )
        results.append(result)

    print("\n" + "=" * 70)
    print("  ALL EPISODES COMPLETE")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  Episode {r.get('episode', '?')}: ERROR - {r['error']}")
        else:
            print(f"  Episode {r['episode']}: {r['n_speakers']} speakers, "
                  f"{r['total_segments']} segments, {r['processing_time_s']}s")


if __name__ == "__main__":
    main()

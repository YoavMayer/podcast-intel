#!/usr/bin/env python3
"""
Full analysis pipeline for a podcast episode.

This script:
  1. Polls the RSS feed until the episode appears (or uses a provided audio URL)
  2. Registers the episode in the episodes registry JSON
  3. Downloads the audio
  4. Transcribes with faster-whisper
  5. Runs speaker diarization
  6. Computes delivery, structure, engagement, and content metrics
  7. Generates reports

Usage:
    # Poll RSS and run full pipeline:
    python tools/run_episode_analysis.py --episode 206 --n-speakers 3 \
        --speakers "Alice,Bob,Charlie"

    # With a known audio URL (skip RSS polling):
    python tools/run_episode_analysis.py --episode 206 --n-speakers 3 \
        --audio-url "https://..." --speakers "Alice,Bob,Charlie"

    # Skip transcription (if transcript already exists):
    python tools/run_episode_analysis.py --episode 206 --skip-transcribe

Environment variables:
    HF_TOKEN              - HuggingFace token for pyannote (optional)
    PODCAST_INTEL_TRANSCRIPTION_DEVICE - "cuda" or "cpu" (default: auto-detect)
"""

import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
EPISODES_JSON = DATA_DIR / "episodes.json"

LOG_FILE = PROJECT_ROOT / "logs" / "episode_analysis.log"


def log(msg: str):
    """Print and log a message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _load_config():
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
#  Step 1: Poll RSS feed for the episode
# ═══════════════════════════════════════════════════════════════════════════

def poll_rss_for_episode(
    episode_num: int,
    rss_url: str,
    max_attempts: int = 48,
    interval_s: int = 900,
) -> dict:
    """
    Poll the RSS feed until the episode appears.
    Default: check every 15 minutes for up to 12 hours.

    Returns the episode metadata dict with audioUrl, title, guid, etc.
    """
    import feedparser

    target_str = str(episode_num)
    target_patterns = [
        f"episode {target_str}",
        f"ep {target_str}",
        f"#{target_str}",
    ]

    for attempt in range(1, max_attempts + 1):
        log(f"RSS poll attempt {attempt}/{max_attempts}...")

        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                title = entry.get("title", "")
                # Check if this entry matches our episode number
                title_lower = title.lower()
                if any(p in title_lower or p in title for p in target_patterns):
                    # Found it - extract metadata
                    audio_url = None
                    file_size = None
                    if hasattr(entry, "enclosures") and entry.enclosures:
                        for enc in entry.enclosures:
                            if enc.get("type", "").startswith("audio/"):
                                audio_url = enc.get("href") or enc.get("url")
                                file_size = enc.get("length")
                                break

                    if not audio_url:
                        continue

                    duration_str = getattr(entry, "itunes_duration", "")
                    pub_date = getattr(entry, "published", "")
                    guid = entry.get("id") or entry.get("guid", "")

                    log(f"Found episode {episode_num} in RSS: {title}")
                    return {
                        "title": title,
                        "epNum": episode_num,
                        "pubDate": pub_date,
                        "guid": guid,
                        "audioUrl": audio_url,
                        "durationStr": duration_str,
                        "fileSizeBytes": int(file_size) if file_size else None,
                    }
        except Exception as e:
            log(f"RSS fetch error: {e}")

        if attempt < max_attempts:
            log(f"Episode {episode_num} not in RSS yet. Next check in {interval_s}s...")
            time.sleep(interval_s)

    raise RuntimeError(
        f"Episode {episode_num} not found in RSS after {max_attempts} attempts "
        f"({max_attempts * interval_s / 3600:.1f} hours)"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2: Register episode in JSON
# ═══════════════════════════════════════════════════════════════════════════

def register_episode(episode_meta: dict) -> None:
    """Add episode to the episodes JSON registry (at the top)."""
    EPISODES_JSON.parent.mkdir(parents=True, exist_ok=True)

    if EPISODES_JSON.exists():
        episodes = json.loads(EPISODES_JSON.read_text(encoding="utf-8"))
    else:
        episodes = []

    # Check if already registered
    for ep in episodes:
        if ep.get("epNum") == episode_meta["epNum"]:
            log(f"Episode {episode_meta['epNum']} already registered, updating audioUrl")
            ep["audioUrl"] = episode_meta["audioUrl"]
            if episode_meta.get("guid"):
                ep["guid"] = episode_meta["guid"]
            EPISODES_JSON.write_text(
                json.dumps(episodes, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return

    # Parse duration
    duration_min = 0.0
    dur_str = episode_meta.get("durationStr", "")
    if dur_str:
        parts = dur_str.split(":")
        if len(parts) == 3:
            duration_min = int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
        elif len(parts) == 2:
            duration_min = int(parts[0]) + int(parts[1]) / 60

    entry = {
        "title": episode_meta["title"],
        "epNum": episode_meta["epNum"],
        "pubDate": episode_meta.get("pubDate", ""),
        "guid": episode_meta.get("guid", ""),
        "audioUrl": episode_meta["audioUrl"],
        "durationStr": dur_str,
        "durationMin": round(duration_min, 1),
    }

    episodes.insert(0, entry)
    EPISODES_JSON.write_text(
        json.dumps(episodes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"Registered episode {episode_meta['epNum']}: {episode_meta['title']}")


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3: Download audio
# ═══════════════════════════════════════════════════════════════════════════

def download_audio(audio_url: str, episode_num: int) -> Path:
    """Download episode audio. Returns path to MP3 file."""
    import requests

    audio_dir = DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dest = audio_dir / f"episode_{episode_num}.mp3"

    if dest.exists():
        log(f"Audio already downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    log(f"Downloading audio from {audio_url[:80]}...")
    resp = requests.get(audio_url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (8192 * 256) == 0:
                pct = downloaded / total * 100
                log(f"  Download: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)")

    log(f"Download complete: {dest.stat().st_size / 1e6:.1f} MB")
    return dest


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4: Transcription (faster-whisper)
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_episode(
    audio_path: Path,
    episode_num: int,
    language: str = "en",
    model_id: str = "small",
) -> Path:
    """
    Transcribe audio using faster-whisper.
    Returns path to transcript.json.
    """
    output_dir = REPORTS_DIR / f"episode_{episode_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcript.json"

    if transcript_path.exists():
        segs = json.loads(transcript_path.read_text(encoding="utf-8"))
        log(f"Transcript already exists: {len(segs)} segments")
        return transcript_path

    log(f"Loading faster-whisper model ({model_id})...")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper not installed. Install with:\n"
            "  pip install faster-whisper"
        )

    # Auto-detect device
    device = os.environ.get("PODCAST_INTEL_TRANSCRIPTION_DEVICE", "")
    if not device:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    log(f"Using device: {device}, compute_type: {compute_type}")

    model = WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
    )

    log(f"Transcribing {audio_path.name}...")
    start_time = time.time()

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        word_timestamps=True,
    )

    segments = []
    for seg in segments_iter:
        words_data = []
        if seg.words:
            for w in seg.words:
                words_data.append({
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word,
                    "probability": round(w.probability, 3),
                })

        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words_data,
        })

        if len(segments) % 200 == 0:
            log(f"  Transcribed {len(segments)} segments...")

    elapsed = time.time() - start_time
    log(f"Transcription complete: {len(segments)} segments in {elapsed:.0f}s")

    # Save transcript
    transcript_path.write_text(
        json.dumps(segments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"Saved transcript: {transcript_path}")

    # Save plain text version
    text_path = output_dir / "transcript.txt"
    full_text = "\n".join(seg["text"] for seg in segments)
    text_path.write_text(full_text, encoding="utf-8")
    log(f"Saved plain text: {text_path}")

    return transcript_path


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5: Diarization (delegates to diarize_episode.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_diarization(episode_num: int, n_speakers: int, speakers: list[str], keep_audio: bool = True) -> dict:
    """Run speaker diarization using the diarize_episode tool."""
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import diarize_episode

    # Register speakers for this episode
    diarize_episode.EPISODE_SPEAKERS[episode_num] = speakers

    log(f"Running diarization: {n_speakers} speakers, names={speakers}")

    result = diarize_episode.run_diarization_pipeline(
        episode_num=episode_num,
        n_speakers=n_speakers,
        use_pyannote=bool(os.environ.get("HF_TOKEN")),
        keep_audio=keep_audio,
    )

    if "error" in result:
        log(f"Diarization error: {result['error']}")
    else:
        log(f"Diarization complete: {result['total_segments']} segments, "
            f"{result['processing_time_s']}s")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Step 6: Metric computation
# ═══════════════════════════════════════════════════════════════════════════

def run_metrics(episode_num: int) -> dict:
    """Run all metric computation scripts."""
    results = {}
    scripts = [
        ("delivery", "src/compute_delivery_metrics.py"),
        ("structure", "src/compute_structure_metrics.py"),
        ("engagement", "src/engagement_analysis.py"),
        ("content", "scripts/content_relevance_metrics.py"),
    ]

    for name, script_path in scripts:
        full_path = PROJECT_ROOT / script_path
        if not full_path.exists():
            log(f"  Skipping {name} metrics: {script_path} not found")
            results[name] = {"status": "skipped", "reason": "script_not_found"}
            continue

        log(f"  Computing {name} metrics...")
        try:
            result = subprocess.run(
                [sys.executable, str(full_path), "--episode", str(episode_num)],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode == 0:
                log(f"  {name} metrics: OK")
                results[name] = {"status": "ok", "stdout": result.stdout[-500:]}
            else:
                log(f"  {name} metrics: FAILED (rc={result.returncode})")
                log(f"  stderr: {result.stderr[-300:]}")
                results[name] = {
                    "status": "error",
                    "returncode": result.returncode,
                    "stderr": result.stderr[-500:],
                }
        except subprocess.TimeoutExpired:
            log(f"  {name} metrics: TIMEOUT")
            results[name] = {"status": "timeout"}
        except Exception as e:
            log(f"  {name} metrics: EXCEPTION {e}")
            results[name] = {"status": "exception", "error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Step 7: Report generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_reports(episode_num: int) -> dict:
    """Generate HTML reports and panel chemistry analysis."""
    results = {}
    report_scripts = [
        ("one_pager", "tools/generate_one_pager.py"),
        ("panel_chemistry", "tools/analyze_panel_chemistry.py"),
    ]

    for name, script_path in report_scripts:
        full_path = PROJECT_ROOT / script_path
        if not full_path.exists():
            log(f"  Skipping {name}: {script_path} not found")
            results[name] = {"status": "skipped"}
            continue

        log(f"  Generating {name}...")
        try:
            cmd = [sys.executable, str(full_path)]
            if name == "one_pager":
                cmd.extend(["--episode", str(episode_num)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode == 0:
                log(f"  {name}: OK")
                results[name] = {"status": "ok"}
            else:
                log(f"  {name}: FAILED (rc={result.returncode})")
                results[name] = {
                    "status": "error",
                    "stderr": result.stderr[-500:],
                }
        except Exception as e:
            log(f"  {name}: EXCEPTION {e}")
            results[name] = {"status": "exception", "error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    episode_num: int,
    n_speakers: int,
    speakers: list[str],
    audio_url: str | None = None,
    title: str | None = None,
    skip_transcribe: bool = False,
    skip_metrics: bool = False,
    keep_audio: bool = True,
    rss_poll_interval: int = 900,
    rss_max_attempts: int = 48,
    language: str = "en",
    model_id: str = "small",
    rss_url: str = "",
) -> dict:
    """Run the complete analysis pipeline."""
    pipeline_start = time.time()
    results = {"episode": episode_num, "steps": {}}

    log(f"{'='*70}")
    log(f"  FULL ANALYSIS PIPELINE: Episode {episode_num}")
    log(f"  Speakers: {n_speakers} - {speakers}")
    log(f"  Audio URL: {'provided' if audio_url else 'will poll RSS'}")
    log(f"{'='*70}")

    # ── Step 1: Get episode metadata ─────────────────────────────────────
    if audio_url:
        episode_meta = {
            "title": title or f"Episode {episode_num}",
            "epNum": episode_num,
            "audioUrl": audio_url,
            "pubDate": datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "guid": "",
            "durationStr": "",
        }
        log("Using provided audio URL")
    else:
        if not rss_url:
            raise RuntimeError(
                "No --audio-url provided and no RSS URL configured. "
                "Set rss_url in podcast.yaml or pass --audio-url."
            )
        log("Polling RSS feed for episode...")
        episode_meta = poll_rss_for_episode(
            episode_num,
            rss_url=rss_url,
            max_attempts=rss_max_attempts,
            interval_s=rss_poll_interval,
        )

    results["steps"]["rss"] = {"status": "ok", "title": episode_meta.get("title")}

    # ── Step 2: Register in JSON ─────────────────────────────────────────
    register_episode(episode_meta)
    results["steps"]["register"] = {"status": "ok"}

    # ── Step 3: Download audio ───────────────────────────────────────────
    audio_path = download_audio(episode_meta["audioUrl"], episode_num)
    results["steps"]["download"] = {
        "status": "ok",
        "size_mb": round(audio_path.stat().st_size / 1e6, 1),
    }

    # ── Step 4: Transcription ────────────────────────────────────────────
    if skip_transcribe:
        transcript_path = REPORTS_DIR / f"episode_{episode_num}" / "transcript.json"
        if not transcript_path.exists():
            log("ERROR: --skip-transcribe but no transcript found!")
            results["steps"]["transcribe"] = {"status": "error", "reason": "not_found"}
            return results
        log("Skipping transcription (--skip-transcribe)")
        results["steps"]["transcribe"] = {"status": "skipped"}
    else:
        transcript_path = transcribe_episode(
            audio_path, episode_num, language=language, model_id=model_id,
        )
        segs = json.loads(transcript_path.read_text(encoding="utf-8"))
        results["steps"]["transcribe"] = {
            "status": "ok",
            "segments": len(segs),
        }

    # ── Step 5: Diarization ──────────────────────────────────────────────
    log("\n-- DIARIZATION --")
    diarization_result = run_diarization(episode_num, n_speakers, speakers, keep_audio)
    results["steps"]["diarization"] = diarization_result

    # ── Step 6: Metrics ──────────────────────────────────────────────────
    if skip_metrics:
        log("Skipping metrics (--skip-metrics)")
        results["steps"]["metrics"] = {"status": "skipped"}
    else:
        log("\n-- METRICS --")
        metrics_result = run_metrics(episode_num)
        results["steps"]["metrics"] = metrics_result

    # ── Step 7: Reports ──────────────────────────────────────────────────
    log("\n-- REPORTS --")
    report_result = generate_reports(episode_num)
    results["steps"]["reports"] = report_result

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    results["total_time_s"] = round(elapsed, 1)
    results["total_time_min"] = round(elapsed / 60, 1)
    results["completed_at"] = datetime.now().isoformat()

    # Save pipeline results
    output_dir = REPORTS_DIR / f"episode_{episode_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "pipeline_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log(f"\n{'='*70}")
    log(f"  PIPELINE COMPLETE: Episode {episode_num}")
    log(f"  Total time: {elapsed/60:.1f} minutes")
    log(f"  Results: {results_path}")
    log(f"{'='*70}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full analysis pipeline for podcast episodes"
    )
    parser.add_argument(
        "--episode", "-e", type=int, required=True,
        help="Episode number (e.g., 206)"
    )
    parser.add_argument(
        "--n-speakers", "-n", type=int, default=3,
        help="Number of speakers (default: 3)"
    )
    parser.add_argument(
        "--speakers", "-s", type=str, default="",
        help="Comma-separated speaker names (ordered by expected talk time)"
    )
    parser.add_argument(
        "--audio-url", type=str, default=None,
        help="Direct audio URL (skip RSS polling)"
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Episode title (used when providing --audio-url)"
    )
    parser.add_argument(
        "--skip-transcribe", action="store_true",
        help="Skip transcription (use existing transcript)"
    )
    parser.add_argument(
        "--skip-metrics", action="store_true",
        help="Skip metric computation"
    )
    parser.add_argument(
        "--keep-audio", action="store_true", default=True,
        help="Keep downloaded audio file (default: True)"
    )
    parser.add_argument(
        "--rss-poll-interval", type=int, default=900,
        help="RSS poll interval in seconds (default: 900 = 15 min)"
    )
    parser.add_argument(
        "--rss-max-attempts", type=int, default=48,
        help="Max RSS poll attempts (default: 48 = 12 hours)"
    )
    parser.add_argument(
        "--delay", type=int, default=0,
        help="Delay start by N seconds (for scheduling)"
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Transcription language code (e.g., en, he). Default: from podcast.yaml or en"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Whisper model ID (e.g., small, large-v3). Default: from podcast.yaml or small"
    )

    args = parser.parse_args()

    # Load config for defaults
    config = _load_config()
    language = args.language or config.get("language", "en")
    model_id = args.model or config.get("models", {}).get("transcription", "small")
    rss_url = config.get("rss_url", "")

    if args.delay > 0:
        target_time = datetime.now().timestamp() + args.delay
        target_str = datetime.fromtimestamp(target_time).strftime("%H:%M:%S")
        log(f"Delaying start by {args.delay}s (will begin at ~{target_str})")
        time.sleep(args.delay)

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()] if args.speakers else []
    if not speakers:
        # Try to get default speakers from config
        cfg_speakers = config.get("speakers", [])
        if cfg_speakers:
            speakers = [s.get("name", s) if isinstance(s, dict) else str(s) for s in cfg_speakers]
        else:
            speakers = [f"Speaker {i+1}" for i in range(args.n_speakers)]

    run_full_pipeline(
        episode_num=args.episode,
        n_speakers=args.n_speakers,
        speakers=speakers,
        audio_url=args.audio_url,
        title=args.title,
        skip_transcribe=args.skip_transcribe,
        skip_metrics=args.skip_metrics,
        keep_audio=args.keep_audio,
        rss_poll_interval=args.rss_poll_interval,
        rss_max_attempts=args.rss_max_attempts,
        language=language,
        model_id=model_id,
        rss_url=rss_url,
    )


if __name__ == "__main__":
    main()

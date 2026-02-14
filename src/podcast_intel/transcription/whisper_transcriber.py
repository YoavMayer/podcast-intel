"""
Real transcription pipeline using faster-whisper on CPU.

Transcribes podcast audio files using the faster-whisper library with
CTranslate2 backend. Supports CPU-only operation with int8 quantization.

The transcription model and language are configurable via podcast.yaml
or environment variables.
"""

import time
import re
import sys
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any

from faster_whisper import WhisperModel

from podcast_intel.transcription.transcribe import (
    TranscriptionInterface,
    TranscriptionResult,
)
from podcast_intel.models.database import Database
from podcast_intel.config import get_config


def detect_language(text: str) -> str:
    """Detect if text is Hebrew, English, or mixed."""
    hebrew_chars = sum(1 for c in text if "\u0590" <= c <= "\u05FF")
    latin_chars = sum(1 for c in text if "A" <= c <= "Z" or "a" <= c <= "z")
    total = hebrew_chars + latin_chars
    if total == 0:
        return "en"
    hebrew_ratio = hebrew_chars / total
    if hebrew_ratio > 0.8:
        return "he"
    elif hebrew_ratio < 0.2:
        return "en"
    return "mixed"


class WhisperTranscriber(TranscriptionInterface):
    """
    Production transcriber using faster-whisper.

    Supports CPU (int8) and GPU (float16) inference.
    Uses VAD filtering for better segment quality.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            print(f"Loading Whisper model '{self.model_size}' "
                  f"(device={self.device}, compute={self.compute_type})...")
            start = time.time()
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            print(f"Model loaded in {time.time() - start:.1f}s")
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str = "en",
        diarize: bool = True,
    ) -> TranscriptionResult:
        model = self._get_model()

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        segments = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (seg.words or [])
                ],
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
            })

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            duration=info.duration,
        )

    def get_word_timestamps(self, audio_path: Path) -> List[Dict[str, Any]]:
        result = self.transcribe(audio_path)
        words = []
        for seg in result.segments:
            words.extend(seg.get("words", []))
        return words


def transcribe_episode(
    audio_path: Path,
    db: Database,
    episode_id: int,
    model_size: str = "small",
    compute_type: str = "int8",
) -> dict:
    """Transcribe an episode and store results in the database."""
    transcriber = WhisperTranscriber(
        model_size=model_size,
        device="cpu",
        compute_type=compute_type,
    )

    # Update episode status
    with db.get_connection() as conn:
        db.update_episode_status(conn, episode_id, "processing")

    print(f"Transcribing {audio_path.name}...")
    start_transcribe = time.time()

    result = transcriber.transcribe(audio_path)

    segment_count = 0
    total_words = 0
    filler_count = 0
    silence_events = []
    prev_end = 0.0

    with db.get_connection() as conn:
        for seg in result.segments:
            segment_count += 1
            text = seg["text"]
            word_count = len(text.split())
            total_words += word_count
            language = detect_language(text)

            # Detect silence gaps
            if prev_end > 0 and seg["start"] - prev_end > 2.0:
                gap = seg["start"] - prev_end
                silence_events.append({
                    "start": prev_end,
                    "end": seg["start"],
                    "duration": gap,
                })
                db.insert_silence_event(
                    conn,
                    episode_id=episode_id,
                    start_time=prev_end,
                    end_time=seg["start"],
                    duration=gap,
                    event_type="dead_air" if gap > 5.0 else "long_pause",
                )

            # Insert segment
            seg_id = db.insert_segment(
                conn,
                episode_id=episode_id,
                start_time=seg["start"],
                end_time=seg["end"],
                text=text,
                word_count=word_count,
                language=language,
                confidence=seg.get("avg_logprob"),
            )

            prev_end = seg["end"]

            if segment_count % 50 == 0:
                elapsed = time.time() - start_transcribe
                print(f"  [{segment_count} segments, {elapsed:.0f}s elapsed, "
                      f"{seg['end']:.0f}s of audio processed]")

        db.update_episode_status(conn, episode_id, "completed")

    transcribe_time = time.time() - start_transcribe

    stats = {
        "episode_id": episode_id,
        "segments": segment_count,
        "total_words": total_words,
        "fillers_detected": filler_count,
        "silence_events": len(silence_events),
        "audio_duration_s": result.duration,
        "transcribe_time_s": round(transcribe_time, 1),
        "speed_ratio": round(result.duration / transcribe_time, 2) if transcribe_time > 0 else 0,
        "model_size": model_size,
    }

    print(f"\nTranscription complete!")
    print(f"  Segments: {segment_count}")
    print(f"  Words: {total_words}")
    print(f"  Silence events: {len(silence_events)}")
    print(f"  Audio duration: {result.duration:.0f}s")
    print(f"  Processing time: {transcribe_time:.0f}s")
    print(f"  Speed: {stats['speed_ratio']}x real-time")

    return stats


def download_episode(url: str, output_path: Path) -> Path:
    """Download an episode MP3 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"Audio already downloaded: {output_path} ({size_mb:.1f} MB)")
        return output_path

    print(f"Downloading to {output_path}...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (10 * 1024 * 1024) < 65536:
                pct = downloaded / total * 100
                print(f"  {downloaded / 1024 / 1024:.0f} MB / "
                      f"{total / 1024 / 1024:.0f} MB ({pct:.0f}%)")

    print(f"Download complete: {output_path} ({downloaded / 1024 / 1024:.1f} MB)")
    return output_path


def main():
    """Download and transcribe an episode from config."""
    config = get_config()
    db = Database(config.db_path)
    db.initialize()

    ep_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"

    print(f"=== Podcast Transcription Pipeline ===")
    print(f"Model: {model_size} (CPU, int8)")
    print(f"Episode ID: {ep_id}")
    print()

    # Get episode from DB
    with db.get_connection() as conn:
        episode = db.get_episode_by_id(conn, ep_id)

    if not episode:
        print(f"Episode {ep_id} not found in database.")
        sys.exit(1)

    # Download
    audio_path = config.audio_dir / f"ep_{episode['guid'][:8]}.mp3"
    download_episode(episode["audio_url"], audio_path)

    # Transcribe
    print()
    stats = transcribe_episode(
        audio_path=audio_path,
        db=db,
        episode_id=ep_id,
        model_size=model_size,
    )

    print(f"\n{'='*50}")
    print(f"FINAL STATS: {stats}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

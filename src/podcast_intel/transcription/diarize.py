"""
Speaker diarization using MFCC features and spectral clustering.

CPU-only approach: extracts audio features per segment, clusters them
into N speakers, and updates the database with speaker assignments.
"""

import sys
import time
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from podcast_intel.models.database import Database
from podcast_intel.config import get_config


def extract_segment_embedding(audio_path: str, start: float, end: float, sr: int = 16000) -> np.ndarray:
    """Extract a fixed-size audio embedding for a segment using MFCCs."""
    duration = end - start
    if duration < 0.3:
        return None

    # Load just this segment
    y, _ = librosa.load(audio_path, sr=sr, offset=start, duration=duration)
    if len(y) < sr * 0.3:  # less than 0.3s of audio
        return None

    # Extract MFCCs (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)

    # Also extract delta and delta-delta MFCCs for better speaker discrimination
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Compute statistics over time: mean + std for each coefficient
    features = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1),
        mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1),
    ])

    # Add spectral features for better discrimination
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    f0, _, _ = librosa.pyin(y, sr=sr, fmin=60, fmax=400)
    f0_clean = f0[~np.isnan(f0)]
    f0_mean = np.mean(f0_clean) if len(f0_clean) > 0 else 150.0
    f0_std = np.std(f0_clean) if len(f0_clean) > 0 else 30.0

    extra = np.array([
        spectral_centroid.mean(), spectral_centroid.std(),
        spectral_bandwidth.mean(), spectral_bandwidth.std(),
        f0_mean, f0_std,
    ])

    return np.concatenate([features, extra])


def diarize_episode(
    audio_path: Path,
    db: Database,
    episode_id: int,
    n_speakers: int = 3,
) -> dict:
    """
    Run speaker diarization on an episode.

    Extracts audio features per segment, clusters into speakers,
    and updates the database with speaker assignments.
    """
    print(f"Diarizing episode {episode_id}: {audio_path.name}")
    print(f"  Target speakers: {n_speakers}")
    start_time = time.time()

    # Get segments from DB
    import sqlite3
    conn_raw = sqlite3.connect(str(db.db_path))
    conn_raw.row_factory = sqlite3.Row
    segments = conn_raw.execute(
        "SELECT id, start_time, end_time, word_count FROM segments WHERE episode_id = ? ORDER BY start_time",
        (episode_id,)
    ).fetchall()
    conn_raw.close()

    print(f"  Segments to process: {len(segments)}")

    # Extract embeddings for each segment
    embeddings = []
    valid_segment_ids = []
    audio_str = str(audio_path)

    for i, seg in enumerate(segments):
        emb = extract_segment_embedding(audio_str, seg["start_time"], seg["end_time"])
        if emb is not None:
            embeddings.append(emb)
            valid_segment_ids.append(seg["id"])

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(segments)} segments, {elapsed:.0f}s elapsed]")

    print(f"  Valid embeddings: {len(embeddings)}/{len(segments)}")

    if len(embeddings) < n_speakers:
        print("  ERROR: Not enough valid segments for clustering")
        return {"error": "insufficient_segments"}

    # Normalize features
    X = np.array(embeddings)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try Agglomerative Clustering (more robust for speaker diarization)
    clustering = AgglomerativeClustering(
        n_clusters=n_speakers,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(X_scaled)

    # Compute silhouette score for quality assessment
    sil_score = silhouette_score(X_scaled, labels, metric="cosine")
    print(f"  Silhouette score: {sil_score:.3f} (>0.3 = good separation)")

    # Analyze clusters to assign speaker names
    # Count words per cluster to rank speakers by talk time
    cluster_stats = {}
    for idx, seg_id in enumerate(valid_segment_ids):
        label = labels[idx]
        seg = next(s for s in segments if s["id"] == seg_id)
        if label not in cluster_stats:
            cluster_stats[label] = {"segments": 0, "words": 0, "duration": 0.0}
        cluster_stats[label]["segments"] += 1
        cluster_stats[label]["words"] += seg["word_count"]
        cluster_stats[label]["duration"] += seg["end_time"] - seg["start_time"]

    # Sort by total duration (most talk time first)
    sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]["duration"], reverse=True)

    # Map cluster labels to speaker names
    speaker_names = ["Speaker_A", "Speaker_B", "Speaker_C", "Speaker_D", "Speaker_E"]
    cluster_to_speaker = {}
    for rank, (cluster_label, stats) in enumerate(sorted_clusters):
        name = speaker_names[rank] if rank < len(speaker_names) else f"Speaker_{rank+1}"
        cluster_to_speaker[cluster_label] = name

    # Insert speakers and update segments in DB
    with db.get_connection() as conn:
        speaker_id_map = {}
        for rank, (cluster_label, stats) in enumerate(sorted_clusters):
            name = cluster_to_speaker[cluster_label]
            speaker_id = db.insert_speaker(conn, name=name, is_host=(rank < 2))
            speaker_id_map[cluster_label] = speaker_id

        # Update segments with speaker_id
        for idx, seg_id in enumerate(valid_segment_ids):
            label = labels[idx]
            speaker_id = speaker_id_map[label]
            conn.execute(
                "UPDATE segments SET speaker_id = ? WHERE id = ?",
                (speaker_id, seg_id),
            )

    elapsed = time.time() - start_time

    # Print results
    print(f"\n  Diarization complete in {elapsed:.0f}s")
    print(f"  Speaker assignments:")
    for cluster_label, stats in sorted_clusters:
        name = cluster_to_speaker[cluster_label]
        dur_min = stats["duration"] / 60
        pct = stats["duration"] / sum(s["duration"] for s in cluster_stats.values()) * 100
        print(f"    {name}: {stats['segments']} segs, {stats['words']} words, "
              f"{dur_min:.1f} min ({pct:.1f}%)")

    return {
        "episode_id": episode_id,
        "n_speakers": n_speakers,
        "silhouette_score": round(sil_score, 3),
        "speakers": {
            cluster_to_speaker[cl]: {
                "segments": stats["segments"],
                "words": stats["words"],
                "duration_s": round(stats["duration"], 1),
            }
            for cl, stats in sorted_clusters
        },
        "processing_time_s": round(elapsed, 1),
    }


def main():
    """Diarize episodes from the database."""
    config = get_config()
    db = Database(config.db_path)

    n_speakers = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    # Map episode IDs to audio files
    import sqlite3
    conn = sqlite3.connect(str(config.db_path))
    conn.row_factory = sqlite3.Row
    episodes = conn.execute("SELECT id, guid, title FROM episodes ORDER BY id").fetchall()
    conn.close()

    for ep in episodes:
        audio_path = config.audio_dir / f"ep_{ep['guid'][:8]}.mp3"
        if not audio_path.exists():
            print(f"Skipping episode {ep['id']}: audio not found at {audio_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Episode {ep['id']}: {ep['title']}")
        print(f"{'='*60}")

        stats = diarize_episode(audio_path, db, ep["id"], n_speakers=n_speakers)
        print(f"\nStats: {stats}")


if __name__ == "__main__":
    main()

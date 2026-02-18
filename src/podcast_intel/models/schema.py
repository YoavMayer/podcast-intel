"""
SQLite schema and database initialization.

Defines the complete database schema for the Podcast Intelligence system,
including tables for episodes, segments, speakers, entities, metrics,
embeddings, and coaching notes. Provides functions to create and initialize
the database with all tables and indexes.

Provides functions to create and initialize the database with all tables and indexes.
"""

import sqlite3
from pathlib import Path
from typing import Optional


# Complete SQLite schema for podcast intelligence system
SCHEMA_SQL = """
-- ============================================================
-- EPISODES: Core episode metadata from RSS feed
-- ============================================================
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    guid            TEXT    NOT NULL UNIQUE,
    title           TEXT    NOT NULL,
    description     TEXT,
    pub_date        TEXT    NOT NULL,  -- ISO-8601 format
    audio_url       TEXT    NOT NULL,
    audio_path      TEXT,              -- local filesystem path to downloaded MP3
    duration_seconds INTEGER,
    file_size_bytes  INTEGER,
    episode_type    TEXT    DEFAULT 'full' CHECK (episode_type IN ('full', 'trailer', 'bonus')),
    transcription_status TEXT DEFAULT 'pending' CHECK (transcription_status IN ('pending', 'processing', 'completed', 'failed')),
    pqs_score       REAL,             -- Podcast Quality Score (0-100)
    created_at      TEXT    DEFAULT (datetime('now')),
    updated_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_pub_date ON episodes(pub_date);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(transcription_status);

-- ============================================================
-- SPEAKERS: Identified panelists/hosts
-- ============================================================
CREATE TABLE IF NOT EXISTS speakers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    name_localized  TEXT,              -- Localized display name (optional)
    voice_embedding BLOB,             -- pyannote speaker embedding for identification
    is_host         INTEGER DEFAULT 0 CHECK (is_host IN (0, 1)),
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_speakers_name ON speakers(name);

-- ============================================================
-- SEGMENTS: Diarized transcript segments (one per speaker turn)
-- ============================================================
CREATE TABLE IF NOT EXISTS segments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    speaker_id      INTEGER REFERENCES speakers(id),
    start_time      REAL    NOT NULL,  -- seconds from episode start
    end_time        REAL    NOT NULL,  -- seconds from episode start
    text            TEXT    NOT NULL,  -- UTF-8 transcript text
    word_count      INTEGER NOT NULL DEFAULT 0,
    language        TEXT    DEFAULT 'en' CHECK (length(language) BETWEEN 2 AND 10),
    sentiment_score REAL,              -- -1.0 to 1.0
    confidence      REAL,              -- ASR confidence 0.0-1.0
    created_at      TEXT    DEFAULT (datetime('now')),
    CHECK (end_time > start_time)
);

CREATE INDEX IF NOT EXISTS idx_segments_episode ON segments(episode_id);
CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(speaker_id);
CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(episode_id, start_time);

-- ============================================================
-- ENTITIES: Canonical entity records (persons, organizations, etc.)
-- ============================================================
CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT    NOT NULL,
    name_localized  TEXT,
    entity_type     TEXT    NOT NULL CHECK (entity_type IN ('person', 'organization', 'location', 'event', 'other')),
    external_id     TEXT,              -- Wikidata QID or other external identifier
    metadata_json   TEXT,              -- additional structured data as JSON
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name, entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

-- ============================================================
-- ENTITY_MENTIONS: Links entities to specific segments
-- ============================================================
CREATE TABLE IF NOT EXISTS entity_mentions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id       INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    segment_id      INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    mention_text    TEXT    NOT NULL,  -- the actual surface form used in speech
    start_offset    INTEGER,           -- character offset within segment text
    confidence      REAL    DEFAULT 1.0,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_episode ON entity_mentions(episode_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_segment ON entity_mentions(segment_id);

-- ============================================================
-- METRICS: Per-episode and per-speaker computed metrics
-- ============================================================
CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    speaker_id      INTEGER REFERENCES speakers(id),  -- NULL = episode-level metric
    metric_name     TEXT    NOT NULL,
    metric_value    REAL    NOT NULL,
    metric_unit     TEXT,              -- e.g., 'wpm', 'percent', 'count', 'score'
    computed_at     TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_metrics_episode ON metrics(episode_id);
CREATE INDEX IF NOT EXISTS idx_metrics_speaker ON metrics(speaker_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_metrics_unique ON metrics(episode_id, speaker_id, metric_name);

-- ============================================================
-- EMBEDDINGS_METADATA: Tracks chunks indexed in the vector store
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings_metadata (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    segment_id      INTEGER REFERENCES segments(id),
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT    NOT NULL,
    speaker_id      INTEGER REFERENCES speakers(id),
    start_time      REAL    NOT NULL,
    end_time        REAL    NOT NULL,
    token_count     INTEGER NOT NULL,
    vector_store_id TEXT    NOT NULL,  -- ID in ChromaDB/Qdrant
    embedding_model TEXT    NOT NULL DEFAULT 'bge-m3',
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_embeddings_episode ON embeddings_metadata(episode_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_id ON embeddings_metadata(vector_store_id);

-- ============================================================
-- FILLER_WORDS: Individual filler word occurrences
-- ============================================================
CREATE TABLE IF NOT EXISTS filler_words (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id      INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    speaker_id      INTEGER REFERENCES speakers(id),
    filler_text     TEXT    NOT NULL,  -- the actual filler word, e.g., "um", "like"
    position_offset INTEGER,           -- character offset in segment text
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_filler_words_episode ON filler_words(episode_id);
CREATE INDEX IF NOT EXISTS idx_filler_words_speaker ON filler_words(speaker_id);
CREATE INDEX IF NOT EXISTS idx_filler_words_text ON filler_words(filler_text);

-- ============================================================
-- SILENCE_EVENTS: Dead air and significant silence instances
-- ============================================================
CREATE TABLE IF NOT EXISTS silence_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    start_time      REAL    NOT NULL,  -- seconds from episode start
    end_time        REAL    NOT NULL,  -- seconds from episode start
    duration        REAL    NOT NULL,  -- seconds
    event_type      TEXT    NOT NULL DEFAULT 'dead_air' CHECK (event_type IN ('dead_air', 'long_pause', 'technical')),
    preceding_speaker_id INTEGER REFERENCES speakers(id),
    following_speaker_id INTEGER REFERENCES speakers(id),
    created_at      TEXT    DEFAULT (datetime('now')),
    CHECK (end_time > start_time),
    CHECK (duration > 0)
);

CREATE INDEX IF NOT EXISTS idx_silence_episode ON silence_events(episode_id);
CREATE INDEX IF NOT EXISTS idx_silence_time ON silence_events(episode_id, start_time);

-- ============================================================
-- COACHING_NOTES: LLM-generated per-speaker coaching feedback
-- ============================================================
CREATE TABLE IF NOT EXISTS coaching_notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    speaker_id      INTEGER NOT NULL REFERENCES speakers(id),
    strengths       TEXT    NOT NULL,  -- JSON array of strength observations
    improvements    TEXT    NOT NULL,  -- JSON array of improvement suggestions
    trends          TEXT,              -- JSON object with trend observations
    generated_by    TEXT    NOT NULL DEFAULT 'gpt-4o-mini',  -- model used
    generated_at    TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_coaching_episode ON coaching_notes(episode_id);
CREATE INDEX IF NOT EXISTS idx_coaching_speaker ON coaching_notes(speaker_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_coaching_unique ON coaching_notes(episode_id, speaker_id);
"""


def create_all_tables(db_path: Path) -> None:
    """
    Create database and all tables with indexes.

    Initializes a new SQLite database with the complete schema.
    This function is idempotent - safe to call multiple times.

    The database is configured for UTF-8 support to properly handle multilingual text.

    Args:
        db_path: Path to the SQLite database file to create/initialize

    Example:
        >>> from pathlib import Path
        >>> db_path = Path("data/db/podcast_intel.db")
        >>> create_all_tables(db_path)
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect and configure for UTF-8
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA encoding = 'UTF-8'")
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        # Execute the schema
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def get_table_names(db_path: Path) -> list[str]:
    """
    Get list of all tables in the database.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        List of table names

    Example:
        >>> tables = get_table_names(Path("data/db/podcast_intel.db"))
        >>> print(tables)
        ['episodes', 'speakers', 'segments', ...]
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def get_table_info(db_path: Path, table_name: str) -> list[tuple]:
    """
    Get schema information for a specific table.

    Args:
        db_path: Path to the SQLite database file
        table_name: Name of the table to inspect

    Returns:
        List of column information tuples (cid, name, type, notnull, dflt_value, pk)

    Example:
        >>> info = get_table_info(Path("data/db/podcast_intel.db"), "episodes")
        >>> for col in info:
        ...     print(f"{col[1]}: {col[2]}")
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return cursor.fetchall()
    finally:
        conn.close()


def drop_all_tables(db_path: Path) -> None:
    """
    Drop all tables in the database.

    WARNING: This is a destructive operation that will delete all data.
    Use only for testing or when you need to recreate the schema.

    Args:
        db_path: Path to the SQLite database file

    Example:
        >>> # Only use in test scenarios!
        >>> drop_all_tables(Path("data/db/test.db"))
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = OFF")

    try:
        # Get all tables
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Drop each table
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()

"""
Database management and data access layer.

Provides a Database class for managing SQLite connections and helper methods
for common operations like inserting episodes, segments, entities, and metrics.
Includes proper error handling, connection management, and UTF-8 support for
multilingual text.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from datetime import datetime

from .schema import create_all_tables


class Database:
    """
    Database connection and query management.

    Provides methods for managing the database connection, executing queries,
    and common CRUD operations for episodes, segments, speakers, entities,
    and metrics. All connections are configured for UTF-8 encoding to properly
    handle multilingual text.

    Example:
        >>> db = Database(Path("data/db/podcast.db"))
        >>> db.initialize()
        >>> with db.get_connection() as conn:
        ...     episode_id = db.insert_episode(conn, guid="ep-001", title="Test", ...)
    """

    def __init__(self, db_path: Path):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """
        Initialize database schema.

        Creates all tables and indexes if they don't exist.
        Safe to call multiple times (idempotent).
        """
        create_all_tables(self.db_path)

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Automatically handles connection cleanup and ensures UTF-8 encoding
        and foreign key constraints are enabled.

        Yields:
            sqlite3.Connection: Database connection with row factory set

        Example:
            >>> with db.get_connection() as conn:
            ...     cursor = conn.execute("SELECT * FROM episodes")
            ...     for row in cursor:
            ...         print(row['title'])
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute("PRAGMA encoding = 'UTF-8'")
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def execute_script(self, script: str) -> None:
        """
        Execute a SQL script (multiple statements).

        Args:
            script: SQL script to execute

        Raises:
            sqlite3.Error: If script execution fails
        """
        with self.get_connection() as conn:
            conn.executescript(script)

    def insert_episode(
        self,
        conn: sqlite3.Connection,
        guid: str,
        title: str,
        pub_date: str,
        audio_url: str,
        description: Optional[str] = None,
        audio_path: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        file_size_bytes: Optional[int] = None,
        episode_type: str = "full",
    ) -> int:
        """
        Insert a new episode record.

        Args:
            conn: Database connection
            guid: Unique episode GUID from RSS feed
            title: Episode title
            pub_date: Publication date (ISO-8601 format)
            audio_url: URL to audio file
            description: Episode description (optional)
            audio_path: Local path to downloaded audio (optional)
            duration_seconds: Episode duration in seconds (optional)
            file_size_bytes: Audio file size in bytes (optional)
            episode_type: Episode type (full/trailer/bonus)

        Returns:
            int: ID of inserted episode

        Raises:
            sqlite3.IntegrityError: If episode with same GUID already exists
        """
        cursor = conn.execute(
            """
            INSERT INTO episodes (
                guid, title, description, pub_date, audio_url,
                audio_path, duration_seconds, file_size_bytes, episode_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                guid,
                title,
                description,
                pub_date,
                audio_url,
                audio_path,
                duration_seconds,
                file_size_bytes,
                episode_type,
            ),
        )
        return cursor.lastrowid

    def get_episode_by_guid(
        self, conn: sqlite3.Connection, guid: str
    ) -> Optional[sqlite3.Row]:
        """
        Retrieve an episode by its GUID.

        Args:
            conn: Database connection
            guid: Episode GUID

        Returns:
            Episode row or None if not found
        """
        cursor = conn.execute("SELECT * FROM episodes WHERE guid = ?", (guid,))
        return cursor.fetchone()

    def get_episode_by_id(
        self, conn: sqlite3.Connection, episode_id: int
    ) -> Optional[sqlite3.Row]:
        """
        Retrieve an episode by its ID.

        Args:
            conn: Database connection
            episode_id: Episode ID

        Returns:
            Episode row or None if not found
        """
        cursor = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        return cursor.fetchone()

    def update_episode_status(
        self, conn: sqlite3.Connection, episode_id: int, status: str
    ) -> None:
        """
        Update episode transcription status.

        Args:
            conn: Database connection
            episode_id: Episode ID
            status: New status (pending/processing/completed/failed)
        """
        conn.execute(
            "UPDATE episodes SET transcription_status = ?, updated_at = datetime('now') WHERE id = ?",
            (status, episode_id),
        )

    def insert_speaker(
        self,
        conn: sqlite3.Connection,
        name: str,
        name_localized: Optional[str] = None,
        is_host: bool = False,
        voice_embedding: Optional[bytes] = None,
    ) -> int:
        """
        Insert or get a speaker record.

        Args:
            conn: Database connection
            name: Speaker name (unique)
            name_localized: Localized display name (optional)
            is_host: Whether this speaker is a host
            voice_embedding: Voice embedding bytes (optional)

        Returns:
            int: ID of speaker (existing or newly inserted)
        """
        # Try to get existing speaker
        cursor = conn.execute("SELECT id FROM speakers WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        # Insert new speaker
        cursor = conn.execute(
            """
            INSERT INTO speakers (name, name_localized, is_host, voice_embedding)
            VALUES (?, ?, ?, ?)
            """,
            (name, name_localized, 1 if is_host else 0, voice_embedding),
        )
        return cursor.lastrowid

    def get_speaker_by_name(
        self, conn: sqlite3.Connection, name: str
    ) -> Optional[sqlite3.Row]:
        """
        Retrieve a speaker by name.

        Args:
            conn: Database connection
            name: Speaker name

        Returns:
            Speaker row or None if not found
        """
        cursor = conn.execute("SELECT * FROM speakers WHERE name = ?", (name,))
        return cursor.fetchone()

    def insert_segment(
        self,
        conn: sqlite3.Connection,
        episode_id: int,
        start_time: float,
        end_time: float,
        text: str,
        speaker_id: Optional[int] = None,
        word_count: Optional[int] = None,
        language: str = "en",
        sentiment_score: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """
        Insert a transcript segment.

        Args:
            conn: Database connection
            episode_id: Episode ID
            start_time: Start time in seconds
            end_time: End time in seconds
            text: Transcript text (UTF-8)
            speaker_id: Speaker ID (optional)
            word_count: Number of words (optional, computed if None)
            language: Language code (he/en/mixed)
            sentiment_score: Sentiment score -1.0 to 1.0 (optional)
            confidence: ASR confidence 0.0-1.0 (optional)

        Returns:
            int: ID of inserted segment
        """
        # Compute word count if not provided
        if word_count is None:
            word_count = len(text.split())

        cursor = conn.execute(
            """
            INSERT INTO segments (
                episode_id, speaker_id, start_time, end_time, text,
                word_count, language, sentiment_score, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                speaker_id,
                start_time,
                end_time,
                text,
                word_count,
                language,
                sentiment_score,
                confidence,
            ),
        )
        return cursor.lastrowid

    def get_segments_by_episode(
        self, conn: sqlite3.Connection, episode_id: int
    ) -> List[sqlite3.Row]:
        """
        Retrieve all segments for an episode, ordered by start time.

        Args:
            conn: Database connection
            episode_id: Episode ID

        Returns:
            List of segment rows
        """
        cursor = conn.execute(
            "SELECT * FROM segments WHERE episode_id = ? ORDER BY start_time",
            (episode_id,),
        )
        return cursor.fetchall()

    def insert_entity(
        self,
        conn: sqlite3.Connection,
        canonical_name: str,
        entity_type: str,
        name_localized: Optional[str] = None,
        external_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Insert or get an entity record.

        Args:
            conn: Database connection
            canonical_name: Canonical entity name
            entity_type: Entity type (player/club/competition/manager/venue/event/other)
            name_localized: Localized display name (optional)
            external_id: External ID like Wikidata QID (optional)
            metadata: Additional metadata as dict (optional)

        Returns:
            int: ID of entity (existing or newly inserted)
        """
        # Try to get existing entity
        cursor = conn.execute(
            "SELECT id FROM entities WHERE canonical_name = ? AND entity_type = ?",
            (canonical_name, entity_type),
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        # Insert new entity
        metadata_json = json.dumps(metadata) if metadata else None
        cursor = conn.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, name_localized, external_id, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (canonical_name, entity_type, name_localized, external_id, metadata_json),
        )
        return cursor.lastrowid

    def insert_entity_mention(
        self,
        conn: sqlite3.Connection,
        entity_id: int,
        segment_id: int,
        episode_id: int,
        mention_text: str,
        start_offset: Optional[int] = None,
        confidence: float = 1.0,
    ) -> int:
        """
        Insert an entity mention.

        Args:
            conn: Database connection
            entity_id: Entity ID
            segment_id: Segment ID
            episode_id: Episode ID
            mention_text: The actual text used in speech
            start_offset: Character offset in segment (optional)
            confidence: Confidence score 0.0-1.0

        Returns:
            int: ID of inserted mention
        """
        cursor = conn.execute(
            """
            INSERT INTO entity_mentions (
                entity_id, segment_id, episode_id, mention_text, start_offset, confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (entity_id, segment_id, episode_id, mention_text, start_offset, confidence),
        )
        return cursor.lastrowid

    def insert_metric(
        self,
        conn: sqlite3.Connection,
        episode_id: int,
        metric_name: str,
        metric_value: float,
        speaker_id: Optional[int] = None,
        metric_unit: Optional[str] = None,
    ) -> int:
        """
        Insert or update a metric.

        Uses INSERT OR REPLACE to handle the unique constraint on
        (episode_id, speaker_id, metric_name).

        Args:
            conn: Database connection
            episode_id: Episode ID
            metric_name: Metric name (e.g., 'speaking_pace_wpm', 'filler_rate_pct')
            metric_value: Numeric metric value
            speaker_id: Speaker ID for per-speaker metrics (optional)
            metric_unit: Unit of measurement (optional)

        Returns:
            int: ID of inserted/updated metric
        """
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO metrics (episode_id, speaker_id, metric_name, metric_value, metric_unit)
            VALUES (?, ?, ?, ?, ?)
            """,
            (episode_id, speaker_id, metric_name, metric_value, metric_unit),
        )
        return cursor.lastrowid

    def get_metrics_by_episode(
        self, conn: sqlite3.Connection, episode_id: int
    ) -> List[sqlite3.Row]:
        """
        Retrieve all metrics for an episode.

        Args:
            conn: Database connection
            episode_id: Episode ID

        Returns:
            List of metric rows
        """
        cursor = conn.execute(
            "SELECT * FROM metrics WHERE episode_id = ? ORDER BY metric_name",
            (episode_id,),
        )
        return cursor.fetchall()

    def insert_filler_word(
        self,
        conn: sqlite3.Connection,
        segment_id: int,
        episode_id: int,
        filler_text: str,
        speaker_id: Optional[int] = None,
        position_offset: Optional[int] = None,
    ) -> int:
        """
        Insert a filler word occurrence.

        Args:
            conn: Database connection
            segment_id: Segment ID
            episode_id: Episode ID
            filler_text: The filler word (e.g., "um", "like")
            speaker_id: Speaker ID (optional)
            position_offset: Character offset in segment (optional)

        Returns:
            int: ID of inserted filler word
        """
        cursor = conn.execute(
            """
            INSERT INTO filler_words (segment_id, episode_id, speaker_id, filler_text, position_offset)
            VALUES (?, ?, ?, ?, ?)
            """,
            (segment_id, episode_id, speaker_id, filler_text, position_offset),
        )
        return cursor.lastrowid

    def insert_silence_event(
        self,
        conn: sqlite3.Connection,
        episode_id: int,
        start_time: float,
        end_time: float,
        duration: float,
        event_type: str = "dead_air",
        preceding_speaker_id: Optional[int] = None,
        following_speaker_id: Optional[int] = None,
    ) -> int:
        """
        Insert a silence event.

        Args:
            conn: Database connection
            episode_id: Episode ID
            start_time: Start time in seconds
            end_time: End time in seconds
            duration: Duration in seconds
            event_type: Event type (dead_air/long_pause/technical)
            preceding_speaker_id: Speaker before silence (optional)
            following_speaker_id: Speaker after silence (optional)

        Returns:
            int: ID of inserted silence event
        """
        cursor = conn.execute(
            """
            INSERT INTO silence_events (
                episode_id, start_time, end_time, duration, event_type,
                preceding_speaker_id, following_speaker_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                start_time,
                end_time,
                duration,
                event_type,
                preceding_speaker_id,
                following_speaker_id,
            ),
        )
        return cursor.lastrowid

    def insert_coaching_note(
        self,
        conn: sqlite3.Connection,
        episode_id: int,
        speaker_id: int,
        strengths: List[str],
        improvements: List[str],
        trends: Optional[Dict[str, Any]] = None,
        generated_by: str = "gpt-4o-mini",
    ) -> int:
        """
        Insert or update a coaching note.

        Uses INSERT OR REPLACE to handle the unique constraint on
        (episode_id, speaker_id).

        Args:
            conn: Database connection
            episode_id: Episode ID
            speaker_id: Speaker ID
            strengths: List of strength observations
            improvements: List of improvement suggestions
            trends: Trend observations as dict (optional)
            generated_by: Model used for generation

        Returns:
            int: ID of inserted/updated coaching note
        """
        strengths_json = json.dumps(strengths)
        improvements_json = json.dumps(improvements)
        trends_json = json.dumps(trends) if trends else None

        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO coaching_notes (
                episode_id, speaker_id, strengths, improvements, trends, generated_by
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                speaker_id,
                strengths_json,
                improvements_json,
                trends_json,
                generated_by,
            ),
        )
        return cursor.lastrowid

    def get_all_episodes(
        self, conn: sqlite3.Connection, limit: Optional[int] = None
    ) -> List[sqlite3.Row]:
        """
        Retrieve all episodes, ordered by publication date (newest first).

        Args:
            conn: Database connection
            limit: Maximum number of episodes to return (optional)

        Returns:
            List of episode rows
        """
        query = "SELECT * FROM episodes ORDER BY pub_date DESC"
        if limit:
            query += f" LIMIT {limit}"
        cursor = conn.execute(query)
        return cursor.fetchall()

    def get_entity_mention_counts(
        self, conn: sqlite3.Connection, entity_type: Optional[str] = None, limit: int = 50
    ) -> List[Tuple[str, str, int]]:
        """
        Get entity mention counts, optionally filtered by entity type.

        Args:
            conn: Database connection
            entity_type: Filter by entity type (optional)
            limit: Maximum number of results

        Returns:
            List of (canonical_name, entity_type, mention_count) tuples
        """
        if entity_type:
            query = """
                SELECT e.canonical_name, e.entity_type, COUNT(em.id) as mention_count
                FROM entities e
                JOIN entity_mentions em ON em.entity_id = e.id
                WHERE e.entity_type = ?
                GROUP BY e.id
                ORDER BY mention_count DESC
                LIMIT ?
            """
            cursor = conn.execute(query, (entity_type, limit))
        else:
            query = """
                SELECT e.canonical_name, e.entity_type, COUNT(em.id) as mention_count
                FROM entities e
                JOIN entity_mentions em ON em.entity_id = e.id
                GROUP BY e.id
                ORDER BY mention_count DESC
                LIMIT ?
            """
            cursor = conn.execute(query, (limit,))

        return [(row[0], row[1], row[2]) for row in cursor.fetchall()]

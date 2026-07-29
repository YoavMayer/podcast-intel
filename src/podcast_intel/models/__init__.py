"""
Data models and database management.

Provides SQLite schema management, Pydantic data models, and
database access layers for episodes, segments, entities, and metrics.
"""

from podcast_intel.models.database import Database
from podcast_intel.models.entities import Entity, Episode, Segment, Speaker
from podcast_intel.models.schema import SCHEMA_SQL, create_all_tables, get_table_names

__all__ = [
    "Database",
    "create_all_tables",
    "get_table_names",
    "SCHEMA_SQL",
    "Episode",
    "Segment",
    "Entity",
    "Speaker",
]

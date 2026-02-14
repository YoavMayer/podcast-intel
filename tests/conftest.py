"""
Shared test fixtures.

Provides pytest fixtures for common test resources including:
- Mock configuration
- Temporary database
- Sample episodes and segments
- Mock transcription data
"""

import pytest
from pathlib import Path
import tempfile
from typing import List, Dict, Any

from podcast_intel.config import Config
from podcast_intel.models.database import Database
from podcast_intel.models.entities import Episode, Segment, Speaker


@pytest.fixture
def temp_dir():
    """
    Create temporary directory for test files.

    Yields:
        Path: Temporary directory path
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """
    Create test configuration with temporary paths.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Config: Test configuration
    """
    # Implementation placeholder
    pass


@pytest.fixture
def test_db(temp_dir: Path) -> Database:
    """
    Create test database with schema.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Database: Initialized test database
    """
    # Implementation placeholder
    pass


@pytest.fixture
def sample_episodes() -> List[Episode]:
    """
    Create sample episode data for testing.

    Returns:
        List of Episode objects
    """
    # Implementation placeholder
    pass


@pytest.fixture
def sample_segments() -> List[Segment]:
    """
    Create sample segment data for testing.

    Returns:
        List of Segment objects
    """
    # Implementation placeholder
    pass


@pytest.fixture
def sample_speakers() -> List[Speaker]:
    """
    Create sample speaker data for testing.

    Returns:
        List of Speaker objects
    """
    # Implementation placeholder
    pass


@pytest.fixture
def mock_transcript() -> Dict[str, Any]:
    """
    Create mock transcript data for testing.

    Returns:
        Dictionary with segments and metadata
    """
    # Implementation placeholder
    pass

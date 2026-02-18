"""
Mock ingestion generating realistic test data.

Provides mock episode data generation for testing and development without
requiring actual RSS feed access or audio downloads. Generates realistic
episode titles, descriptions, and metadata.

The mock data uses English by default but the framework supports any language
via the preset/config system.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import random

from ..models.database import Database
from ..config import get_config


def generate_mock_episodes(db: Database, count: int = 5) -> List[int]:
    """
    Generate and insert mock episode metadata for testing.

    Creates realistic mock episodes with titles about podcast topics,
    complete with descriptions, speakers, and metadata. Inserts episodes
    and speakers into the database.

    Args:
        db: Database instance for inserting mock data
        count: Number of mock episodes to generate (default: 5)

    Returns:
        List of inserted episode IDs

    Example:
        >>> db = Database(Path("data/db/test.db"))
        >>> db.initialize()
        >>> episode_ids = generate_mock_episodes(db, count=3)
        >>> print(f"Created {len(episode_ids)} mock episodes")
    """
    # Mock episode data: realistic English podcast titles
    mock_titles = [
        "Episode 201 - Season Premiere Discussion",
        "Episode 202 - Deep Dive: Industry Trends",
        "Episode 203 - Guest Interview Special",
        "Episode 204 - Weekly Roundup and Analysis",
        "Episode 205 - Listener Q&A Session",
        "Episode 206 - Year in Review",
        "Episode 207 - Behind the Scenes",
        "Episode 208 - Expert Panel Discussion",
        "Episode 209 - Breaking News Analysis",
        "Episode 210 - Community Spotlight",
    ]

    mock_descriptions = [
        "Kicking off the new season with a wide-ranging discussion. We cover the biggest stories, set expectations, and outline what listeners can look forward to.",
        "A deep dive into the latest industry trends. What is changing, who are the key players driving innovation, and what does it mean for the future?",
        "A special guest joins us to share their journey, insights, and predictions. An in-depth conversation covering career highlights and lessons learned.",
        "Our weekly roundup covering the top stories, listener feedback, and analysis of the most talked-about events of the past week.",
        "We answer your burning questions! From hot takes to thoughtful inquiries, the hosts tackle listener-submitted topics in this interactive episode.",
        "Looking back at the year that was. Highlights, surprises, disappointments, and the moments that defined the past twelve months.",
        "A behind-the-scenes look at how the show is made. Production secrets, bloopers, and a candid conversation about the creative process.",
        "An expert panel weighs in on the most pressing topics of the moment. Multiple perspectives, lively debate, and actionable takeaways.",
        "Breaking down the biggest news story of the week. What happened, why it matters, and what comes next. Our hosts provide context and analysis.",
        "Shining a spotlight on our community. Listener stories, fan contributions, and a celebration of the people who make this show possible.",
    ]

    # Base date for episodes (recent dates)
    base_date = datetime.now() - timedelta(days=30)

    episode_ids = []

    with db.get_connection() as conn:
        # First, insert speakers
        speaker_names = [
            ("Alex", True),
            ("Jordan", True),
            ("Sam", False),
        ]

        speaker_ids = {}
        for english_name, is_host in speaker_names:
            speaker_id = db.insert_speaker(
                conn,
                name=english_name,
                is_host=is_host
            )
            speaker_ids[english_name] = speaker_id
            print(f"Inserted speaker: {english_name} - ID: {speaker_id}")

        # Generate episodes
        for i in range(min(count, len(mock_titles))):
            # Create realistic episode data
            title = mock_titles[i]
            description = mock_descriptions[i]

            # Publication date: spread episodes over last 30 days
            pub_date = base_date + timedelta(days=i * 6)
            pub_date_str = pub_date.strftime("%Y-%m-%dT%H:%M:%S+00:00")

            # Duration between 40-90 minutes (2400-5400 seconds)
            duration_seconds = random.randint(2400, 5400)

            # Generate GUID
            guid = f"4f8c8e1e-{1000+i:04x}-4a2b-8c3d-{random.randint(100000000000, 999999999999):012x}"

            # Audio URL
            audio_url = f"https://example.com/podcast/play/{random.randint(10000000, 99999999)}/audio-{guid}.mp3"

            # File size: roughly 1MB per minute
            file_size_bytes = duration_seconds * 1024 * 1024 // 60

            # Insert episode
            episode_id = db.insert_episode(
                conn,
                guid=guid,
                title=title,
                description=description,
                pub_date=pub_date_str,
                audio_url=audio_url,
                duration_seconds=duration_seconds,
                file_size_bytes=file_size_bytes,
                episode_type="full"
            )

            episode_ids.append(episode_id)
            print(f"Inserted episode {i+1}/{count}: {title} - ID: {episode_id}")

    return episode_ids


def generate_mock_audio_path(episode_id: str) -> str:
    """
    Generate mock audio file path for testing.

    Args:
        episode_id: Episode identifier

    Returns:
        Mock file path string

    Example:
        >>> path = generate_mock_audio_path("ep-001")
        >>> print(path)
        data/audio/ep-001.mp3
    """
    return f"data/audio/{episode_id}.mp3"


def create_test_episode(
    title: str,
    duration_seconds: int,
    pub_date: datetime
) -> Dict[str, Any]:
    """
    Create a single test episode with specified parameters.

    Args:
        title: Episode title
        duration_seconds: Episode duration
        pub_date: Publication date

    Returns:
        Episode metadata dictionary

    Example:
        >>> episode = create_test_episode(
        ...     "Episode 100 - Year in Review",
        ...     3600,
        ...     datetime(2024, 12, 15)
        ... )
        >>> print(episode["title"])
    """
    guid = f"test-{random.randint(100000, 999999)}"

    return {
        "guid": guid,
        "title": title,
        "description": f"Description of {title}",
        "pub_date": pub_date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "audio_url": f"https://example.com/audio/{guid}.mp3",
        "duration_seconds": duration_seconds,
        "file_size_bytes": duration_seconds * 1024 * 1024 // 60,
        "episode_type": "full"
    }


def main():
    """
    Main entry point for mock ingestion script.

    Initializes database and generates mock episodes.
    Can be run with: python -m podcast_intel.ingestion.mock_ingest
    """
    print("=== Mock Ingestion System ===")
    print()

    # Get configuration
    config = get_config()
    print(f"Database path: {config.db_path}")
    print()

    # Initialize database
    db = Database(config.db_path)
    db.initialize()
    print("Database initialized successfully")
    print()

    # Generate mock episodes
    print("Generating mock episodes...")
    episode_ids = generate_mock_episodes(db, count=5)
    print()

    print(f"Successfully created {len(episode_ids)} mock episodes")
    print(f"Episode IDs: {episode_ids}")
    print()

    # Show summary
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM episodes")
        total_episodes = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM speakers")
        total_speakers = cursor.fetchone()[0]

        print("Database Summary:")
        print(f"  Total episodes: {total_episodes}")
        print(f"  Total speakers: {total_speakers}")

        # Show latest episodes
        print()
        print("Latest episodes:")
        cursor = conn.execute(
            "SELECT id, title, duration_seconds, pub_date FROM episodes ORDER BY pub_date DESC LIMIT 5"
        )
        for row in cursor.fetchall():
            mins = row[2] // 60 if row[2] else 0
            print(f"  [{row[0]}] {row[1]} ({mins} min) - {row[3][:10]}")


if __name__ == "__main__":
    main()

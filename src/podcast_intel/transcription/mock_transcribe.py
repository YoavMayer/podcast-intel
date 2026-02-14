"""
Mock transcription with realistic English segments.

Test implementation generating realistic transcript segments
with speaker labels, timestamps, and word-level data. Useful for
development and testing without GPU requirements.

Generates podcast content with natural conversation patterns,
speaker-specific filler word patterns, and realistic pacing/silence
simulation. The default language is English but Hebrew and other
languages can be configured via presets.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random

from podcast_intel.transcription.transcribe import (
    TranscriptionInterface,
    TranscriptionResult
)
from podcast_intel.models.database import Database
from podcast_intel.config import get_config


# ---------------------------------------------------------------------------
# English segment templates organized by topic category.
# Curly-brace placeholders are filled at generation time.
# ---------------------------------------------------------------------------

MATCH_ANALYSIS_TEMPLATES = [
    "The match against {team} was {adjective}, the {formation} worked really well",
    "In the second half against {team} we saw significant improvement in build-up play",
    "The pressing from {team} in the first half was very aggressive, we couldn't get out",
    "At the end of the day, the clean sheet against {team} is the most important thing",
    "We saw a quick counter-attack, {team} couldn't recover from it",
    "Our set pieces against {team} were much more dangerous this time",
    "The xG from the match against {team} says we should have won by a bigger margin",
    "That was a clear penalty, the VAR didn't do its job in the match against {team}",
]

PLAYER_DISCUSSION_TEMPLATES = [
    "I think {player} played phenomenally, his {stat} was {value}",
    "{player} gave everything on the pitch, you can see his progress",
    "The hat-trick from {player} was a historic moment, three world-class goals",
    "You have to give credit to {player}, he was absolutely outstanding",
    "The performance from {player} reminds me of his best period",
    "{player} was the most dangerous winger in the league this month",
    "The stats for {player} speak for themselves, {stat} of {value}",
    "The question is whether {player} can maintain this level all season",
]

TACTICAL_ANALYSIS_TEMPLATES = [
    "The {formation} allows the midfield to control the game effectively",
    "The problem with the {formation} is that the defender is left exposed",
    "In build-up play he asks the goalkeeper to start, it's risky but effective",
    "The offside trap only works when the entire line is synchronized",
    "The idea behind the short corner kick is to create numerical superiority",
    "The pressing is based on specific triggers when the ball reaches the {position}",
    "The transition from {formation} to 4-2-3-1 shows the tactical flexibility",
    "The striker needs to drop deeper in build-up to create an overload in midfield",
]

TRANSFER_TALK_TEMPLATES = [
    "I heard they want to sign {player} from {club}, that could be an excellent signing",
    "If {player} really leaves for {club}, it would be a disaster for us",
    "This transfer window must bring a new {position}, it's our weakest area",
    "The transfer fee for {player} was excessive, but that's normal money in the league now",
    "Rumors about {player} from {club}, I believe it could work in our system",
    "{club} want {value} million for {player}, and that's just not a reasonable price",
]

GENERAL_OPINION_TEMPLATES = [
    "I disagree, there's a problem with the {concept} of the team",
    "Look, in today's game you have to be in the top four to compete",
    "The table position doesn't reflect our performances, the xG tells a different story",
    "After the cup, we can focus on the league, and that should be the priority",
    "The cup is an opportunity to give minutes to the youth players",
    "Every season starts with hope, this year things are looking much better",
    "The crowd at the stadium was phenomenal, the atmosphere made the difference",
    "You need patience with this project, it takes time to build a team",
]

# All templates combined for random selection
ALL_TEMPLATES = (
    MATCH_ANALYSIS_TEMPLATES
    + PLAYER_DISCUSSION_TEMPLATES
    + TACTICAL_ANALYSIS_TEMPLATES
    + TRANSFER_TALK_TEMPLATES
    + GENERAL_OPINION_TEMPLATES
)

# ---------------------------------------------------------------------------
# Generic entities for template filling
# ---------------------------------------------------------------------------

PLAYERS = [
    "Martinez",
    "Fernandez",
    "Johnson",
    "Williams",
    "Anderson",
]

CLUBS = [
    "Arsenal",
    "Chelsea",
    "Manchester City",
    "Liverpool",
]

FORMATIONS = [
    "3-4-3",
    "4-2-3-1",
    "4-3-3",
    "3-5-2",
]

POSITIONS = [
    "striker",
    "midfielder",
    "defender",
    "winger",
    "goalkeeper",
]

ADJECTIVES = [
    "incredible",
    "amazing",
    "disappointing",
    "exciting",
    "tough",
    "perfect",
    "tense",
    "dramatic",
    "poor",
    "impressive",
]

STATS = [
    "xG",
    "pass completion",
    "pressing index",
    "distance covered",
    "tackles won",
    "chances created",
    "clean sheet",
    "assist record",
]

STAT_VALUES = [
    "above average",
    "highest in the league",
    "outstanding",
    "0.85",
    "92%",
    "12.3 km",
    "very impressive",
    "personal best",
]

# ---------------------------------------------------------------------------
# Speaker-specific filler word distributions
# ---------------------------------------------------------------------------

FILLER_WORDS_ALL = [
    "um", "uh", "like", "you know", "I mean", "basically",
    "right?", "so", "well", "okay", "actually",
]

# Speaker filler word weights: (filler, weight)
# Alex (host) - moderate fillers, favours "like"
ALEX_FILLER_WEIGHTS = {
    "like": 5, "um": 3, "so": 3, "well": 2, "right?": 2,
    "basically": 1, "okay": 1, "you know": 1, "I mean": 1,
    "uh": 1, "actually": 1,
}

# Jordan (host) - more fillers, favours "you know" and "I mean"
JORDAN_FILLER_WEIGHTS = {
    "you know": 5, "I mean": 4, "um": 3, "so": 3, "like": 3,
    "uh": 2, "actually": 2, "well": 2, "right?": 2,
    "basically": 1, "okay": 1,
}

# Sam (guest) - fewer fillers, more measured, uses "actually" and "um"
SAM_FILLER_WEIGHTS = {
    "actually": 5, "um": 4, "so": 2, "right?": 2, "basically": 2,
    "well": 1, "like": 1, "I mean": 1, "okay": 1,
    "uh": 1, "you know": 1,
}

# Speaker profile config
SPEAKER_PROFILES = {
    "Alex": {
        "is_host": True,
        "filler_weights": ALEX_FILLER_WEIGHTS,
        "filler_rate": 0.35,
        "pace_factor": 1.0,
        "style": "host_moderate",
    },
    "Jordan": {
        "is_host": True,
        "filler_weights": JORDAN_FILLER_WEIGHTS,
        "filler_rate": 0.55,
        "pace_factor": 0.85,
        "style": "host_passionate",
    },
    "Sam": {
        "is_host": False,
        "filler_weights": SAM_FILLER_WEIGHTS,
        "filler_rate": 0.25,
        "pace_factor": 1.2,
        "style": "guest_analytical",
    },
}

# ---------------------------------------------------------------------------
# Host transition / question templates
# ---------------------------------------------------------------------------

HOST_TRANSITION_TEMPLATES = [
    "So let's move on to the next topic, what do you guys think about the {concept}?",
    "Wait, before we continue, {name} what do you think about this?",
    "Okay, question for {name}, how do you see the situation with {concept}?",
    "Let's talk about the next match against {team}, what are the expectations?",
    "Now, the topic everyone's talking about, the {concept}",
    "Before we wrap up, we have to talk about {player}, what's going on with him?",
]

# Jordan's passionate reactions
JORDAN_PASSIONATE_TEMPLATES = [
    "No way! {player} should have received the ball there!",
    "Listen, I'm telling you, this {concept} is going to change everything!",
    "I can't believe the VAR didn't give that penalty!",
    "Mate, {player} is just on another level, no question about it!",
    "That's insane! {team} couldn't stop our counter-attack!",
    "I'm hearing the rumors about {player} and I'm getting excited, honestly!",
]

# Sam's analytical style
SAM_ANALYTICAL_TEMPLATES = [
    "If you look at the data, the xG for {team} shows an interesting picture",
    "Tactically, the {formation} gives an advantage in midfield possession",
    "The data shows that {player} is top 5 in the league for {stat}",
    "You need to look at this analytically, the pressing metrics show {team} dropped off",
    "From a statistical standpoint, the {stat} for {player} is {value}",
    "If you run a regression analysis on the performances, there's a clear positive trend",
]

# English terms common in football analysis
ENGLISH_TERMS = [
    "Premier League", "top 4", "VAR", "pressing", "build-up play",
    "set pieces", "counter-attack", "xG", "clean sheet", "hat-trick",
    "penalty", "corner kick", "offside", "substitution", "formation",
    "3-4-3", "4-2-3-1", "midfield", "striker", "goalkeeper",
    "defender", "winger",
]


def _weighted_choice(weights: Dict[str, int]) -> str:
    """Select a random item from a dict of {item: weight} pairs."""
    items = list(weights.keys())
    item_weights = list(weights.values())
    return random.choices(items, weights=item_weights, k=1)[0]


def _fill_template(template: str) -> str:
    """Fill placeholders in a template string with random entities."""
    result = template
    if "{team}" in result:
        result = result.replace("{team}", random.choice(CLUBS))
    if "{player}" in result:
        result = result.replace("{player}", random.choice(PLAYERS))
    if "{club}" in result:
        result = result.replace("{club}", random.choice(CLUBS))
    if "{formation}" in result:
        result = result.replace("{formation}", random.choice(FORMATIONS))
    if "{adjective}" in result:
        result = result.replace("{adjective}", random.choice(ADJECTIVES))
    if "{stat}" in result:
        result = result.replace("{stat}", random.choice(STATS))
    if "{value}" in result:
        result = result.replace("{value}", random.choice(STAT_VALUES))
    if "{concept}" in result:
        concepts = [
            "pressing", "build-up play", "transfer window", "top 4 race",
            "VAR decisions", "set pieces", "youth development", "midfield balance",
            "defensive structure", "attacking transitions",
        ]
        result = result.replace("{concept}", random.choice(concepts))
    if "{position}" in result:
        result = result.replace("{position}", random.choice(POSITIONS))
    if "{name}" in result:
        names = list(SPEAKER_PROFILES.keys())
        result = result.replace("{name}", random.choice(names))
    return result


def _detect_language(text: str) -> str:
    """
    Determine language label for a segment based on script analysis.

    Returns:
        'en' for English text (default for this mock).
    """
    has_hebrew = False
    has_latin = False
    for ch in text:
        if "\u0590" <= ch <= "\u05FF":
            has_hebrew = True
        elif ch.isalpha() and ch.isascii():
            has_latin = True
        if has_hebrew and has_latin:
            return "mixed"
    if has_hebrew:
        return "he"
    if has_latin:
        return "en"
    return "en"


def _count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


class MockTranscriber(TranscriptionInterface):
    """
    Mock transcriber generating realistic test data.

    Generates transcript segments with:
    - Multiple speakers (2-3 per episode)
    - Realistic timestamps and durations
    - Natural filler words with speaker-specific patterns
    - Football/podcast terminology

    Example:
        >>> transcriber = MockTranscriber(num_speakers=3)
        >>> result = transcriber.transcribe(Path("audio.mp3"))
        >>> print(result.segments[0]["text"])
    """

    def __init__(self, num_speakers: int = 3, segments_per_minute: int = 4):
        """
        Initialize mock transcriber.

        Args:
            num_speakers: Number of speakers to simulate (2 or 3)
            segments_per_minute: Average segments per minute of audio
        """
        self.num_speakers = min(num_speakers, 3)
        self.segments_per_minute = segments_per_minute

        # Build speaker list from profiles
        speaker_names = list(SPEAKER_PROFILES.keys())[:self.num_speakers]
        self.speakers = {
            i: SPEAKER_PROFILES[name]
            for i, name in enumerate(speaker_names)
        }
        self.speaker_names = speaker_names

    def transcribe(
        self,
        audio_path: Path,
        language: str = "en",
        diarize: bool = True
    ) -> TranscriptionResult:
        """
        Generate mock transcription with segments.

        Produces 15-25 segments representing roughly 60 minutes of podcast
        content, with varied speaker turns, filler words, and realistic
        timestamps.

        Args:
            audio_path: Path to audio file (used for duration estimation)
            language: Language code (default 'en')
            diarize: Include speaker diarization labels

        Returns:
            TranscriptionResult with mock segments and diarization data
        """
        # Determine total duration: use filename hint or default ~60 min
        total_duration = 3600.0  # default 60 minutes
        num_segments = random.randint(15, 25)

        segments = []
        diarization = []
        current_time = random.uniform(1.0, 5.0)  # small intro offset

        # Generate turn sequence ensuring natural conversation flow
        speaker_sequence = self._generate_speaker_sequence(num_segments)

        for seg_idx in range(num_segments):
            speaker_idx = speaker_sequence[seg_idx]
            profile = self.speakers[speaker_idx]
            speaker_name = self.speaker_names[speaker_idx]

            # Generate segment text based on speaker style
            text = self.generate_segment(speaker_idx)

            # Compute realistic duration based on word count and pace
            word_count = _count_words(text)
            # English speakers ~140-180 words per minute
            base_wpm = random.uniform(140.0, 180.0)
            duration = (word_count / base_wpm) * 60.0 * profile["pace_factor"]
            # Clamp duration to 5-45 seconds
            duration = max(5.0, min(45.0, duration))

            start_time = round(current_time, 2)
            end_time = round(current_time + duration, 2)

            # Determine language
            seg_language = _detect_language(text)

            # Confidence score
            confidence = round(random.uniform(0.85, 0.98), 3)

            # Sentiment: slight positive bias for discussion
            sentiment = round(random.uniform(-0.3, 0.8), 3)

            segment = {
                "id": seg_idx,
                "start": start_time,
                "end": end_time,
                "text": text,
                "speaker": f"SPEAKER_{speaker_idx:02d}" if diarize else None,
                "speaker_name": speaker_name if diarize else None,
                "words": _count_words(text),
                "language": seg_language,
                "confidence": confidence,
                "sentiment": sentiment,
            }
            segments.append(segment)

            if diarize:
                diarization.append({
                    "speaker": f"SPEAKER_{speaker_idx:02d}",
                    "speaker_name": speaker_name,
                    "start": start_time,
                    "end": end_time,
                })

            # Advance time with a small inter-segment gap
            gap = random.uniform(0.3, 2.5)
            current_time = end_time + gap

        # Adjust total_duration to match generated content
        total_duration = current_time + random.uniform(1.0, 5.0)

        return TranscriptionResult(
            segments=segments,
            language=language,
            duration=round(total_duration, 2),
            diarization=diarization,
        )

    def get_word_timestamps(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Generate mock word-level timestamps.

        Creates a full transcription and then breaks each segment into
        word-level entries with realistic sub-second offsets.

        Args:
            audio_path: Path to audio file

        Returns:
            List of dicts with 'word', 'start', 'end', 'confidence', 'speaker'
        """
        result = self.transcribe(audio_path)
        word_timestamps = []

        for segment in result.segments:
            words = segment["text"].split()
            if not words:
                continue

            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_duration = seg_end - seg_start

            # Distribute words evenly across segment duration
            word_duration = seg_duration / len(words)

            for w_idx, word in enumerate(words):
                w_start = round(seg_start + w_idx * word_duration, 3)
                w_end = round(w_start + word_duration * random.uniform(0.6, 0.95), 3)
                # Per-word confidence with slight variation
                w_confidence = round(
                    min(1.0, max(0.5, segment["confidence"] + random.uniform(-0.1, 0.05))),
                    3
                )

                word_timestamps.append({
                    "word": word,
                    "start": w_start,
                    "end": w_end,
                    "confidence": w_confidence,
                    "speaker": segment.get("speaker"),
                })

        return word_timestamps

    def generate_segment(self, speaker_id: int) -> str:
        """
        Generate realistic text segment for a speaker.

        Selects templates appropriate to the speaker's style, fills
        placeholders with entities, and injects filler words
        according to the speaker's filler distribution.

        Args:
            speaker_id: Speaker index (0, 1, or 2)

        Returns:
            Text segment string
        """
        # Alias for backwards compatibility
        return self.generate_hebrew_segment(speaker_id)

    def generate_hebrew_segment(self, speaker_id: int) -> str:
        """
        Generate realistic text segment with natural speech patterns.

        Selects templates appropriate to the speaker's style, fills
        placeholders with entities, and injects filler words
        according to the speaker's filler distribution.

        Args:
            speaker_id: Speaker index (0, 1, or 2)

        Returns:
            Text with natural speech patterns
        """
        profile = self.speakers[speaker_id]

        # Choose template pool based on speaker style
        if profile["style"] == "host_moderate":
            pool = (
                HOST_TRANSITION_TEMPLATES
                + MATCH_ANALYSIS_TEMPLATES
                + GENERAL_OPINION_TEMPLATES
                + PLAYER_DISCUSSION_TEMPLATES
            )
        elif profile["style"] == "host_passionate":
            pool = (
                JORDAN_PASSIONATE_TEMPLATES
                + MATCH_ANALYSIS_TEMPLATES
                + TRANSFER_TALK_TEMPLATES
                + PLAYER_DISCUSSION_TEMPLATES
            )
        else:
            pool = (
                SAM_ANALYTICAL_TEMPLATES
                + TACTICAL_ANALYSIS_TEMPLATES
                + PLAYER_DISCUSSION_TEMPLATES
                + GENERAL_OPINION_TEMPLATES
            )

        template = random.choice(pool)
        text = _fill_template(template)

        # Possibly chain a second sentence for longer segments
        if random.random() < 0.45:
            second_template = random.choice(ALL_TEMPLATES)
            second_text = _fill_template(second_template)
            text = text + ". " + second_text

        # Inject filler words according to speaker pattern
        text = self._inject_fillers(text, speaker_id)

        return text

    def _inject_fillers(self, text: str, speaker_id: int) -> str:
        """
        Inject filler words into text based on speaker filler patterns.

        Fillers are inserted at natural break points (beginning, before
        commas, between sentences) to simulate real speech disfluencies.

        Args:
            text: The base segment text
            speaker_id: Speaker index for profile lookup

        Returns:
            Text with filler words naturally inserted
        """
        profile = self.speakers[speaker_id]
        filler_weights = profile["filler_weights"]
        filler_rate = profile["filler_rate"]

        # Possibly add a leading filler
        if random.random() < filler_rate:
            filler = _weighted_choice(filler_weights)
            text = filler + ", " + text

        # Possibly add a mid-sentence filler at a comma or period boundary
        if ", " in text and random.random() < filler_rate * 0.6:
            parts = text.split(", ", 1)
            if len(parts) == 2:
                filler = _weighted_choice(filler_weights)
                text = parts[0] + ", " + filler + ", " + parts[1]

        # Possibly add a trailing confirmation filler
        if random.random() < filler_rate * 0.3:
            trailing_fillers = ["right?", "you know", "basically"]
            text = text + ", " + random.choice(trailing_fillers)

        return text

    def _generate_speaker_sequence(self, num_segments: int) -> List[int]:
        """
        Generate a natural speaker turn sequence for a conversation.

        Ensures that the host (speaker 0) opens, speakers alternate
        with some natural clustering (a speaker may talk 1-3 turns in a row),
        and the host periodically reclaims the floor.

        Args:
            num_segments: Number of segments to generate turns for

        Returns:
            List of speaker indices, one per segment
        """
        speaker_ids = list(self.speakers.keys())
        sequence = []

        # Host opens
        current_speaker = 0
        sequence.append(current_speaker)

        for i in range(1, num_segments):
            # Host re-enters periodically to guide the conversation
            if i % 5 == 0:
                current_speaker = 0
            elif random.random() < 0.6:
                # Switch to a different speaker
                other_speakers = [s for s in speaker_ids if s != current_speaker]
                current_speaker = random.choice(other_speakers)
            # else: same speaker continues (natural multi-turn speaking)

            sequence.append(current_speaker)

        return sequence


# ---------------------------------------------------------------------------
# Database-integrated mock transcription function
# ---------------------------------------------------------------------------

def _find_filler_words_in_text(
    text: str,
) -> List[Tuple[str, int]]:
    """
    Scan text for known filler words and return their positions.

    Args:
        text: Segment text to scan

    Returns:
        List of (filler_text, character_offset) tuples
    """
    found = []
    for filler in FILLER_WORDS_ALL:
        start = 0
        while True:
            idx = text.find(filler, start)
            if idx == -1:
                break
            # Verify it appears as a standalone token (preceded/followed by
            # space, comma, or string boundary)
            before_ok = (idx == 0) or (text[idx - 1] in " ,.")
            after_end = idx + len(filler)
            after_ok = (after_end >= len(text)) or (text[after_end] in " ,.")
            if before_ok and after_ok:
                found.append((filler, idx))
            start = idx + len(filler)
    return found


def _get_speaker_ids(db: Database) -> Dict[str, int]:
    """
    Retrieve speaker name-to-id mapping from the database.

    Args:
        db: Database instance

    Returns:
        Dict mapping speaker English name to database ID
    """
    speaker_ids = {}
    with db.get_connection() as conn:
        for name in SPEAKER_PROFILES:
            row = db.get_speaker_by_name(conn, name)
            if row:
                speaker_ids[name] = row["id"]
    return speaker_ids


def generate_mock_transcription(db: Database, episode_id: int) -> int:
    """
    Generate a complete mock transcription for an episode and store it.

    Creates 15-25 realistic podcast segments with:
    - Speaker-attributed text
    - Filler word detection and storage
    - Silence/dead-air event generation
    - Transcription status update

    Args:
        db: Database instance for storing generated data
        episode_id: ID of the episode to transcribe

    Returns:
        Number of segments created
    """
    # Retrieve episode metadata for duration
    with db.get_connection() as conn:
        episode = db.get_episode_by_id(conn, episode_id)
    if episode is None:
        raise ValueError(f"Episode {episode_id} not found in database")

    episode_duration = episode["duration_seconds"] or 3600

    # Get speaker IDs from database
    speaker_ids = _get_speaker_ids(db)
    if not speaker_ids:
        raise ValueError("No speakers found in database. Run mock_ingest first.")

    # Create the transcriber and generate transcript
    transcriber = MockTranscriber(num_speakers=len(speaker_ids))
    result = transcriber.transcribe(
        audio_path=Path(f"data/audio/episode_{episode_id}.mp3"),
        language="en",
        diarize=True,
    )

    # Map speaker labels to database IDs
    speaker_name_to_id = speaker_ids
    speaker_names_ordered = list(SPEAKER_PROFILES.keys())[:len(speaker_ids)]

    segment_count = 0
    total_fillers = 0
    segment_db_ids = []

    with db.get_connection() as conn:
        # Mark episode as processing
        db.update_episode_status(conn, episode_id, "processing")

        # Scale timestamps to match episode duration
        generated_duration = result.duration
        time_scale = episode_duration / generated_duration if generated_duration > 0 else 1.0

        for seg in result.segments:
            # Scale times to actual episode duration
            start_time = round(seg["start"] * time_scale, 2)
            end_time = round(seg["end"] * time_scale, 2)

            # Resolve speaker
            speaker_name = seg.get("speaker_name")
            db_speaker_id = speaker_name_to_id.get(speaker_name)

            # Determine language label
            seg_language = seg.get("language", "en")

            # Sentiment from generation
            sentiment = seg.get("sentiment", 0.0)

            # Confidence from generation
            confidence = seg.get("confidence", 0.9)

            # Insert segment
            segment_id = db.insert_segment(
                conn,
                episode_id=episode_id,
                start_time=start_time,
                end_time=end_time,
                text=seg["text"],
                speaker_id=db_speaker_id,
                word_count=seg.get("words"),
                language=seg_language,
                sentiment_score=sentiment,
                confidence=confidence,
            )
            segment_db_ids.append(segment_id)
            segment_count += 1

            # Detect and store filler words
            fillers = _find_filler_words_in_text(seg["text"])
            for filler_text, offset in fillers:
                db.insert_filler_word(
                    conn,
                    segment_id=segment_id,
                    episode_id=episode_id,
                    filler_text=filler_text,
                    speaker_id=db_speaker_id,
                    position_offset=offset,
                )
                total_fillers += 1

        # Generate silence events (2-4 per episode)
        num_silences = random.randint(2, 4)
        _generate_silence_events(
            db, conn, episode_id, episode_duration,
            result.segments, time_scale, speaker_name_to_id, num_silences
        )

        # Mark episode as completed
        db.update_episode_status(conn, episode_id, "completed")

    return segment_count


def _generate_silence_events(
    db: Database,
    conn: Any,
    episode_id: int,
    episode_duration: float,
    segments: List[Dict[str, Any]],
    time_scale: float,
    speaker_name_to_id: Dict[str, int],
    num_events: int,
) -> int:
    """
    Generate and store silence/dead-air events between segments.

    Places silence events at plausible positions in the episode timeline,
    assigning preceding and following speakers based on nearby segments.

    Args:
        db: Database instance
        conn: Active database connection
        episode_id: Episode ID
        episode_duration: Total episode duration in seconds
        segments: List of generated segment dicts
        time_scale: Scaling factor from generated to actual timestamps
        speaker_name_to_id: Speaker name to DB ID mapping
        num_events: Number of silence events to create

    Returns:
        Number of silence events created
    """
    event_types = ["dead_air", "long_pause"]
    events_created = 0

    # Find gaps between segments where silence can be placed
    candidate_gaps = []
    for i in range(len(segments) - 1):
        gap_start = segments[i]["end"] * time_scale
        gap_end = segments[i + 1]["start"] * time_scale
        if gap_end > gap_start:
            candidate_gaps.append((i, gap_start, gap_end))

    # Also add some random silence positions within the episode
    for _ in range(num_events * 2):
        pos = random.uniform(60.0, episode_duration - 60.0)
        candidate_gaps.append((-1, pos, pos))

    random.shuffle(candidate_gaps)

    for gap_idx, gap_start, gap_end in candidate_gaps:
        if events_created >= num_events:
            break

        event_type = random.choice(event_types)

        if event_type == "dead_air":
            duration = round(random.uniform(3.0, 8.0), 2)
        else:
            duration = round(random.uniform(1.5, 4.0), 2)

        silence_start = round(gap_start, 2)
        silence_end = round(silence_start + duration, 2)

        # Ensure silence fits within episode
        if silence_end > episode_duration:
            continue

        # Find preceding and following speakers
        preceding_speaker_id = None
        following_speaker_id = None

        if gap_idx >= 0 and gap_idx < len(segments):
            preceding_name = segments[gap_idx].get("speaker_name")
            preceding_speaker_id = speaker_name_to_id.get(preceding_name)
        if gap_idx >= 0 and gap_idx + 1 < len(segments):
            following_name = segments[gap_idx + 1].get("speaker_name")
            following_speaker_id = speaker_name_to_id.get(following_name)

        # For random positions without gap context, pick random speakers
        if preceding_speaker_id is None:
            speaker_ids_list = list(speaker_name_to_id.values())
            if speaker_ids_list:
                preceding_speaker_id = random.choice(speaker_ids_list)
        if following_speaker_id is None:
            speaker_ids_list = list(speaker_name_to_id.values())
            if speaker_ids_list:
                following_speaker_id = random.choice(speaker_ids_list)

        db.insert_silence_event(
            conn,
            episode_id=episode_id,
            start_time=silence_start,
            end_time=silence_end,
            duration=duration,
            event_type=event_type,
            preceding_speaker_id=preceding_speaker_id,
            following_speaker_id=following_speaker_id,
        )
        events_created += 1

    return events_created


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point for mock transcription script.

    Initializes the database, retrieves all episodes, generates mock
    transcriptions for each, and prints summary statistics.

    Can be run with: python -m podcast_intel.transcription.mock_transcribe
    """
    print("=== Mock Transcription System ===")
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

    # Get all episodes
    with db.get_connection() as conn:
        episodes = db.get_all_episodes(conn)

    if not episodes:
        print("No episodes found. Run mock_ingest first:")
        print("  python -m podcast_intel.ingestion.mock_ingest")
        return

    print(f"Found {len(episodes)} episodes to transcribe")
    print()

    # Run mock transcription for each episode
    total_segments = 0
    episode_results = []

    for episode in episodes:
        episode_id = episode["id"]
        title = episode["title"]
        status = episode["transcription_status"]

        if status == "completed":
            print(f"  [{episode_id}] {title} - already transcribed, skipping")
            continue

        print(f"  [{episode_id}] Transcribing: {title}...")
        segment_count = generate_mock_transcription(db, episode_id)
        total_segments += segment_count
        episode_results.append((episode_id, title, segment_count))
        print(f"         -> {segment_count} segments created")

    print()

    # Print summary statistics
    print("=" * 55)
    print("TRANSCRIPTION SUMMARY")
    print("=" * 55)
    print()

    if episode_results:
        avg_segments = total_segments / len(episode_results)
        print(f"  Episodes transcribed : {len(episode_results)}")
        print(f"  Total segments       : {total_segments}")
        print(f"  Average per episode  : {avg_segments:.1f}")
    else:
        print("  No new episodes were transcribed.")

    print()

    # Detailed database stats
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM segments")
        total_db_segments = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM filler_words")
        total_fillers = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM silence_events")
        total_silences = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE transcription_status = 'completed'"
        )
        completed_episodes = cursor.fetchone()[0]

        # Per-speaker segment counts
        cursor = conn.execute(
            """
            SELECT s.name, COUNT(seg.id) as seg_count
            FROM speakers s
            LEFT JOIN segments seg ON seg.speaker_id = s.id
            GROUP BY s.id
            ORDER BY seg_count DESC
            """
        )
        speaker_stats = cursor.fetchall()

        # Language distribution
        cursor = conn.execute(
            """
            SELECT language, COUNT(*) as cnt
            FROM segments
            GROUP BY language
            ORDER BY cnt DESC
            """
        )
        lang_stats = cursor.fetchall()

        # Top filler words
        cursor = conn.execute(
            """
            SELECT filler_text, COUNT(*) as cnt
            FROM filler_words
            GROUP BY filler_text
            ORDER BY cnt DESC
            LIMIT 5
            """
        )
        top_fillers = cursor.fetchall()

    print("  Database Statistics:")
    print(f"    Total segments in DB    : {total_db_segments}")
    print(f"    Total filler words      : {total_fillers}")
    print(f"    Total silence events    : {total_silences}")
    print(f"    Completed episodes      : {completed_episodes}")
    print()

    print("  Speaker Breakdown:")
    for row in speaker_stats:
        name = row[0]
        count = row[1]
        print(f"    {name}: {count} segments")
    print()

    print("  Language Distribution:")
    for row in lang_stats:
        lang = row[0]
        count = row[1]
        label = {"he": "Hebrew", "en": "English", "mixed": "Mixed"}.get(lang, lang)
        print(f"    {label}: {count} segments")
    print()

    if top_fillers:
        print("  Top Filler Words:")
        for row in top_fillers:
            print(f"    {row[0]}: {row[1]} occurrences")
    print()

    print("Mock transcription complete.")


if __name__ == "__main__":
    main()

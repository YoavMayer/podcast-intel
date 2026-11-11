"""
Explicit state machine for an episode's transcription status.

The ``episodes.transcription_status`` column used to be a free string that
each transcriber set by hand. This module names the states, lists the moves
that are allowed between them, and gives one function that checks a move
before it happens. The stored values are unchanged, so existing databases
read exactly as before.

Lifecycle::

    pending    -> processing
    processing -> completed | failed | pending (stale reset)
    failed     -> processing (retry) | pending
    completed  -> pending (explicit reset)

A ``processing`` row left behind by a killed process can be moved to
``failed`` or back to ``pending``; it can never jump straight to another
``processing`` run, which would hide the crash.
"""

from __future__ import annotations

from enum import Enum


class TranscriptionStatus(str, Enum):
    """Transcription processing status, stored as its lowercase value."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def parse(cls, value: str | TranscriptionStatus) -> TranscriptionStatus:
        """
        Read a status from the database or a caller.

        Accepts an enum member or any old string spelling (surrounding
        whitespace and upper case are tolerated). Raises ``ValueError`` for
        anything that is not a known status.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(f"Unknown transcription status: {value!r}")


#: For each state, the states it may move to next.
TRANSCRIPTION_TRANSITIONS: dict[TranscriptionStatus, frozenset[TranscriptionStatus]] = {
    TranscriptionStatus.PENDING: frozenset({TranscriptionStatus.PROCESSING}),
    TranscriptionStatus.PROCESSING: frozenset(
        {
            TranscriptionStatus.COMPLETED,
            TranscriptionStatus.FAILED,
            TranscriptionStatus.PENDING,
        }
    ),
    TranscriptionStatus.COMPLETED: frozenset({TranscriptionStatus.PENDING}),
    TranscriptionStatus.FAILED: frozenset(
        {TranscriptionStatus.PROCESSING, TranscriptionStatus.PENDING}
    ),
}


class IllegalTransitionError(ValueError):
    """Raised when a status change is not in the transition table."""

    def __init__(self, current: TranscriptionStatus, new: TranscriptionStatus) -> None:
        allowed = sorted(s.value for s in TRANSCRIPTION_TRANSITIONS[current])
        super().__init__(
            f"Illegal transcription status change {current.value!r} -> {new.value!r}; "
            f"allowed from {current.value!r}: {allowed}"
        )
        self.current = current
        self.new = new


def can_transition(
    current: str | TranscriptionStatus, new: str | TranscriptionStatus
) -> bool:
    """Return True if moving from ``current`` to ``new`` is allowed."""
    return TranscriptionStatus.parse(new) in TRANSCRIPTION_TRANSITIONS[
        TranscriptionStatus.parse(current)
    ]


def check_transition(
    current: str | TranscriptionStatus, new: str | TranscriptionStatus
) -> TranscriptionStatus:
    """
    Validate a status change and return the new status as an enum member.

    Raises:
        ValueError: if either value is not a known status
        IllegalTransitionError: if the move is not in the table
    """
    cur = TranscriptionStatus.parse(current)
    nxt = TranscriptionStatus.parse(new)
    if nxt not in TRANSCRIPTION_TRANSITIONS[cur]:
        raise IllegalTransitionError(cur, nxt)
    return nxt


__all__ = [
    "TranscriptionStatus",
    "TRANSCRIPTION_TRANSITIONS",
    "IllegalTransitionError",
    "can_transition",
    "check_transition",
]

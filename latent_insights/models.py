"""
Data models — dataclasses for the computation engine.
Pydantic is only used at API boundaries (see api/schemas.py).
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# --- Enums ---


class ThreadStatus(str, Enum):
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETE = "complete"
    ERROR = "error"


class MoveType(str, Enum):
    # Coordinator-picked analytical moves
    SCOPE = "SCOPE"
    FORAGE = "FORAGE"
    FRAME = "FRAME"
    INTERROGATE = "INTERROGATE"
    SYNTHESIZE = "SYNTHESIZE"
    STUCK = "STUCK"
    DONE = "DONE"
    # Mixed-initiative moves — steps contributed by the human or
    # representing the thread's waiting state. These aren't chosen by
    # the coordinator; they're committed directly by the runner when a
    # human posts guidance or when a thread enters a terminal waiting
    # state. They live in the step timeline alongside the analytical
    # moves so the UI can render every step uniformly.
    HUMAN_INPUT = "HUMAN_INPUT"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"


class CoordinatorStatus(str, Enum):
    CONTINUE = "CONTINUE"
    STUCK = "STUCK"
    DONE = "DONE"


# --- DB records ---


@dataclass
class Session:
    id: str
    dataset_path: str
    table_name: str = "dataset"
    schema_summary: str | None = None
    scout_output: dict | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Thread:
    id: str
    session_id: str
    seed_question: str
    motivation: str
    entry_point: str
    status: ThreadStatus = ThreadStatus.RUNNING
    summary: str | None = None
    error: str | None = None
    running_summary: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# --- Agent I/O ---


@dataclass
class ScoutQuestion:
    question: str
    motivation: str
    entry_point: str
    difficulty: str  # simple | moderate | deep


@dataclass
class ScoutOutput:
    exploration_notes: str
    questions: list[ScoutQuestion]


@dataclass
class CoordinatorDecision:
    assessment: str
    next_move: MoveType
    rationale: str
    status: CoordinatorStatus
    worker_instruction: str | None = None
    # When STUCK:
    question_for_human: str | None = None
    context: str | None = None


@dataclass
class WorkerResult:
    result: str
    view_requested: dict | None = None
    llm_calls: list[dict] | None = None


# --- Events ---


@dataclass
class StreamEvent:
    """Queue envelope around a render-ready ``FeedEntry``.

    ``data`` carries the entry's ``model_dump(exclude_none=True)`` — the
    SSE serializer writes it verbatim as the ``data:`` line, and the
    snapshot mapper (`api/feed.py::session_to_feed`) produces the same
    shape from a static ``SessionResponse``. ``session_id`` /
    ``thread_id`` / ``event_type`` are mirrored at the envelope level for
    routing; ``timestamp`` matches the entry's timestamp.
    """

    session_id: str
    thread_id: str
    event_type: str
    message: str
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

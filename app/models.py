"""
Data models — dataclasses for the computation engine.
Pydantic is only used at API boundaries (see api/schemas.py).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# --- Enums ---


class ThreadStatus(str, Enum):
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETE = "complete"


class MoveType(str, Enum):
    SCOPE = "SCOPE"
    FORAGE = "FORAGE"
    FRAME = "FRAME"
    INTERROGATE = "INTERROGATE"
    SYNTHESIZE = "SYNTHESIZE"


class CoordinatorStatus(str, Enum):
    CONTINUE = "CONTINUE"
    STUCK = "STUCK"
    DONE = "DONE"


# --- DB records ---


@dataclass
class Session:
    id: str
    dataset_path: str
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
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Step:
    id: str
    thread_id: str
    step_number: int
    move: MoveType
    instruction: str
    result_summary: str
    result_details: str | None = None
    view_created: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


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
    queries_executed: list[dict]
    summary: str
    details: str | None = None
    view_requested: dict | None = None


# --- Events ---


@dataclass
class ThreadEvent:
    session_id: str
    thread_id: str
    event_type: str  # step_completed, thread_waiting, thread_complete, scout_done
    payload: dict = field(default_factory=dict)

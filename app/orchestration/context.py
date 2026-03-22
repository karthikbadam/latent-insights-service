"""Thread context — carries all state between state machine steps."""

import time
from dataclasses import dataclass, field
from threading import Event
from typing import Any

from app.config import AppConfig
from app.core.llm import LLMClient
from app.core.queue import Queue
from app.core.state import StateStore
from app.core.tracing import Span, TraceStore
from app.models import CoordinatorDecision, Thread


@dataclass
class ThreadContext:
    """All state needed to drive one analytical thread through its lifecycle."""

    # Fixed for thread lifetime
    config: AppConfig
    llm: LLMClient
    session_db: Any
    queue: Queue
    state: StateStore
    trace_store: TraceStore
    thread: Thread
    schema_summary: str

    # Mutable across steps
    step_number: int = 0
    step_span: Span | None = None
    step_start: float = 0.0
    decision: CoordinatorDecision | None = None
    human_messages: list[str] = field(default_factory=list)
    thread_start: float = field(default_factory=time.monotonic)

    # Worker state (reset each worker step)
    worker_messages: list[dict] = field(default_factory=list)
    worker_instruction: str = ""
    current_model: str = ""
    consecutive_errors: int = 0
    attempts: int = 0
    llm_calls: list[dict] = field(default_factory=list)
    thread_views: str = "(none)"
    coordinator_ms: int = 0
    move_history: list[str] = field(default_factory=list)

    # Completion signaling (for tests and callers that need to wait)
    done_event: Event = field(default_factory=Event)

    @property
    def tid(self) -> str:
        return self.thread.id[:8]

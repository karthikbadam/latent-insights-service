"""
InvestigationStore — unified in-memory store for sessions, threads, and steps.

Replaces the split StateStore + TraceStore with a single store.

Persistence: ``data/sessions/{session_id}.json`` with the same shape as
the REST ``GET /sessions/{id}`` response (flat top-level fields, flat
``StepEvent`` dicts), so saved files and the live API snapshot are 1:1.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from latent_insights.models import Session, Thread, ThreadStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step — one coordinator->worker cycle with its events
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One coordinator->worker cycle.

    Fields mirror ``api.schemas.StepResponse`` so save/load to the JSON
    snapshot format is a straight serialization. ``events`` is a list of
    flat dicts matching ``api.schemas.StepEvent`` (keys: ``type``,
    ``timestamp``, plus type-specific fields like ``sql``/``response``).
    """

    thread_id: str                       # internal — not on wire
    step_number: int = 0
    move: str = ""
    instruction: str = ""
    assessment: str = ""
    rationale: str = ""
    result: str = ""
    view_created: str | None = None
    events: list[dict] = field(default_factory=list)
    status: str = "ok"                   # internal: "ok" | "stuck" | "error"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.end_time is None:
            return None
        return round((self.end_time - self.start_time) * 1000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# InvestigationStore
# ---------------------------------------------------------------------------

class InvestigationStore:
    """Unified store for sessions, threads, steps, and pending messages."""

    def __init__(self, data_dir: str = "data"):
        self._sessions: dict[str, Session] = {}
        self._threads: dict[str, Thread] = {}
        self._session_threads: dict[str, list[str]] = {}
        self._steps: dict[str, list[Step]] = {}
        self._pending_messages: dict[str, list[dict]] = {}
        self._data_dir = data_dir

    # --- Sessions ---

    def create_session(self, dataset_path: str, table_name: str = "dataset") -> Session:
        session = Session(id=generate_id(), dataset_path=dataset_path, table_name=table_name)
        self._sessions[session.id] = session
        self._session_threads[session.id] = []
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> list[Session]:
        return list(self._sessions.values())

    def update_session_table_name(self, session_id: str, table_name: str):
        session = self._sessions.get(session_id)
        if session:
            session.table_name = table_name

    def update_session_schema(self, session_id: str, schema_summary: str):
        session = self._sessions.get(session_id)
        if session:
            session.schema_summary = schema_summary

    def update_session_scout(self, session_id: str, scout_output: dict):
        session = self._sessions.get(session_id)
        if session:
            session.scout_output = scout_output

    # --- Threads ---

    def create_thread(
        self,
        session_id: str,
        seed_question: str,
        motivation: str = "",
        entry_point: str = "",
    ) -> Thread:
        thread = Thread(
            id=generate_id(),
            session_id=session_id,
            seed_question=seed_question,
            motivation=motivation,
            entry_point=entry_point,
        )
        self._threads[thread.id] = thread
        if session_id not in self._session_threads:
            self._session_threads[session_id] = []
        self._session_threads[session_id].append(thread.id)
        return thread

    def get_thread(self, thread_id: str) -> Thread | None:
        return self._threads.get(thread_id)

    def get_threads(self, session_id: str) -> list[Thread]:
        thread_ids = self._session_threads.get(session_id, [])
        return [self._threads[tid] for tid in thread_ids if tid in self._threads]

    def update_thread_status(
        self,
        thread_id: str,
        status: ThreadStatus,
        summary: str | None = None,
        error: str | None = None,
    ):
        thread = self._threads.get(thread_id)
        if thread is None:
            return
        thread.status = status
        thread.updated_at = datetime.utcnow()
        if summary is not None:
            thread.summary = summary
        if error is not None:
            thread.error = error

    def update_thread_running_summary(self, thread_id: str, running_summary: str):
        thread = self._threads.get(thread_id)
        if thread:
            thread.running_summary = running_summary

    # --- Pending messages ---

    def push_pending_message(
        self, thread_id: str, content: str, target: str = "thread",
    ):
        if thread_id not in self._pending_messages:
            self._pending_messages[thread_id] = []
        self._pending_messages[thread_id].append({
            "content": content,
            "target": target,
            "timestamp": time.time(),
        })

    def push_pivot_marker(self, thread_id: str):
        """Push a sentinel that triggers a runner pivot without committing
        a new HUMAN_INPUT step. Used when the route handler has already
        committed the step inline and just needs the RUNNING thread to
        flush its in-flight step at the next callback boundary.
        """
        self._pending_messages.setdefault(thread_id, []).append(
            {"committed": True}
        )

    def drain_pending_messages(self, thread_id: str) -> list[dict]:
        return self._pending_messages.pop(thread_id, [])

    def has_pending_messages(self, thread_id: str) -> bool:
        """Peek — does this thread have pending human messages?

        Used by the runner's flush-and-pivot logic to avoid draining
        when there's nothing to commit.
        """
        return bool(self._pending_messages.get(thread_id))

    # --- Steps ---

    def start_step(self, thread_id: str) -> Step:
        """Append a new in-progress step to the thread and return it.

        ``step_number`` is assigned based on position. Callers populate
        ``move`` / ``instruction`` / ``result`` / ``view_created``
        directly on the returned dataclass as the step progresses.
        """
        if thread_id not in self._steps:
            self._steps[thread_id] = []
        step = Step(
            thread_id=thread_id,
            step_number=len(self._steps[thread_id]) + 1,
        )
        self._steps[thread_id].append(step)
        return step

    def end_step(self, step: Step, status: str = "ok"):
        step.end_time = time.time()
        step.status = status

    def add_event(
        self,
        step: Step,
        event: dict,
        timestamp: float | None = None,
    ):
        """Append a flat ``StepEvent``-shaped dict to the step.

        Callers build the event dict with the fields appropriate for the
        event type (``llm_call`` / ``tool_call`` / ``human_message``) —
        see ``api.schemas.StepEvent`` for the canonical shape.
        ``type`` and ``timestamp`` are ensured on the stored record.
        """
        record = dict(event)
        record.setdefault("timestamp", timestamp if timestamp is not None else time.time())
        if "type" not in record:
            raise ValueError("event dict must include a 'type' key")
        step.events.append(record)

    def get_steps(self, thread_id: str) -> list[Step]:
        return self._steps.get(thread_id, [])

    def clear_steps(self, thread_id: str):
        self._steps.pop(thread_id, None)

    # --- History formatting ---

    def format_thread_history(
        self,
        thread_id: str,
        running_summary: str | None = None,
        full_window: int = 3,
    ) -> str:
        """Render the thread's step timeline as a prompt block.

        Human-contributed steps (``move="HUMAN_INPUT"``) and terminal
        waiting steps (``move="WAITING_FOR_HUMAN"``) are included the
        same way analytical steps are — they have a ``move`` and a
        ``result`` like any other step. No side-channel ``human_messages``
        parameter is needed; a mixed-initiative timeline is just a list
        of steps.
        """
        steps = self.get_steps(thread_id)
        if not steps:
            preamble = f"Summary so far: {running_summary}\n\n" if running_summary else ""
            return preamble + "(No steps yet — this is the first move)"

        parts = []

        if running_summary:
            parts.append(f"**Summary of earlier analysis:**\n{running_summary}")

        total = len(steps)
        cutoff = max(1, total - full_window)

        for i, step in enumerate(steps, 1):
            move = step.move or "?"
            instruction = step.instruction
            result = step.result

            if i == 1 or i > cutoff:
                parts.append(
                    f"Step {i} [{move}]:\n"
                    f'  Instruction: "{instruction}"\n'
                    f"  Result: {result}"
                )
            else:
                first_sentence = result.split(".")[0].strip() + "." if result else ""
                parts.append(f"Step {i} [{move}]: {first_sentence}")

        return "\n\n".join(parts)

    def summarize_history(
        self,
        thread_id: str,
        llm,
        model: str,
        seed_question: str,
        threshold: int = 5,
    ) -> str | None:
        steps = self.get_steps(thread_id)
        if len(steps) < threshold:
            return None

        history_parts = []
        for i, step in enumerate(steps, 1):
            move = step.move or "?"
            result = step.result
            history_parts.append(f"Step {i} [{move}]: {result}")

        history_text = "\n\n".join(history_parts)

        messages = [
            {"role": "system", "content": (
                "You are a research assistant. Summarize the analytical thread history below "
                "into 3-5 sentences. Preserve key findings, hypotheses tested, and data patterns "
                "discovered. Be specific about numbers and results."
            )},
            {"role": "user", "content": (
                f"Thread question: {seed_question}\n\n"
                f"History ({len(steps)} steps):\n\n{history_text}\n\n"
                "Summarize the progress so far."
            )},
        ]

        response = llm.call(
            model=model,
            messages=messages,
            role="summarizer",
            temperature=0.0,
            max_tokens=512,
        )
        return response.content

    # --- Persistence -----------------------------------------------------
    #
    # Disk shape matches api.schemas.SessionResponse — flat top-level
    # fields, threads/steps nested, and events in their canonical
    # ``StepEvent`` shape. The only field dropped on save is ``urls``,
    # which is derived from the request at GET time and can't be
    # reconstructed server-side.

    def _step_to_dict(self, step: Step) -> dict:
        return {
            "step_number": step.step_number,
            "move": step.move,
            "instruction": step.instruction,
            "assessment": step.assessment,
            "rationale": step.rationale,
            "result": step.result,
            "view_created": step.view_created,
            "duration_ms": step.duration_ms,
            "start_time": step.start_time,
            "end_time": step.end_time,
            "events": list(step.events),
        }

    def _thread_to_dict(self, thread: Thread) -> dict:
        steps = self.get_steps(thread.id)
        return {
            "id": thread.id,
            "seed_question": thread.seed_question,
            "motivation": thread.motivation,
            "status": thread.status.value,
            "summary": thread.summary,
            "running_summary": thread.running_summary,
            "error": thread.error,
            "steps": [
                self._step_to_dict(s) for s in steps if s.end_time is not None
            ],
            "updated_at": thread.updated_at.isoformat() if thread.updated_at else "",
        }

    def save_session(self, session_id: str):
        """Write the session snapshot to disk in the API response shape."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        sessions_dir = os.path.join(self._data_dir, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        threads = self.get_threads(session_id)
        scout_questions = (
            session.scout_output.get("questions") if session.scout_output else None
        )

        data = {
            "id": session.id,
            "dataset_path": session.dataset_path,
            # table_name is internal server state — not on the API wire
            # but needed to rehydrate the DuckDB connection on restart.
            # Frontend consumers of /saved can ignore it.
            "table_name": session.table_name,
            "schema_summary": session.schema_summary,
            "scout_questions": scout_questions,
            "threads": [self._thread_to_dict(t) for t in threads],
            "created_at": session.created_at.isoformat() if session.created_at else "",
        }

        filepath = os.path.join(sessions_dir, f"{session_id}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"State saved: {filepath} ({len(threads)} threads)")

    def save_all(self):
        for session_id in self._sessions:
            self.save_session(session_id)

    def load_session(self, session_id: str) -> Session | None:
        """Reload a session from its saved JSON snapshot.

        Accepts the current flat ``SessionResponse`` shape on disk. Any
        timing split that existed internally (coordinator_ms vs
        worker_ms per step) is not preserved — only the aggregate
        ``duration_ms`` round-trips. That's fine: those metrics are only
        used for log lines on the live run.
        """
        filepath = os.path.join(self._data_dir, "sessions", f"{session_id}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath) as f:
            data = json.load(f)

        # Prefer the saved table_name; fall back to re-deriving from the
        # dataset path for files saved before this field was written.
        dataset_path = data["dataset_path"]
        table_name = data.get("table_name")
        if not table_name:
            try:
                from latent_insights.db.connection import table_name_from_path
                table_name = table_name_from_path(dataset_path)
            except Exception:
                table_name = "dataset"

        session = Session(
            id=data["id"],
            dataset_path=dataset_path,
            table_name=table_name,
            schema_summary=data.get("schema_summary"),
            scout_output=(
                {"questions": data["scout_questions"], "exploration_notes": ""}
                if data.get("scout_questions") is not None
                else None
            ),
        )
        self._sessions[session.id] = session
        self._session_threads[session.id] = []

        for td in data.get("threads", []):
            # ``updated_at`` serializes as an ISO string; tolerate empty/missing.
            updated_at = td.get("updated_at") or ""
            try:
                updated_dt = datetime.fromisoformat(updated_at) if updated_at else datetime.utcnow()
            except ValueError:
                updated_dt = datetime.utcnow()

            thread = Thread(
                id=td["id"],
                session_id=session.id,
                seed_question=td["seed_question"],
                motivation=td.get("motivation") or "",
                entry_point=td.get("entry_point", ""),
                status=ThreadStatus(td.get("status", "running")),
                summary=td.get("summary"),
                error=td.get("error"),
                running_summary=td.get("running_summary"),
                updated_at=updated_dt,
            )
            self._threads[thread.id] = thread
            self._session_threads[session.id].append(thread.id)

            step_records = td.get("steps") or []
            steps: list[Step] = []
            for sd in step_records:
                duration_ms = sd.get("duration_ms") or 0
                # Prefer persisted timestamps; fall back to deriving from
                # duration_ms for legacy snapshots that didn't save them.
                start_time = sd.get("start_time")
                end_time = sd.get("end_time")
                if start_time is None or end_time is None:
                    end_time = end_time if end_time is not None else time.time()
                    start_time = start_time if start_time is not None else (
                        end_time - (duration_ms / 1000.0)
                    )
                steps.append(Step(
                    thread_id=thread.id,
                    step_number=sd.get("step_number", len(steps) + 1),
                    move=sd.get("move") or "",
                    instruction=sd.get("instruction") or "",
                    assessment=sd.get("assessment") or "",
                    rationale=sd.get("rationale") or "",
                    result=sd.get("result") or "",
                    view_created=sd.get("view_created"),
                    events=list(sd.get("events") or []),
                    status="ok",
                    start_time=start_time,
                    end_time=end_time,
                ))
            if steps:
                self._steps[thread.id] = steps

        logger.info(f"State loaded: {filepath}")
        return session

    # --- Counts ---

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def thread_count(self) -> int:
        return len(self._threads)

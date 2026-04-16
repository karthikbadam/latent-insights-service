"""
InvestigationStore — unified in-memory store for sessions, threads, and spans.

Replaces the split StateStore (state.py) + TraceStore (tracing.py) with a
single store and a single-file-per-session JSON persistence model.

Persistence: ``data/sessions/{session_id}.json`` containing the full session
graph: ``{session: {...}, threads: [{...thread_fields, spans: [...]}]}``.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime

from latent_insights.models import Session, Thread, ThreadStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Span — step-level trace record (moved from tracing.py)
# ---------------------------------------------------------------------------

def _generate_span_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class Span:
    trace_id: str
    span_id: str = field(default_factory=_generate_span_id)
    parent_span_id: str | None = None
    name: str = ""
    kind: str = "step"
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "ok"
    status_message: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# InvestigationStore
# ---------------------------------------------------------------------------

class InvestigationStore:
    """Unified store for sessions, threads, spans, and pending messages."""

    def __init__(self, data_dir: str = "data"):
        self._sessions: dict[str, Session] = {}
        self._threads: dict[str, Thread] = {}
        self._session_threads: dict[str, list[str]] = {}
        self._spans: dict[str, list[Span]] = {}
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

    def drain_pending_messages(self, thread_id: str) -> list[dict]:
        return self._pending_messages.pop(thread_id, [])

    # --- Spans ---

    def start_span(
        self,
        trace_id: str,
        name: str,
        kind: str = "step",
        parent_span_id: str | None = None,
        attributes: dict | None = None,
    ) -> Span:
        span = Span(
            trace_id=trace_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )
        if trace_id not in self._spans:
            self._spans[trace_id] = []
        self._spans[trace_id].append(span)
        return span

    def end_span(
        self,
        span: Span,
        status: str = "ok",
        status_message: str | None = None,
    ):
        span.end_time = time.time()
        span.status = status
        span.status_message = status_message

    def add_event(
        self,
        span: Span,
        name: str,
        attributes: dict | None = None,
        timestamp: float | None = None,
    ):
        span.events.append({
            "name": name,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "attributes": attributes or {},
        })

    def get_spans(self, trace_id: str) -> list[Span]:
        return self._spans.get(trace_id, [])

    def get_step_spans(self, trace_id: str) -> list[Span]:
        return [s for s in self.get_spans(trace_id) if s.kind == "step"]

    def clear_spans(self, trace_id: str):
        self._spans.pop(trace_id, None)

    # --- History formatting (ported from tracing.py) ---

    def format_thread_history(
        self,
        trace_id: str,
        human_messages: list[dict] | list[str] | None = None,
        running_summary: str | None = None,
        full_window: int = 3,
    ) -> str:
        def _content(m) -> str:
            return m.get("content", "") if isinstance(m, dict) else str(m)

        steps = self.get_step_spans(trace_id)
        if not steps:
            preamble = f"Summary so far: {running_summary}\n\n" if running_summary else ""
            if human_messages:
                parts = [f'[Human input]: "{_content(msg)}"' for msg in human_messages]
                return preamble + "\n\n".join(parts) if preamble else "\n\n".join(parts)
            return preamble + "(No steps yet — this is the first move)"

        parts = []

        if running_summary:
            parts.append(f"**Summary of earlier analysis:**\n{running_summary}")

        total = len(steps)
        cutoff = max(1, total - full_window)

        for i, span in enumerate(steps, 1):
            move = span.attributes.get("move", "?")
            instruction = span.attributes.get("instruction", "")
            result = span.attributes.get("result", "")

            if i == 1 or i > cutoff:
                parts.append(
                    f"Step {i} [{move}]:\n"
                    f'  Instruction: "{instruction}"\n'
                    f"  Result: {result}"
                )
            else:
                first_sentence = result.split(".")[0].strip() + "." if result else ""
                parts.append(f"Step {i} [{move}]: {first_sentence}")

        if human_messages:
            for msg in human_messages:
                parts.append(f'[Human input]: "{_content(msg)}"')

        return "\n\n".join(parts)

    def summarize_history(
        self,
        trace_id: str,
        llm,
        model: str,
        seed_question: str,
        threshold: int = 5,
    ) -> str | None:
        steps = self.get_step_spans(trace_id)
        if len(steps) < threshold:
            return None

        history_parts = []
        for i, span in enumerate(steps, 1):
            move = span.attributes.get("move", "?")
            result = span.attributes.get("result", "")
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

    # --- Persistence (single JSON file per session) ---

    def save_session(self, session_id: str):
        """Write session + threads + spans to a single JSON file."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        sessions_dir = os.path.join(self._data_dir, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        threads = self.get_threads(session_id)
        thread_records = []
        for t in threads:
            spans = self.get_spans(t.id)
            thread_records.append({
                **asdict(t),
                "spans": [asdict(s) for s in spans],
            })

        data = {
            "session": asdict(session),
            "threads": thread_records,
        }

        filepath = os.path.join(sessions_dir, f"{session_id}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"State saved: {filepath} ({len(threads)} threads)")

    def save_all(self):
        """Save all sessions to disk."""
        for session_id in self._sessions:
            self.save_session(session_id)

    def load_session(self, session_id: str) -> Session | None:
        """Reload session + threads + spans from JSON file."""
        filepath = os.path.join(self._data_dir, "sessions", f"{session_id}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath) as f:
            data = json.load(f)

        session_data = data["session"]
        session = Session(
            id=session_data["id"],
            dataset_path=session_data["dataset_path"],
            table_name=session_data.get("table_name", "dataset"),
            schema_summary=session_data.get("schema_summary"),
            scout_output=session_data.get("scout_output"),
        )
        self._sessions[session.id] = session
        self._session_threads[session.id] = []

        for td in data.get("threads", []):
            thread = Thread(
                id=td["id"],
                session_id=td["session_id"],
                seed_question=td["seed_question"],
                motivation=td.get("motivation", ""),
                entry_point=td.get("entry_point", ""),
                status=ThreadStatus(td.get("status", "running")),
                summary=td.get("summary"),
                error=td.get("error"),
                running_summary=td.get("running_summary"),
            )
            self._threads[thread.id] = thread
            self._session_threads[session.id].append(thread.id)

            # Reload spans for this thread
            span_records = td.get("spans", [])
            if span_records:
                spans = []
                for sd in span_records:
                    span = Span(
                        trace_id=sd["trace_id"],
                        span_id=sd["span_id"],
                        parent_span_id=sd.get("parent_span_id"),
                        name=sd.get("name", ""),
                        kind=sd.get("kind", "step"),
                        attributes=sd.get("attributes", {}),
                        events=sd.get("events", []),
                        status=sd.get("status", "ok"),
                        status_message=sd.get("status_message"),
                        start_time=float(sd.get("start_time", 0)),
                        end_time=float(sd["end_time"]) if sd.get("end_time") else None,
                    )
                    spans.append(span)
                self._spans[thread.id] = spans

        logger.info(f"State loaded: {filepath}")
        return session

    # --- Counts ---

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def thread_count(self) -> int:
        return len(self._threads)

"""
StateStore — in-memory session and thread state with file persistence.

All session/thread state lives here. DuckDB is only used as a read-only
dataset query engine by workers. State is dumped to JSON on thread
completion and app shutdown.
"""

import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime

from app.models import Session, Thread, ThreadStatus

logger = logging.getLogger(__name__)


def generate_id() -> str:
    return uuid.uuid4().hex[:12]


class StateStore:
    def __init__(self, data_dir: str = "data"):
        self._sessions: dict[str, Session] = {}
        self._threads: dict[str, Thread] = {}
        self._session_threads: dict[str, list[str]] = {}
        self._data_dir = data_dir

    # --- Sessions ---

    def create_session(self, dataset_path: str, table_name: str = "dataset") -> Session:
        session = Session(id=generate_id(), dataset_path=dataset_path, table_name=table_name)
        self._sessions[session.id] = session
        self._session_threads[session.id] = []
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

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

    # --- Persistence ---

    def dump_session(self, session_id: str):
        """Write session + threads to a JSON file."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        state_dir = os.path.join(self._data_dir, "state")
        os.makedirs(state_dir, exist_ok=True)

        threads = self.get_threads(session_id)
        data = {
            "session": asdict(session),
            "threads": [asdict(t) for t in threads],
        }

        filepath = os.path.join(state_dir, f"{session_id}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"State dumped: {filepath} ({len(threads)} threads)")

    def dump_all(self):
        """Dump all sessions to disk."""
        for session_id in self._sessions:
            self.dump_session(session_id)

    def load_session(self, session_id: str) -> Session | None:
        """Reload session + threads from JSON file."""
        filepath = os.path.join(self._data_dir, "state", f"{session_id}.json")
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

        logger.info(f"State loaded: {filepath}")
        return session

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def thread_count(self) -> int:
        return len(self._threads)

"""Session lifecycle — upload, profile, scout, spawn threads."""

import logging
import time
from dataclasses import asdict

from app.agents.profiler import Profiler
from app.agents.scout import Scout
from app.config import AppConfig
from app.core.llm import LLMClient
from app.core.queue import Queue
from app.core.state import StateStore
from app.core.tracing import TraceStore
from app.db.connection import Database
from app.models import StreamEvent, ThreadStatus
from app.orchestration.thread import ThreadRunner

logger = logging.getLogger(__name__)


class SessionFlow:
    """Orchestrates session lifecycle: profile, scout, spawn threads."""

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        db: Database,
        queue: Queue,
        state: StateStore,
        trace_store: TraceStore,
    ):
        self.config = config
        self.llm = llm
        self.db = db
        self.queue = queue
        self.state = state
        self.trace_store = trace_store
        self.profiler = Profiler(llm, config.models.profiler)
        self.scout = Scout(llm, config.models.scout)

    def create(self, session_id: str, dataset_path: str) -> str:
        """
        Full session creation flow (runs as background task):
        1. Create session DB with dataset loaded
        2. Run profiler -> store schema_summary
        3. Run scout -> store scout_output
        4. Spawn threads for scout questions
        """
        session_start = time.monotonic()

        logger.info(f"Session {session_id} flow starting for {dataset_path}")

        session_db, table_name = self.db.create_session_db(session_id, dataset_path)
        self.state.update_session_table_name(session_id, table_name)

        # Run profiler
        t0 = time.monotonic()
        schema_summary = self.profiler.call(session_db, table_name)
        profiler_ms = round((time.monotonic() - t0) * 1000)

        self.state.update_session_schema(session_id, schema_summary)
        logger.info(f"Session {session_id} profiled ({profiler_ms}ms)")

        # Run scout
        t0 = time.monotonic()
        scout_output = self.scout.call(
            schema_summary=schema_summary,
            table_name=table_name,
            session_db=session_db,
        )
        scout_ms = round((time.monotonic() - t0) * 1000)

        self.state.update_session_scout(session_id, asdict(scout_output))

        self.queue.emit(StreamEvent(
            session_id=session_id,
            thread_id="",
            event_type="scout_done",
            message=f"Scout found {len(scout_output.questions)} questions",
            data={
                "question_count": len(scout_output.questions),
                "questions": [
                    {
                        "question": q.question,
                        "motivation": q.motivation,
                        "entry_point": q.entry_point,
                        "difficulty": q.difficulty,
                    }
                    for q in scout_output.questions
                ],
            },
        ))

        setup_elapsed = round(time.monotonic() - session_start, 2)
        logger.info(
            f"Session {session_id} scouted: {len(scout_output.questions)} questions "
            f"(profiler={profiler_ms}ms scout={scout_ms}ms setup={setup_elapsed}s)"
        )

        session_db.close()

        self._spawn_threads(session_id, scout_output.questions, schema_summary)

        self.state.dump_session(session_id)
        return session_id

    def continue_(self, session_id: str) -> str:
        """
        Continue an existing session:
        1. Resume any WAITING threads with a default message
        2. Re-run scout with existing findings as context → spawn new threads
        """
        session = self.state.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        schema_summary = session.schema_summary
        if not schema_summary:
            raise ValueError(f"Session {session_id} has no schema — run initial setup first")

        threads = self.state.get_threads(session_id)

        # 1. Resume all WAITING and COMPLETE threads
        resumable = [t for t in threads if t.status in (ThreadStatus.WAITING, ThreadStatus.COMPLETE)]
        for t in resumable:
            message = (
                "Continue the analysis. Try a different approach if you were stuck."
                if t.status == ThreadStatus.WAITING
                else "The previous analysis is complete. Dig deeper — are there follow-up questions, edge cases, or subgroups worth investigating?"
            )
            thread_db = self.db.open_session_connection(session_id)
            runner = ThreadRunner(
                config=self.config,
                llm=self.llm,
                session_db=thread_db,
                queue=self.queue,
                state=self.state,
                trace_store=self.trace_store,
                thread=t,
                schema_summary=schema_summary,
                human_messages=[message],
            )
            runner.resume()

        logger.info(f"Session {session_id} resumed {len(resumable)} threads")

        # 2. Re-run scout with existing questions/findings as context
        existing_questions = [t.seed_question for t in threads]
        existing_findings = [
            f"- {t.seed_question}: {t.summary}"
            for t in threads
            if t.summary
        ]

        prior_context = ""
        if existing_questions:
            prior_context += "\n\n## Already investigated\n\nDo NOT repeat these questions:\n"
            prior_context += "\n".join(f"- {q}" for q in existing_questions)
        if existing_findings:
            prior_context += "\n\n## Findings so far\n\n"
            prior_context += "\n".join(existing_findings)
            prior_context += "\n\nUse these findings to ask deeper follow-up questions."

        scout_schema = schema_summary + prior_context

        session_db = self.db.open_session_connection(session_id)
        t0 = time.monotonic()
        scout_output = self.scout.call(
            schema_summary=scout_schema,
            table_name=session.table_name,
            session_db=session_db,
        )
        scout_ms = round((time.monotonic() - t0) * 1000)
        session_db.close()

        # Filter out questions that are too similar to existing ones
        existing_lower = {q.lower().strip() for q in existing_questions}
        new_questions = [
            q for q in scout_output.questions
            if q.question.lower().strip() not in existing_lower
        ]

        logger.info(
            f"Session {session_id} continue: scout found {len(scout_output.questions)} questions, "
            f"{len(new_questions)} new ({scout_ms}ms)"
        )

        self.queue.emit(StreamEvent(
            session_id=session_id,
            thread_id="",
            event_type="scout_done",
            message=f"Scout found {len(new_questions)} new questions",
            data={
                "question_count": len(new_questions),
                "questions": [
                    {
                        "question": q.question,
                        "motivation": q.motivation,
                        "entry_point": q.entry_point,
                        "difficulty": q.difficulty,
                    }
                    for q in new_questions
                ],
                "resumed_threads": len(resumable),
            },
        ))

        self._spawn_threads(session_id, new_questions, schema_summary)

        self.state.dump_session(session_id)
        return session_id

    def _spawn_threads(self, session_id: str, questions, schema_summary: str):
        """Create and start threads for a list of scout questions."""
        for q in questions:
            thread = self.state.create_thread(
                session_id, q.question, q.motivation, q.entry_point,
            )
            thread_db = self.db.open_session_connection(session_id)
            runner = ThreadRunner(
                config=self.config,
                llm=self.llm,
                session_db=thread_db,
                queue=self.queue,
                state=self.state,
                trace_store=self.trace_store,
                thread=thread,
                schema_summary=schema_summary,
            )
            runner.start()

"""Session lifecycle — upload, profile, scout, spawn threads."""

import logging
import time
from dataclasses import asdict

from latent_insights.agents.profiler import Profiler
from latent_insights.agents.scout import Scout
from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMClient
from latent_insights.core.queue import Queue
from latent_insights.core.store import InvestigationStore
from latent_insights.db.connection import Database
from latent_insights.models import ScoutQuestion, StreamEvent, ThreadStatus
from latent_insights.orchestration.runner import RunnerMode, ThreadRunner

logger = logging.getLogger(__name__)


class SessionFlow:
    """Orchestrates session lifecycle: profile, scout, spawn threads."""

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        db: Database,
        queue: Queue,
        store: InvestigationStore,
    ):
        self.config = config
        self.llm = llm
        self.db = db
        self.queue = queue
        self.store = store
        self.profiler = Profiler(llm, config.models.profiler)
        self.scout = Scout(llm, config.models.scout)

    def create(self, session_id: str, dataset_path: str) -> str:
        """
        Full session creation flow (runs as background task):
        1. Create session DB with dataset loaded
        2. Run profiler -> store schema_summary
        3. Optionally run scout -> store scout_output (controlled by question_source)
        4. Spawn threads for scout/initial questions
        """
        session_start = time.monotonic()
        question_source = self.config.question_source

        logger.info(
            f"Session {session_id} flow starting for {dataset_path} "
            f"(question_source={question_source})"
        )

        session_db, table_name = self.db.create_session_db(session_id, dataset_path)
        self.store.update_session_table_name(session_id, table_name)

        # Run profiler
        t0 = time.monotonic()
        schema_summary = self.profiler.call(session_db, table_name)
        profiler_ms = round((time.monotonic() - t0) * 1000)

        self.store.update_session_schema(session_id, schema_summary)
        logger.info(f"Session {session_id} profiled ({profiler_ms}ms)")

        self.queue.emit(StreamEvent(
            session_id=session_id,
            thread_id="",
            event_type="schema_summary_ready",
            message="Dataset profiled.",
            data={"schema_summary": schema_summary},
        ))

        # Close read-write connection — all further access is read-only
        session_db.close()

        # Spawn initial_questions threads immediately (before scout)
        initial_questions = self.config.initial_questions
        if initial_questions:
            initial_q_objects = [
                ScoutQuestion(question=q, motivation="", entry_point="", difficulty="moderate")
                for q in initial_questions
            ]
            self._spawn_threads(session_id, initial_q_objects, schema_summary)
            logger.info(f"Session {session_id} spawned {len(initial_questions)} initial question threads")

        # Skip scout if question_source is "human"
        if question_source == "human":
            setup_elapsed = round(time.monotonic() - session_start, 2)
            logger.info(
                f"Session {session_id} ready for human questions "
                f"(profiler={profiler_ms}ms setup={setup_elapsed}s)"
            )
            self.queue.emit(StreamEvent(
                session_id=session_id,
                thread_id="",
                event_type="session_ready",
                message="Session profiled. Waiting for human questions.",
                data={
                    "question_source": "human",
                },
            ))
            self.store.save_session(session_id)
            return session_id

        # Run scout with a read-only connection
        scout_schema = schema_summary
        scout_context = self.config.scout_context
        if scout_context:
            scout_schema = (
                f"## User guidance\n\n{scout_context}\n\n"
                f"Use the guidance above to focus your question discovery.\n\n"
                f"{schema_summary}"
            )

        scout_db = self.db.open_session_connection(session_id)
        t0 = time.monotonic()
        scout_output = self.scout.call(
            schema_summary=scout_schema,
            table_name=table_name,
            session_db=scout_db,
            num_questions=self.config.num_scout_seed_questions,
        )
        scout_ms = round((time.monotonic() - t0) * 1000)
        scout_db.close()

        self.store.update_session_scout(session_id, asdict(scout_output))

        # Apply max_threads budget to scout questions
        scout_questions = scout_output.questions
        max_threads = self.config.max_threads
        if max_threads is not None:
            remaining = max(0, max_threads - len(initial_questions))
            scout_questions = scout_questions[:remaining]

        self.queue.emit(StreamEvent(
            session_id=session_id,
            thread_id="",
            event_type="scout_done",
            message=f"Scout found {len(scout_output.questions)} questions, spawning {len(scout_questions)}",
            data={
                "question_count": len(scout_questions),
                "questions": [
                    {
                        "question": q.question,
                        "motivation": q.motivation,
                        "entry_point": q.entry_point,
                        "difficulty": q.difficulty,
                    }
                    for q in scout_questions
                ],
            },
        ))

        setup_elapsed = round(time.monotonic() - session_start, 2)
        logger.info(
            f"Session {session_id} scouted: {len(scout_output.questions)} questions, "
            f"spawning {len(scout_questions)} "
            f"(profiler={profiler_ms}ms scout={scout_ms}ms setup={setup_elapsed}s)"
        )

        self._spawn_threads(session_id, scout_questions, schema_summary)

        self.store.save_session(session_id)
        return session_id

    def continue_(self, session_id: str) -> str:
        """
        Continue an existing session:
        1. Resume any WAITING threads with a default message
        2. Re-run scout with existing findings as context -> spawn new threads
        """
        session = self.store.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        schema_summary = session.schema_summary
        if not schema_summary:
            raise ValueError(f"Session {session_id} has no schema — run initial setup first")

        threads = self.store.get_threads(session_id)

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
                store=self.store,
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

        scout_context = self.config.scout_context
        if scout_context:
            scout_schema = (
                f"## User guidance\n\n{scout_context}\n\n"
                f"Use the guidance above to focus your question discovery.\n\n"
                f"{scout_schema}"
            )

        session_db = self.db.open_session_connection(session_id)
        t0 = time.monotonic()
        scout_output = self.scout.call(
            schema_summary=scout_schema,
            table_name=session.table_name,
            session_db=session_db,
            num_questions=self.config.num_scout_seed_questions,
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

        self.store.save_session(session_id)
        return session_id

    def _spawn_threads(self, session_id: str, questions, schema_summary: str):
        """Create and start threads for a list of questions.

        Pattern dispatch:
        - "coordinator_worker" (default): one ThreadRunner per question
        - "fan_out": spawn all questions as fan-out with post-hoc synthesis
        - "human_in_the_loop": each thread pauses after every step
        """
        pattern = self.config.default_pattern
        if not questions:
            return

        if pattern == "fan_out":
            from latent_insights.orchestration.patterns import fan_out_with_synthesis
            question_texts = [q.question for q in questions]
            fan_out_with_synthesis(
                questions=question_texts,
                session_id=session_id,
                config=self.config,
                llm=self.llm,
                db=self.db,
                queue=self.queue,
                store=self.store,
                schema_summary=schema_summary,
            )
            return

        mode = (
            RunnerMode.STEP_AND_PAUSE
            if pattern == "human_in_the_loop"
            else RunnerMode.LOOP_UNTIL_DONE
        )

        for q in questions:
            thread = self.store.create_thread(
                session_id, q.question, q.motivation, q.entry_point,
            )
            thread_db = self.db.open_session_connection(session_id)
            runner = ThreadRunner(
                config=self.config,
                llm=self.llm,
                session_db=thread_db,
                queue=self.queue,
                store=self.store,
                thread=thread,
                schema_summary=schema_summary,
                mode=mode,
            )
            runner.start()

"""Thread state machine — coordinator-worker cycle via futures.

Each step (coordinator LLM call, worker LLM call, SQL execution) is submitted
as a future to the pool. Pool threads are released between steps so other
analytical threads can make progress.
"""

import logging
import time
from threading import Event
from typing import Any

from openai import APITimeoutError

from app.agents.coordinator import Coordinator
from app.agents.worker import Worker
from app.config import AppConfig
from app.core.llm import LLMClient
from app.core.parsing import detect_degeneration
from app.core.queue import Queue
from app.core.state import StateStore
from app.core.tracing import Span, TraceStore
from app.models import (
    CoordinatorDecision,
    CoordinatorStatus,
    MoveType,
    StreamEvent,
    Thread,
    ThreadStatus,
    WorkerResult,
)

logger = logging.getLogger(__name__)


class ThreadRunner:
    """Drives one analytical thread through its coordinator-worker lifecycle."""

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        session_db: Any,
        queue: Queue,
        state: StateStore,
        trace_store: TraceStore,
        thread: Thread,
        schema_summary: str,
        human_messages: list[str] | None = None,
    ):
        self.config = config
        self.llm = llm
        self.session_db = session_db
        self.queue = queue
        self.state = state
        self.trace_store = trace_store
        self.thread = thread
        self.schema_summary = schema_summary
        self.human_messages = human_messages or []

        # Step state
        self.step_number: int = 0
        self.step_span: Span | None = None
        self.step_start: float = 0.0
        self.decision: CoordinatorDecision | None = None
        self.coordinator_ms: int = 0
        self.move_history: list[str] = []
        self.thread_start: float = time.monotonic()

        # Error tracking
        self.error_count: int = 0

        # Completion signaling
        self.done_event: Event = Event()

        # Agents
        self.coordinator = Coordinator(
            llm=llm,
            model=config.models.coordinator,
            temperature=config.temperatures.coordinator,
            queue=queue,
            session_id=thread.session_id,
            thread_id=thread.id,
        )
        self.worker = Worker(
            llm=llm,
            model=config.models.worker,
            fallback_model=config.models.worker_fallback,
            schema_summary=schema_summary,
            session_db=session_db,
            config=config,
            queue=queue,
            session_id=thread.session_id,
            thread_id=thread.id,
        )

    @property
    def tid(self) -> str:
        return self.thread.id[:8]

    # --- Public API ---

    def start(self):
        """Kick off the thread state machine. Non-blocking — returns immediately."""
        self._run_step()

    def resume(self, human_messages: list[str] | None = None):
        """Resume a stuck thread. Human messages guide the next step."""
        if human_messages:
            self.human_messages = human_messages

        if not self.trace_store.get_spans(self.thread.id):
            self.trace_store.load_trace(self.thread.id, self.thread.session_id)

        self.state.update_thread_status(self.thread.id, ThreadStatus.RUNNING)
        self.start()

    # --- State machine ---

    def _run_step(self):
        """Begin a new coordinator-worker step by scheduling the coordinator call."""
        self.step_number = len(self.trace_store.get_step_spans(self.thread.id)) + 1
        self.step_start = time.monotonic()
        self.step_span = self.trace_store.start_span(
            trace_id=self.thread.id,
            name=f"step_{self.step_number}",
            kind="step",
        )

        if self.step_number == 1:
            self.queue.emit(StreamEvent(
                session_id=self.thread.session_id,
                thread_id=self.thread.id,
                event_type="thread_start",
                message=self.thread.seed_question,
                data={
                    "seed_question": self.thread.seed_question,
                    "motivation": self.thread.motivation,
                    "entry_point": self.thread.entry_point,
                },
            ))

        future = self.queue.schedule(
            fn=self._coordinator_call,
            args=(),
            task_id=f"coord-{self.tid}-{self.step_number}",
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            description=f"Coordinator step {self.step_number}: {self.thread.seed_question[:60]}",
        )
        future.add_done_callback(lambda f: self._safe_callback(self._on_coordinator_done, f))

    def _coordinator_call(self):
        """Run coordinator LLM call. Executes on pool thread."""
        thread_history = self.trace_store.format_thread_history(
            self.thread.id, self.human_messages, running_summary=self.thread.running_summary,
        )
        thread_views = self._get_thread_views()

        t0 = time.monotonic()
        decision, log = self.coordinator.call(
            seed_question=self.thread.seed_question,
            motivation=self.thread.motivation,
            entry_point=self.thread.entry_point,
            schema_summary=self.schema_summary,
            thread_history=thread_history,
        )
        coordinator_ms = round((time.monotonic() - t0) * 1000)
        log["duration_ms"] = coordinator_ms
        return decision, log, coordinator_ms, thread_views

    def _on_coordinator_done(self, future):
        """Process coordinator result and schedule worker or finalize."""
        decision, coord_log, coordinator_ms, thread_views = future.result()

        self.decision = decision
        self.coordinator_ms = coordinator_ms

        self.trace_store.add_event(self.step_span, "llm_call", {
            "agent": "coordinator",
            "model": coord_log["model"],
            "duration_ms": coordinator_ms,
            "input_tokens": coord_log.get("input_tokens"),
            "output_tokens": coord_log.get("output_tokens"),
            "response": coord_log.get("response"),
        })

        logger.info(
            f"Thread {self.thread.id} coordinator: {decision.status.value} "
            f"-> {decision.next_move.value} ({coordinator_ms}ms)"
        )

        if decision.status == CoordinatorStatus.STUCK:
            if self.step_number <= 2:
                logger.warning(
                    f"Thread {self.tid} STUCK on step {self.step_number} — overriding to FORAGE"
                )
                decision.status = CoordinatorStatus.CONTINUE
                decision.next_move = MoveType.FORAGE
                decision.worker_instruction = (
                    f"Try a different exploratory approach to answer: {self.thread.seed_question}"
                )
            else:
                self._finalize("stuck")
                return

        # Initialize worker for this step
        self.worker.start(
            instruction=decision.worker_instruction or "",
            thread_views=thread_views,
        )

        self.queue.emit(StreamEvent(
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            event_type="step_start",
            message=decision.worker_instruction or "",
            data={
                "move": decision.next_move.value,
                "step_number": self.step_number,
                "instruction": decision.worker_instruction or "",
            },
        ))

        self._schedule_worker_call()

    def _schedule_worker_call(self):
        """Submit the next worker LLM call as a future."""
        future = self.queue.schedule(
            fn=self.worker.call,
            args=(),
            task_id=f"worker-{self.tid}-{self.step_number}-{self.worker.attempts}",
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            description=f"Worker call {self.worker.attempts}: {self.worker.instruction[:60]}",
        )
        future.add_done_callback(lambda f: self._safe_callback(self._on_worker_done, f))

    def _on_worker_done(self, future):
        """Process worker LLM response — tool calls or final answer."""
        try:
            response, call_ms = future.result()
        except APITimeoutError:
            logger.warning(f"Worker LLM call timed out for thread {self.thread.id}")
            self.worker.handle_timeout()
            self._schedule_worker_call()
            return

        result = self.worker.handle_response(response, call_ms)
        if result is None:
            self._schedule_worker_call()
        else:
            self._complete_step(result)

    def _complete_step(self, worker_result: WorkerResult):
        """A worker step finished. Record it and decide next action."""
        step_ms = round((time.monotonic() - self.step_start) * 1000)
        worker_ms = step_ms - self.coordinator_ms

        if worker_result.llm_calls:
            for call in worker_result.llm_calls:
                self.trace_store.add_event(self.step_span, "llm_call", call)

        self.step_span.attributes.update({
            "move": self.decision.next_move.value,
            "instruction": self.decision.worker_instruction,
            "result": worker_result.result,
            "coordinator_ms": self.coordinator_ms,
            "worker_ms": worker_ms,
        })
        self.trace_store.end_span(self.step_span)

        logger.info(
            f"Thread {self.thread.id} step {self.step_number} "
            f"({self.decision.next_move.value}): "
            f"coordinator={self.coordinator_ms}ms worker={worker_ms}ms "
            f"total={step_ms}ms"
        )

        self.queue.emit(StreamEvent(
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            event_type="step_complete",
            message=worker_result.result,
            data={
                "step_number": self.step_number,
                "move": self.decision.next_move.value,
                "result": worker_result.result,
            },
        ))

        if self.step_number > 1 and self.step_number % 5 == 0:
            self._maybe_summarize()

        # Move repetition guard
        self.move_history.append(self.decision.next_move.value)
        max_same = self.config.max_repeated_moves
        if (
            len(self.move_history) >= max_same
            and len(set(self.move_history[-max_same:])) == 1
            and self.decision.status != CoordinatorStatus.DONE
        ):
            move_name = self.move_history[-1]
            logger.warning(
                f"Thread {self.thread.id} repeated {move_name} {max_same} times — forcing STUCK"
            )
            self.decision.status = CoordinatorStatus.STUCK
            self.decision.question_for_human = (
                f"Thread has repeated {move_name} {max_same} times without progress. "
                "What direction should the analysis take?"
            )
            self.decision.context = f"Last {max_same} moves: {', '.join(self.move_history[-max_same:])}"
            self._finalize("stuck")
            return

        if self.decision.status == CoordinatorStatus.DONE:
            self._finalize("complete", worker_result=worker_result)
        else:
            self.human_messages = []
            self._run_step()

    def _finalize(
        self,
        status: str,
        worker_result: WorkerResult | None = None,
        error: Exception | None = None,
    ):
        """Finalize thread — complete, stuck, or error. Handles trace, state, events, cleanup."""
        if status == "complete" and worker_result:
            thread_elapsed = round(time.monotonic() - self.thread_start, 2)
            logger.info(
                f"Thread {self.thread.id} complete: "
                f"{self.step_number} steps in {thread_elapsed}s"
            )
            self.trace_store.flush_to_file(self.thread.id, self.thread.session_id)
            self.trace_store.clear_trace(self.thread.id)
            self.state.update_thread_status(
                self.thread.id, ThreadStatus.COMPLETE,
                summary=worker_result.result,
            )
            self.state.dump_session(self.thread.session_id)
            self.queue.emit(StreamEvent(
                session_id=self.thread.session_id,
                thread_id=self.thread.id,
                event_type="thread_complete",
                message=worker_result.result,
                data={
                    "summary": worker_result.result,
                    "total_seconds": thread_elapsed,
                    "step_count": self.step_number,
                },
            ))

        elif status == "stuck":
            decision = self.decision
            if self.step_span:
                self.step_span.attributes.update({
                    "move": decision.next_move.value if decision else "STUCK",
                    "instruction": (decision.question_for_human or "") if decision else "",
                    "result": f"STUCK: {decision.context or ''}" if decision else "STUCK",
                })
                self.trace_store.end_span(self.step_span, status="stuck")

            self.trace_store.flush_to_file(self.thread.id, self.thread.session_id)
            self.trace_store.clear_trace(self.thread.id)
            self.state.update_thread_status(self.thread.id, ThreadStatus.WAITING)
            self.state.dump_session(self.thread.session_id)
            self.queue.emit(StreamEvent(
                session_id=self.thread.session_id,
                thread_id=self.thread.id,
                event_type="thread_waiting",
                message=(decision.question_for_human or "Thread needs guidance.") if decision else "Thread needs guidance.",
                data={
                    "question": decision.question_for_human if decision else None,
                    "context": decision.context if decision else None,
                },
            ))

        elif status == "error" and error:
            error_msg = f"{type(error).__name__}: {error}"
            logger.error(
                f"Thread {self.thread.id} error: {error_msg}",
                exc_info=True,
            )
            try:
                if self.step_span:
                    self.step_span.attributes.update({
                        "move": self.decision.next_move.value if self.decision else "ERROR",
                        "instruction": (self.decision.worker_instruction or "") if self.decision else "",
                        "result": f"Error: {error_msg}",
                        "error": error_msg,
                    })
                    self.trace_store.end_span(self.step_span, status="error")
                self.trace_store.flush_to_file(self.thread.id, self.thread.session_id)
                self.trace_store.clear_trace(self.thread.id)
            except Exception:
                pass

            self.state.update_thread_status(self.thread.id, ThreadStatus.WAITING)
            self.state.dump_session(self.thread.session_id)
            self.queue.emit(StreamEvent(
                session_id=self.thread.session_id,
                thread_id=self.thread.id,
                event_type="thread_waiting",
                message=f"Thread encountered an error: {error_msg}. How should it proceed?",
                data={
                    "question": f"Thread encountered an error: {error_msg}. How should it proceed?",
                    "context": error_msg,
                },
            ))

        # Cleanup: close DB and signal completion
        try:
            self.session_db.close()
        except Exception:
            pass
        self.done_event.set()

    def _safe_callback(self, handler, future):
        """Wrap a done callback to catch exceptions and route to _finalize."""
        try:
            handler(future)
        except Exception as e:
            self.error_count += 1
            if self.error_count < 3:
                logger.warning(
                    f"Thread {self.tid} error (attempt {self.error_count}), retrying step: {e}"
                )
                try:
                    if self.step_span:
                        self.step_span.attributes.update({
                            "move": "ERROR",
                            "result": f"Error: {e}",
                        })
                        self.trace_store.end_span(self.step_span, status="error")
                    self._run_step()
                except Exception:
                    logger.error(f"Thread {self.tid} failed to retry after error", exc_info=True)
                    self._finalize("error", error=e)
            else:
                logger.error(f"Thread {self.thread.id} callback error: {e}", exc_info=True)
                try:
                    self._finalize("error", error=e)
                except Exception:
                    logger.error(f"Thread {self.thread.id} failed to finalize after error", exc_info=True)
                    self.done_event.set()

    def _maybe_summarize(self):
        """Summarize thread history if it's getting long."""
        try:
            summary = self.trace_store.summarize_history(
                trace_id=self.thread.id,
                llm=self.llm,
                model=self.config.models.coordinator,
                seed_question=self.thread.seed_question,
            )
            if summary and not detect_degeneration(summary):
                self.state.update_thread_running_summary(self.thread.id, summary)
                self.thread.running_summary = summary
                logger.info(f"Thread {self.thread.id} history summarized")
            elif summary:
                logger.warning(f"Thread {self.thread.id} summary discarded — degenerate output detected")
        except Exception as e:
            logger.warning(f"History summarization failed: {e}")

    def _get_thread_views(self) -> str:
        """List existing views for this thread."""
        try:
            rows = self.session_db.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_type = 'VIEW' AND table_name LIKE ?
            """, [f"thread_{self.thread.id}_%"]).fetchall()
            if rows:
                return "\n".join(r[0] for r in rows)
        except Exception:
            pass
        return "(none)"

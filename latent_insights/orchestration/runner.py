"""
ThreadRunner — continuation-passing coordinator-worker state machine.

Each LLM call (coordinator, worker turn, summarizer) and each SQL tool
call is submitted to the ``Queue`` as its own task, chained via
``future.add_done_callback``. Pool threads are released between calls so
other analytical threads can make progress — important when the number
of active threads exceeds ``Queue.max_workers``.

Replaces the LangGraph ``StateGraph`` and the blocking while-loop with a
single class. All event emission goes through ``Recorder``; all
persistent state lives on ``InvestigationStore``; the only transient
per-thread state is the handful of instance attributes set in
``_start_step``.
"""

import logging
import time
from concurrent.futures import Future
from datetime import timezone
from enum import Enum
from threading import Event, Lock
from typing import Any

from latent_insights.agents.coordinator import Coordinator
from latent_insights.agents.worker import Worker
from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMClient, is_transient_llm_error
from latent_insights.core.parsing import detect_degeneration
from latent_insights.core.queue import Queue
from latent_insights.core.recorder import Recorder
from latent_insights.core.store import InvestigationStore, Span
from latent_insights.models import (
    CoordinatorDecision,
    CoordinatorStatus,
    MoveType,
    StreamEvent,
    Thread,
    ThreadStatus,
)

logger = logging.getLogger(__name__)


class RunnerMode(str, Enum):
    LOOP_UNTIL_DONE = "loop_until_done"
    STEP_AND_PAUSE = "step_and_pause"


class ThreadRunner:
    """Drives one analytical thread through its coordinator-worker lifecycle.

    Continuation-passing: the control flow is a chain of small callbacks,
    each submitted to the ``Queue``'s thread pool. Pool slots are held
    only for the duration of a single LLM call or SQL execution.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        llm: LLMClient,
        session_db: Any,
        queue: Queue,
        store: InvestigationStore,
        thread: Thread,
        schema_summary: str,
        human_messages: list | None = None,
        mode: RunnerMode = RunnerMode.LOOP_UNTIL_DONE,
    ):
        self.config = config
        self.queue = queue
        self.store = store
        self.thread = thread
        self.session_db = session_db
        self.schema_summary = schema_summary
        self.human_messages: list = list(human_messages or [])
        self.mode = mode

        self.recorder = Recorder(store, queue, thread.session_id, thread.id)
        self.done_event: Event = Event()

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
        self.llm = llm

        # Persistent loop state (survives across callbacks)
        self.step_number: int = 0
        self.move_history: list[str] = []
        self.thread_start: float = 0.0

        # Per-step state (reset by _start_step)
        self._span: Span | None = None
        self._decision: CoordinatorDecision | None = None
        self._coordinator_ms: int = 0
        self._step_start: float = 0.0

        # Per-turn tool-call coordination (multiple SQL tasks per LLM turn)
        self._pending_sql: int = 0
        self._pending_sql_lock: Lock = Lock()

    @property
    def tid(self) -> str:
        return self.thread.id[:8]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Kick off the thread. Non-blocking — returns immediately."""
        self.thread_start = time.monotonic()
        self._start_step()

    def resume(self, human_messages: list | None = None):
        """Resume a WAITING/COMPLETE thread."""
        if human_messages:
            self.human_messages = list(human_messages)

        if not self.store.get_spans(self.thread.id):
            self.store.load_session(self.thread.session_id)

        self.step_number = len(self.store.get_step_spans(self.thread.id))
        self.store.update_thread_status(self.thread.id, ThreadStatus.RUNNING)
        self.recorder.thread_resumed(self.step_number, self.human_messages)
        self.start()

    # ------------------------------------------------------------------
    # Step lifecycle — each blocking call is scheduled on the pool,
    # callbacks chain the next action.
    # ------------------------------------------------------------------

    def _start_step(self):
        """Begin a new coordinator step. Schedules the coordinator LLM call."""
        self.step_number += 1
        self._step_start = time.monotonic()

        # Drain injected messages from the interrupt API
        injected = self.store.drain_pending_messages(self.thread.id)
        if injected:
            self.human_messages.extend(injected)
            logger.info(
                f"Thread {self.tid} received {len(injected)} injected message(s)"
            )

        # Start span for this step
        self._span = self.store.start_span(
            trace_id=self.thread.id,
            name=f"step_{self.step_number}",
            kind="step",
        )

        # thread_start fires on step 1, stamped with created_at so the UI
        # orders it before the first step_start (see graph.py history).
        if self.step_number == 1:
            thread_obj = self.store.get_thread(self.thread.id)
            if thread_obj:
                created_ts = thread_obj.created_at.replace(
                    tzinfo=timezone.utc
                ).timestamp()
            else:
                created_ts = time.time()
            self.recorder.thread_start(
                self.thread.seed_question,
                self.thread.motivation,
                self.thread.entry_point,
                timestamp=created_ts,
            )

        # Record human messages on the span so they sort into the step
        # timeline with their original wall-clock timestamp.
        for msg in self.human_messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                target = msg.get("target", "thread")
                ts = msg.get("timestamp")
            else:
                content = str(msg)
                target = "thread"
                ts = None
            self.recorder.human_message(
                self._span, content=content, target=target, timestamp=ts,
            )

        # Schedule the coordinator LLM call as a pool task
        self._schedule(
            fn=self._do_coordinator_call,
            callback=self._on_coordinator_done,
            task_id=f"coord-{self.tid}-{self.step_number}",
            description=f"Coordinator step {self.step_number}: {self.thread.seed_question[:60]}",
        )

    def _do_coordinator_call(self):
        """Runs on a pool thread — blocks only for the coordinator LLM call."""
        thread_history = self.store.format_thread_history(
            self.thread.id,
            self.human_messages,
            running_summary=self.thread.running_summary,
        )
        t0 = time.monotonic()
        decision, log = self.coordinator.call(
            seed_question=self.thread.seed_question,
            motivation=self.thread.motivation,
            entry_point=self.thread.entry_point,
            schema_summary=self.schema_summary,
            thread_history=thread_history,
        )
        log["duration_ms"] = round((time.monotonic() - t0) * 1000)
        return decision, log

    def _on_coordinator_done(self, future: Future):
        """Callback after coordinator LLM call. Decide worker / finalize."""
        decision, coord_log = future.result()
        coordinator_ms = coord_log["duration_ms"]

        self._decision = decision
        self._coordinator_ms = coordinator_ms

        logger.info(
            f"Thread {self.thread.id} coordinator: {decision.status.value} "
            f"-> {decision.next_move.value} ({coordinator_ms}ms)"
        )

        # Early-stuck override at step <= 2 (before emitting SSE so the
        # event's `move` matches the committed final move).
        if decision.status == CoordinatorStatus.STUCK and self.step_number <= 2:
            logger.warning(
                f"Thread {self.tid} STUCK on step {self.step_number} — overriding to FORAGE"
            )
            decision.status = CoordinatorStatus.CONTINUE
            decision.next_move = MoveType.FORAGE
            decision.worker_instruction = (
                f"Try a different exploratory approach to answer: {self.thread.seed_question}"
            )

        final_move = decision.next_move.value

        self.recorder.llm_call(
            self._span,
            step_number=self.step_number,
            move=final_move,
            agent="coordinator",
            model=coord_log["model"],
            input_tokens=coord_log.get("input_tokens"),
            output_tokens=coord_log.get("output_tokens"),
            duration_ms=coordinator_ms,
            response=coord_log.get("response", ""),
        )
        self.recorder.step_start(
            self.step_number, final_move, decision.worker_instruction or "",
        )

        # Genuinely STUCK (post step 2)
        if decision.status == CoordinatorStatus.STUCK:
            self.store.end_span(self._span, status="stuck")
            step_count = len(self.store.get_step_spans(self.thread.id))
            self.recorder.thread_waiting(
                reason="coordinator_stuck",
                question=decision.question_for_human or "Thread needs guidance.",
                context=decision.context,
                step_count=step_count,
            )
            self._finish()
            return

        # Move repetition guard — commits the move to history first so the
        # guard sees it. Trip on DONE only if the last N moves are equal.
        self.move_history.append(final_move)
        max_same = self.config.max_repeated_moves
        if (
            len(self.move_history) >= max_same
            and len(set(self.move_history[-max_same:])) == 1
            and decision.status != CoordinatorStatus.DONE
        ):
            logger.warning(
                f"Thread {self.thread.id} repeated {final_move} {max_same} times — forcing STUCK"
            )
            self.store.end_span(self._span, status="stuck")
            step_count = len(self.store.get_step_spans(self.thread.id))
            self.recorder.thread_waiting(
                reason="repeated_moves",
                question="Thread repeated the same move too many times — needs guidance.",
                step_count=step_count,
            )
            self._finish()
            return

        # Initialize worker for this step and schedule its first LLM call
        thread_views = self._get_thread_views()
        self.worker.start(
            instruction=decision.worker_instruction or "",
            thread_views=thread_views,
            step_number=self.step_number,
            move=final_move,
        )
        self._schedule_worker_call()

    # --- Worker turn: LLM call + optional tool calls ----------------------

    def _schedule_worker_call(self):
        """Schedule one worker LLM call as a pool task."""
        self._schedule(
            fn=self.worker.call,
            callback=self._on_worker_llm_done,
            task_id=f"worker-{self.tid}-{self.step_number}-{self.worker.attempts}",
            description=(
                f"Worker LLM call {self.worker.attempts}: "
                f"{self.worker.instruction[:60]}"
            ),
        )

    def _on_worker_llm_done(self, future: Future):
        """Callback after a worker LLM turn. Either schedules SQL tasks for
        the tool calls, loops back for another LLM call, or completes the
        step with a ``WorkerResult``.
        """
        response, call_ms = future.result()

        if response.tool_calls:
            # Split: extract SQL tasks and schedule each as its own pool task.
            sql_tasks = self.worker.prepare_tool_calls(response, call_ms)

            # If nothing runnable came out (all malformed / unknown tools),
            # the worker already appended tool error messages — loop back
            # to the LLM for another turn.
            if not sql_tasks:
                self._schedule_worker_call()
                return

            # Initialize per-turn SQL coordination. The last SQL to finish
            # will apply the error guardrails and schedule the next LLM call.
            with self._pending_sql_lock:
                self._pending_sql = len(sql_tasks)

            for idx, task in enumerate(sql_tasks):
                self._schedule(
                    fn=self._do_sql,
                    args=(task["tool_call_id"], task["sql"]),
                    callback=self._on_sql_done,
                    task_id=(
                        f"sql-{self.tid}-{self.step_number}-"
                        f"{self.worker.attempts}-{idx}"
                    ),
                    description=f"SQL: {task['sql'][:60]}",
                )
            return

        # No tool calls — final answer path
        result = self.worker.handle_final(response, call_ms)
        if result is None:
            # Retry-on-malformed-JSON / empty-response path
            self._schedule_worker_call()
        else:
            self._complete_step(result)

    def _do_sql(self, tool_call_id: str, sql: str):
        """Runs on a pool thread — blocks only for this one SQL execution."""
        t0 = time.monotonic()
        result_text = Worker.execute_sql(self.session_db, sql)
        sql_ms = round((time.monotonic() - t0) * 1000)
        return tool_call_id, sql, result_text, sql_ms

    def _on_sql_done(self, future: Future):
        """Callback after one SQL execution. Records the result; when all
        SQL for this LLM turn have completed, apply error guardrails and
        schedule the next worker LLM call.
        """
        tool_call_id, sql, result_text, sql_ms = future.result()
        self.worker.record_tool_result(tool_call_id, sql, result_text, sql_ms)

        with self._pending_sql_lock:
            self._pending_sql -= 1
            last = self._pending_sql == 0

        if last:
            self.worker.apply_error_guardrails()
            self._schedule_worker_call()

    # --- Step completion / thread finalization ---------------------------

    def _complete_step(self, result):
        """Worker turn produced a final result. Close span, emit SSE, decide next."""
        worker_ms = round((time.monotonic() - self._step_start) * 1000)

        if result.llm_calls:
            for call in result.llm_calls:
                self.store.add_event(self._span, call["type"], call)
        self._span.attributes.update({
            "move": self._decision.next_move.value,
            "instruction": self._decision.worker_instruction,
            "result": result.result,
            "coordinator_ms": self._coordinator_ms,
            "worker_ms": worker_ms,
        })
        self.store.end_span(self._span)

        logger.info(
            f"Thread {self.thread.id} step {self.step_number} "
            f"({self._decision.next_move.value}): "
            f"coordinator={self._coordinator_ms}ms worker={worker_ms}ms"
        )

        self.recorder.step_complete(
            self.step_number,
            self._decision.next_move.value,
            self._decision.worker_instruction or "",
            result.result,
            self._coordinator_ms + worker_ms,
        )

        # Human messages are consumed by the step that saw them.
        self.human_messages = []

        # HITL: pause after one step
        if self.mode == RunnerMode.STEP_AND_PAUSE:
            thread_obj = self.store.get_thread(self.thread.id)
            if thread_obj and thread_obj.status != ThreadStatus.COMPLETE:
                step_count = len(self.store.get_step_spans(self.thread.id))
                self.store.update_thread_status(self.thread.id, ThreadStatus.WAITING)
                self.store.save_session(self.thread.session_id)
                self.queue.emit(StreamEvent(
                    session_id=self.thread.session_id,
                    thread_id=self.thread.id,
                    event_type="thread_waiting",
                    message=(
                        f"Step complete ({self._decision.next_move.value}). "
                        "Review and send a message to continue."
                    ),
                    data={
                        "pattern": "human_in_the_loop",
                        "last_move": self._decision.next_move.value,
                        "last_result": result.result,
                        "step_number": step_count,
                    },
                ))
            self._finish()
            return

        # DONE: finalize
        if self._decision.status == CoordinatorStatus.DONE:
            thread_elapsed = round((time.monotonic() - self.thread_start) * 1000)
            logger.info(
                f"Thread {self.thread.id} complete: {self.step_number} steps in {thread_elapsed}ms"
            )
            self.recorder.thread_complete(
                result.result, thread_elapsed, self.step_number,
            )
            self._finish()
            return

        # Periodic summarization every 5 steps — scheduled as its own LLM task.
        if self.step_number > 1 and self.step_number % 5 == 0:
            self._schedule_summarize()
            return

        # Continue with the next coordinator step
        self._start_step()

    # --- Summarization (optional LLM task) -------------------------------

    def _schedule_summarize(self):
        """Schedule the history-summary LLM call as its own pool task,
        then continue with the next coordinator step regardless of outcome.
        """
        self._schedule(
            fn=self._do_summarize,
            callback=self._on_summarize_done,
            task_id=f"summarize-{self.tid}-{self.step_number}",
            description=f"Summarize history at step {self.step_number}",
        )

    def _do_summarize(self):
        return self.store.summarize_history(
            trace_id=self.thread.id,
            llm=self.llm,
            model=self.config.models.coordinator,
            seed_question=self.thread.seed_question,
        )

    def _on_summarize_done(self, future: Future):
        try:
            summary = future.result()
            if summary and not detect_degeneration(summary):
                self.store.update_thread_running_summary(self.thread.id, summary)
                logger.info(f"Thread {self.thread.id} history summarized")
            elif summary:
                logger.warning(
                    f"Thread {self.thread.id} summary discarded — degenerate output"
                )
        except Exception as e:
            logger.warning(f"History summarization failed: {e}")
        # Always continue the loop — summarization failure is non-fatal.
        self._start_step()

    # ------------------------------------------------------------------
    # Scheduling helper: wraps queue.schedule and routes callback errors
    # through _handle_error so a single try/except in each callback isn't
    # needed.
    # ------------------------------------------------------------------

    def _schedule(
        self,
        *,
        fn,
        callback,
        task_id: str,
        description: str,
        args: tuple = (),
    ) -> Future:
        future = self.queue.schedule(
            fn=fn,
            args=args,
            task_id=task_id,
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            description=description,
        )
        future.add_done_callback(lambda f: self._safe_callback(callback, f))
        return future

    def _safe_callback(self, handler, future: Future):
        """Invoke a callback, routing any exception (from the pool task OR
        the callback itself) through the shared error path.
        """
        try:
            handler(future)
        except Exception as e:
            self._handle_error(e)

    # ------------------------------------------------------------------
    # Error + finalization helpers
    # ------------------------------------------------------------------

    def _handle_error(self, e: Exception):
        """Route any loop error to thread_waiting with a classified reason."""
        if self.done_event.is_set():
            # Already finalized by a concurrent path — avoid double-emit.
            logger.debug(f"Thread {self.thread.id} error after finalize: {e}")
            return

        reason = "retry_exhausted" if is_transient_llm_error(e) else "unexpected_error"
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(
            f"Thread {self.thread.id} loop error ({reason}): {error_msg}",
            exc_info=True,
        )

        try:
            question = (
                "The LLM provider was unreachable after multiple retries. "
                "Send a message when you want the thread to try again."
                if reason == "retry_exhausted"
                else f"Thread encountered an error: {error_msg}"
            )
            step_count = len(self.store.get_step_spans(self.thread.id))
            self.recorder.thread_waiting(
                reason=reason,
                question=question,
                context=error_msg,
                error=error_msg,
                step_count=step_count,
                span_status="error",
            )
        except Exception:
            logger.error(
                f"Thread {self.thread.id} failed to finalize after loop error",
                exc_info=True,
            )
        finally:
            self._finish()

    def _finish(self):
        """Release DB, set done_event. Idempotent."""
        if self.done_event.is_set():
            return
        try:
            self.session_db.close()
        except Exception:
            pass
        self.done_event.set()

    def _get_thread_views(self) -> str:
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

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
from latent_insights.core.llm import (
    LLMClient,
    is_context_length_error,
    is_transient_llm_error,
)
from latent_insights.core.parsing import detect_degeneration
from latent_insights.core.queue import Queue
from latent_insights.core.recorder import Recorder
from latent_insights.core.store import InvestigationStore, Step
from latent_insights.models import (
    CoordinatorDecision,
    CoordinatorStatus,
    MoveType,
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
        mode: RunnerMode = RunnerMode.LOOP_UNTIL_DONE,
        on_done: "callable | None" = None,
    ):
        self.config = config
        self.queue = queue
        self.store = store
        self.thread = thread
        self.session_db = session_db
        self.schema_summary = schema_summary
        self.mode = mode
        # One-shot completion hook, invoked once from ``_finish`` after
        # any terminal path (complete / waiting / error). Patterns that
        # coordinate across multiple threads (e.g. fan-out synthesis)
        # register a callback here so they don't have to hold a pool
        # worker blocked on ``done_event.wait``.
        self.on_done = on_done

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
        # True once ``thread_start`` has been emitted — controls the
        # one-shot emission even when the first few rows committed are
        # HUMAN_INPUT steps drained from the pending queue.
        self._thread_start_emitted: bool = False

        # Per-step state (reset by _start_step)
        self._step: Step | None = None
        self._decision: CoordinatorDecision | None = None
        self._coordinator_ms: int = 0
        self._step_start: float = 0.0

        # Per-turn tool-call coordination (multiple SQL tasks per LLM turn)
        self._pending_sql: int = 0
        self._pending_sql_lock: Lock = Lock()

        # Context-length recovery budget. When the model says the prompt
        # is too large, we close the current step as an error row whose
        # ``result`` explains the failure — ``format_thread_history``
        # surfaces that result to the next coordinator call via the same
        # path any other step's result uses, so the coordinator sees the
        # overflow and picks a simpler next move. Capped at
        # ``_max_context_recoveries`` per thread to bound retries.
        self._context_recoveries: int = 0
        self._max_context_recoveries: int = 2

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

    def resume(self):
        """Resume a WAITING/COMPLETE thread.

        The caller is responsible for pushing any human guidance to the
        store's pending queue **before** calling resume — the runner
        drains pending messages into ``HUMAN_INPUT`` steps on its first
        ``_start_step``, ahead of the coordinator step.
        """
        if not self.store.get_steps(self.thread.id):
            self.store.load_session(self.thread.session_id)

        self.step_number = len(self.store.get_steps(self.thread.id))
        # thread_start already fired on the original run; don't re-fire.
        self._thread_start_emitted = True
        self.store.update_thread_status(self.thread.id, ThreadStatus.RUNNING)
        self.recorder.thread_resumed(self.step_number)
        self.start()

    # ------------------------------------------------------------------
    # Step lifecycle — each blocking call is scheduled on the pool,
    # callbacks chain the next action.
    # ------------------------------------------------------------------

    def _start_step(self):
        """Begin a new coordinator step. Schedules the coordinator LLM call.

        Order matters: emit ``thread_start`` first (if this is the very
        first step), then drain any queued human messages as
        ``HUMAN_INPUT`` steps, then commit the coordinator's own step
        row. That way the timeline reads:
        ``thread_start → [HUMAN_INPUT steps] → coordinator step`` — the
        human guidance lands ahead of the coordinator move that will
        react to it.
        """
        # 1. thread_start (one-shot, on the first call).
        if not self._thread_start_emitted:
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
            self._thread_start_emitted = True

        # 2. Any pre-queued human messages become HUMAN_INPUT steps
        # before the coordinator step gets created.
        self._drain_pending_as_steps()

        # 3. Start the coordinator step row. Sync self.step_number to
        # whatever the store assigned so the counter stays consistent
        # regardless of how many HUMAN_INPUT steps were drained above.
        self._step = self.store.start_step(self.thread.id)
        self.step_number = self._step.step_number
        self._step_start = time.monotonic()

        # 4. Schedule the coordinator LLM call as a pool task.
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
        # Human input posted mid-LLM-call? Flush the in-flight step and
        # pivot — the coordinator's decision is discarded; a fresh
        # coordinator step runs with the HUMAN_INPUT step in history.
        if self._flush_and_pivot():
            return
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

        # Stamp the chosen move + instruction on the in-progress step row
        # so downstream consumers (history formatting, API snapshots if
        # the thread terminates mid-step) see the committed move.
        self._step.move = final_move
        self._step.instruction = decision.worker_instruction or ""

        self.recorder.llm_call(
            self._step,
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
            self.step_number,
            final_move,
            decision.worker_instruction or "",
            assessment=decision.assessment or "",
            rationale=decision.rationale or "",
            status=decision.status.value,
        )

        # Genuinely STUCK (post step 2)
        if decision.status == CoordinatorStatus.STUCK:
            self.store.end_step(self._step, status="stuck")
            step_count = len(self.store.get_steps(self.thread.id))
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
            self.store.end_step(self._step, status="stuck")
            step_count = len(self.store.get_steps(self.thread.id))
            self.recorder.thread_waiting(
                reason="repeated_moves",
                question="Thread repeated the same move too many times — needs guidance.",
                step_count=step_count,
            )
            self._finish()
            return

        # Initialize worker for this step and schedule its first LLM call.
        # SYNTHESIZE is handled specially: the worker gets the condensed
        # thread history (the same view the coordinator sees) and no tool,
        # so it can only summarize what the previous steps found — no new
        # queries.
        thread_views = self._get_thread_views()
        worker_history = ""
        if final_move == "SYNTHESIZE":
            worker_history = self.store.format_thread_history(
                self.thread.id,
                running_summary=self.thread.running_summary,
            )
        self.worker.start(
            instruction=decision.worker_instruction or "",
            thread_views=thread_views,
            step_number=self.step_number,
            move=final_move,
            thread_history=worker_history,
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
        if self._flush_and_pivot():
            return
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

        If human input landed during SQL execution we still record this
        SQL's result (losing it would be wasteful) but the flush-and-pivot
        check fires on the LAST SQL of the turn so the pivot happens in
        place of the next worker LLM call.
        """
        tool_call_id, sql, result_text, sql_ms = future.result()
        self.worker.record_tool_result(tool_call_id, sql, result_text, sql_ms)

        with self._pending_sql_lock:
            self._pending_sql -= 1
            last = self._pending_sql == 0

        if last:
            if self._flush_and_pivot():
                return
            self.worker.apply_error_guardrails()
            self._schedule_worker_call()

    # --- Step completion / thread finalization ---------------------------

    def _complete_step(self, result):
        """Worker turn produced a final result. Finalize step, emit SSE, decide next."""
        worker_ms = round((time.monotonic() - self._step_start) * 1000)

        # The worker batched its per-turn events into ``result.llm_calls``
        # (llm_call + tool_call records). They're already in the canonical
        # flat ``StepEvent`` shape, so just append each one.
        if result.llm_calls:
            for call in result.llm_calls:
                self.store.add_event(self._step, call)
        # Promote the final result + view onto the step row (move and
        # instruction were already stamped in _on_coordinator_done).
        self._step.result = result.result
        if getattr(result, "view_requested", None):
            view = result.view_requested
            if isinstance(view, dict):
                self._step.view_created = view.get("name")
        self.store.end_step(self._step)

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

        # HITL: pause after one step. Route through recorder.thread_waiting
        # so the pause is represented as a WAITING_FOR_HUMAN step in the
        # timeline — same shape as any other terminal waiting state.
        if self.mode == RunnerMode.STEP_AND_PAUSE:
            thread_obj = self.store.get_thread(self.thread.id)
            if thread_obj and thread_obj.status != ThreadStatus.COMPLETE:
                step_count = len(self.store.get_steps(self.thread.id))
                question = (
                    f"Step complete ({self._decision.next_move.value}). "
                    "Review and send a message to continue."
                )
                self.recorder.thread_waiting(
                    reason="human_review",
                    question=question,
                    context=result.result,
                    step_count=step_count,
                    span_status="ok",
                )
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

        # Periodic summarization at the configured cadence. ``<= 0`` or
        # an unset interval disables it entirely.
        every = self.config.summarize_every_steps
        if every and every > 0 and self.step_number > 1 and self.step_number % every == 0:
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
            thread_id=self.thread.id,
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
        # If human input arrived while summarizing, pivot now; otherwise
        # continue the loop — summarization failure is non-fatal.
        if self._flush_and_pivot():
            return
        self._start_step()

    # ------------------------------------------------------------------
    # Mixed-initiative: commit pending human messages as HUMAN_INPUT
    # steps, and flush-and-pivot when input arrives mid-step.
    # ------------------------------------------------------------------

    def _drain_pending_as_steps(self) -> bool:
        """Drain the store's pending-message queue into HUMAN_INPUT steps.

        Each drained message becomes its own ``HUMAN_INPUT`` step via
        ``recorder.human_input_step`` — visible in the timeline, in
        snapshots, and on SSE just like any other step. Returns True if
        anything was committed.
        """
        injected = self.store.drain_pending_messages(self.thread.id)
        if not injected:
            return False
        for msg in injected:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                target = msg.get("target", "thread")
                ts = msg.get("timestamp")
            else:
                content = str(msg)
                target = "thread"
                ts = None
            self.recorder.human_input_step(
                content, target=target, timestamp=ts,
            )
        logger.info(
            f"Thread {self.tid} committed {len(injected)} HUMAN_INPUT step(s)"
        )
        return True

    def _flush_and_pivot(self) -> bool:
        """Interrupt the current step with any pending human input.

        Called at the top of every scheduling callback. If human input
        has been posted since the last callback:
          1. Close the in-flight step (if still open) with
             ``status="flushed"``.
          2. Drain the pending queue into HUMAN_INPUT step(s).
          3. Kick off a fresh coordinator step.
          4. Return True so the caller short-circuits its default
             "schedule the next natural task" flow.

        If no pending input, returns False and the callback proceeds
        normally.
        """
        if not self.store.has_pending_messages(self.thread.id):
            return False

        # Close the in-flight step with a self-describing result. This
        # becomes visible in format_thread_history for the next
        # coordinator call.
        if self._step is not None and self._step.end_time is None:
            if not self._step.result:
                self._step.result = (
                    "Step flushed — human input received mid-step"
                )
            self.store.end_step(self._step, status="flushed")
            self._step = None

        self._drain_pending_as_steps()

        # Pivot to a fresh coordinator step. The newly-committed
        # HUMAN_INPUT step(s) are already in the store, so the
        # coordinator sees them via format_thread_history.
        self._start_step()
        return True

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
        """Route any loop error to a recovery path or ``thread_waiting``.

        Context-length errors get a recovery budget: we close the current
        step with an error result, inject a hint into the thread history
        as a ``human_message``, and schedule the next step. The
        coordinator sees the hint in ``format_thread_history`` and can
        pick a simpler move. Once the budget is exhausted we fall
        through to ``thread_waiting``.
        """
        if self.done_event.is_set():
            # Already finalized by a concurrent path — avoid double-emit.
            logger.debug(f"Thread {self.thread.id} error after finalize: {e}")
            return

        if (
            is_context_length_error(e)
            and self._context_recoveries < self._max_context_recoveries
        ):
            self._context_recoveries += 1
            logger.warning(
                f"Thread {self.thread.id} context-length exceeded "
                f"(recovery {self._context_recoveries}/{self._max_context_recoveries}); "
                "injecting hint and continuing"
            )
            self._recover_from_context_overflow(e)
            return

        reason = "retry_exhausted" if is_transient_llm_error(e) else "unexpected_error"
        if is_context_length_error(e):
            reason = "context_exhausted"
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(
            f"Thread {self.thread.id} loop error ({reason}): {error_msg}",
            exc_info=True,
        )

        try:
            if reason == "retry_exhausted":
                question = (
                    "The LLM provider was unreachable after multiple retries. "
                    "Send a message when you want the thread to try again."
                )
            elif reason == "context_exhausted":
                question = (
                    "This thread's context has grown too large for the model "
                    "even after compressing its history. Send a narrower "
                    "follow-up question and it will restart."
                )
            else:
                question = f"Thread encountered an error: {error_msg}"
            step_count = len(self.store.get_steps(self.thread.id))
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

    def _recover_from_context_overflow(self, exc: Exception):
        """Close the failing step and continue the loop.

        The step's ``result`` is set to a self-describing message. The
        coordinator reads it via ``format_thread_history`` on the next
        step — the same path any other step result takes — and can pick
        a simpler next move without a separate hint channel.
        """
        error_msg = f"{type(exc).__name__}: {exc}"

        if self._step is not None and self._step.end_time is None:
            self._step.result = (
                "Context overflow: this step's prompt exceeded the model's "
                "context window. The next move should narrow the data — "
                "fewer columns, tighter filters, aggressive aggregation, or "
                "LIMIT — so the tool result stays short. "
                f"Error: {error_msg}"
            )
            self.store.end_step(self._step, status="error")

        # Persist so a crash mid-recovery still leaves a coherent snapshot.
        self.store.save_session(self.thread.session_id)

        # Continue with the next step.
        self._start_step()

    def _finish(self):
        """Release DB, set done_event, fire on_done. Idempotent."""
        if self.done_event.is_set():
            return
        try:
            self.session_db.close()
        except Exception:
            pass
        self.done_event.set()
        if self.on_done is not None:
            try:
                self.on_done()
            except Exception:
                logger.exception(
                    f"Thread {self.thread.id} on_done callback raised; "
                    "continuing"
                )

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

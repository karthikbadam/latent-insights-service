"""
ThreadLoop — plain while-loop driving the coordinator-worker analysis cycle.

Replaces the LangGraph StateGraph (graph.py) + ThreadRunner (thread.py) with
a straightforward loop. All event emission goes through ``Recorder``, all
state lives on the ``InvestigationStore`` + local variables.
"""

import logging
import time
from dataclasses import asdict
from datetime import timezone
from enum import Enum
from threading import Event
from typing import Any

from latent_insights.agents.coordinator import Coordinator
from latent_insights.agents.worker import Worker
from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMClient
from latent_insights.core.parsing import detect_degeneration
from latent_insights.core.queue import Queue
from latent_insights.core.recorder import Recorder
from latent_insights.core.store import InvestigationStore
from latent_insights.models import (
    CoordinatorStatus,
    MoveType,
    StreamEvent,
    Thread,
    ThreadStatus,
)

logger = logging.getLogger(__name__)


class LoopMode(str, Enum):
    LOOP_UNTIL_DONE = "loop_until_done"
    STEP_AND_PAUSE = "step_and_pause"


class ThreadLoop:
    """Drives one analytical thread through its coordinator-worker lifecycle.

    A plain while-loop that:
    - Calls the coordinator to decide the next move
    - Runs the worker to execute it
    - Handles stuck detection, move repetition guards, HITL pausing
    - Emits all events through a Recorder (dual span + SSE writes)
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
        mode: LoopMode = LoopMode.LOOP_UNTIL_DONE,
    ):
        self.config = config
        self.queue = queue
        self.store = store
        self.thread = thread
        self.session_db = session_db
        self.schema_summary = schema_summary
        self.human_messages = human_messages or []
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

    @property
    def tid(self) -> str:
        return self.thread.id[:8]

    # --- Public API ---

    def start(self):
        """Kick off the thread loop. Non-blocking — returns immediately."""
        future = self.queue.schedule(
            fn=self._run,
            args=(),
            task_id=f"thread-{self.tid}",
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            description=f"Thread: {self.thread.seed_question[:60]}",
        )
        future.add_done_callback(lambda f: self._on_done(f))

    def resume(self, human_messages: list | None = None):
        """Resume a waiting/complete thread with optional human messages."""
        if human_messages:
            self.human_messages = human_messages

        # Reload spans if not in memory
        if not self.store.get_spans(self.thread.id):
            self.store.load_session(self.thread.session_id)

        existing_steps = len(self.store.get_step_spans(self.thread.id))
        self._resume_step_offset = existing_steps

        self.store.update_thread_status(self.thread.id, ThreadStatus.RUNNING)
        self.recorder.thread_resumed(existing_steps, self.human_messages)
        self.start()

    # --- Internal ---

    def _run(self):
        """Execute the analysis loop. Runs on a pool thread."""
        try:
            self._loop()
        except Exception as e:
            self._handle_error(e)
        finally:
            try:
                self.session_db.close()
            except Exception:
                pass
            self.done_event.set()

    def _loop(self):
        thread_id = self.thread.id
        session_id = self.thread.session_id
        step_number = getattr(self, "_resume_step_offset", 0)
        move_history: list[str] = []
        human_messages = list(self.human_messages)
        thread_start = time.monotonic()

        while True:
            step_number += 1

            # Drain pending messages injected mid-run
            injected = self.store.drain_pending_messages(thread_id)
            if injected:
                human_messages.extend(injected)
                logger.info(f"Thread {self.tid} received {len(injected)} injected message(s)")

            # Format history from spans
            thread_history = self.store.format_thread_history(
                thread_id, human_messages,
            )

            # Emit thread_start on first step
            if step_number == 1:
                thread_obj = self.store.get_thread(thread_id)
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

            # Start trace span for this step
            span = self.store.start_span(
                trace_id=thread_id, name=f"step_{step_number}", kind="step",
            )

            # Record human messages on the span
            for msg in human_messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    target = msg.get("target", "thread")
                    ts = msg.get("timestamp")
                else:
                    content = str(msg)
                    target = "thread"
                    ts = None
                self.recorder.human_message(
                    span, content=content, target=target, timestamp=ts,
                )

            # --- Coordinator ---
            t0 = time.monotonic()
            decision, coord_log = self.coordinator.call(
                seed_question=self.thread.seed_question,
                motivation=self.thread.motivation,
                entry_point=self.thread.entry_point,
                schema_summary=self.schema_summary,
                thread_history=thread_history,
            )
            coordinator_ms = round((time.monotonic() - t0) * 1000)
            coord_log["duration_ms"] = coordinator_ms

            logger.info(
                f"Thread {thread_id} coordinator: {decision.status.value} "
                f"-> {decision.next_move.value} ({coordinator_ms}ms)"
            )

            # Early stuck override at step <= 2
            if decision.status == CoordinatorStatus.STUCK and step_number <= 2:
                logger.warning(f"Thread {self.tid} STUCK on step {step_number} — overriding to FORAGE")
                decision.status = CoordinatorStatus.CONTINUE
                decision.next_move = MoveType.FORAGE
                decision.worker_instruction = (
                    f"Try a different exploratory approach to answer: {self.thread.seed_question}"
                )

            final_move = decision.next_move.value

            # Record coordinator LLM call
            self.recorder.llm_call(
                span,
                step_number=step_number,
                move=final_move,
                agent="coordinator",
                model=coord_log["model"],
                input_tokens=coord_log.get("input_tokens"),
                output_tokens=coord_log.get("output_tokens"),
                duration_ms=coordinator_ms,
                response=coord_log.get("response", ""),
            )

            self.recorder.step_start(step_number, final_move, decision.worker_instruction or "")

            # --- STUCK (post step 2) ---
            if decision.status == CoordinatorStatus.STUCK:
                step_count = len(self.store.get_step_spans(thread_id))
                self.store.end_span(span, status="stuck")
                self.recorder.thread_waiting(
                    reason="coordinator_stuck",
                    question=decision.question_for_human or "Thread needs guidance.",
                    context=decision.context,
                    step_count=step_count,
                )
                return

            # --- Move repetition guard ---
            move_history.append(final_move)
            max_same = self.config.max_repeated_moves
            if (
                len(move_history) >= max_same
                and len(set(move_history[-max_same:])) == 1
                and decision.status != CoordinatorStatus.DONE
            ):
                logger.warning(
                    f"Thread {thread_id} repeated {final_move} {max_same} times — forcing STUCK"
                )
                step_count = len(self.store.get_step_spans(thread_id))
                self.store.end_span(span, status="stuck")
                self.recorder.thread_waiting(
                    reason="repeated_moves",
                    question="Thread repeated the same move too many times — needs guidance.",
                    step_count=step_count,
                )
                return

            # --- Worker ---
            thread_views = self._get_thread_views()
            self.worker.start(
                instruction=decision.worker_instruction or "",
                thread_views=thread_views,
                step_number=step_number,
                move=final_move,
            )

            step_start = time.monotonic()
            result = None
            while result is None:
                response, call_ms = self.worker.call()
                result = self.worker.handle_response(response, call_ms)
            worker_ms = round((time.monotonic() - step_start) * 1000)

            # Record worker events on span
            if result.llm_calls:
                for call in result.llm_calls:
                    self.store.add_event(span, call["type"], call)
            span.attributes.update({
                "move": final_move,
                "instruction": decision.worker_instruction,
                "result": result.result,
                "coordinator_ms": coordinator_ms,
                "worker_ms": worker_ms,
            })
            self.store.end_span(span)

            logger.info(
                f"Thread {thread_id} step {step_number} ({final_move}): "
                f"coordinator={coordinator_ms}ms worker={worker_ms}ms"
            )

            self.recorder.step_complete(
                step_number, final_move,
                decision.worker_instruction or "", result.result,
                coordinator_ms + worker_ms,
            )

            # Clear human messages after first step uses them
            human_messages = []

            # --- HITL: pause after one step ---
            if self.mode == LoopMode.STEP_AND_PAUSE:
                thread_obj = self.store.get_thread(thread_id)
                if thread_obj and thread_obj.status != ThreadStatus.COMPLETE:
                    step_count = len(self.store.get_step_spans(thread_id))
                    self.store.update_thread_status(thread_id, ThreadStatus.WAITING)
                    self.store.save_session(session_id)
                    self.queue.emit(StreamEvent(
                        session_id=session_id,
                        thread_id=thread_id,
                        event_type="thread_waiting",
                        message=f"Step complete ({final_move}). Review and send a message to continue.",
                        data={
                            "pattern": "human_in_the_loop",
                            "last_move": final_move,
                            "last_result": result.result,
                            "step_number": step_count,
                        },
                    ))
                return

            # --- Done? ---
            if decision.status == CoordinatorStatus.DONE:
                thread_elapsed = round((time.monotonic() - thread_start) * 1000)
                logger.info(
                    f"Thread {thread_id} complete: {step_number} steps in {thread_elapsed}ms"
                )
                self.recorder.thread_complete(result.result, thread_elapsed, step_number)
                return

            # --- Summarize every 5 steps ---
            if step_number > 1 and step_number % 5 == 0:
                self._summarize(step_number)

    def _summarize(self, step_number: int):
        try:
            summary = self.store.summarize_history(
                trace_id=self.thread.id,
                llm=self.coordinator.llm,
                model=self.config.models.coordinator,
                seed_question=self.thread.seed_question,
            )
            if summary and not detect_degeneration(summary):
                self.store.update_thread_running_summary(self.thread.id, summary)
                logger.info(f"Thread {self.thread.id} history summarized")
            elif summary:
                logger.warning(f"Thread {self.thread.id} summary discarded — degenerate output")
        except Exception as e:
            logger.warning(f"History summarization failed: {e}")

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

    def _handle_error(self, e: Exception):
        from latent_insights.core.llm import is_transient_llm_error

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

    def _on_done(self, future):
        """Handle future completion. Errors are already handled in _run."""
        try:
            future.result()
        except Exception:
            pass  # Already handled in _run -> _handle_error
        finally:
            self.done_event.set()

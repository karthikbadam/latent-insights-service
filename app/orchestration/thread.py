"""Thread loop — coordinator-worker cycle until completion."""

import logging
import time

from app.agents.coordinator import run_coordinator
from app.agents.worker import run_worker
from app.config import AppConfig
from app.core.llm import LLMClient
from app.core.queue import Queue
from app.core.state import StateStore
from app.core.tracing import TraceStore
from app.models import (
    CoordinatorStatus,
    StreamEvent,
    Thread,
    ThreadStatus,
)

logger = logging.getLogger(__name__)


def _get_thread_views(session_db, thread_id: str) -> str:
    """List existing views for this thread."""
    try:
        rows = session_db.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_type = 'VIEW' AND table_name LIKE ?
        """, [f"thread_{thread_id}_%"]).fetchall()
        if rows:
            return "\n".join(r[0] for r in rows)
    except Exception:
        pass
    return "(none)"


def run_thread_loop(
    config: AppConfig,
    llm: LLMClient,
    session_db,
    queue: Queue,
    state: StateStore,
    trace_store: TraceStore,
    thread: Thread,
    schema_summary: str,
    human_messages: list[str] | None = None,
):
    """
    Run the coordinator-worker loop for a thread until DONE or STUCK.

    Steps are buffered in memory as spans. On completion, they're
    flushed to JSONL trace files and state is dumped.
    """
    if human_messages is None:
        human_messages = []

    thread_start = time.monotonic()
    tid = thread.id[:8]

    try:
        while True:
            step_start = time.monotonic()
            step_number = len(trace_store.get_step_spans(thread.id)) + 1

            step_span = trace_store.start_span(
                trace_id=thread.id,
                name=f"step_{step_number}",
                kind="step",
            )

            # Build history from in-memory spans
            thread_history = trace_store.format_thread_history(
                thread.id, human_messages, running_summary=thread.running_summary,
            )
            thread_views = _get_thread_views(session_db, thread.id)

            # Coordinator decides
            t0 = time.monotonic()
            decision, coordinator_log = run_coordinator(
                llm=llm,
                model=config.models.coordinator,
                seed_question=thread.seed_question,
                motivation=thread.motivation,
                entry_point=thread.entry_point,
                schema_summary=schema_summary,
                thread_history=thread_history,
                temperature=config.temperatures.coordinator,
                queue=queue,
                session_id=thread.session_id,
                thread_id=thread.id,
            )
            coordinator_ms = round((time.monotonic() - t0) * 1000)
            coordinator_log["duration_ms"] = coordinator_ms

            trace_store.add_event(step_span, "coordinator", {
                "model": config.models.coordinator,
                "duration_ms": coordinator_ms,
                "status": decision.status.value,
                "next_move": decision.next_move.value,
                "assessment": decision.assessment,
                "rationale": decision.rationale,
            })

            logger.info(
                f"Thread {thread.id} coordinator: {decision.status.value} "
                f"-> {decision.next_move.value} ({coordinator_ms}ms)"
            )

            # Handle STUCK
            if decision.status == CoordinatorStatus.STUCK:
                step_span.attributes.update({
                    "move": decision.next_move.value,
                    "instruction": decision.question_for_human or "",
                    "result": f"STUCK: {decision.context or ''}",
                })
                trace_store.end_span(step_span, status="stuck")

                trace_store.flush_to_file(thread.id, thread.session_id)
                trace_store.clear_trace(thread.id)
                state.update_thread_status(thread.id, ThreadStatus.WAITING)
                state.dump_session(thread.session_id)

                queue.emit(StreamEvent(
                    session_id=thread.session_id,
                    thread_id=thread.id,
                    event_type="waiting",
                    message=f"[{tid}] STUCK: {decision.question_for_human}",
                    data={
                        "question": decision.question_for_human,
                        "context": decision.context,
                    },
                ))
                return

            # Worker executes
            queue.emit(StreamEvent(
                session_id=thread.session_id,
                thread_id=thread.id,
                event_type="step",
                message=f"[{tid}] {decision.next_move.value}: {decision.worker_instruction[:120]}...",
                data={"move": decision.next_move.value, "step_number": step_number},
            ))

            t0 = time.monotonic()
            worker_result = run_worker(
                llm=llm,
                model=config.models.worker,
                fallback_model=config.models.worker_fallback,
                worker_instruction=decision.worker_instruction,
                schema_summary=schema_summary,
                session_db=session_db,
                thread_views=thread_views,
                max_retries=config.max_worker_retries,
                max_consecutive_errors=config.max_consecutive_errors,
                timeout=config.llm_timeout,
                queue=queue,
                session_id=thread.session_id,
                thread_id=thread.id,
            )
            worker_ms = round((time.monotonic() - t0) * 1000)

            step_ms = round((time.monotonic() - step_start) * 1000)

            trace_store.add_event(step_span, "worker", {
                "model": config.models.worker,
                "duration_ms": worker_ms,
                "result_preview": worker_result.result[:200],
            })
            if worker_result.llm_calls:
                for call in worker_result.llm_calls:
                    trace_store.add_event(step_span, "llm_call", call)

            step_span.attributes.update({
                "move": decision.next_move.value,
                "instruction": decision.worker_instruction,
                "result": worker_result.result,
                "coordinator_ms": coordinator_ms,
                "worker_ms": worker_ms,
            })
            trace_store.end_span(step_span)

            logger.info(
                f"Thread {thread.id} step {step_number} "
                f"({decision.next_move.value}): "
                f"coordinator={coordinator_ms}ms worker={worker_ms}ms "
                f"total={step_ms}ms"
            )

            # Emit step completion
            queue.emit(StreamEvent(
                session_id=thread.session_id,
                thread_id=thread.id,
                event_type="step",
                message=worker_result.result,
            ))

            # Summarize history periodically to keep context manageable
            if step_number > 1 and step_number % 5 == 0:
                _maybe_summarize(
                    trace_store, state, thread, llm, config,
                )

            # Handle DONE
            if decision.status == CoordinatorStatus.DONE:
                thread_elapsed = round(time.monotonic() - thread_start, 2)
                logger.info(
                    f"Thread {thread.id} complete: "
                    f"{step_number} steps in {thread_elapsed}s"
                )

                trace_store.flush_to_file(thread.id, thread.session_id)
                trace_store.clear_trace(thread.id)
                state.update_thread_status(
                    thread.id, ThreadStatus.COMPLETE,
                    summary=worker_result.result,
                )
                state.dump_session(thread.session_id)

                queue.emit(StreamEvent(
                    session_id=thread.session_id,
                    thread_id=thread.id,
                    event_type="complete",
                    message=(
                        f"[{tid}] COMPLETE ({step_number} steps, {thread_elapsed}s): "
                        f"{worker_result.result[:120]}"
                    ),
                    data={
                        "summary": worker_result.result,
                        "total_seconds": thread_elapsed,
                        "step_count": step_number,
                    },
                ))
                return

            # Clear human messages after first loop iteration
            human_messages = []

    except Exception as e:
        thread_elapsed = round(time.monotonic() - thread_start, 2)
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(
            f"Thread {thread.id} failed after {thread_elapsed}s: {error_msg}",
            exc_info=True,
        )
        try:
            trace_store.flush_to_file(thread.id, thread.session_id)
            trace_store.clear_trace(thread.id)
            state.update_thread_status(
                thread.id, ThreadStatus.ERROR,
                error=error_msg,
            )
            state.dump_session(thread.session_id)
            queue.emit(StreamEvent(
                session_id=thread.session_id,
                thread_id=thread.id,
                event_type="error",
                message=f"[{tid}] ERROR: {error_msg}",
                data={"error": error_msg},
            ))
        except Exception:
            logger.error(f"Failed to mark thread {thread.id} as errored", exc_info=True)
    finally:
        # Close per-thread connection
        try:
            session_db.close()
        except Exception:
            pass


def _maybe_summarize(
    trace_store: TraceStore,
    state: StateStore,
    thread: Thread,
    llm: LLMClient,
    config: AppConfig,
):
    """Summarize thread history if it's getting long."""
    try:
        summary = trace_store.summarize_history(
            trace_id=thread.id,
            llm=llm,
            model=config.models.coordinator,
            seed_question=thread.seed_question,
        )
        if summary:
            state.update_thread_running_summary(thread.id, summary)
            thread.running_summary = summary
            logger.info(f"Thread {thread.id} history summarized")
    except Exception as e:
        logger.warning(f"History summarization failed: {e}")


def resume_thread(
    config: AppConfig,
    llm: LLMClient,
    session_db,
    queue: Queue,
    state: StateStore,
    trace_store: TraceStore,
    thread_id: str,
    human_message: str,
    schema_summary: str,
):
    """Resume a stuck thread with a human message."""
    thread = state.get_thread(thread_id)
    if thread is None:
        raise ValueError(f"Thread {thread_id} not found")

    # Reload trace from JSONL if not in memory
    if not trace_store.get_spans(thread_id):
        trace_store.load_trace(thread_id, thread.session_id)

    state.update_thread_status(thread_id, ThreadStatus.RUNNING)

    run_thread_loop(
        config=config,
        llm=llm,
        session_db=session_db,
        queue=queue,
        state=state,
        trace_store=trace_store,
        thread=thread,
        schema_summary=schema_summary,
        human_messages=[human_message],
    )

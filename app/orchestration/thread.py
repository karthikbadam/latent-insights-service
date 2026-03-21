"""Thread loop — coordinator-worker cycle until completion."""

import logging
import time

from app.agents.coordinator import run_coordinator
from app.agents.worker import run_worker
from app.config import AppConfig
from app.core.llm import LLMClient
from app.core.queue import Queue
from app.db import queries
from app.db.mcp import create_thread_view
from app.models import (
    CoordinatorStatus,
    Thread,
    ThreadEvent,
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


async def run_thread_loop(
    config: AppConfig,
    llm: LLMClient,
    main_db,
    session_db,
    queue: Queue,
    thread: Thread,
    schema_summary: str,
    human_messages: list[str] | None = None,
):
    """
    Run the coordinator-worker loop for a thread until DONE or STUCK.

    On STUCK: sets thread status to WAITING, emits event, returns.
    On DONE: runs final SYNTHESIZE step, sets COMPLETE, emits event, returns.
    No artificial step limits.
    """
    if human_messages is None:
        human_messages = []

    thread_start = time.monotonic()

    try:
        while True:
            step_start = time.monotonic()

            # Load current state
            steps = queries.get_steps(main_db, thread.id)
            thread_history = queries.format_thread_history(steps, human_messages)
            thread_views = _get_thread_views(session_db, thread.id)

            # Coordinator decides
            t0 = time.monotonic()
            decision = await run_coordinator(
                llm=llm,
                model=config.models.coordinator,
                seed_question=thread.seed_question,
                motivation=thread.motivation,
                entry_point=thread.entry_point,
                schema_summary=schema_summary,
                thread_history=thread_history,
                temperature=config.temperatures.coordinator,
            )
            coordinator_ms = round((time.monotonic() - t0) * 1000)

            logger.info(
                f"Thread {thread.id} coordinator: {decision.status.value} "
                f"-> {decision.next_move.value} ({coordinator_ms}ms)"
            )

            # Handle STUCK
            if decision.status == CoordinatorStatus.STUCK:
                thread_elapsed = round(time.monotonic() - thread_start, 2)
                logger.info(f"Thread {thread.id} waiting after {thread_elapsed}s total")

                async with queue.db_write_lock:
                    queries.update_thread_status(main_db, thread.id, ThreadStatus.WAITING)

                await queue.emit(ThreadEvent(
                    session_id=thread.session_id,
                    thread_id=thread.id,
                    event_type="thread_waiting",
                    payload={
                        "question": decision.question_for_human,
                        "context": decision.context,
                    },
                ))
                return

            # Worker executes
            t0 = time.monotonic()
            worker_result = await run_worker(
                llm=llm,
                model=config.models.worker,
                fallback_model=config.models.worker_fallback,
                worker_instruction=decision.worker_instruction,
                schema_summary=schema_summary,
                session_db=session_db,
                thread_views=thread_views,
                cache_ttl_hours=config.cache.worker,
                max_retries=config.max_worker_retries,
            )
            worker_ms = round((time.monotonic() - t0) * 1000)

            # Handle view creation
            view_name = None
            if worker_result.view_requested:
                try:
                    view_name = create_thread_view(
                        session_db,
                        thread.id,
                        worker_result.view_requested["name"],
                        worker_result.view_requested["sql"],
                    )
                except Exception as e:
                    logger.warning(f"View creation failed: {e}")

            step_number = len(steps) + 1
            step_ms = round((time.monotonic() - step_start) * 1000)

            logger.info(
                f"Thread {thread.id} step {step_number} "
                f"({decision.next_move.value}): "
                f"coordinator={coordinator_ms}ms worker={worker_ms}ms "
                f"total={step_ms}ms"
            )

            # Build LLM call log for this step
            step_llm_calls = []
            step_llm_calls.append({
                "agent": "coordinator",
                "duration_ms": coordinator_ms,
            })
            if worker_result.llm_calls:
                for call in worker_result.llm_calls:
                    call["agent"] = "worker"
                step_llm_calls.extend(worker_result.llm_calls)

            # Persist step
            async with queue.db_write_lock:
                queries.append_step(
                    main_db,
                    thread.id,
                    decision.next_move,
                    decision.worker_instruction,
                    worker_result.result,
                    view_created=view_name,
                    duration_ms=step_ms,
                    llm_calls=step_llm_calls,
                )

            # Emit step event
            await queue.emit(ThreadEvent(
                session_id=thread.session_id,
                thread_id=thread.id,
                event_type="step_completed",
                payload={
                    "step_number": step_number,
                    "move": decision.next_move.value,
                    "result": worker_result.result,
                    "timing": {
                        "coordinator_ms": coordinator_ms,
                        "worker_ms": worker_ms,
                        "step_ms": step_ms,
                    },
                },
            ))

            # Handle DONE
            if decision.status == CoordinatorStatus.DONE:
                thread_elapsed = round(time.monotonic() - thread_start, 2)
                logger.info(
                    f"Thread {thread.id} complete: "
                    f"{step_number} steps in {thread_elapsed}s"
                )

                async with queue.db_write_lock:
                    queries.update_thread_status(
                        main_db, thread.id, ThreadStatus.COMPLETE,
                        summary=worker_result.result,
                    )

                await queue.emit(ThreadEvent(
                    session_id=thread.session_id,
                    thread_id=thread.id,
                    event_type="thread_complete",
                    payload={
                        "summary": worker_result.result,
                        "total_seconds": thread_elapsed,
                        "step_count": step_number,
                    },
                ))
                return

            # Clear human messages after first loop iteration (they've been seen)
            human_messages = []

    except Exception as e:
        thread_elapsed = round(time.monotonic() - thread_start, 2)
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(
            f"Thread {thread.id} failed after {thread_elapsed}s: {error_msg}",
            exc_info=True,
        )
        try:
            async with queue.db_write_lock:
                queries.update_thread_status(
                    main_db, thread.id, ThreadStatus.ERROR,
                    error=error_msg,
                )
            await queue.emit(ThreadEvent(
                session_id=thread.session_id,
                thread_id=thread.id,
                event_type="thread_error",
                payload={"error": error_msg},
            ))
        except Exception:
            logger.error(f"Failed to mark thread {thread.id} as errored", exc_info=True)


async def resume_thread(
    config: AppConfig,
    llm: LLMClient,
    main_db,
    session_db,
    queue: Queue,
    thread_id: str,
    human_message: str,
    schema_summary: str,
):
    """Resume a stuck thread with a human message."""
    thread = queries.get_thread(main_db, thread_id)
    if thread is None:
        raise ValueError(f"Thread {thread_id} not found")

    async with queue.db_write_lock:
        queries.update_thread_status(main_db, thread_id, ThreadStatus.RUNNING)

    await run_thread_loop(
        config=config,
        llm=llm,
        main_db=main_db,
        session_db=session_db,
        queue=queue,
        thread=thread,
        schema_summary=schema_summary,
        human_messages=[human_message],
    )

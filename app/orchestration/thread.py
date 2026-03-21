"""Thread loop — coordinator-worker cycle until completion."""

import logging

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

    while True:
        # Load current state
        steps = queries.get_steps(main_db, thread.id)
        thread_history = queries.format_thread_history(steps, human_messages)
        thread_views = _get_thread_views(session_db, thread.id)

        # Coordinator decides
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

        logger.info(
            f"Thread {thread.id} coordinator: {decision.status.value} "
            f"-> {decision.next_move.value}"
        )

        # Handle STUCK
        if decision.status == CoordinatorStatus.STUCK:
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

        # Persist step
        async with queue.db_write_lock:
            queries.append_step(
                main_db,
                thread.id,
                decision.next_move,
                decision.worker_instruction,
                worker_result.summary,
                result_details=worker_result.details,
                view_created=view_name,
            )

        # Emit step event
        await queue.emit(ThreadEvent(
            session_id=thread.session_id,
            thread_id=thread.id,
            event_type="step_completed",
            payload={
                "step_number": len(steps) + 1,
                "move": decision.next_move.value,
                "summary": worker_result.summary,
            },
        ))

        # Handle DONE
        if decision.status == CoordinatorStatus.DONE:
            async with queue.db_write_lock:
                queries.update_thread_status(main_db, thread.id, ThreadStatus.COMPLETE)

            await queue.emit(ThreadEvent(
                session_id=thread.session_id,
                thread_id=thread.id,
                event_type="thread_complete",
                payload={"summary": worker_result.summary},
            ))
            return

        # Clear human messages after first loop iteration (they've been seen)
        human_messages = []


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

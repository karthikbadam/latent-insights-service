"""Session lifecycle — upload, profile, scout, spawn threads."""

import asyncio
import logging
import time
from dataclasses import asdict
from functools import partial

from app.agents.profiler import run_profiler
from app.agents.scout import run_scout
from app.config import AppConfig
from app.core.llm import LLMClient
from app.core.queue import Queue
from app.core.state import StateStore
from app.core.tracing import TraceStore
from app.db.connection import Database
from app.models import StreamEvent
from app.orchestration.thread import run_thread_loop

logger = logging.getLogger(__name__)


async def create_session_flow(
    config: AppConfig,
    llm: LLMClient,
    db: Database,
    queue: Queue,
    state: StateStore,
    trace_store: TraceStore,
    session_id: str,
    dataset_path: str,
):
    """
    Full session creation flow (runs as background task):
    1. Create session DB with dataset loaded
    2. Run profiler -> store schema_summary
    3. Run scout -> store scout_output
    4. Spawn threads for top N scout questions
    """
    session_start = time.monotonic()

    logger.info(f"Session {session_id} flow starting for {dataset_path}")

    # Create session DB with dataset (run in thread to avoid blocking event loop)
    loop = asyncio.get_running_loop()
    session_db, table_name = await loop.run_in_executor(
        None, partial(db.create_session_db, session_id, dataset_path)
    )

    state.update_session_table_name(session_id, table_name)

    # Run profiler
    t0 = time.monotonic()
    schema_summary = await run_profiler(
        llm=llm,
        model=config.models.profiler,
        session_db=session_db,
        table_name=table_name,
    )
    profiler_ms = round((time.monotonic() - t0) * 1000)

    state.update_session_schema(session_id, schema_summary)

    logger.info(f"Session {session_id} profiled ({profiler_ms}ms)")

    # Run scout
    t0 = time.monotonic()
    scout_output = await run_scout(
        llm=llm,
        model=config.models.scout,
        schema_summary=schema_summary,
        table_name=table_name,
        session_db=session_db,
    )
    scout_ms = round((time.monotonic() - t0) * 1000)

    state.update_session_scout(session_id, asdict(scout_output))

    await queue.emit(StreamEvent(
        session_id=session_id,
        thread_id="",
        event_type="scout_done",
        message=f"Scout found {len(scout_output.questions)} questions",
        data={"question_count": len(scout_output.questions)},
    ))

    setup_elapsed = round(time.monotonic() - session_start, 2)
    logger.info(
        f"Session {session_id} scouted: {len(scout_output.questions)} questions "
        f"(profiler={profiler_ms}ms scout={scout_ms}ms setup={setup_elapsed}s)"
    )

    # Close the initial session_db — each thread gets its own connection
    await loop.run_in_executor(None, session_db.close)

    # Spawn threads in batches — process all questions, batch_size at a time
    batch_size = config.default_seed_threads
    questions = scout_output.questions

    for batch_start in range(0, len(questions), batch_size):
        batch = questions[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(questions) + batch_size - 1) // batch_size
        logger.info(
            f"Session {session_id} spawning batch {batch_num}/{total_batches} "
            f"({len(batch)} threads)"
        )

        batch_tasks = []
        for q in batch:
            thread = state.create_thread(
                session_id, q.question, q.motivation, q.entry_point,
            )

            thread_db = await loop.run_in_executor(
                None, partial(db.open_session_connection, session_id)
            )

            task = queue.schedule(
                coro=run_thread_loop(
                    config=config,
                    llm=llm,
                    session_db=thread_db,
                    queue=queue,
                    state=state,
                    trace_store=trace_store,
                    thread=thread,
                    schema_summary=schema_summary,
                ),
                task_id=f"thread-{thread.id}",
                session_id=session_id,
                thread_id=thread.id,
                description=f"Thread: {q.question[:60]}",
            )
            batch_tasks.append(task)

        # Wait for this batch to complete before starting the next
        if batch_tasks:
            await asyncio.gather(*batch_tasks, return_exceptions=True)

    await loop.run_in_executor(None, partial(state.dump_session, session_id))
    return session_id

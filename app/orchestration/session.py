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
from app.db import queries
from app.db.connection import Database
from app.models import ThreadEvent
from app.orchestration.thread import run_thread_loop

logger = logging.getLogger(__name__)


async def create_session_flow(
    config: AppConfig,
    llm: LLMClient,
    db: Database,
    queue: Queue,
    session_id: str,
    dataset_path: str,
):
    """
    Full session creation flow (runs as background task):
    1. Create session DB with dataset loaded
    2. Run profiler -> store schema_summary
    3. Run scout -> store scout_output
    4. Spawn threads for top N scout questions

    Session record must already exist in main DB.
    """
    session_start = time.monotonic()
    main_db = db.get_main_db()

    logger.info(f"Session {session_id} flow starting for {dataset_path}")

    # 2. Create session DB with dataset (run in thread to avoid blocking event loop)
    loop = asyncio.get_running_loop()
    session_db, table_name = await loop.run_in_executor(
        None, partial(db.create_session_db, session_id, dataset_path)
    )

    # Store the resolved table name
    async with queue.db_write_lock:
        queries.update_session_table_name(main_db, session_id, table_name)

    # MCP server disabled — workers use direct SQL via run_sql tool loop.
    # mcp_server_start('stdio') blocks on stdin, causing the process to hang.
    # await loop.run_in_executor(None, setup_mcp_server, session_db)
    # await loop.run_in_executor(None, publish_table, session_db, table_name)

    # 3. Run profiler
    t0 = time.monotonic()
    schema_summary = await run_profiler(
        llm=llm,
        model=config.models.profiler,
        session_db=session_db,
        table_name=table_name,
        cache_ttl_hours=config.cache.profiler,
    )
    profiler_ms = round((time.monotonic() - t0) * 1000)

    async with queue.db_write_lock:
        queries.update_session_schema(main_db, session_id, schema_summary)

    logger.info(f"Session {session_id} profiled ({profiler_ms}ms)")

    # 4. Run scout
    t0 = time.monotonic()
    scout_output = await run_scout(
        llm=llm,
        model=config.models.scout,
        schema_summary=schema_summary,
        table_name=table_name,
        session_db=session_db,
    )
    scout_ms = round((time.monotonic() - t0) * 1000)

    async with queue.db_write_lock:
        queries.update_session_scout(main_db, session_id, asdict(scout_output))

    await queue.emit(ThreadEvent(
        session_id=session_id,
        thread_id="",
        event_type="scout_done",
        payload={"question_count": len(scout_output.questions)},
    ))

    setup_elapsed = round(time.monotonic() - session_start, 2)
    logger.info(
        f"Session {session_id} scouted: {len(scout_output.questions)} questions "
        f"(profiler={profiler_ms}ms scout={scout_ms}ms setup={setup_elapsed}s)"
    )

    # 5. Spawn threads for top N questions
    num_threads = min(config.default_seed_threads, len(scout_output.questions))
    for i in range(num_threads):
        q = scout_output.questions[i]
        async with queue.db_write_lock:
            thread = queries.create_thread(
                main_db, session_id, q.question, q.motivation, q.entry_point,
            )

        queue.schedule(
            coro=run_thread_loop(
                config=config,
                llm=llm,
                main_db=main_db,
                session_db=session_db,
                queue=queue,
                thread=thread,
                schema_summary=schema_summary,
            ),
            task_id=f"thread-{thread.id}",
            session_id=session_id,
            thread_id=thread.id,
            description=f"Thread: {q.question[:60]}",
        )

    return session_id

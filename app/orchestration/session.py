"""Session lifecycle — upload, profile, scout, spawn threads."""

import logging
from dataclasses import asdict

from app.agents.profiler import run_profiler
from app.agents.scout import run_scout
from app.config import AppConfig
from app.core.llm import LLMClient
from app.core.queue import Queue
from app.db import queries
from app.db.connection import Database
from app.db.mcp import publish_table, setup_mcp_server
from app.models import ThreadEvent
from app.orchestration.thread import run_thread_loop

logger = logging.getLogger(__name__)


async def create_session_flow(
    config: AppConfig,
    llm: LLMClient,
    db: Database,
    queue: Queue,
    dataset_path: str,
) -> str:
    """
    Full session creation flow:
    1. Create session record in main DB
    2. Create session DB with dataset loaded
    3. Run profiler -> store schema_summary
    4. Run scout -> store scout_output
    5. Spawn threads for top N scout questions

    Returns session_id.
    """
    main_db = db.get_main_db()

    # 1. Create session record
    async with queue.db_write_lock:
        session = queries.create_session(main_db, dataset_path)

    session_id = session.id
    logger.info(f"Session {session_id} created for {dataset_path}")

    # 2. Create session DB with dataset
    session_db = db.create_session_db(session_id, dataset_path)

    # Set up MCP server on session DB (optional, degrades gracefully)
    setup_mcp_server(session_db)
    publish_table(session_db, "dataset")

    # 3. Run profiler
    schema_summary = await run_profiler(
        llm=llm,
        model=config.models.profiler,
        session_db=session_db,
        cache_ttl_hours=config.cache.profiler,
    )

    async with queue.db_write_lock:
        queries.update_session_schema(main_db, session_id, schema_summary)

    logger.info(f"Session {session_id} profiled")

    # 4. Run scout
    scout_output = await run_scout(
        llm=llm,
        model=config.models.scout,
        schema_summary=schema_summary,
        session_db=session_db,
    )

    async with queue.db_write_lock:
        queries.update_session_scout(main_db, session_id, asdict(scout_output))

    await queue.emit(ThreadEvent(
        session_id=session_id,
        thread_id="",
        event_type="scout_done",
        payload={"question_count": len(scout_output.questions)},
    ))

    logger.info(f"Session {session_id} scouted: {len(scout_output.questions)} questions")

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

"""
API routes — thin HTTP layer over orchestration.
"""

import logging
import os
import time

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.api.schemas import (
    CacheStats,
    CreateThreadRequest,
    DatasetInfo,
    PostMessageRequest,
    SessionResponse,
    StepResponse,
    SystemStats,
    TaskResponse,
    ThreadDetailResponse,
    ThreadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_start_time = time.time()


def _get_state():
    """Get global app state from main module."""
    from app import main
    if not main.config or not main.llm or not main.db or not main.queue_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return main.config, main.llm, main.db, main.queue_instance


# --- Dataset Management ---


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset file."""
    config, _, _, _ = _get_state()

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    upload_dir = os.path.join(config.data_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return DatasetInfo(
        name=file.filename,
        path=file_path,
        size_bytes=len(content),
        source="upload",
    )


@router.get("/datasets")
async def list_datasets() -> list[DatasetInfo]:
    """List all available datasets (uploads + samples)."""
    config, _, _, _ = _get_state()
    datasets = []

    for source in ("uploads", "samples"):
        dir_path = os.path.join(config.data_dir, source)
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if fname.endswith(".csv"):
                full_path = os.path.join(dir_path, fname)
                datasets.append(DatasetInfo(
                    name=fname,
                    path=full_path,
                    size_bytes=os.path.getsize(full_path),
                    source=source.rstrip("s"),  # "upload" or "sample"
                ))

    return datasets


@router.get("/datasets/samples")
async def list_sample_datasets() -> list[DatasetInfo]:
    """List sample datasets only."""
    config, _, _, _ = _get_state()
    samples_dir = os.path.join(config.data_dir, "samples")
    datasets = []

    if os.path.isdir(samples_dir):
        for fname in os.listdir(samples_dir):
            if fname.endswith(".csv"):
                full_path = os.path.join(samples_dir, fname)
                datasets.append(DatasetInfo(
                    name=fname,
                    path=full_path,
                    size_bytes=os.path.getsize(full_path),
                    source="sample",
                ))

    return datasets


# --- Sessions ---


@router.post("/sessions")
async def create_session(file: UploadFile = File(...)):
    """Create a new analysis session with a dataset upload."""
    config, llm, db, queue = _get_state()

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    upload_dir = os.path.join(config.data_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    from app.orchestration.session import create_session_flow

    # Schedule the full session flow as a background task
    async def session_task():
        return await create_session_flow(config, llm, db, queue, file_path)

    # Run synchronously to get the session_id, then background threads
    session_id = await create_session_flow(config, llm, db, queue, file_path)

    return {"session_id": session_id, "status": "created"}


@router.post("/sessions/from-dataset")
async def create_session_from_dataset(dataset_path: str):
    """Create a session from an existing dataset path."""
    config, llm, db, queue = _get_state()

    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")

    from app.orchestration.session import create_session_flow

    session_id = await create_session_flow(config, llm, db, queue, dataset_path)
    return {"session_id": session_id, "status": "created"}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session state including scout questions and thread list."""
    _, _, db, _ = _get_state()
    from app.db import queries

    main_db = db.get_main_db()
    session = queries.get_session(main_db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    threads = queries.get_threads(main_db, session_id)

    return SessionResponse(
        id=session.id,
        thread_count=len(threads),
        schema_summary=session.schema_summary,
        scout_questions=session.scout_output.get("questions") if session.scout_output else None,
    )


@router.get("/sessions/{session_id}/threads")
async def get_threads(session_id: str):
    """Get all threads for a session."""
    _, _, db, _ = _get_state()
    from app.db import queries

    main_db = db.get_main_db()
    threads = queries.get_threads(main_db, session_id)

    return [
        ThreadResponse(
            id=t.id,
            seed_question=t.seed_question,
            status=t.status.value,
            step_count=len(queries.get_steps(main_db, t.id)),
            updated_at=t.updated_at.isoformat() if t.updated_at else "",
        )
        for t in threads
    ]


@router.post("/sessions/{session_id}/threads")
async def create_thread(session_id: str, request: CreateThreadRequest):
    """Create a user-initiated thread with a custom question."""
    config, llm, db, queue = _get_state()
    from app.db import queries
    from app.orchestration.thread import run_thread_loop

    main_db = db.get_main_db()
    session = queries.get_session(main_db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.schema_summary is None:
        raise HTTPException(status_code=400, detail="Session profiling not complete yet")

    thread = queries.create_thread(
        main_db, session_id, request.question, request.motivation or "", "",
    )

    session_db = db.create_session_db(session_id, session.dataset_path)

    queue.schedule(
        coro=run_thread_loop(
            config=config,
            llm=llm,
            main_db=main_db,
            session_db=session_db,
            queue=queue,
            thread=thread,
            schema_summary=session.schema_summary,
        ),
        task_id=f"thread-{thread.id}",
        session_id=session_id,
        thread_id=thread.id,
        description=f"Thread: {request.question[:60]}",
    )

    return ThreadResponse(
        id=thread.id,
        seed_question=thread.seed_question,
        status=thread.status.value,
        step_count=0,
        updated_at=thread.updated_at.isoformat() if thread.updated_at else "",
    )


# --- Threads ---


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get full thread detail including all steps."""
    _, _, db, _ = _get_state()
    from app.db import queries

    main_db = db.get_main_db()
    thread = queries.get_thread(main_db, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    steps = queries.get_steps(main_db, thread_id)

    return ThreadDetailResponse(
        id=thread.id,
        session_id=thread.session_id,
        seed_question=thread.seed_question,
        motivation=thread.motivation,
        entry_point=thread.entry_point,
        status=thread.status.value,
        steps=[
            StepResponse(
                id=s.id,
                step_number=s.step_number,
                move=s.move.value,
                instruction=s.instruction,
                result_summary=s.result_summary,
                result_details=s.result_details,
                view_created=s.view_created,
            )
            for s in steps
        ],
    )


@router.post("/threads/{thread_id}/messages")
async def post_message(thread_id: str, request: PostMessageRequest):
    """Post a human message to a stuck thread, resuming it."""
    config, llm, db, queue = _get_state()
    from app.db import queries
    from app.orchestration.thread import resume_thread

    main_db = db.get_main_db()
    thread = queries.get_thread(main_db, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.status.value != "waiting":
        raise HTTPException(status_code=400, detail="Thread is not waiting for input")

    session = queries.get_session(main_db, thread.session_id)
    session_db = db.create_session_db(thread.session_id, session.dataset_path)

    queue.schedule(
        coro=resume_thread(
            config=config,
            llm=llm,
            main_db=main_db,
            session_db=session_db,
            queue=queue,
            thread_id=thread_id,
            human_message=request.content,
            schema_summary=session.schema_summary or "",
        ),
        task_id=f"resume-{thread_id}",
        session_id=thread.session_id,
        thread_id=thread_id,
        description=f"Resume: {thread.seed_question[:40]}",
    )

    return {"status": "resumed", "thread_id": thread_id}


# --- System Monitoring ---


@router.get("/system/health")
async def system_health():
    """Extended health check."""
    try:
        config, llm, db, queue = _get_state()
        main_db = db.get_main_db()
        main_db.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "db_connected": db_ok,
        "llm_configured": bool(config.openrouter_api_key) if db_ok else False,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@router.get("/system/tasks")
async def list_tasks(session_id: str | None = None) -> list[TaskResponse]:
    """List active async tasks."""
    _, _, _, queue = _get_state()
    tasks = queue.get_active_tasks(session_id)
    return [
        TaskResponse(
            task_id=t.task.get_name(),
            session_id=t.session_id,
            thread_id=t.thread_id,
            description=t.description,
        )
        for t in tasks
    ]


@router.get("/system/stats")
async def system_stats() -> SystemStats:
    """System-wide statistics."""
    _, _, db, _ = _get_state()
    main_db = db.get_main_db()

    session_count = main_db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    thread_count = main_db.execute("SELECT COUNT(*) FROM threads").fetchone()[0]
    step_count = main_db.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    cache_entries = main_db.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]

    return SystemStats(
        session_count=session_count,
        thread_count=thread_count,
        step_count=step_count,
        cache_entries=cache_entries,
    )


@router.get("/system/cache")
async def cache_stats() -> list[CacheStats]:
    """LLM cache statistics by role."""
    _, _, db, _ = _get_state()
    main_db = db.get_main_db()

    rows = main_db.execute("""
        SELECT role,
               COUNT(*) as cached_calls,
               COALESCE(SUM(input_tokens), 0) as input_tokens_saved,
               COALESCE(SUM(output_tokens), 0) as output_tokens_saved
        FROM llm_cache
        GROUP BY role
    """).fetchall()

    return [
        CacheStats(
            role=r[0], cached_calls=r[1],
            input_tokens_saved=r[2], output_tokens_saved=r[3],
        )
        for r in rows
    ]

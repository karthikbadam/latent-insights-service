"""
API routes — thin HTTP layer over orchestration.
"""

import logging
import os

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from app.api.schemas import (
    CreateThreadRequest,
    PostMessageRequest,
    SessionResponse,
    SystemStats,
    ThreadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_state():
    """Get global app state from main module."""
    from app import main
    if not main.config or not main.llm or not main.db or not main.queue_instance or not main.state_store:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return main.config, main.llm, main.db, main.queue_instance, main.state_store, main.trace_store


# --- Sessions ---


@router.post("/sessions")
async def create_session(
    file: UploadFile | None = File(None),
    dataset_path: str | None = Query(None),
):
    """Create a new analysis session from file upload or existing dataset path."""
    config, llm, db, queue, state, trace_store = _get_state()

    if file and file.filename:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        upload_dir = os.path.join(config.data_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        resolved_path = os.path.join(upload_dir, file.filename)
        content = await file.read()
        with open(resolved_path, "wb") as f:
            f.write(content)
    elif dataset_path:
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
        resolved_path = dataset_path
    else:
        raise HTTPException(status_code=400, detail="Provide either a file upload or dataset_path")

    from app.db.connection import table_name_from_path
    from app.orchestration.session import create_session_flow

    table_name = table_name_from_path(resolved_path)
    session = state.create_session(resolved_path, table_name)

    queue.schedule(
        coro=create_session_flow(
            config, llm, db, queue, state, trace_store, session.id, resolved_path,
        ),
        task_id=f"session-{session.id}",
        session_id=session.id,
        description=f"Session setup: {os.path.basename(resolved_path)}",
    )

    return {"session_id": session.id, "status": "created"}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full session state with threads."""
    _, _, _, _, state, _ = _get_state()

    session = state.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    threads = state.get_threads(session_id)

    return SessionResponse(
        id=session.id,
        dataset_path=session.dataset_path,
        schema_summary=session.schema_summary,
        scout_questions=session.scout_output.get("questions") if session.scout_output else None,
        threads=[
            ThreadResponse(
                id=t.id,
                seed_question=t.seed_question,
                status=t.status.value,
                summary=t.summary,
                error=t.error,
                updated_at=t.updated_at.isoformat() if t.updated_at else "",
            )
            for t in threads
        ],
        created_at=session.created_at.isoformat() if session.created_at else "",
    )


@router.post("/sessions/{session_id}/threads")
async def create_thread(session_id: str, request: CreateThreadRequest):
    """Create a user-initiated thread with a custom question."""
    config, llm, db, queue, state, trace_store = _get_state()
    from app.orchestration.thread import run_thread_loop

    session = state.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.schema_summary is None:
        raise HTTPException(status_code=400, detail="Session profiling not complete yet")

    thread = state.create_thread(
        session_id, request.question, request.motivation or "", "",
    )

    thread_db = db.open_session_connection(session_id)

    queue.schedule(
        coro=run_thread_loop(
            config=config,
            llm=llm,
            session_db=thread_db,
            queue=queue,
            state=state,
            trace_store=trace_store,
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
        updated_at=thread.updated_at.isoformat() if thread.updated_at else "",
    )


@router.post("/threads/{thread_id}/messages")
async def post_message(thread_id: str, request: PostMessageRequest):
    """Post a human message to a stuck thread, resuming it."""
    config, llm, db, queue, state, trace_store = _get_state()
    from app.orchestration.thread import resume_thread

    thread = state.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.status.value != "waiting":
        raise HTTPException(status_code=400, detail="Thread is not waiting for input")

    session = state.get_session(thread.session_id)
    thread_db = db.open_session_connection(thread.session_id)

    queue.schedule(
        coro=resume_thread(
            config=config,
            llm=llm,
            session_db=thread_db,
            queue=queue,
            state=state,
            trace_store=trace_store,
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


# --- System ---


@router.get("/system/stats")
async def system_stats() -> SystemStats:
    """Session and thread counts."""
    _, _, _, _, state, _ = _get_state()

    return SystemStats(
        session_count=state.session_count,
        thread_count=state.thread_count,
    )

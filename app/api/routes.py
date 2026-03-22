"""
API routes — thin HTTP layer over orchestration.
"""

import asyncio
import logging
import os
from functools import partial

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Query

from app.api.schemas import (
    CreateThreadRequest,
    PostMessageRequest,
    SessionResponse,
    SessionUrls,
    StepResponse,
    SystemStats,
    ThreadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _get_state():
    """Get global app state from main module."""
    from app import main
    if not main.config or not main.llm or not main.db or not main.queue_instance or not main.state_store:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return main.config, main.llm, main.db, main.queue_instance, main.state_store, main.trace_store


async def _steps_from_trace(trace_store, thread) -> list[StepResponse]:
    """Convert TraceStore spans to StepResponse list for API."""
    spans = trace_store.get_step_spans(thread.id)
    if not spans:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, partial(trace_store.load_trace, thread.id, thread.session_id),
        )
        spans = trace_store.get_step_spans(thread.id)

    # Only include completed steps (in-progress spans have no attributes yet)
    spans = [s for s in spans if s.end_time is not None]

    steps = []
    for i, span in enumerate(spans, 1):
        attrs = span.attributes
        duration_ms = None
        if span.end_time and span.start_time:
            duration_ms = round((span.end_time - span.start_time) * 1000)
        steps.append(StepResponse(
            step_number=i,
            move=attrs.get("move", ""),
            instruction=attrs.get("instruction", ""),
            result=attrs.get("result", ""),
            view_created=attrs.get("view_created"),
            duration_ms=duration_ms,
        ))
    return steps


# --- Sessions ---


@router.post("/sessions")
async def create_session(
    request: Request,
    file: UploadFile | None = File(None),
    dataset_path: str | None = Query(None),
):
    """Create a new analysis session from file upload or existing dataset path."""
    config, llm, db, queue, state, trace_store = _get_state()

    if file and file.filename:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        upload_dir = os.path.join(config.data_dir, "uploads")
        resolved_path = os.path.join(upload_dir, file.filename)
        content = await file.read()

        def _write_upload():
            os.makedirs(upload_dir, exist_ok=True)
            with open(resolved_path, "wb") as f:
                f.write(content)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_upload)
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

    base = _base_url(request)
    return {
        "session_id": session.id,
        "status": "created",
        "urls": {
            "self": f"{base}/api/sessions/{session.id}",
            "events": f"{base}/api/sessions/{session.id}/events",
            "threads": f"{base}/api/sessions/{session.id}/threads",
        },
    }


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    """Get full session state with threads and steps."""
    _, _, _, _, state, trace_store = _get_state()

    session = state.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    threads = state.get_threads(session_id)

    thread_responses = []
    for t in threads:
        steps = await _steps_from_trace(trace_store, t)
        thread_responses.append(ThreadResponse(
            id=t.id,
            seed_question=t.seed_question,
            motivation=t.motivation,
            status=t.status.value,
            summary=t.summary,
            running_summary=t.running_summary,
            error=t.error,
            steps=steps,
            updated_at=t.updated_at.isoformat() if t.updated_at else "",
        ))

    return SessionResponse(
        id=session.id,
        dataset_path=session.dataset_path,
        schema_summary=session.schema_summary,
        scout_questions=session.scout_output.get("questions") if session.scout_output else None,
        threads=thread_responses,
        urls=SessionUrls(
            self=f"{_base_url(request)}/api/sessions/{session_id}",
            events=f"{_base_url(request)}/api/sessions/{session_id}/events",
            threads=f"{_base_url(request)}/api/sessions/{session_id}/threads",
        ),
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

    loop = asyncio.get_running_loop()
    thread_db = await loop.run_in_executor(None, partial(db.open_session_connection, session_id))

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
        motivation=thread.motivation,
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
    loop = asyncio.get_running_loop()
    thread_db = await loop.run_in_executor(None, partial(db.open_session_connection, thread.session_id))

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

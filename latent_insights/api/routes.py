"""
API routes — thin HTTP layer over orchestration.
"""

import logging
import os

import json as json_mod

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File, Query

from latent_insights.api.schemas import (
    CreateThreadRequest,
    GraphStateResponse,
    PatternInfo,
    PostMessageRequest,
    RunPatternRequest,
    RunPatternResponse,
    SessionConfig,
    SessionResponse,
    SessionSummary,
    SessionUrls,
    StepEvent,
    StepResponse,
    SystemStats,
    ThreadResponse,
)

from latent_insights.models import StreamEvent, ThreadStatus

logger = logging.getLogger(__name__)

router = APIRouter()


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _get_state(request: Request):
    """Get app state from request.app.state (set during lifespan)."""
    s = request.app.state
    if not hasattr(s, "config") or not s.config:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return s.config, s.llm, s.db, s.queue, s.state_store, s.trace_store


def _steps_from_trace(trace_store, thread) -> list[StepResponse]:
    """Convert TraceStore spans to StepResponse list for API."""
    spans = trace_store.get_step_spans(thread.id)
    if not spans:
        trace_store.load_trace(thread.id, thread.session_id)
        spans = trace_store.get_step_spans(thread.id)

    # Only include completed steps (in-progress spans have no attributes yet)
    spans = [s for s in spans if s.end_time is not None]

    steps = []
    for i, span in enumerate(spans, 1):
        attrs = span.attributes
        duration_ms = None
        if span.end_time and span.start_time:
            duration_ms = round((span.end_time - span.start_time) * 1000)

        # Build interleaved event timeline from span events
        step_events = []
        for event in span.events:
            event_attrs = event.get("attributes", {})
            step_events.append(StepEvent(
                type=event["name"],
                timestamp=event.get("timestamp", 0),
                agent=event_attrs.get("agent"),
                model=event_attrs.get("model"),
                duration_ms=event_attrs.get("duration_ms"),
                input_tokens=event_attrs.get("input_tokens"),
                output_tokens=event_attrs.get("output_tokens"),
                sql=event_attrs.get("sql"),
                tool_result=event_attrs.get("tool_result"),
                response=event_attrs.get("response"),
            ))
        step_events.sort(key=lambda e: e.timestamp)

        steps.append(StepResponse(
            step_number=i,
            move=attrs.get("move", ""),
            instruction=attrs.get("instruction", ""),
            result=attrs.get("result", ""),
            view_created=attrs.get("view_created"),
            duration_ms=duration_ms,
            events=step_events,
        ))
    return steps


# --- Sessions ---


@router.post("/sessions")
def create_session(
    request: Request,
    file: UploadFile | None = File(None),
    dataset_path: str | None = Query(None),
    config_json: str | None = Form(None, alias="config"),
):
    """Create a new analysis session from file upload or existing dataset path."""
    config, llm, db, queue, state, trace_store = _get_state(request)

    # Parse per-session config overrides
    session_config = config
    if config_json:
        try:
            raw = json_mod.loads(config_json)
        except json_mod.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in config field")
        parsed = SessionConfig(**raw)
        session_config = config.with_overrides(parsed.model_dump())

    if file and file.filename:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        upload_dir = os.path.join(session_config.data_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        resolved_path = os.path.join(upload_dir, file.filename)
        content = file.file.read()
        with open(resolved_path, "wb") as f:
            f.write(content)
    elif dataset_path:
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
        resolved_path = dataset_path
    else:
        raise HTTPException(status_code=400, detail="Provide either a file upload or dataset_path")

    from latent_insights.db.connection import table_name_from_path
    from latent_insights.orchestration.session import SessionFlow

    table_name = table_name_from_path(resolved_path)
    session = state.create_session(resolved_path, table_name)

    flow = SessionFlow(session_config, llm, db, queue, state, trace_store)
    queue.schedule(
        fn=flow.create,
        args=(session.id, resolved_path),
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


@router.get("/sessions/{session_id}/saved")
def get_saved_session(session_id: str, request: Request):
    """Return a previously saved session from data/sessions/."""
    config, *_ = _get_state(request)
    stored = os.path.join(config.data_dir, "sessions", f"{session_id}.json")
    if not os.path.exists(stored):
        raise HTTPException(status_code=404, detail="Saved session not found")
    with open(stored) as f:
        return json_mod.load(f)


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    """Get full session state with threads and steps."""
    _, _, _, _, state, trace_store = _get_state(request)

    session = state.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    threads = state.get_threads(session_id)

    thread_responses = []
    for t in threads:
        steps = _steps_from_trace(trace_store, t)
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
def create_thread(session_id: str, request: Request, body: CreateThreadRequest):
    """Create a user-initiated thread with a custom question."""
    config, llm, db, queue, state, trace_store = _get_state(request)
    from latent_insights.orchestration.thread import ThreadRunner

    session = state.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.schema_summary is None:
        raise HTTPException(status_code=400, detail="Session profiling not complete yet")

    thread = state.create_thread(
        session_id, body.question, body.motivation or "", "",
    )

    thread_db = db.open_session_connection(session_id)

    runner = ThreadRunner(
        config=config,
        llm=llm,
        session_db=thread_db,
        queue=queue,
        state=state,
        trace_store=trace_store,
        thread=thread,
        schema_summary=session.schema_summary,
    )
    runner.start()

    return ThreadResponse(
        id=thread.id,
        seed_question=thread.seed_question,
        motivation=thread.motivation,
        status=thread.status.value,
        updated_at=thread.updated_at.isoformat() if thread.updated_at else "",
    )


@router.post("/sessions/{session_id}/continue")
def continue_session(session_id: str, request: Request):
    """Continue a session: resume stuck threads + scout new questions."""
    config, llm, db, queue, state, trace_store = _get_state(request)
    from latent_insights.orchestration.session import SessionFlow

    session = state.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.schema_summary is None:
        raise HTTPException(status_code=400, detail="Session profiling not complete yet")

    flow = SessionFlow(config, llm, db, queue, state, trace_store)
    queue.schedule(
        fn=flow.continue_,
        args=(session_id,),
        task_id=f"continue-{session_id}",
        session_id=session_id,
        description=f"Continue session {session_id}",
    )

    threads = state.get_threads(session_id)
    resumable = [t for t in threads if t.status.value in ("waiting", "complete")]

    return {
        "status": "continuing",
        "session_id": session_id,
        "threads_resumed": len(resumable),
    }


@router.post("/threads/{thread_id}/messages")
def post_message(thread_id: str, request: Request, body: PostMessageRequest):
    """Post a human message to a thread.

    - If thread is WAITING or COMPLETE: resumes the thread with the message.
    - If thread is RUNNING: injects the message into the next coordinator step
      (non-blocking interrupt).
    """
    config, llm, db, queue, state, trace_store = _get_state(request)

    thread = state.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread.status == ThreadStatus.RUNNING:
        # Inject message into the running thread — picked up at next coordinator step
        state.push_pending_message(thread_id, body.content)
        queue.emit(StreamEvent(
            session_id=thread.session_id,
            thread_id=thread_id,
            event_type="message_injected",
            message=body.content,
            data={"content": body.content, "target": "thread"},
        ))
        return {"status": "injected", "thread_id": thread_id}

    if thread.status.value not in ("waiting", "complete"):
        raise HTTPException(status_code=400, detail=f"Thread status '{thread.status.value}' does not accept messages")

    from latent_insights.orchestration.thread import ThreadRunner

    session = state.get_session(thread.session_id)
    thread_db = db.open_session_connection(thread.session_id)

    runner = ThreadRunner(
        config=config,
        llm=llm,
        session_db=thread_db,
        queue=queue,
        state=state,
        trace_store=trace_store,
        thread=thread,
        schema_summary=session.schema_summary or "",
    )
    runner.resume(human_messages=[body.content])

    return {"status": "resumed", "thread_id": thread_id}


@router.post("/sessions/{session_id}/messages")
def post_session_message(session_id: str, request: Request, body: PostMessageRequest):
    """Broadcast a human message to all running threads in a session.

    The message is injected into each running thread and picked up at
    the next coordinator step. Also resumes any waiting threads.
    """
    config, llm, db, queue, state, trace_store = _get_state(request)

    session = state.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    threads = state.get_threads(session_id)
    injected_ids = []
    resumed_ids = []

    for t in threads:
        if t.status == ThreadStatus.RUNNING:
            state.push_pending_message(t.id, body.content)
            injected_ids.append(t.id)
        elif t.status == ThreadStatus.WAITING:
            from latent_insights.orchestration.thread import ThreadRunner
            thread_db = db.open_session_connection(session_id)
            runner = ThreadRunner(
                config=config, llm=llm, session_db=thread_db, queue=queue,
                state=state, trace_store=trace_store, thread=t,
                schema_summary=session.schema_summary or "",
            )
            runner.resume(human_messages=[body.content])
            resumed_ids.append(t.id)

    queue.emit(StreamEvent(
        session_id=session_id,
        thread_id="",
        event_type="message_injected",
        message=body.content,
        data={
            "content": body.content,
            "target": "session",
            "injected_threads": injected_ids,
            "resumed_threads": resumed_ids,
        },
    ))

    return {
        "status": "delivered",
        "session_id": session_id,
        "injected_threads": len(injected_ids),
        "resumed_threads": len(resumed_ids),
    }


@router.get("/sessions")
def list_sessions(request: Request):
    """List all sessions with metadata."""
    _, _, _, _, state, _ = _get_state(request)

    sessions = state.get_all_sessions()
    summaries = []
    for s in sessions:
        threads = state.get_threads(s.id)
        status_counts: dict[str, int] = {}
        for t in threads:
            status_counts[t.status.value] = status_counts.get(t.status.value, 0) + 1
        summaries.append(SessionSummary(
            id=s.id,
            dataset_path=s.dataset_path,
            table_name=s.table_name,
            thread_count=len(threads),
            status_counts=status_counts,
            created_at=s.created_at.isoformat() if s.created_at else "",
        ))

    return summaries


@router.get("/threads/{thread_id}")
def get_thread(thread_id: str, request: Request):
    """Get a single thread with its steps."""
    _, _, _, _, state, trace_store = _get_state(request)

    thread = state.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    steps = _steps_from_trace(trace_store, thread)
    return ThreadResponse(
        id=thread.id,
        seed_question=thread.seed_question,
        motivation=thread.motivation,
        status=thread.status.value,
        summary=thread.summary,
        running_summary=thread.running_summary,
        error=thread.error,
        steps=steps,
        updated_at=thread.updated_at.isoformat() if thread.updated_at else "",
    )


# --- System ---


@router.get("/system/stats")
def system_stats(request: Request) -> SystemStats:
    """Session and thread counts."""
    _, _, _, _, state, _ = _get_state(request)

    return SystemStats(
        session_count=state.session_count,
        thread_count=state.thread_count,
    )


# --- Patterns ---


@router.get("/patterns")
def list_patterns() -> list[PatternInfo]:
    """List available agentic patterns."""
    from latent_insights.orchestration.patterns import PATTERN_REGISTRY

    return [PatternInfo(**p) for p in PATTERN_REGISTRY.values()]


@router.post("/sessions/{session_id}/patterns/{pattern_name}")
def run_pattern(
    session_id: str,
    pattern_name: str,
    request: Request,
    body: RunPatternRequest,
):
    """Run a named pattern for a session."""
    config, llm, db, queue, state, trace_store = _get_state(request)
    from latent_insights.orchestration.patterns import PATTERN_REGISTRY

    session = state.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.schema_summary is None:
        raise HTTPException(status_code=400, detail="Session profiling not complete yet")
    if pattern_name not in PATTERN_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown pattern: {pattern_name}")

    inputs = body.inputs

    if pattern_name == "coordinator_worker":
        question = inputs.get("question", "")
        if not question:
            raise HTTPException(status_code=400, detail="'question' is required")
        from latent_insights.orchestration.thread import ThreadRunner

        thread = state.create_thread(
            session_id, question, inputs.get("motivation", ""), "",
        )
        thread_db = db.open_session_connection(session_id)
        runner = ThreadRunner(
            config=config, llm=llm, session_db=thread_db, queue=queue,
            state=state, trace_store=trace_store, thread=thread,
            schema_summary=session.schema_summary,
        )
        runner.start()
        return RunPatternResponse(
            thread_id=thread.id, pattern=pattern_name, status="running",
        )

    elif pattern_name == "fan_out":
        questions = inputs.get("questions", [])
        if not questions:
            raise HTTPException(status_code=400, detail="'questions' is required")
        from latent_insights.orchestration.thread import ThreadRunner

        thread_ids = []
        for q in questions:
            thread = state.create_thread(session_id, q, "", "")
            thread_db = db.open_session_connection(session_id)
            runner = ThreadRunner(
                config=config, llm=llm, session_db=thread_db, queue=queue,
                state=state, trace_store=trace_store, thread=thread,
                schema_summary=session.schema_summary,
            )
            runner.start()
            thread_ids.append(thread.id)
        return {"pattern": pattern_name, "status": "running", "thread_ids": thread_ids}

    elif pattern_name == "human_in_the_loop":
        question = inputs.get("question", "")
        if not question:
            raise HTTPException(status_code=400, detail="'question' is required")
        from latent_insights.orchestration.thread import ThreadRunner

        thread = state.create_thread(session_id, question, inputs.get("motivation", ""), "")
        thread_db = db.open_session_connection(session_id)
        runner = ThreadRunner(
            config=config, llm=llm, session_db=thread_db, queue=queue,
            state=state, trace_store=trace_store, thread=thread,
            schema_summary=session.schema_summary,
        )
        runner.start()
        return RunPatternResponse(
            thread_id=thread.id, pattern=pattern_name, status="running",
        )

    raise HTTPException(status_code=400, detail=f"Pattern '{pattern_name}' not yet implemented")


# --- Graph State (debug) ---


@router.get("/threads/{thread_id}/graph-state")
def get_graph_state(thread_id: str, request: Request):
    """Inspect the current LangGraph state for a thread."""
    _, _, _, _, state, trace_store = _get_state(request)

    thread = state.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    spans = trace_store.get_step_spans(thread_id)
    if not spans:
        trace_store.load_trace(thread_id, thread.session_id)
        spans = trace_store.get_step_spans(thread_id)

    move_history = []
    for span in spans:
        move = span.attributes.get("move")
        if move:
            move_history.append(move)

    # Determine current node from thread status
    current_node = None
    if thread.status.value == "running":
        current_node = "coordinator" if not spans or spans[-1].end_time else "worker"

    return GraphStateResponse(
        thread_id=thread_id,
        step_number=len(spans),
        current_node=current_node,
        move_history=move_history,
        status=thread.status.value,
        decision=spans[-1].attributes if spans else None,
    )

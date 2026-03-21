"""Integration tests — full session and thread flows with mock LLM."""

import json
from unittest.mock import AsyncMock

import duckdb
import pytest

from app.config import AppConfig
from app.core.llm import LLMResponse
from app.core.queue import Queue
from app.db import queries
from app.db.schema import create_tables
from app.models import ThreadStatus
from app.orchestration.thread import resume_thread, run_thread_loop


@pytest.fixture
def integration_setup(tmp_path):
    """Set up main DB, session DB, config, mock LLM, and queue."""
    main_db = duckdb.connect(":memory:")
    create_tables(main_db)

    session_db = duckdb.connect(":memory:")
    csv_path = "tests/fixtures/sample_dataset.csv"
    session_db.execute(f"CREATE TABLE dataset AS SELECT * FROM read_csv_auto('{csv_path}')")

    config = AppConfig()
    queue = Queue()

    return {
        "main_db": main_db,
        "session_db": session_db,
        "config": config,
        "queue": queue,
    }


def _make_coordinator_response(status, move, instruction=None, question=None, context=None):
    """Helper to build coordinator mock responses."""
    data = {
        "assessment": f"Assessment for {move}",
        "next_move": move,
        "rationale": "test rationale",
        "status": status,
    }
    if instruction:
        data["worker_instruction"] = instruction
    if question:
        data["question_for_human"] = question
    if context:
        data["context"] = context
    return json.dumps(data)


def _make_worker_response(summary, sql="SELECT 1"):
    """Helper to build worker mock responses."""
    return json.dumps({
        "queries_executed": [{"purpose": "test", "sql": sql, "key_results": "test"}],
        "summary": summary,
        "details": None,
        "view_requested": None,
    })


@pytest.mark.asyncio
async def test_thread_loop_three_steps_done(integration_setup):
    """Thread runs 3 steps then completes."""
    setup = integration_setup
    main_db = setup["main_db"]

    session = queries.create_session(main_db, "test.csv")
    thread = queries.create_thread(main_db, session.id, "Test question?", "Motivation", "Entry")

    # Mock LLM: coordinator returns CONTINUE twice, then DONE
    # Worker always returns immediately (no tool calls)
    mock_llm = AsyncMock()
    call_count = [0]

    async def mock_call(model, messages, role, temperature=0.0, tools=None, cache_ttl_hours=0, max_tokens=4096):
        call_count[0] += 1
        if role == "coordinator":
            step_num = len(queries.get_steps(main_db, thread.id))
            if step_num < 2:
                content = _make_coordinator_response("CONTINUE", "FORAGE", "Run query")
            else:
                content = _make_coordinator_response("DONE", "SYNTHESIZE", "Final summary")
            return LLMResponse(content=content, model=model)
        elif role == "worker":
            if tools:
                # First call with tools — return final answer directly (no tool use)
                return LLMResponse(
                    content=_make_worker_response(f"Step result {call_count[0]}"),
                    model=model, tool_calls=None,
                )
            return LLMResponse(
                content=_make_worker_response(f"Step result {call_count[0]}"),
                model=model,
            )
        return LLMResponse(content="{}", model=model)

    mock_llm.call = mock_call

    await run_thread_loop(
        config=setup["config"],
        llm=mock_llm,
        main_db=main_db,
        session_db=setup["session_db"],
        queue=setup["queue"],
        thread=thread,
        schema_summary="test schema",
    )

    # Verify 3 steps were created
    steps = queries.get_steps(main_db, thread.id)
    assert len(steps) == 3

    # Verify thread is COMPLETE
    final_thread = queries.get_thread(main_db, thread.id)
    assert final_thread.status == ThreadStatus.COMPLETE


@pytest.mark.asyncio
async def test_thread_loop_stuck_then_resume(integration_setup):
    """Thread gets stuck, human replies, thread resumes and completes."""
    setup = integration_setup
    main_db = setup["main_db"]

    session = queries.create_session(main_db, "test.csv")
    thread = queries.create_thread(main_db, session.id, "Hard question?", "Complex", "Start here")

    phase = {"value": "initial"}

    async def mock_call(model, messages, role, temperature=0.0, tools=None, cache_ttl_hours=0, max_tokens=4096):
        if role == "coordinator":
            if phase["value"] == "initial":
                # First run: get stuck after 1 step
                steps = queries.get_steps(main_db, thread.id)
                if len(steps) == 0:
                    return LLMResponse(
                        content=_make_coordinator_response("CONTINUE", "FORAGE", "Explore"),
                        model=model,
                    )
                else:
                    return LLMResponse(
                        content=_make_coordinator_response(
                            "STUCK", "INTERROGATE",
                            question="Is this pattern real?",
                            context="Found something odd",
                        ),
                        model=model,
                    )
            else:
                # After resume: complete
                steps = queries.get_steps(main_db, thread.id)
                if len(steps) < 3:
                    return LLMResponse(
                        content=_make_coordinator_response("CONTINUE", "INTERROGATE", "Verify"),
                        model=model,
                    )
                else:
                    return LLMResponse(
                        content=_make_coordinator_response("DONE", "SYNTHESIZE", "Conclude"),
                        model=model,
                    )
        elif role == "worker":
            return LLMResponse(
                content=_make_worker_response("Result"),
                model=model, tool_calls=None,
            )
        return LLMResponse(content="{}", model=model)

    mock_llm = AsyncMock()
    mock_llm.call = mock_call

    # Run until stuck
    await run_thread_loop(
        config=setup["config"],
        llm=mock_llm,
        main_db=main_db,
        session_db=setup["session_db"],
        queue=setup["queue"],
        thread=thread,
        schema_summary="test schema",
    )

    # Should be WAITING with 1 step
    t = queries.get_thread(main_db, thread.id)
    assert t.status == ThreadStatus.WAITING
    assert len(queries.get_steps(main_db, thread.id)) == 1

    # Resume
    phase["value"] = "resumed"
    await resume_thread(
        config=setup["config"],
        llm=mock_llm,
        main_db=main_db,
        session_db=setup["session_db"],
        queue=setup["queue"],
        thread_id=thread.id,
        human_message="Yes, this is a known effect",
        schema_summary="test schema",
    )

    # Should be COMPLETE with more steps
    t = queries.get_thread(main_db, thread.id)
    assert t.status == ThreadStatus.COMPLETE
    assert len(queries.get_steps(main_db, thread.id)) >= 3


@pytest.mark.asyncio
async def test_thread_emits_events(integration_setup):
    """Verify SSE events are emitted during thread execution."""
    setup = integration_setup
    main_db = setup["main_db"]
    queue = setup["queue"]

    session = queries.create_session(main_db, "test.csv")
    thread = queries.create_thread(main_db, session.id, "Event test?", "", "")

    # Subscribe to events
    event_queue = queue.subscribe(session.id)

    call_count = [0]

    async def mock_call(model, messages, role, temperature=0.0, tools=None, cache_ttl_hours=0, max_tokens=4096):
        call_count[0] += 1
        if role == "coordinator":
            return LLMResponse(
                content=_make_coordinator_response("DONE", "SYNTHESIZE", "wrap up"),
                model=model,
            )
        elif role == "worker":
            return LLMResponse(
                content=_make_worker_response("Final finding"),
                model=model, tool_calls=None,
            )
        return LLMResponse(content="{}", model=model)

    mock_llm = AsyncMock()
    mock_llm.call = mock_call

    await run_thread_loop(
        config=setup["config"],
        llm=mock_llm,
        main_db=main_db,
        session_db=setup["session_db"],
        queue=queue,
        thread=thread,
        schema_summary="test schema",
    )

    # Collect emitted events
    events = []
    while not event_queue.empty():
        events.append(await event_queue.get())

    event_types = [e.event_type for e in events]
    assert "step_completed" in event_types
    assert "thread_complete" in event_types

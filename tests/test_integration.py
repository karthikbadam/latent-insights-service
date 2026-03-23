"""Integration tests — state management, tracing, and thread flows with mock LLM."""

import json
from unittest.mock import MagicMock

import duckdb
import pytest

from app.config import AppConfig
from app.core.llm import LLMResponse
from app.core.queue import Queue
from app.core.state import StateStore, generate_id
from app.core.tracing import TraceStore
from app.models import ThreadStatus
from app.orchestration.thread import ThreadRunner


# --- StateStore unit tests ---


def test_generate_id_format():
    id_ = generate_id()
    assert len(id_) == 12
    assert all(c in "0123456789abcdef" for c in id_)


def test_generate_id_unique():
    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100


def test_create_and_get_session(state_store):
    session = state_store.create_session("/data/test.csv")
    assert session.dataset_path == "/data/test.csv"
    assert session.schema_summary is None

    fetched = state_store.get_session(session.id)
    assert fetched is not None
    assert fetched.id == session.id


def test_get_session_not_found(state_store):
    assert state_store.get_session("nonexistent") is None


def test_update_session_schema(state_store):
    session = state_store.create_session("/data/test.csv")
    state_store.update_session_schema(session.id, "## Schema\nTest summary")

    fetched = state_store.get_session(session.id)
    assert fetched.schema_summary == "## Schema\nTest summary"


def test_update_session_scout(state_store):
    session = state_store.create_session("/data/test.csv")
    scout_data = {"exploration_notes": "test", "questions": [{"q": "why?"}]}
    state_store.update_session_scout(session.id, scout_data)

    fetched = state_store.get_session(session.id)
    assert fetched.scout_output == scout_data


def test_create_and_get_thread(state_store):
    session = state_store.create_session("/data/test.csv")
    thread = state_store.create_thread(session.id, "Why gaps?", "Detection bias", "Check distributions")

    assert thread.seed_question == "Why gaps?"
    assert thread.status == ThreadStatus.RUNNING

    fetched = state_store.get_thread(thread.id)
    assert fetched is not None
    assert fetched.seed_question == "Why gaps?"
    assert fetched.motivation == "Detection bias"


def test_get_threads_ordered(state_store):
    session = state_store.create_session("/data/test.csv")
    state_store.create_thread(session.id, "First")
    state_store.create_thread(session.id, "Second")
    state_store.create_thread(session.id, "Third")

    threads = state_store.get_threads(session.id)
    assert len(threads) == 3
    assert threads[0].seed_question == "First"
    assert threads[2].seed_question == "Third"


def test_update_thread_status(state_store):
    session = state_store.create_session("/data/test.csv")
    thread = state_store.create_thread(session.id, "Test")

    state_store.update_thread_status(thread.id, ThreadStatus.WAITING)
    assert state_store.get_thread(thread.id).status == ThreadStatus.WAITING

    state_store.update_thread_status(thread.id, ThreadStatus.COMPLETE, summary="Done")
    fetched = state_store.get_thread(thread.id)
    assert fetched.status == ThreadStatus.COMPLETE
    assert fetched.summary == "Done"


def test_dump_and_load_session(state_store):
    session = state_store.create_session("/data/test.csv", "test_table")
    state_store.update_session_schema(session.id, "schema text")
    thread = state_store.create_thread(session.id, "Question?", "Why", "Start")
    state_store.update_thread_status(thread.id, ThreadStatus.COMPLETE, summary="Found it")

    state_store.dump_session(session.id)

    # Create a fresh store and load
    fresh = StateStore(data_dir=state_store._data_dir)
    loaded = fresh.load_session(session.id)

    assert loaded is not None
    assert loaded.schema_summary == "schema text"
    assert loaded.table_name == "test_table"

    threads = fresh.get_threads(session.id)
    assert len(threads) == 1
    assert threads[0].status == ThreadStatus.COMPLETE
    assert threads[0].summary == "Found it"


def test_counts(state_store):
    assert state_store.session_count == 0
    assert state_store.thread_count == 0

    s = state_store.create_session("/data/test.csv")
    state_store.create_thread(s.id, "Q1")
    state_store.create_thread(s.id, "Q2")

    assert state_store.session_count == 1
    assert state_store.thread_count == 2


# --- TraceStore unit tests ---


def test_trace_format_empty(tmp_path):
    ts = TraceStore(data_dir=str(tmp_path))
    result = ts.format_thread_history("nonexistent")
    assert "No steps yet" in result


def test_trace_format_windowed(tmp_path):
    ts = TraceStore(data_dir=str(tmp_path))
    trace_id = "test-trace"

    # Create 6 steps
    for i in range(6):
        span = ts.start_span(trace_id, f"step_{i+1}", kind="step")
        span.attributes = {
            "move": "FORAGE",
            "instruction": f"instruction {i+1}",
            "result": f"Result {i+1}. Some detail here.",
        }
        ts.end_span(span)

    result = ts.format_thread_history(trace_id, full_window=3)

    # Step 1 should be full
    assert 'Instruction: "instruction 1"' in result
    # Steps 2-3 should be condensed (only first sentence)
    assert "Step 2 [FORAGE]: Result 2." in result
    assert "Step 3 [FORAGE]: Result 3." in result
    # Steps 4-6 should be full
    assert 'Instruction: "instruction 4"' in result
    assert 'Instruction: "instruction 5"' in result
    assert 'Instruction: "instruction 6"' in result


def test_trace_format_with_summary(tmp_path):
    ts = TraceStore(data_dir=str(tmp_path))
    trace_id = "test-trace"

    span = ts.start_span(trace_id, "step_1", kind="step")
    span.attributes = {"move": "SCOPE", "instruction": "test", "result": "scoped"}
    ts.end_span(span)

    result = ts.format_thread_history(
        trace_id, running_summary="Earlier we found X and Y.",
    )
    assert "Summary of earlier analysis" in result
    assert "Earlier we found X and Y." in result


def test_trace_format_with_human_messages(tmp_path):
    ts = TraceStore(data_dir=str(tmp_path))
    trace_id = "test-trace"

    span = ts.start_span(trace_id, "step_1", kind="step")
    span.attributes = {"move": "FORAGE", "instruction": "Explore", "result": "Found gap"}
    ts.end_span(span)

    result = ts.format_thread_history(trace_id, human_messages=["Check by stellar type"])
    assert "[Human input]" in result
    assert "Check by stellar type" in result


def test_trace_flush_and_load(tmp_path):
    ts = TraceStore(data_dir=str(tmp_path))
    trace_id = "test-trace"
    session_id = "test-session"

    span = ts.start_span(trace_id, "step_1", kind="step")
    span.attributes = {"move": "SCOPE", "instruction": "filter", "result": "done"}
    ts.end_span(span)

    ts.flush_to_file(trace_id, session_id)
    ts.clear_trace(trace_id)
    assert ts.get_spans(trace_id) == []

    loaded = ts.load_trace(trace_id, session_id)
    assert len(loaded) == 1
    assert loaded[0].attributes["move"] == "SCOPE"


# --- Thread state machine integration tests ---


@pytest.fixture
def integration_setup(tmp_path):
    """Set up session DB, config, mock LLM, queue, state, and trace store."""
    session_db = duckdb.connect(":memory:")
    csv_path = "tests/fixtures/sample_dataset.csv"
    session_db.execute(f"CREATE TABLE dataset AS SELECT * FROM read_csv_auto('{csv_path}')")

    config = AppConfig()
    queue = Queue()
    state = StateStore(data_dir=str(tmp_path))
    trace_store = TraceStore(data_dir=str(tmp_path))

    return {
        "session_db": session_db,
        "config": config,
        "queue": queue,
        "state": state,
        "trace_store": trace_store,
    }


def _make_coordinator_response(status, move, instruction=None, question=None, context=None):
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


def _make_worker_response(summary):
    return json.dumps({
        "summary": summary,
        "details": None,
        "view_requested": None,
    })


def _build_runner(setup, thread, session_db=None, human_messages=None):
    """Build a ThreadRunner for testing."""
    return ThreadRunner(
        config=setup["config"],
        llm=MagicMock(),
        session_db=session_db or setup["session_db"],
        queue=setup["queue"],
        state=setup["state"],
        trace_store=setup["trace_store"],
        thread=thread,
        schema_summary="test schema",
        human_messages=human_messages or [],
    )


def test_thread_loop_three_steps_done(integration_setup):
    """Thread runs 3 steps then completes."""
    setup = integration_setup
    state = setup["state"]

    session = state.create_session("test.csv")
    thread = state.create_thread(session.id, "Test question?", "Motivation", "Entry")

    coordinator_calls = [0]

    def mock_call(model, messages, role, temperature=0.0, tools=None, max_tokens=4096, timeout=120.0):
        if role == "coordinator":
            coordinator_calls[0] += 1
            if coordinator_calls[0] < 3:
                content = _make_coordinator_response("CONTINUE", "FORAGE", "Run query")
            else:
                content = _make_coordinator_response("DONE", "SYNTHESIZE", "Final summary")
            return LLMResponse(content=content, model=model)
        elif role == "worker":
            return LLMResponse(
                content=_make_worker_response(f"Step result {coordinator_calls[0]}"),
                model=model, tool_calls=None,
            )
        return LLMResponse(content="{}", model=model)

    runner = _build_runner(setup, thread)
    runner.llm.call = mock_call
    runner.coordinator.llm.call = mock_call
    runner.worker.llm.call = mock_call

    runner.start()
    runner.done_event.wait(timeout=10)

    final_thread = state.get_thread(thread.id)
    assert final_thread.status == ThreadStatus.COMPLETE
    assert final_thread.summary is not None


def test_thread_loop_stuck_then_resume(integration_setup):
    """Thread gets stuck, human replies, thread resumes and completes."""
    setup = integration_setup
    state = setup["state"]

    session = state.create_session("test.csv")
    thread = state.create_thread(session.id, "Hard question?", "Complex", "Start here")

    phase = {"value": "initial"}
    coordinator_calls = [0]

    def mock_call(model, messages, role, temperature=0.0, tools=None, max_tokens=4096, timeout=120.0):
        if role == "coordinator":
            coordinator_calls[0] += 1
            if phase["value"] == "initial":
                if coordinator_calls[0] == 1:
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
                if coordinator_calls[0] <= 4:
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

    # Run until stuck
    runner = _build_runner(setup, thread)
    runner.llm.call = mock_call
    runner.coordinator.llm.call = mock_call
    runner.worker.llm.call = mock_call

    runner.start()
    runner.done_event.wait(timeout=10)

    t = state.get_thread(thread.id)
    assert t.status == ThreadStatus.WAITING

    # Resume with human message
    phase["value"] = "resumed"

    session_db2 = duckdb.connect(":memory:")
    session_db2.execute(
        "CREATE TABLE dataset AS SELECT * FROM read_csv_auto('tests/fixtures/sample_dataset.csv')"
    )

    runner2 = _build_runner(setup, thread, session_db=session_db2, human_messages=["Yes, this is a known effect"])
    runner2.llm.call = mock_call
    runner2.coordinator.llm.call = mock_call
    runner2.worker.llm.call = mock_call

    runner2.resume()
    runner2.done_event.wait(timeout=10)

    t = state.get_thread(thread.id)
    assert t.status == ThreadStatus.COMPLETE


def test_thread_emits_events(integration_setup):
    """Verify SSE events are emitted during thread execution."""
    setup = integration_setup
    state = setup["state"]
    queue = setup["queue"]

    session = state.create_session("test.csv")
    thread = state.create_thread(session.id, "Event test?", "", "")

    event_queue = queue.subscribe(session.id)

    def mock_call(model, messages, role, temperature=0.0, tools=None, max_tokens=4096, timeout=120.0):
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

    runner = _build_runner(setup, thread)
    runner.llm.call = mock_call
    runner.coordinator.llm.call = mock_call
    runner.worker.llm.call = mock_call

    runner.start()
    runner.done_event.wait(timeout=10)

    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())

    event_types = [e.event_type for e in events]
    assert "thread_start" in event_types
    assert "step_start" in event_types
    assert "step_complete" in event_types
    assert "thread_complete" in event_types


def test_thread_move_repetition_guard(integration_setup):
    """Thread that repeats the same move N times gets forced to STUCK/WAITING."""
    setup = integration_setup
    state = setup["state"]

    session = state.create_session("test.csv")
    thread = state.create_thread(session.id, "Repetitive?", "Test", "Start")

    def mock_call(model, messages, role, temperature=0.0, tools=None, max_tokens=4096, timeout=120.0):
        if role == "coordinator":
            return LLMResponse(
                content=_make_coordinator_response("CONTINUE", "FORAGE", "Keep exploring"),
                model=model,
            )
        elif role == "worker":
            return LLMResponse(
                content=_make_worker_response("Same old result"),
                model=model, tool_calls=None,
            )
        return LLMResponse(content="{}", model=model)

    runner = _build_runner(setup, thread)
    runner.llm.call = mock_call
    runner.coordinator.llm.call = mock_call
    runner.worker.llm.call = mock_call

    runner.start()
    runner.done_event.wait(timeout=10)

    t = state.get_thread(thread.id)
    assert t.status == ThreadStatus.WAITING


def test_thread_error_becomes_waiting(integration_setup):
    """Thread errors become WAITING (not ERROR) so human can help."""
    setup = integration_setup
    state = setup["state"]

    session = state.create_session("test.csv")
    thread = state.create_thread(session.id, "Will error?", "Test", "Start")

    call_count = [0]

    def mock_call(model, messages, role, temperature=0.0, tools=None, max_tokens=4096, timeout=120.0):
        call_count[0] += 1
        if role == "coordinator":
            raise RuntimeError("Simulated LLM failure")
        return LLMResponse(content="{}", model=model)

    runner = _build_runner(setup, thread)
    runner.llm.call = mock_call
    runner.coordinator.llm.call = mock_call
    runner.worker.llm.call = mock_call

    runner.start()
    runner.done_event.wait(timeout=10)

    t = state.get_thread(thread.id)
    assert t.status == ThreadStatus.WAITING

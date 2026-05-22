"""Regression tests for the two ordering bugs the FeedEntry refactor fixed.

1. Coordinator ``step_start`` must precede ``llm_call`` for the same step
   so the frontend's "attach to last started step" reducer puts the
   call on the right row.

2. ``message_injected`` is gone; HUMAN_INPUT is committed inline in the
   POST handler so worker events from the previous step can't be
   misattributed to a not-yet-existent HUMAN_INPUT slot.
"""

import json
from unittest.mock import MagicMock

import duckdb
import pytest

from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMResponse
from latent_insights.core.queue import Queue
from latent_insights.core.recorder import Recorder
from latent_insights.core.store import InvestigationStore
from latent_insights.models import ThreadStatus
from latent_insights.orchestration.runner import ThreadRunner


# Reuse the same fixture shape as test_integration.py.
@pytest.fixture
def setup(tmp_path):
    session_db = duckdb.connect(":memory:")
    session_db.execute(
        "CREATE TABLE dataset AS "
        "SELECT * FROM read_csv_auto('tests/fixtures/sample_dataset.csv')"
    )
    return {
        "session_db": session_db,
        "config": AppConfig(),
        "queue": Queue(),
        "store": InvestigationStore(data_dir=str(tmp_path)),
    }


def _coord(status, move, instruction=None, question=None, context=None):
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


def _worker(summary):
    return json.dumps({"summary": summary, "details": None, "view_requested": None})


def _runner(setup, thread):
    return ThreadRunner(
        config=setup["config"],
        llm=MagicMock(),
        session_db=setup["session_db"],
        queue=setup["queue"],
        store=setup["store"],
        thread=thread,
        schema_summary="test schema",
    )


# ---------------------------------------------------------------------------
# Bug 1: step_start must precede llm_call for every step
# ---------------------------------------------------------------------------


def test_coordinator_step_start_precedes_llm_call(setup):
    store = setup["store"]
    queue = setup["queue"]

    session = store.create_session("test.csv")
    thread = store.create_thread(session.id, "Q?", "m", "e")
    event_queue = queue.subscribe(session.id)

    coord_calls = [0]

    def mock_call(model, messages, role, **_kwargs):
        if role == "coordinator":
            coord_calls[0] += 1
            if coord_calls[0] == 1:
                return LLMResponse(
                    content=_coord("CONTINUE", "FORAGE", "explore"), model=model,
                )
            return LLMResponse(
                content=_coord("DONE", "SYNTHESIZE", "wrap"), model=model,
            )
        return LLMResponse(content=_worker("result"), model=model, tool_calls=None)

    runner = _runner(setup, thread)
    runner.coordinator.llm.call = mock_call
    runner.worker.llm.call = mock_call
    runner.start()
    runner.done_event.wait(timeout=10)

    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())

    # For every (thread_id, step_number) pair, step_start's feed_index
    # must be lower than any llm_call's feed_index on the same step.
    starts: dict[tuple[str, int], int] = {}
    for ev in events:
        if ev.event_type != "step_start":
            continue
        key = (ev.thread_id, ev.data.get("step_number"))
        starts[key] = ev.data.get("feed_index", -1)

    for ev in events:
        if ev.event_type != "llm_call":
            continue
        key = (ev.thread_id, ev.data.get("step_number"))
        assert key in starts, f"llm_call on {key} has no step_start"
        assert ev.data.get("feed_index", -1) > starts[key], (
            f"llm_call on {key} came before step_start "
            f"(llm={ev.data.get('feed_index')} vs start={starts[key]})"
        )


# ---------------------------------------------------------------------------
# Bug 2: HUMAN_INPUT committed inline before the response returns
# ---------------------------------------------------------------------------


def test_human_input_lands_in_store_before_runner_drain(setup):
    """Simulates the route handler: commit HUMAN_INPUT via Recorder
    INLINE, then push a pivot marker. The store immediately reflects
    the step; the runner's drain treats the marker as a no-op.
    """
    store = setup["store"]
    queue = setup["queue"]

    session = store.create_session("test.csv")
    thread = store.create_thread(session.id, "Q?", "m", "e")

    # Inline commit (what POST /messages does for a RUNNING thread).
    recorder = Recorder(store, queue, session.id, thread.id)
    recorder.human_input_step("focus on outliers", target="thread")
    store.push_pivot_marker(thread.id)

    # The step is visible NOW, before any runner callback fires.
    steps = store.get_steps(thread.id)
    assert any(s.move == "HUMAN_INPUT" for s in steps), (
        "HUMAN_INPUT step must exist inline, not on next callback"
    )
    assert store.has_pending_messages(thread.id), (
        "pivot marker must keep has_pending_messages true so the runner pivots"
    )


def test_runner_drain_skips_committed_markers(setup):
    """The runner's drain ignores ``{committed: True}`` markers but
    still records that the pivot was requested.
    """
    store = setup["store"]
    queue = setup["queue"]

    session = store.create_session("test.csv")
    thread = store.create_thread(session.id, "Q?", "m", "e")
    event_queue = queue.subscribe(session.id)

    # Pre-commit a HUMAN_INPUT inline, then push the marker.
    Recorder(store, queue, session.id, thread.id).human_input_step(
        "look at A", target="thread",
    )
    store.push_pivot_marker(thread.id)

    coord_calls = [0]

    def mock_call(model, messages, role, **_kwargs):
        if role == "coordinator":
            coord_calls[0] += 1
            if coord_calls[0] == 1:
                return LLMResponse(
                    content=_coord("CONTINUE", "FORAGE", "go"), model=model,
                )
            return LLMResponse(
                content=_coord("DONE", "SYNTHESIZE", "wrap"), model=model,
            )
        return LLMResponse(content=_worker("ok"), model=model, tool_calls=None)

    runner = _runner(setup, thread)
    runner.coordinator.llm.call = mock_call
    runner.worker.llm.call = mock_call
    runner.start()
    runner.done_event.wait(timeout=10)

    # After the run, there is still exactly ONE HUMAN_INPUT step (the
    # one committed inline). The runner's drain did not add a second.
    steps = store.get_steps(thread.id)
    human_steps = [s for s in steps if s.move == "HUMAN_INPUT"]
    assert len(human_steps) == 1, (
        f"expected 1 HUMAN_INPUT step, found {len(human_steps)}"
    )
    assert human_steps[0].result == "look at A"

    # And exactly one human_message SSE event for that step.
    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    human_events = [e for e in events if e.event_type == "human_message"]
    assert len(human_events) == 1


def test_message_injected_event_is_never_emitted(setup):
    """The legacy ``message_injected`` event is dropped from the stream."""
    store = setup["store"]
    queue = setup["queue"]

    session = store.create_session("test.csv")
    thread = store.create_thread(session.id, "Q?", "m", "e")
    event_queue = queue.subscribe(session.id)

    # Simulate the inline-commit path used by the route handler.
    Recorder(store, queue, session.id, thread.id).human_input_step(
        "guide me", target="session",
    )

    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    assert all(e.event_type != "message_injected" for e in events)


def test_thread_status_running_check_at_inline_commit(setup):
    """RUNNING thread: pivot marker keeps has_pending_messages alive so
    the runner's next ``_flush_and_pivot`` boundary fires.
    """
    store = setup["store"]
    queue = setup["queue"]

    session = store.create_session("test.csv")
    thread = store.create_thread(session.id, "Q?", "m", "e")

    # Simulate the RUNNING-thread branch.
    Recorder(store, queue, session.id, thread.id).human_input_step(
        "x", target="thread",
    )
    store.push_pivot_marker(thread.id)

    # Marker stays in pending until the runner drains it.
    assert store.has_pending_messages(thread.id)
    drained = store.drain_pending_messages(thread.id)
    assert drained == [{"committed": True}]
    assert not store.has_pending_messages(thread.id)


# ---------------------------------------------------------------------------
# SSE payload shape — every event carries a feed_index
# ---------------------------------------------------------------------------


def test_every_emitted_event_carries_feed_index(setup):
    store = setup["store"]
    queue = setup["queue"]

    session = store.create_session("test.csv")
    thread = store.create_thread(session.id, "Q?", "m", "e")
    event_queue = queue.subscribe(session.id)

    def mock_call(model, messages, role, **_kwargs):
        if role == "coordinator":
            return LLMResponse(
                content=_coord("DONE", "SYNTHESIZE", "wrap"), model=model,
            )
        return LLMResponse(content=_worker("done"), model=model, tool_calls=None)

    runner = _runner(setup, thread)
    runner.coordinator.llm.call = mock_call
    runner.worker.llm.call = mock_call
    runner.start()
    runner.done_event.wait(timeout=10)

    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())

    assert events, "expected at least one event"
    indices = [e.data["feed_index"] for e in events]
    # Monotonic and dense (each event takes one index).
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices), "feed_indices must be unique"
    assert indices[0] == 0
    assert indices[-1] == len(indices) - 1

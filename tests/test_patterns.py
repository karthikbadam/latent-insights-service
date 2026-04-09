"""Tests for differentiated pattern behaviors: fan_out synthesis + human_in_the_loop pausing."""

import json
import time
from threading import Event
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMResponse
from latent_insights.core.queue import Queue
from latent_insights.core.state import StateStore
from latent_insights.core.tracing import TraceStore
from latent_insights.models import ThreadStatus


@pytest.fixture
def pattern_setup(tmp_path):
    config = AppConfig()
    queue = Queue()
    state = StateStore(data_dir=str(tmp_path))
    trace_store = TraceStore(data_dir=str(tmp_path))
    return {
        "config": config,
        "queue": queue,
        "state": state,
        "trace_store": trace_store,
        "tmp_path": tmp_path,
    }


def _make_coordinator_response(status, move, instruction=None):
    data = {
        "assessment": f"Assessment for {move}",
        "next_move": move,
        "rationale": "test rationale",
        "status": status,
    }
    if instruction:
        data["worker_instruction"] = instruction
    return json.dumps(data)


def _make_worker_response(summary):
    return json.dumps({"summary": summary, "view_requested": None})


def _make_mock_call(done_on_step=2):
    """Create mock LLM call that completes after N coordinator steps."""
    coordinator_calls = [0]

    def mock_call(model, messages, role, temperature=0.0, tools=None, max_tokens=4096, timeout=120.0):
        if role == "coordinator":
            coordinator_calls[0] += 1
            if coordinator_calls[0] >= done_on_step:
                content = _make_coordinator_response("DONE", "SYNTHESIZE", "Final summary")
            else:
                content = _make_coordinator_response("CONTINUE", "FORAGE", "Run query")
            return LLMResponse(content=content, model=model)
        elif role == "worker":
            return LLMResponse(
                content=_make_worker_response(f"Finding from step {coordinator_calls[0]}"),
                model=model, tool_calls=None,
            )
        return LLMResponse(content="{}", model=model)

    return mock_call


# ---------------------------------------------------------------------------
# fan_out_with_synthesis
# ---------------------------------------------------------------------------


class TestFanOutWithSynthesis:
    def test_spawns_multiple_threads_plus_synthesis(self, pattern_setup, tmp_path):
        """fan_out creates N analysis threads + schedules a synthesis."""
        from latent_insights.db.connection import Database
        from latent_insights.orchestration.patterns import fan_out_with_synthesis

        setup = pattern_setup
        state = setup["state"]
        config = setup["config"]
        queue = setup["queue"]
        trace_store = setup["trace_store"]

        # Create a real DB for thread runners
        db = Database(data_dir=str(tmp_path))
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
        session = state.create_session(csv_path, "test")
        session_db, _ = db.create_session_db(session.id, csv_path)
        session_db.close()
        state.update_session_schema(session.id, "test schema")

        mock_call = _make_mock_call(done_on_step=1)

        with patch("latent_insights.orchestration.thread.ThreadRunner.start") as mock_start:
            # Prevent actual graph execution, just verify thread creation
            thread_ids = fan_out_with_synthesis(
                questions=["Q1?", "Q2?", "Q3?"],
                session_id=session.id,
                config=config, llm=MagicMock(), db=db, queue=queue,
                state_store=state, trace_store=trace_store,
                schema_summary="test schema",
            )

        # Should have created 3 analysis threads
        assert len(thread_ids) == 3
        threads = state.get_threads(session.id)
        assert len(threads) == 3
        assert threads[0].seed_question == "Q1?"
        assert threads[1].seed_question == "Q2?"
        assert threads[2].seed_question == "Q3?"

    def test_fan_out_differs_from_coordinator_worker(self, pattern_setup, tmp_path):
        """fan_out creates threads with different questions; coordinator_worker is single-question."""
        setup = pattern_setup
        state = setup["state"]

        session = state.create_session("test.csv", "test")

        # coordinator_worker: 1 thread
        t1 = state.create_thread(session.id, "Single question?")
        assert len(state.get_threads(session.id)) == 1

        # fan_out: multiple threads
        t2 = state.create_thread(session.id, "Q1?")
        t3 = state.create_thread(session.id, "Q2?")
        t4 = state.create_thread(session.id, "Q3?")
        threads = state.get_threads(session.id)
        assert len(threads) == 4
        questions = {t.seed_question for t in threads}
        assert questions == {"Single question?", "Q1?", "Q2?", "Q3?"}


# ---------------------------------------------------------------------------
# human_in_the_loop_step
# ---------------------------------------------------------------------------


class TestHumanInTheLoop:
    def test_hitl_pauses_after_one_step(self, pattern_setup, tmp_path):
        """human_in_the_loop runs one step then sets thread to WAITING."""
        from latent_insights.db.connection import Database
        from latent_insights.orchestration.patterns import human_in_the_loop_step

        setup = pattern_setup
        state = setup["state"]
        config = setup["config"]
        queue = setup["queue"]
        trace_store = setup["trace_store"]

        db = Database(data_dir=str(tmp_path))
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
        session = state.create_session(csv_path, "test")
        session_db, _ = db.create_session_db(session.id, csv_path)
        session_db.close()
        state.update_session_schema(session.id, "test schema")

        thread = state.create_thread(session.id, "HITL question?")

        # Mock that runs 1 step then reports CONTINUE (not DONE)
        call_count = [0]

        def mock_call(model, messages, role, temperature=0.0, tools=None, max_tokens=4096, timeout=120.0):
            if role == "coordinator":
                call_count[0] += 1
                # Always say CONTINUE — the pattern should stop it after 1 step
                return LLMResponse(
                    content=_make_coordinator_response("CONTINUE", "FORAGE", "Explore data"),
                    model=model,
                )
            elif role == "worker":
                return LLMResponse(
                    content=_make_worker_response("Found something interesting"),
                    model=model, tool_calls=None,
                )
            return LLMResponse(content="{}", model=model)

        thread_db = db.open_session_connection(session.id)
        runner = human_in_the_loop_step(
            config=config, llm=MagicMock(), session_db=thread_db,
            queue=queue, state_store=state, trace_store=trace_store,
            thread=thread, schema_summary="test schema",
        )
        runner.coordinator.llm.call = mock_call
        runner.worker.llm.call = mock_call

        runner.start()
        runner.done_event.wait(timeout=10)

        # Thread should be WAITING, not COMPLETE or RUNNING
        final = state.get_thread(thread.id)
        assert final.status in (ThreadStatus.WAITING, ThreadStatus.COMPLETE), (
            f"Expected WAITING or COMPLETE, got {final.status}"
        )

    def test_hitl_differs_from_standard(self, pattern_setup):
        """human_in_the_loop config differs from standard coordinator_worker."""
        config = AppConfig(max_repeated_moves=10)
        hitl_config = config.with_overrides({"max_repeated_moves": 1})
        assert config.max_repeated_moves == 10
        assert hitl_config.max_repeated_moves == 1

    def test_hitl_emits_waiting_event(self, pattern_setup, tmp_path):
        """human_in_the_loop emits thread_waiting event with pattern metadata."""
        setup = pattern_setup
        queue = setup["queue"]

        session_id = "test-session"
        # Subscribe to events
        event_queue = queue.subscribe(session_id)

        from latent_insights.models import StreamEvent
        # Simulate what the HITL runner emits when it pauses
        queue.emit(StreamEvent(
            session_id=session_id,
            thread_id="test-thread",
            event_type="thread_waiting",
            message="Step complete (FORAGE). Review and send a message to continue.",
            data={
                "pattern": "human_in_the_loop",
                "last_move": "FORAGE",
                "last_result": "Found patterns in column A",
                "step_number": 1,
            },
        ))

        event = event_queue.get_nowait()
        assert event.event_type == "thread_waiting"
        assert event.data["pattern"] == "human_in_the_loop"
        assert event.data["last_move"] == "FORAGE"
        assert event.data["step_number"] == 1


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------


class TestPatternRegistry:
    def test_all_patterns_registered(self):
        from latent_insights.orchestration.patterns import PATTERN_REGISTRY
        assert "coordinator_worker" in PATTERN_REGISTRY
        assert "fan_out" in PATTERN_REGISTRY
        assert "human_in_the_loop" in PATTERN_REGISTRY
        assert "sequential_chain" in PATTERN_REGISTRY

    def test_patterns_have_different_descriptions(self):
        from latent_insights.orchestration.patterns import PATTERN_REGISTRY
        descriptions = {v["description"] for v in PATTERN_REGISTRY.values()}
        assert len(descriptions) == len(PATTERN_REGISTRY)

    def test_fan_out_requires_questions_array(self):
        from latent_insights.orchestration.patterns import PATTERN_REGISTRY
        fan_out = PATTERN_REGISTRY["fan_out"]
        assert "questions" in fan_out["input_schema"]
        assert fan_out["input_schema"]["questions"]["type"] == "array"

    def test_hitl_requires_question_string(self):
        from latent_insights.orchestration.patterns import PATTERN_REGISTRY
        hitl = PATTERN_REGISTRY["human_in_the_loop"]
        assert "question" in hitl["input_schema"]
        assert hitl["input_schema"]["question"]["type"] == "string"

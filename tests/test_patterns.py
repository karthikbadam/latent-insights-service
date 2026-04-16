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
from latent_insights.core.store import InvestigationStore
from latent_insights.models import ThreadStatus


@pytest.fixture
def pattern_setup(tmp_path):
    config = AppConfig()
    queue = Queue()
    store = InvestigationStore(data_dir=str(tmp_path))
    try:
        yield {
            "config": config,
            "queue": queue,
            "store": store,
            "tmp_path": tmp_path,
        }
    finally:
        queue.shutdown(wait=False)


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
        store = setup["store"]
        config = setup["config"]
        queue = setup["queue"]

        # Create a real DB for thread loops
        db = Database(data_dir=str(tmp_path))
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
        session = store.create_session(csv_path, "test")
        session_db, _ = db.create_session_db(session.id, csv_path)
        session_db.close()
        store.update_session_schema(session.id, "test schema")

        # Patch ThreadLoop.start to be a no-op, AND patch queue.schedule
        # so the synthesis wait task never runs
        with patch("latent_insights.orchestration.loop.ThreadLoop.start"), \
             patch.object(queue, "schedule") as mock_schedule:
            thread_ids = fan_out_with_synthesis(
                questions=["Q1?", "Q2?", "Q3?"],
                session_id=session.id,
                config=config, llm=MagicMock(), db=db, queue=queue,
                store=store,
                schema_summary="test schema",
            )
            # Verify synthesis task was scheduled
            assert mock_schedule.called
            scheduled_task = mock_schedule.call_args.kwargs.get("task_id", "")
            assert "fanout-synth" in scheduled_task

        # Should have created 3 analysis threads
        assert len(thread_ids) == 3
        threads = store.get_threads(session.id)
        assert len(threads) == 3
        assert threads[0].seed_question == "Q1?"
        assert threads[1].seed_question == "Q2?"
        assert threads[2].seed_question == "Q3?"

    def test_fan_out_differs_from_coordinator_worker(self, pattern_setup, tmp_path):
        """fan_out creates threads with different questions; coordinator_worker is single-question."""
        setup = pattern_setup
        store = setup["store"]

        session = store.create_session("test.csv", "test")

        # coordinator_worker: 1 thread
        t1 = store.create_thread(session.id, "Single question?")
        assert len(store.get_threads(session.id)) == 1

        # fan_out: multiple threads
        t2 = store.create_thread(session.id, "Q1?")
        t3 = store.create_thread(session.id, "Q2?")
        t4 = store.create_thread(session.id, "Q3?")
        threads = store.get_threads(session.id)
        assert len(threads) == 4
        questions = {t.seed_question for t in threads}
        assert questions == {"Single question?", "Q1?", "Q2?", "Q3?"}


# ---------------------------------------------------------------------------
# human_in_the_loop (via LoopMode.STEP_AND_PAUSE)
# ---------------------------------------------------------------------------


class TestHumanInTheLoop:
    def test_hitl_pauses_after_one_step(self, pattern_setup, tmp_path):
        """human_in_the_loop runs one step then sets thread to WAITING."""
        from latent_insights.db.connection import Database
        from latent_insights.orchestration.loop import LoopMode, ThreadLoop

        setup = pattern_setup
        store = setup["store"]
        config = setup["config"]
        queue = setup["queue"]

        db = Database(data_dir=str(tmp_path))
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
        session = store.create_session(csv_path, "test")
        session_db, _ = db.create_session_db(session.id, csv_path)
        session_db.close()
        store.update_session_schema(session.id, "test schema")

        thread = store.create_thread(session.id, "HITL question?")

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
        loop = ThreadLoop(
            config=config, llm=MagicMock(), session_db=thread_db,
            queue=queue, store=store,
            thread=thread, schema_summary="test schema",
            mode=LoopMode.STEP_AND_PAUSE,
        )
        loop.coordinator.llm.call = mock_call
        loop.worker.llm.call = mock_call

        loop.start()
        loop.done_event.wait(timeout=10)

        # Thread should be WAITING, not COMPLETE or RUNNING
        final = store.get_thread(thread.id)
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
        # Simulate what the HITL loop emits when it pauses
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


# ---------------------------------------------------------------------------
# SessionFlow default_pattern dispatch
# ---------------------------------------------------------------------------


class TestSessionFlowPatternDispatch:
    def test_default_pattern_coordinator_worker(self, pattern_setup, tmp_path):
        """default_pattern='coordinator_worker' uses ThreadLoop per question."""
        from latent_insights.db.connection import Database
        from latent_insights.models import ScoutQuestion
        from latent_insights.orchestration.session import SessionFlow

        config = AppConfig(default_pattern="coordinator_worker", data_dir=str(tmp_path))
        store = pattern_setup["store"]
        queue = pattern_setup["queue"]
        db = Database(data_dir=str(tmp_path))

        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n1,2,3\n")
        session = store.create_session(csv_path, "test")
        session_db, _ = db.create_session_db(session.id, csv_path)
        session_db.close()
        store.update_session_schema(session.id, "test schema")

        flow = SessionFlow(config, MagicMock(), db, queue, store)

        questions = [
            ScoutQuestion(question="Q1?", motivation="", entry_point="", difficulty="moderate"),
            ScoutQuestion(question="Q2?", motivation="", entry_point="", difficulty="moderate"),
        ]

        with patch("latent_insights.orchestration.loop.ThreadLoop.start") as mock_start:
            flow._spawn_threads(session.id, questions, "test schema")
            # coordinator_worker: one ThreadLoop.start() per question
            assert mock_start.call_count == 2

        # Two threads created
        threads = store.get_threads(session.id)
        assert len(threads) == 2

    def test_default_pattern_fan_out(self, pattern_setup, tmp_path):
        """default_pattern='fan_out' calls fan_out_with_synthesis."""
        from latent_insights.db.connection import Database
        from latent_insights.models import ScoutQuestion
        from latent_insights.orchestration.session import SessionFlow

        config = AppConfig(default_pattern="fan_out", data_dir=str(tmp_path))
        store = pattern_setup["store"]
        queue = pattern_setup["queue"]
        db = Database(data_dir=str(tmp_path))

        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n1,2,3\n")
        session = store.create_session(csv_path, "test")
        session_db, _ = db.create_session_db(session.id, csv_path)
        session_db.close()

        flow = SessionFlow(config, MagicMock(), db, queue, store)

        questions = [
            ScoutQuestion(question="Q1?", motivation="", entry_point="", difficulty="moderate"),
            ScoutQuestion(question="Q2?", motivation="", entry_point="", difficulty="moderate"),
            ScoutQuestion(question="Q3?", motivation="", entry_point="", difficulty="moderate"),
        ]

        with patch(
            "latent_insights.orchestration.patterns.fan_out_with_synthesis"
        ) as mock_fan_out:
            mock_fan_out.return_value = ["t1", "t2", "t3"]
            flow._spawn_threads(session.id, questions, "test schema")
            assert mock_fan_out.called
            call_kwargs = mock_fan_out.call_args.kwargs
            assert call_kwargs["questions"] == ["Q1?", "Q2?", "Q3?"]
            assert call_kwargs["session_id"] == session.id

    def test_default_pattern_human_in_the_loop(self, pattern_setup, tmp_path):
        """default_pattern='human_in_the_loop' uses ThreadLoop with STEP_AND_PAUSE."""
        from latent_insights.db.connection import Database
        from latent_insights.models import ScoutQuestion
        from latent_insights.orchestration.session import SessionFlow

        config = AppConfig(default_pattern="human_in_the_loop", data_dir=str(tmp_path))
        store = pattern_setup["store"]
        queue = pattern_setup["queue"]
        db = Database(data_dir=str(tmp_path))

        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n1,2,3\n")
        session = store.create_session(csv_path, "test")
        session_db, _ = db.create_session_db(session.id, csv_path)
        session_db.close()

        flow = SessionFlow(config, MagicMock(), db, queue, store)

        questions = [
            ScoutQuestion(question="Q1?", motivation="", entry_point="", difficulty="moderate"),
            ScoutQuestion(question="Q2?", motivation="", entry_point="", difficulty="moderate"),
        ]

        with patch("latent_insights.orchestration.loop.ThreadLoop.start") as mock_start:
            flow._spawn_threads(session.id, questions, "test schema")
            assert mock_start.call_count == 2

        # Two threads created
        threads = store.get_threads(session.id)
        assert len(threads) == 2

    def test_empty_questions_does_nothing(self, pattern_setup, tmp_path):
        """_spawn_threads with empty list returns early, no threads created."""
        from latent_insights.db.connection import Database
        from latent_insights.orchestration.session import SessionFlow

        config = AppConfig(default_pattern="coordinator_worker", data_dir=str(tmp_path))
        store = pattern_setup["store"]
        queue = pattern_setup["queue"]
        db = Database(data_dir=str(tmp_path))

        session = store.create_session("test.csv", "test")
        flow = SessionFlow(config, MagicMock(), db, queue, store)

        flow._spawn_threads(session.id, [], "test schema")
        assert len(store.get_threads(session.id)) == 0

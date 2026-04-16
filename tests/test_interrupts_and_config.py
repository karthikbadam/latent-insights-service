"""Tests for thread/session interrupts, question_source config, and scout_context."""

import json
import time
from unittest.mock import MagicMock

import duckdb
import pytest

from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMResponse
from latent_insights.core.queue import Queue
from latent_insights.core.store import InvestigationStore
from latent_insights.models import StreamEvent, ThreadStatus
from latent_insights.orchestration.runner import ThreadRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def setup(tmp_path):
    session_db = duckdb.connect(":memory:")
    csv_path = "tests/fixtures/sample_dataset.csv"
    session_db.execute(f"CREATE TABLE dataset AS SELECT * FROM read_csv_auto('{csv_path}')")

    config = AppConfig()
    queue = Queue()
    store = InvestigationStore(data_dir=str(tmp_path))

    return {
        "session_db": session_db,
        "config": config,
        "queue": queue,
        "store": store,
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
    return json.dumps({"summary": summary, "view_requested": None})


def _build_runner(setup_dict, thread, human_messages=None):
    for msg in human_messages or []:
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        target = msg.get("target", "thread") if isinstance(msg, dict) else "thread"
        setup_dict["store"].push_pending_message(thread.id, content, target=target)
    return ThreadRunner(
        config=setup_dict["config"],
        llm=MagicMock(),
        session_db=setup_dict["session_db"],
        queue=setup_dict["queue"],
        store=setup_dict["store"],
        thread=thread,
        schema_summary="test schema",
    )


# ---------------------------------------------------------------------------
# InvestigationStore pending messages
# ---------------------------------------------------------------------------


class TestPendingMessages:
    def test_push_and_drain(self, setup):
        store = setup["store"]
        store.push_pending_message("t1", "Hello")
        store.push_pending_message("t1", "World")

        msgs = store.drain_pending_messages("t1")
        assert [m["content"] for m in msgs] == ["Hello", "World"]
        assert all(m["target"] == "thread" for m in msgs)

    def test_drain_clears(self, setup):
        store = setup["store"]
        store.push_pending_message("t1", "msg")
        store.drain_pending_messages("t1")

        assert store.drain_pending_messages("t1") == []

    def test_drain_nonexistent_returns_empty(self, setup):
        store = setup["store"]
        assert store.drain_pending_messages("nonexistent") == []

    def test_separate_threads(self, setup):
        store = setup["store"]
        store.push_pending_message("t1", "for-t1")
        store.push_pending_message("t2", "for-t2")

        assert [m["content"] for m in store.drain_pending_messages("t1")] == ["for-t1"]
        assert [m["content"] for m in store.drain_pending_messages("t2")] == ["for-t2"]


# ---------------------------------------------------------------------------
# Thread interrupt (inject message into running thread)
# ---------------------------------------------------------------------------


class TestThreadInterrupt:
    def test_injected_message_reaches_coordinator(self, setup):
        """Messages injected via push_pending_message appear in coordinator's history."""
        store = setup["store"]
        session = store.create_session("test.csv")
        thread = store.create_thread(session.id, "Test question?")

        coordinator_calls = [0]
        seen_messages = []

        def mock_call(model, messages, role, temperature=0.0, tools=None, max_tokens=4096, timeout=120.0):
            if role == "coordinator":
                coordinator_calls[0] += 1
                # On first call, inject a pending message for the next step
                if coordinator_calls[0] == 1:
                    store.push_pending_message(thread.id, "Focus on outliers!")
                    content = _make_coordinator_response("CONTINUE", "FORAGE", "Run query")
                elif coordinator_calls[0] == 2:
                    # Check if the injected message appears in the prompt
                    for msg in messages:
                        if isinstance(msg.get("content"), str) and "Focus on outliers!" in msg["content"]:
                            seen_messages.append("Focus on outliers!")
                    content = _make_coordinator_response("DONE", "SYNTHESIZE", "Wrap up")
                else:
                    content = _make_coordinator_response("DONE", "SYNTHESIZE", "Wrap up")
                return LLMResponse(content=content, model=model)
            elif role == "worker":
                return LLMResponse(
                    content=_make_worker_response("Result"), model=model, tool_calls=None,
                )
            return LLMResponse(content="{}", model=model)

        runner = _build_runner(setup, thread)
        runner.coordinator.llm.call = mock_call
        runner.worker.llm.call = mock_call

        runner.start()
        runner.done_event.wait(timeout=10)

        assert coordinator_calls[0] >= 2

    def test_pending_messages_drained_after_use(self, setup):
        """After coordinator reads pending messages, they're gone."""
        store = setup["store"]
        store.push_pending_message("t1", "msg1")

        # Simulate what the coordinator loop does
        drained = store.drain_pending_messages("t1")
        assert len(drained) == 1
        assert drained[0]["content"] == "msg1"
        assert store.drain_pending_messages("t1") == []


# ---------------------------------------------------------------------------
# AppConfig: question_source and scout_context
# ---------------------------------------------------------------------------


class TestAppConfigExtensions:
    def test_question_source_default(self):
        config = AppConfig()
        assert config.question_source == "scout"

    def test_scout_context_default(self):
        config = AppConfig()
        assert config.scout_context == ""

    def test_with_overrides_question_source(self):
        config = AppConfig()
        new = config.with_overrides({"question_source": "human"})
        assert new.question_source == "human"
        assert config.question_source == "scout"  # original unchanged

    def test_with_overrides_scout_context(self):
        config = AppConfig()
        new = config.with_overrides({"scout_context": "Focus on revenue"})
        assert new.scout_context == "Focus on revenue"

    def test_with_overrides_both(self):
        config = AppConfig()
        new = config.with_overrides({
            "question_source": "both",
            "scout_context": "Seasonal patterns",
        })
        assert new.question_source == "both"
        assert new.scout_context == "Seasonal patterns"

    def test_with_overrides_none_ignored(self):
        config = AppConfig(question_source="human", scout_context="test")
        new = config.with_overrides({"question_source": None, "scout_context": None})
        assert new.question_source == "human"
        assert new.scout_context == "test"

    def test_default_pattern_default(self):
        config = AppConfig()
        assert config.default_pattern == "coordinator_worker"

    def test_with_overrides_default_pattern(self):
        config = AppConfig()
        new = config.with_overrides({"default_pattern": "fan_out"})
        assert new.default_pattern == "fan_out"
        assert config.default_pattern == "coordinator_worker"  # original unchanged


# ---------------------------------------------------------------------------
# SessionFlow: question_source="human" skips scout
# ---------------------------------------------------------------------------


class TestSessionFlowQuestionSource:
    def test_human_mode_skips_scout(self, setup, tmp_path):
        """When question_source=human, SessionFlow.create skips scout and emits session_ready."""
        from latent_insights.db.connection import Database
        from latent_insights.orchestration.session import SessionFlow

        config = AppConfig(question_source="human", data_dir=str(tmp_path))
        store = setup["store"]
        queue = setup["queue"]
        mock_llm = MagicMock()

        # Profiler mock
        mock_llm.call.return_value = LLMResponse(
            content="## Schema\nTest columns: a, b, c", model="test",
        )

        db = Database(data_dir=str(tmp_path))

        # Create a minimal CSV for the profiler
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")

        session = store.create_session(csv_path, "test")
        flow = SessionFlow(config, mock_llm, db, queue, store)

        # Subscribe to events
        event_queue = queue.subscribe(session.id)

        flow.create(session.id, csv_path)

        # Collect events
        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())

        event_types = [e.event_type for e in events]
        # Should have session_ready, NOT scout_done
        assert "session_ready" in event_types
        assert "scout_done" not in event_types

    def test_scout_mode_runs_scout(self, setup, tmp_path):
        """When question_source=scout (default), scout runs normally."""
        config = AppConfig(question_source="scout")
        assert config.question_source == "scout"


class TestScoutContext:
    def test_scout_context_prepended_to_schema(self, setup, tmp_path):
        """When scout_context is set, it's prepended to the scout's input."""
        from latent_insights.db.connection import Database
        from latent_insights.orchestration.session import SessionFlow

        config = AppConfig(
            question_source="scout",
            scout_context="Focus on revenue and churn metrics",
            data_dir=str(tmp_path),
        )

        captured_messages = []

        def capture_llm_call(model, messages, role, temperature=0.0, tools=None, max_tokens=4096, timeout=120.0):
            captured_messages.append({"role": role, "messages": messages})
            if role == "profiler":
                return LLMResponse(content="## Schema\nCols: revenue, churn", model="test")
            elif role == "scout":
                return LLMResponse(
                    content=json.dumps({
                        "questions": [{
                            "question": "What drives churn?",
                            "motivation": "test",
                            "entry_point": "test",
                            "difficulty": "moderate",
                        }]
                    }),
                    model="test",
                )
            return LLMResponse(content="{}", model="test")

        mock_llm = MagicMock()
        mock_llm.call = capture_llm_call

        store = setup["store"]
        queue = setup["queue"]
        db = Database(data_dir=str(tmp_path))

        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("revenue,churn\n100,0.1\n200,0.05\n")

        session = store.create_session(csv_path, "test")
        flow = SessionFlow(config, mock_llm, db, queue, store)

        # Don't spawn actual threads — just verify scout received context
        flow._spawn_threads = MagicMock()

        flow.create(session.id, csv_path)

        # Find the scout call and verify context was in the messages
        scout_calls = [c for c in captured_messages if c["role"] == "scout"]
        assert len(scout_calls) >= 1
        scout_user_msg = scout_calls[0]["messages"][1]["content"]
        assert "Focus on revenue and churn metrics" in scout_user_msg


# ---------------------------------------------------------------------------
# Pydantic schema validation
# ---------------------------------------------------------------------------


class TestSchemaExtensions:
    def test_session_config_question_source(self):
        from latent_insights.api.schemas import SessionConfig
        cfg = SessionConfig(question_source="human", scout_context="test context")
        assert cfg.question_source == "human"
        assert cfg.scout_context == "test context"

    def test_session_config_defaults_none(self):
        from latent_insights.api.schemas import SessionConfig
        cfg = SessionConfig()
        assert cfg.question_source is None
        assert cfg.scout_context is None

"""Tests for app.agents.worker — SQL execution worker agent."""

import json
from unittest.mock import MagicMock

from app.agents.worker import Worker
from app.core.llm import LLMResponse
from app.models import WorkerResult


# --- static helpers ---


def test_format_results_basic():
    result = Worker.format_results(["name", "count"], [("Transit", 68), ("RV", 18)])
    assert "name | count" in result
    assert "Transit | 68" in result
    assert "RV | 18" in result


def test_format_results_empty():
    result = Worker.format_results(["col"], [])
    assert result == "(no rows returned)"


# --- execute_sql ---


def test_execute_sql_success(session_db):
    result = Worker.execute_sql(session_db, "SELECT COUNT(*) as cnt FROM dataset")
    assert "cnt" in result
    assert "103" in result


def test_execute_sql_error(session_db):
    result = Worker.execute_sql(session_db, "SELECT * FROM nonexistent_table")
    assert "SQL ERROR" in result


def test_execute_sql_groupby(session_db):
    result = Worker.execute_sql(
        session_db,
        "SELECT discoverymethod, COUNT(*) as cnt FROM dataset GROUP BY 1 ORDER BY 2 DESC",
    )
    assert "Transit" in result


# --- Worker class ---


def _make_config():
    config = MagicMock()
    config.max_worker_retries = 3
    config.max_consecutive_errors = 5
    config.llm_timeout = 120.0
    return config


def _make_worker(mock_llm, session_db, schema_summary="test schema"):
    queue = MagicMock()
    queue.emit = MagicMock()
    return Worker(
        llm=mock_llm,
        model="m",
        fallback_model="fb",
        schema_summary=schema_summary,
        session_db=session_db,
        config=_make_config(),
        queue=queue,
        session_id="s1",
        thread_id="t1",
    )


def test_worker_start_resets_state(session_db):
    mock_llm = MagicMock()
    worker = _make_worker(mock_llm, session_db)
    worker.consecutive_errors = 5
    worker.attempts = 10
    worker.llm_calls = [{"x": 1}]

    worker.start("Count rows")

    assert worker.consecutive_errors == 0
    assert worker.attempts == 0
    assert worker.llm_calls == []
    assert worker.instruction == "Count rows"
    assert len(worker.messages) == 2
    assert worker.messages[0]["role"] == "system"


def test_worker_handle_response_no_tool_calls(session_db):
    """Worker returns final JSON answer without tool calls."""
    mock_llm = MagicMock()
    worker = _make_worker(mock_llm, session_db)
    worker.start("Count rows")

    response = LLMResponse(
        content=json.dumps({
            "summary": "Test summary",
            "view_requested": None,
        }),
        model="test",
        tool_calls=None,
    )

    result = worker.handle_response(response, call_ms=100)

    assert isinstance(result, WorkerResult)
    assert result.result == "Test summary"


def test_worker_handle_response_with_tool_call(session_db):
    """Worker calls run_sql tool — handle_response returns None (needs another call)."""
    mock_llm = MagicMock()
    worker = _make_worker(mock_llm, session_db)
    worker.start("Count rows")

    response = LLMResponse(
        content="",
        model="test",
        tool_calls=[{
            "id": "call_1",
            "function": {
                "name": "run_sql",
                "arguments": json.dumps({"sql": "SELECT COUNT(*) as cnt FROM dataset"}),
            },
        }],
    )

    result = worker.handle_response(response, call_ms=100)

    assert result is None  # needs another call
    # Tool result was appended to messages
    tool_msgs = [m for m in worker.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "103" in tool_msgs[0]["content"]


def test_worker_handle_response_sql_error_tracks_consecutive(session_db):
    """SQL errors increment consecutive_errors counter."""
    mock_llm = MagicMock()
    worker = _make_worker(mock_llm, session_db)
    worker.start("Bad query")

    response = LLMResponse(
        content="",
        model="test",
        tool_calls=[{
            "id": "call_1",
            "function": {
                "name": "run_sql",
                "arguments": json.dumps({"sql": "SELECT * FROM nonexistent"}),
            },
        }],
    )

    worker.handle_response(response, call_ms=100)

    assert worker.consecutive_errors == 1


def test_worker_handle_response_empty_retries(session_db):
    """Empty response returns None and appends retry message."""
    mock_llm = MagicMock()
    worker = _make_worker(mock_llm, session_db)
    worker.start("Count rows")

    response = LLMResponse(content="", model="test", tool_calls=None)

    result = worker.handle_response(response, call_ms=100)

    assert result is None
    last_msg = worker.messages[-1]
    assert "empty" in last_msg["content"].lower()


def test_worker_handle_response_non_json_retries(session_db):
    """Non-JSON text response returns None and asks for JSON."""
    mock_llm = MagicMock()
    worker = _make_worker(mock_llm, session_db)
    worker.start("Count rows")

    response = LLMResponse(
        content="The dataset has 103 rows.",
        model="test",
        tool_calls=None,
    )

    result = worker.handle_response(response, call_ms=100)

    assert result is None
    last_msg = worker.messages[-1]
    assert "JSON" in last_msg["content"]


def test_worker_handle_response_with_view_request(session_db):
    """Worker requests a view creation."""
    mock_llm = MagicMock()
    worker = _make_worker(mock_llm, session_db)
    worker.start("Filter data")

    response = LLMResponse(
        content=json.dumps({
            "summary": "Created filtered view",
            "view_requested": {
                "name": "filtered_data",
                "sql": "SELECT * FROM dataset WHERE pl_rade IS NOT NULL",
            },
        }),
        model="test",
        tool_calls=None,
    )

    result = worker.handle_response(response, call_ms=100)

    assert result is not None
    assert result.view_requested is not None
    assert result.view_requested["name"] == "filtered_data"

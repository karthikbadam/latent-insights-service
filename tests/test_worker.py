"""Tests for app.agents.worker — SQL execution worker agent."""

import json
from unittest.mock import MagicMock


from app.agents.worker import _execute_sql, _format_results, run_worker
from app.core.llm import LLMResponse
from app.models import WorkerResult


# --- _format_results ---


def test_format_results_basic():
    result = _format_results(["name", "count"], [("Transit", 68), ("RV", 18)])
    assert "name | count" in result
    assert "Transit | 68" in result
    assert "RV | 18" in result


def test_format_results_empty():
    result = _format_results(["col"], [])
    assert result == "(no rows returned)"


# --- _execute_sql ---


def test_execute_sql_success(session_db):
    result = _execute_sql(session_db, "SELECT COUNT(*) as cnt FROM dataset")
    assert "cnt" in result
    assert "103" in result


def test_execute_sql_error(session_db):
    result = _execute_sql(session_db, "SELECT * FROM nonexistent_table")
    assert "SQL ERROR" in result


def test_execute_sql_groupby(session_db):
    result = _execute_sql(
        session_db,
        "SELECT discoverymethod, COUNT(*) as cnt FROM dataset GROUP BY 1 ORDER BY 2 DESC",
    )
    assert "Transit" in result


# --- run_worker tool-use loop ---


def test_run_worker_no_tool_calls(session_db, schema_summary):
    """Worker that returns final answer without tool calls."""
    final_response = json.dumps({
        "summary": "Test summary",
        "details": None,
        "view_requested": None,
    })

    mock_llm = MagicMock()
    mock_llm.call.return_value = LLMResponse(
        content=final_response, model="test", tool_calls=None,
    )

    result = run_worker(
        llm=mock_llm, model="m", fallback_model="fb",
        worker_instruction="Count rows",
        schema_summary=schema_summary, session_db=session_db,
    )

    assert isinstance(result, WorkerResult)
    assert result.result == "Test summary"


def test_run_worker_with_tool_call(session_db, schema_summary):
    """Worker that calls run_sql tool, then returns final answer."""
    mock_llm = MagicMock()

    # First call: LLM wants to run SQL
    tool_call_response = LLMResponse(
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

    # Second call: LLM returns final answer
    final_response = LLMResponse(
        content=json.dumps({
            "summary": "The dataset contains 103 rows.",
            "details": None,
            "view_requested": None,
        }),
        model="test",
        tool_calls=None,
    )

    mock_llm.call.side_effect = [tool_call_response, final_response]

    result = run_worker(
        llm=mock_llm, model="m", fallback_model="fb",
        worker_instruction="Count the rows",
        schema_summary=schema_summary, session_db=session_db,
    )

    assert result.result == "The dataset contains 103 rows."
    assert mock_llm.call.call_count == 2

    # Verify tool result was passed back
    second_call_messages = mock_llm.call.call_args_list[1].kwargs.get("messages") or \
        mock_llm.call.call_args_list[1][1].get("messages", mock_llm.call.call_args_list[1][0][1] if len(mock_llm.call.call_args_list[1][0]) > 1 else [])
    tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "103" in tool_msgs[0]["content"]


def test_run_worker_sql_error_in_tool(session_db, schema_summary):
    """Worker gets SQL error, LLM sees it and self-corrects."""
    mock_llm = MagicMock()

    # First call: LLM tries bad SQL
    bad_sql_response = LLMResponse(
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

    # Second call: LLM sees error, tries correct SQL
    good_sql_response = LLMResponse(
        content="",
        model="test",
        tool_calls=[{
            "id": "call_2",
            "function": {
                "name": "run_sql",
                "arguments": json.dumps({"sql": "SELECT COUNT(*) as cnt FROM dataset"}),
            },
        }],
    )

    # Third call: LLM returns final answer
    final_response = LLMResponse(
        content=json.dumps({
            "summary": "Found 103 rows after correcting query.",
            "details": None,
            "view_requested": None,
        }),
        model="test",
        tool_calls=None,
    )

    mock_llm.call.side_effect = [bad_sql_response, good_sql_response, final_response]

    result = run_worker(
        llm=mock_llm, model="m", fallback_model="fb",
        worker_instruction="Count rows",
        schema_summary=schema_summary, session_db=session_db,
    )

    assert "103" in result.result
    assert mock_llm.call.call_count == 3


def test_run_worker_with_view_request(session_db, schema_summary):
    """Worker requests a view creation."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = LLMResponse(
        content=json.dumps({
            "summary": "Created filtered view",
            "details": None,
            "view_requested": {
                "name": "filtered_data",
                "sql": "SELECT * FROM dataset WHERE pl_rade IS NOT NULL",
            },
        }),
        model="test",
        tool_calls=None,
    )

    result = run_worker(
        llm=mock_llm, model="m", fallback_model="fb",
        worker_instruction="Filter data",
        schema_summary=schema_summary, session_db=session_db,
    )

    assert result.view_requested is not None
    assert result.view_requested["name"] == "filtered_data"

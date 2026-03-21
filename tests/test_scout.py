"""Tests for app.agents.scout — question discovery agent."""

import pytest

from app.agents.scout import _run_exploratory_queries, run_scout
from app.models import ScoutOutput
from tests.conftest import make_mock_llm


def test_run_exploratory_queries_shape(session_db):
    result = _run_exploratory_queries(session_db)
    assert "103 rows" in result
    assert "19 columns" in result


def test_run_exploratory_queries_column_overview(session_db):
    result = _run_exploratory_queries(session_db)
    assert "pl_name" in result
    assert "discoverymethod" in result


def test_run_exploratory_queries_distributions(session_db):
    result = _run_exploratory_queries(session_db)
    # discoverymethod has <15 distinct values — should show distribution
    assert "Transit" in result


def test_run_exploratory_queries_numeric_stats(session_db):
    result = _run_exploratory_queries(session_db)
    assert "min=" in result
    assert "max=" in result


@pytest.mark.asyncio
async def test_run_scout_basic(schema_summary):
    mock = make_mock_llm("scout_response.json")
    result = await run_scout(llm=mock, model="test", schema_summary=schema_summary)

    assert isinstance(result, ScoutOutput)
    assert len(result.exploration_notes) > 0
    assert len(result.questions) > 0


@pytest.mark.asyncio
async def test_run_scout_with_session_db(session_db, schema_summary):
    mock = make_mock_llm("scout_response.json")
    result = await run_scout(
        llm=mock, model="test", schema_summary=schema_summary, session_db=session_db,
    )

    assert isinstance(result, ScoutOutput)

    # Verify exploration data was included in prompt
    call_args = mock.call.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
    user_msg = messages[1]["content"]
    assert "exploratory queries" in user_msg
    assert "103 rows" in user_msg


@pytest.mark.asyncio
async def test_run_scout_without_session_db(schema_summary):
    mock = make_mock_llm("scout_response.json")
    result = await run_scout(llm=mock, model="test", schema_summary=schema_summary)

    assert isinstance(result, ScoutOutput)

    # Without session_db, no exploration data
    call_args = mock.call.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
    user_msg = messages[1]["content"]
    assert "exploratory queries" not in user_msg


@pytest.mark.asyncio
async def test_scout_question_fields(schema_summary):
    mock = make_mock_llm("scout_response.json")
    result = await run_scout(llm=mock, model="test", schema_summary=schema_summary)

    for q in result.questions:
        assert q.question, "question must be non-empty"
        assert q.motivation, "motivation must be non-empty"
        assert q.entry_point, "entry_point must be non-empty"
        assert q.difficulty in ("simple", "moderate", "deep")

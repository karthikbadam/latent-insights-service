"""Tests for app.agents.profiler — dataset profiling agent."""

import pytest

from app.agents.profiler import _gather_column_stats, run_profiler
from tests.conftest import make_mock_llm


def test_gather_column_stats_has_all_columns(session_db):
    info = session_db.execute("DESCRIBE dataset").fetchall()
    stats = _gather_column_stats(session_db, "dataset", info)

    assert "pl_name" in stats
    assert "discoverymethod" in stats
    assert "pl_orbper" in stats
    assert "pl_bmasse" in stats


def test_gather_column_stats_numeric(session_db):
    info = session_db.execute("DESCRIBE dataset").fetchall()
    stats = _gather_column_stats(session_db, "dataset", info)

    # pl_orbper is numeric — should have min, max, mean
    assert "min=" in stats
    assert "max=" in stats
    assert "mean=" in stats


def test_gather_column_stats_categorical(session_db):
    info = session_db.execute("DESCRIBE dataset").fetchall()
    stats = _gather_column_stats(session_db, "dataset", info)

    # discoverymethod has <20 unique — should show value counts
    assert "Transit" in stats


def test_gather_column_stats_null_rates(session_db):
    info = session_db.execute("DESCRIBE dataset").fetchall()
    stats = _gather_column_stats(session_db, "dataset", info)

    # pl_orbeccen has NULLs — should show non-100% rate
    lines = stats.split("\n")
    eccen_line = [line for line in lines if line.startswith("pl_orbeccen")][0]
    assert "/103" in eccen_line  # not all 103 are non-null


@pytest.mark.asyncio
async def test_run_profiler(session_db):
    mock = make_mock_llm("profiler_response.json")
    result = await run_profiler(
        llm=mock, model="test-model", session_db=session_db,
    )

    assert isinstance(result, str)
    assert len(result) > 0
    mock.call.assert_called_once()

    # Verify the prompt contains column stats
    call_args = mock.call.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
    user_msg = messages[1]["content"]
    assert "103 rows" in user_msg
    assert "19 columns" in user_msg
    assert "min=" in user_msg


@pytest.mark.asyncio
async def test_run_profiler_output_contains_fixture_content(session_db):
    mock = make_mock_llm("profiler_response.json")
    result = await run_profiler(llm=mock, model="test-model", session_db=session_db)

    # Output should be the fixture content
    assert "Dataset summary" in result
    assert "Column profiles" in result

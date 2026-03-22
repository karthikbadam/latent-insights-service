"""Shared test fixtures for Latent Insights."""

import json
import os
from unittest.mock import MagicMock

import duckdb
import pytest

from app.core.llm import LLMClient, LLMResponse
from app.core.state import StateStore

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def state_store(tmp_path):
    """In-memory StateStore backed by tmp_path for persistence tests."""
    return StateStore(data_dir=str(tmp_path))


@pytest.fixture
def session_db():
    """In-memory DuckDB with sample dataset loaded as 'dataset' table."""
    db = duckdb.connect(":memory:")
    csv_path = os.path.join(FIXTURES_DIR, "sample_dataset.csv")
    db.execute(f"CREATE TABLE dataset AS SELECT * FROM read_csv_auto('{csv_path}')")
    return db


@pytest.fixture
def schema_summary():
    """Pre-built schema summary string for agent tests."""
    fixture_path = os.path.join(FIXTURES_DIR, "profiler_response.json")
    with open(fixture_path) as f:
        return json.load(f)["content"]


def make_mock_llm(fixture_name: str) -> MagicMock:
    """Create a mock LLMClient that returns a canned response from a fixture file."""
    fixture_path = os.path.join(FIXTURES_DIR, fixture_name)
    with open(fixture_path) as f:
        data = json.load(f)

    response = LLMResponse(
        content=data["content"],
        model="test-model",
        input_tokens=100,
        output_tokens=200,
        cached=False,
    )

    mock = MagicMock(spec=LLMClient)
    mock.call.return_value = response
    mock.call_with_retry.return_value = response
    return mock


@pytest.fixture
def mock_llm():
    """Default mock LLM returning profiler response."""
    return make_mock_llm("profiler_response.json")

"""Tests for app.core.llm — LLMClient cache and retry behavior."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import duckdb
import pytest

from app.core.llm import LLMClient, LLMResponse


@pytest.fixture
def cache_db():
    db = duckdb.connect(":memory:")
    db.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key     VARCHAR PRIMARY KEY,
            model         VARCHAR NOT NULL,
            role          VARCHAR NOT NULL,
            response      JSON NOT NULL,
            input_tokens  INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            created_at    TIMESTAMP DEFAULT current_timestamp,
            ttl_hours     INTEGER DEFAULT 24
        )
    """)
    return db


def test_compute_cache_key_deterministic():
    messages = [{"role": "user", "content": "hello"}]
    key1 = LLMClient.compute_cache_key("model-a", messages)
    key2 = LLMClient.compute_cache_key("model-a", messages)
    assert key1 == key2
    assert len(key1) == 64  # SHA256 hex


def test_compute_cache_key_varies_with_model():
    messages = [{"role": "user", "content": "hello"}]
    key1 = LLMClient.compute_cache_key("model-a", messages)
    key2 = LLMClient.compute_cache_key("model-b", messages)
    assert key1 != key2


def test_compute_cache_key_varies_with_temperature():
    messages = [{"role": "user", "content": "hello"}]
    key1 = LLMClient.compute_cache_key("model-a", messages, temperature=0.0)
    key2 = LLMClient.compute_cache_key("model-a", messages, temperature=0.5)
    assert key1 != key2


def test_compute_cache_key_varies_with_messages():
    key1 = LLMClient.compute_cache_key("m", [{"role": "user", "content": "a"}])
    key2 = LLMClient.compute_cache_key("m", [{"role": "user", "content": "b"}])
    assert key1 != key2


def test_cache_hit(cache_db):
    client = LLMClient(api_key="test", base_url="http://test")
    client.set_cache_db(cache_db)

    key = "test_key_123"
    cache_db.execute("""
        INSERT INTO llm_cache (cache_key, model, role, response, input_tokens, output_tokens)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [key, "test-model", "worker", json.dumps({"content": "cached result"}), 50, 100])

    result = client._check_cache(key, ttl_hours=24)
    assert result is not None
    assert result.content == "cached result"
    assert result.cached is True


def test_cache_miss_expired(cache_db):
    client = LLMClient(api_key="test", base_url="http://test")
    client.set_cache_db(cache_db)

    key = "old_key"
    old_time = datetime.utcnow() - timedelta(hours=48)
    cache_db.execute("""
        INSERT INTO llm_cache (cache_key, model, role, response, input_tokens, output_tokens, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [key, "test-model", "worker", json.dumps({"content": "old"}), 50, 100, old_time])

    result = client._check_cache(key, ttl_hours=24)
    assert result is None


def test_cache_write(cache_db):
    client = LLMClient(api_key="test", base_url="http://test")
    client.set_cache_db(cache_db)

    response = LLMResponse(content="new result", model="m", input_tokens=10, output_tokens=20)
    client._write_cache("write_key", "m", "worker", response, ttl_hours=24)

    row = cache_db.execute("SELECT response FROM llm_cache WHERE cache_key = ?", ["write_key"]).fetchone()
    assert row is not None
    data = json.loads(row[0])
    assert data["content"] == "new result"


@pytest.mark.asyncio
async def test_call_skips_cache_nonzero_temp(cache_db):
    client = LLMClient(api_key="test", base_url="http://test")
    client.set_cache_db(cache_db)

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "response"
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

    client._client = AsyncMock()
    client._client.chat.completions.create = AsyncMock(return_value=mock_completion)

    result = await client.call(
        model="m", messages=[{"role": "user", "content": "hi"}],
        role="test", temperature=0.5, cache_ttl_hours=24,
    )

    assert result.content == "response"
    assert result.cached is False
    # Verify no cache write happened
    row = cache_db.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
    assert row[0] == 0


@pytest.mark.asyncio
async def test_call_uses_cache_on_hit(cache_db):
    client = LLMClient(api_key="test", base_url="http://test")
    client.set_cache_db(cache_db)

    messages = [{"role": "user", "content": "cached call"}]
    key = LLMClient.compute_cache_key("m", messages, temperature=0.0)

    cache_db.execute("""
        INSERT INTO llm_cache (cache_key, model, role, response, input_tokens, output_tokens)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [key, "m", "test", json.dumps({"content": "from cache"}), 10, 20])

    client._client = AsyncMock()

    result = await client.call(
        model="m", messages=messages, role="test",
        temperature=0.0, cache_ttl_hours=24,
    )

    assert result.content == "from cache"
    assert result.cached is True
    client._client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_call_with_retry_uses_fallback():
    client = LLMClient(api_key="test", base_url="http://test")

    call_count = 0
    models_used = []

    async def mock_call(model, messages, role, temperature, tools=None, cache_ttl_hours=0, max_tokens=4096):
        nonlocal call_count
        call_count += 1
        models_used.append(model)
        if call_count < 3:
            raise Exception(f"API error {call_count}")
        return LLMResponse(content="success", model=model, input_tokens=10, output_tokens=20)

    client.call = mock_call

    result = await client.call_with_retry(
        model="primary", fallback_model="fallback",
        messages=[{"role": "user", "content": "test"}],
        role="test", max_retries=3,
    )

    assert result.content == "success"
    assert models_used[-1] == "fallback"
    assert call_count == 3

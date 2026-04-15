"""Tests for app.core.llm — LLMClient call and retry behavior."""

from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from latent_insights.core.llm import LLMClient, is_transient_llm_error


def _make_completion(content: str = "response", tool_calls=None):
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = content
    mock_completion.choices[0].message.tool_calls = tool_calls
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    return mock_completion


def _make_client_with_mock(create_mock):
    client = LLMClient(
        api_key="test", base_url="http://test",
        max_transient_retries=3, transient_backoff_base=0.0,
    )
    client._client = MagicMock()
    client._client.chat.completions.create = create_mock
    return client


def test_call_returns_response():
    client = _make_client_with_mock(MagicMock(return_value=_make_completion("response")))

    result = client.call(
        model="m", messages=[{"role": "user", "content": "hi"}],
        role="test", temperature=0.5,
    )

    assert result.content == "response"
    assert result.input_tokens == 10
    assert result.output_tokens == 20


def test_call_captures_tool_calls():
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "run_sql"
    mock_tc.function.arguments = '{"sql": "SELECT 1"}'

    client = _make_client_with_mock(
        MagicMock(return_value=_make_completion(content="", tool_calls=[mock_tc])),
    )

    result = client.call(
        model="m", messages=[{"role": "user", "content": "hi"}], role="test",
    )

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "run_sql"


def test_call_retries_transient_then_succeeds():
    """APIConnectionError, 429, 5xx, timeouts are retried silently."""
    request = httpx.Request("POST", "http://test")
    response_503 = httpx.Response(status_code=503, request=request)

    calls = []

    def create(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise APIConnectionError(request=request)
        if len(calls) == 2:
            raise APIStatusError("boom", response=response_503, body=None)
        return _make_completion("finally")

    client = _make_client_with_mock(create)
    result = client.call(
        model="m", messages=[{"role": "user", "content": "hi"}], role="test",
    )

    assert result.content == "finally"
    assert len(calls) == 3


def test_call_raises_on_transient_retry_exhaustion():
    """After max_transient_retries + 1 attempts, the last transient error propagates."""
    request = httpx.Request("POST", "http://test")

    def always_fail(**kwargs):
        raise APIConnectionError(request=request)

    client = _make_client_with_mock(always_fail)

    with pytest.raises(APIConnectionError):
        client.call(
            model="m", messages=[{"role": "user", "content": "hi"}], role="test",
        )


def test_call_does_not_retry_non_transient_errors():
    """Auth / bad-request errors propagate immediately."""
    request = httpx.Request("POST", "http://test")
    response_401 = httpx.Response(status_code=401, request=request)

    calls = []

    def create(**kwargs):
        calls.append(1)
        raise APIStatusError("unauth", response=response_401, body=None)

    client = _make_client_with_mock(create)

    with pytest.raises(APIStatusError):
        client.call(
            model="m", messages=[{"role": "user", "content": "hi"}], role="test",
        )
    assert len(calls) == 1  # no retries


def test_is_transient_llm_error_classification():
    request = httpx.Request("POST", "http://test")
    response_503 = httpx.Response(status_code=503, request=request)
    response_400 = httpx.Response(status_code=400, request=request)

    assert is_transient_llm_error(APIConnectionError(request=request))
    assert is_transient_llm_error(APITimeoutError(request=request))
    assert is_transient_llm_error(
        APIStatusError("x", response=response_503, body=None)
    )
    assert not is_transient_llm_error(
        APIStatusError("x", response=response_400, body=None)
    )
    assert not is_transient_llm_error(ValueError("random bug"))

"""Tests for app.core.llm — LLMClient call and retry behavior."""

from unittest.mock import MagicMock


from app.core.llm import LLMClient, LLMResponse


def test_call_returns_response():
    client = LLMClient(api_key="test", base_url="http://test")

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "response"
    mock_completion.choices[0].message.tool_calls = None
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

    client._client = MagicMock()
    client._client.chat.completions.create = MagicMock(return_value=mock_completion)

    result = client.call(
        model="m", messages=[{"role": "user", "content": "hi"}],
        role="test", temperature=0.5,
    )

    assert result.content == "response"
    assert result.input_tokens == 10
    assert result.output_tokens == 20


def test_call_captures_tool_calls():
    client = LLMClient(api_key="test", base_url="http://test")

    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "run_sql"
    mock_tc.function.arguments = '{"sql": "SELECT 1"}'

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = ""
    mock_completion.choices[0].message.tool_calls = [mock_tc]
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

    client._client = MagicMock()
    client._client.chat.completions.create = MagicMock(return_value=mock_completion)

    result = client.call(
        model="m", messages=[{"role": "user", "content": "hi"}],
        role="test",
    )

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "run_sql"


def test_call_with_retry_uses_fallback():
    client = LLMClient(api_key="test", base_url="http://test")

    call_count = 0
    models_used = []

    def mock_call(model, messages, role, temperature, tools=None, max_tokens=4096):
        nonlocal call_count
        call_count += 1
        models_used.append(model)
        if call_count < 3:
            raise Exception(f"API error {call_count}")
        return LLMResponse(content="success", model=model, input_tokens=10, output_tokens=20)

    client.call = mock_call

    result = client.call_with_retry(
        model="primary", fallback_model="fallback",
        messages=[{"role": "user", "content": "test"}],
        role="test", max_retries=3,
    )

    assert result.content == "success"
    assert models_used[-1] == "fallback"
    assert call_count == 3

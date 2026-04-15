"""
LLM client — OpenAI-compatible API integration (OpenRouter, Ollama, etc).

All agent calls go through here. Transient network errors (connection resets,
rate limits, 5xx, timeouts) are retried transparently with exponential
backoff. Callers only see exceptions once retries are exhausted or when the
error is non-transient (auth, malformed request, etc).
"""

import logging
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

logger = logging.getLogger(__name__)


# HTTP statuses we treat as transient (worth retrying).
_RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class LLMResponse:
    """Standardized response from any LLM call."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    tool_calls: list | None = None


def is_transient_llm_error(exc: BaseException) -> bool:
    """True if the exception is a retryable network/server-side failure.

    Callers that catch exceptions raised from LLMClient.call can use this
    to tell retry-exhausted transient failures apart from other bugs.
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in _RETRYABLE_STATUSES
    return False


# Internal alias kept terse for use inside this module.
_is_transient = is_transient_llm_error


class LLMClient:
    """
    LLM client for OpenAI-compatible APIs (OpenRouter, Ollama, etc).

    `call()` retries transient errors (connection resets, rate limits, 5xx,
    timeouts) with exponential backoff. If all retries fail, the last
    exception is re-raised so the caller can classify it as retry-exhausted.

    Usage:
        client = LLMClient(api_key, base_url)
        response = client.call(
            model="anthropic/claude-3.5-haiku",
            messages=[{"role": "user", "content": "hello"}],
            role="worker",
            temperature=0.0,
        )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        app_name: str = "",
        app_url: str = "",
        think: bool = True,
        max_transient_retries: int = 3,
        transient_backoff_base: float = 1.0,
    ):
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._app_name = app_name
        self._app_url = app_url
        self._think = think
        self._max_transient_retries = max_transient_retries
        self._transient_backoff_base = transient_backoff_base

    def call(
        self,
        model: str,
        messages: list[dict],
        role: str = "default",
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> LLMResponse:
        """
        Make an LLM call. Retries transient failures with exponential
        backoff; non-transient errors (auth, 4xx other than the retryable
        set) raise immediately.
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_headers": {},
            "timeout": timeout,
        }
        if self._app_url:
            kwargs["extra_headers"]["HTTP-Referer"] = self._app_url
        if self._app_name:
            kwargs["extra_headers"]["X-Title"] = self._app_name
        if tools:
            kwargs["tools"] = tools
        if not self._think:
            kwargs["extra_body"] = {"think": False}

        logger.info(f"LLM call: model={model} role={role} temp={temperature}")

        max_attempts = self._max_transient_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                completion = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                if not _is_transient(exc) or attempt == max_attempts:
                    raise
                last_exc = exc
                backoff = self._transient_backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    f"LLM transient error on attempt {attempt}/{max_attempts} "
                    f"({type(exc).__name__}): retrying in {backoff:.1f}s"
                )
                time.sleep(backoff)
        else:  # pragma: no cover — the for-else runs only if loop completes without break
            assert last_exc is not None
            raise last_exc

        choice = completion.choices[0]
        content = choice.message.content or ""
        usage = completion.usage

        raw_tool_calls = None
        if choice.message.tool_calls:
            raw_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        response = LLMResponse(
            content=content,
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached=False,
            tool_calls=raw_tool_calls,
        )

        logger.info(
            f"LLM response: {response.input_tokens} in / "
            f"{response.output_tokens} out"
        )
        return response

"""
LLM client — OpenAI-compatible API integration (OpenRouter, Ollama, etc).

All agent calls go through here.
"""

import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM call."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    tool_calls: list | None = None


class LLMClient:
    """
    LLM client for OpenAI-compatible APIs (OpenRouter, Ollama, etc).

    Usage:
        client = LLMClient(api_key, base_url)
        response = await client.call(
            model="anthropic/claude-3.5-haiku",
            messages=[{"role": "user", "content": "hello"}],
            role="worker",
            temperature=0.0,
        )
    """

    def __init__(self, api_key: str, base_url: str, app_name: str = "", app_url: str = ""):
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._app_name = app_name
        self._app_url = app_url

    async def call(
        self,
        model: str,
        messages: list[dict],
        role: str = "default",
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Make an LLM call through OpenRouter.

        Args:
            model: OpenRouter model ID (e.g., "anthropic/claude-3.5-haiku")
            messages: Chat messages array
            role: Agent role name (for logging)
            temperature: Sampling temperature
            tools: Optional tool definitions
            max_tokens: Max output tokens
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_headers": {},
        }
        if self._app_url:
            kwargs["extra_headers"]["HTTP-Referer"] = self._app_url
        if self._app_name:
            kwargs["extra_headers"]["X-Title"] = self._app_name
        if tools:
            kwargs["tools"] = tools

        logger.info(f"LLM call: model={model} role={role} temp={temperature}")
        completion = await self._client.chat.completions.create(**kwargs)

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

    async def call_with_retry(
        self,
        model: str,
        fallback_model: str,
        messages: list[dict],
        role: str = "default",
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        max_retries: int = 3,
    ) -> LLMResponse:
        """Call with retry, escalating to fallback model on failure."""
        last_error = None

        for attempt in range(max_retries):
            try:
                current_model = model if attempt < max_retries - 1 else fallback_model
                return await self.call(
                    model=current_model,
                    messages=messages,
                    role=role,
                    temperature=temperature,
                    tools=tools,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
                )

        raise last_error

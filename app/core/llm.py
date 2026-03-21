"""
LLM client — OpenRouter integration with DuckDB-backed cache.

This is fully working infrastructure. All agent calls go through here.
"""

import hashlib
import json
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
    OpenRouter LLM client with DuckDB-backed response cache.

    Usage:
        client = LLMClient(config)
        client.set_cache_db(db_connection)  # optional, enables caching

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
        self._cache_db = None

    def set_cache_db(self, db):
        """Attach a DuckDB connection for response caching."""
        self._cache_db = db
        self._ensure_cache_table()

    def _ensure_cache_table(self):
        if self._cache_db is None:
            return
        self._cache_db.execute("""
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

    @staticmethod
    def compute_cache_key(
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Deterministic cache key from the full request signature."""
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools or [],
            "temperature": temperature,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _check_cache(self, cache_key: str, ttl_hours: int) -> LLMResponse | None:
        """Check DuckDB cache for a hit."""
        if self._cache_db is None or ttl_hours <= 0:
            return None
        try:
            row = self._cache_db.execute("""
                SELECT response, model, input_tokens, output_tokens
                FROM llm_cache
                WHERE cache_key = ?
                  AND created_at > current_timestamp - INTERVAL (? || ' hours')
            """, [cache_key, ttl_hours]).fetchone()
            if row:
                logger.debug(f"Cache hit: {cache_key[:12]}...")
                data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                return LLMResponse(
                    content=data["content"],
                    model=row[1],
                    input_tokens=row[2],
                    output_tokens=row[3],
                    cached=True,
                )
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    def _write_cache(
        self,
        cache_key: str,
        model: str,
        role: str,
        response: LLMResponse,
        ttl_hours: int,
    ):
        """Write response to DuckDB cache."""
        if self._cache_db is None or ttl_hours <= 0:
            return
        try:
            self._cache_db.execute("""
                INSERT OR REPLACE INTO llm_cache
                (cache_key, model, role, response, input_tokens, output_tokens, ttl_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                cache_key,
                model,
                role,
                json.dumps({"content": response.content}),
                response.input_tokens,
                response.output_tokens,
                ttl_hours,
            ])
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    async def call(
        self,
        model: str,
        messages: list[dict],
        role: str = "default",
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        cache_ttl_hours: int = 0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Make an LLM call through OpenRouter, with optional caching.

        Args:
            model: OpenRouter model ID (e.g., "anthropic/claude-3.5-haiku")
            messages: Chat messages array
            role: Agent role name (for cache tracking)
            temperature: Sampling temperature
            tools: Optional tool definitions
            cache_ttl_hours: Cache TTL in hours (0 = no cache)
            max_tokens: Max output tokens
        """
        # Check cache for deterministic calls
        cache_key = None
        if cache_ttl_hours > 0 and temperature == 0.0:
            cache_key = self.compute_cache_key(model, messages, tools, temperature)
            cached = self._check_cache(cache_key, cache_ttl_hours)
            if cached:
                return cached

        # Build request kwargs
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

        # Make the call
        logger.info(f"LLM call: model={model} role={role} temp={temperature}")
        completion = await self._client.chat.completions.create(**kwargs)

        # Extract response
        choice = completion.choices[0]
        content = choice.message.content or ""
        usage = completion.usage

        # Capture tool calls if present
        raw_tool_calls = None
        if choice.message.tool_calls:
            raw_tool_calls = [
                {
                    "id": tc.id,
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

        # Write to cache
        if cache_key and cache_ttl_hours > 0:
            self._write_cache(cache_key, model, role, response, cache_ttl_hours)

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
        cache_ttl_hours: int = 0,
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
                    cache_ttl_hours=cache_ttl_hours,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
                )

        raise last_error

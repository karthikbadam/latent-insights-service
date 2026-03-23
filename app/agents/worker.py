"""
Worker agent — executes one analytical step via SQL.

Manages its own message history, retry logic, tool-use loop, and event emission.
Works with any dataset — no domain-specific assumptions.
"""

import json
import logging
import re
import time

from app.agents.base import Agent
from app.core.llm import LLMClient
from app.core.parsing import detect_degeneration, parse_worker_response
from app.core.queue import Queue
from app.models import StreamEvent, WorkerResult

logger = logging.getLogger(__name__)


class Worker(Agent):
    """Executes analytical steps via SQL tool-use loop."""

    SYSTEM_PROMPT = """\
You are a data analysis worker. You receive an analytical instruction
and execute it against a DuckDB database using SQL.

## Dataset schema

{schema_summary}

## Available thread views

{thread_views}

## Your instruction

{worker_instruction}

## How to work

1. Plan your query — columns, filters, aggregations.
2. Use the run_sql tool to execute SQL (DuckDB dialect). Supports CTEs, window functions,
   PERCENTILE_CONT, QUALIFY, HISTOGRAM(), APPROX_QUANTILE, etc.
3. You may call run_sql multiple times to explore, refine, and validate.
4. If asked to create a filtered subset, include view definition in your final response.
5. Summarize for a technical reader. Lead with findings and method.

## Output format

When done querying, return your final answer as JSON (no tool call):

{{
  "summary": "2-4 sentence narrative of findings. Lead with most interesting finding. Include methodology notes, NULL caveats, secondary findings.",
  "view_requested": {{"name": "...", "sql": "..."}} or null
}}

## Rules
- Check NULL rates before computing stats.
- Comparing groups: absolute numbers AND effect size.
- Log scale thinking for values spanning orders of magnitude.
- If a SQL query errors because a function does not exist, do NOT search for alternative \
function names. Instead, rewrite your analysis using basic SQL math (arithmetic, CASE, \
aggregates like AVG/STDDEV_POP/CORR). DuckDB is a SQL engine, not a statistics package — \
anything not built into standard SQL must be computed manually.
- Do NOT use DuckDB extensions like ml, spatial, or stats (e.g. linear_regression, kmeans, \
ols). Stick to standard SQL: aggregates, window functions, CTEs, CASE expressions.
"""

    RUN_SQL_TOOL = {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a read-only SQL query against the DuckDB database. "
                "Returns column names and all result rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute (DuckDB dialect)",
                    }
                },
                "required": ["sql"],
            },
        },
    }

    def __init__(
        self,
        llm: LLMClient,
        model: str,
        fallback_model: str,
        schema_summary: str,
        session_db,
        config,
        queue: Queue,
        session_id: str,
        thread_id: str,
    ):
        super().__init__(llm, model)
        self.fallback_model = fallback_model
        self.schema_summary = schema_summary
        self.session_db = session_db
        self.config = config
        self.queue = queue
        self.session_id = session_id
        self.thread_id = thread_id

        # Per-step state (reset in start())
        self.messages: list[dict] = []
        self.instruction: str = ""
        self.current_model: str = model
        self.consecutive_errors: int = 0
        self.attempts: int = 0
        self.llm_calls: list[dict] = []

    @property
    def role(self) -> str:
        return "worker"

    def start(self, instruction: str, thread_views: str = "(none)"):
        """Initialize worker state for a new step."""
        self.instruction = instruction
        self.current_model = self.model
        self.consecutive_errors = 0
        self.attempts = 0
        self.llm_calls = []

        prompt = self.SYSTEM_PROMPT.format(
            schema_summary=self.schema_summary,
            thread_views=thread_views,
            worker_instruction=instruction,
        )
        self.messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Execute this analysis and return results."},
        ]

    def call(self) -> tuple:
        """Single LLM call. Returns (response, call_ms). Raises APITimeoutError on timeout."""
        self.attempts += 1
        if self.attempts > 50:
            raise ValueError("Worker exceeded 50 LLM turns without producing a result")

        if self.consecutive_errors >= self.config.max_worker_retries:
            self.current_model = self.fallback_model

        t0 = time.monotonic()
        response = self.llm.call(
            model=self.current_model,
            messages=self.messages,
            role=self.role,
            temperature=0.0,
            tools=[self.RUN_SQL_TOOL],
            timeout=self.config.llm_timeout,
        )
        call_ms = round((time.monotonic() - t0) * 1000)

        has_tools = bool(response.tool_calls)
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="llm_call",
            message=f"Worker {'executing SQL' if has_tools else 'summarizing'} ({call_ms}ms)",
            data={
                "role": self.role,
                "model": self.current_model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "duration_ms": call_ms,
                "has_tool_calls": has_tools,
            },
        ))

        return response, call_ms

    def handle_timeout(self):
        """Handle an APITimeoutError by appending a retry message."""
        self.consecutive_errors += 1
        self.messages.append({
            "role": "user",
            "content": "Your previous response timed out. Simplify your approach and respond more concisely.",
        })

    def handle_response(self, response, call_ms: int) -> WorkerResult | None:
        """Process worker LLM response. Returns WorkerResult when done, None if another call needed."""
        if response.tool_calls:
            return self._handle_tool_calls(response, call_ms)
        return self._handle_final(response, call_ms)

    def _handle_final(self, response, call_ms: int) -> WorkerResult | None:
        """Worker returned a final text response (no tool calls)."""
        self.llm_calls.append({
            "agent": self.role,
            "type": "response",
            "duration_ms": call_ms,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "response": response.content if response.content else "",
        })

        if not response.content or not response.content.strip():
            logger.warning("Worker returned empty response, requesting JSON output")
            self.messages.append({"role": "assistant", "content": response.content or ""})
            self.messages.append({
                "role": "user",
                "content": "Your response was empty. Please provide your final answer as JSON matching the output format specified in the system prompt.",
            })
            return None

        if detect_degeneration(response.content):
            logger.warning(f"Worker output degeneration detected for thread {self.thread_id}")
            self.messages.append({"role": "assistant", "content": response.content})
            self.messages.append({
                "role": "user",
                "content": "Your output contained repeated/degenerate text. Please provide a concise, clean JSON response with your findings so far.",
            })
            return None

        # Check if response looks like it's attempting JSON (contains { })
        has_json_block = bool(re.search(r"\{.*\}", response.content, re.DOTALL))

        if has_json_block:
            try:
                worker_result = parse_worker_response(response.content)
            except (ValueError, json.JSONDecodeError):
                logger.warning("Worker returned malformed JSON, requesting reformat")
                self.messages.append({"role": "assistant", "content": response.content})
                self.messages.append({
                    "role": "user",
                    "content": "Your response contained JSON but it was malformed. Please reformat as valid JSON matching the output format.",
                })
                return None
            worker_result.llm_calls = self.llm_calls
            return worker_result
        else:
            # Intermediate reasoning — no JSON, just thinking. Continue the loop.
            logger.info(f"Worker intermediate reasoning ({len(response.content)} chars)")
            self.messages.append({"role": "assistant", "content": response.content})
            return None

    def _handle_tool_calls(self, response, call_ms: int) -> None:
        """Worker wants to execute SQL tools. Always returns None (needs another call)."""
        assistant_msg = {"role": "assistant", "content": response.content or None}
        assistant_msg["tool_calls"] = response.tool_calls
        self.messages.append(assistant_msg)

        tool_results = []
        for tool_call in response.tool_calls:
            func = tool_call["function"]
            if func["name"] == "run_sql":
                args = json.loads(func["arguments"])
                sql = args.get("sql", "")
                logger.info(f"Worker executing SQL: {sql[:200]}")
                t_sql = time.monotonic()
                result_text = self.execute_sql(self.session_db, sql)
                sql_ms = round((time.monotonic() - t_sql) * 1000)

                self.queue.emit(StreamEvent(
                    session_id=self.session_id,
                    thread_id=self.thread_id,
                    event_type="tool_call",
                    message=sql,
                    data={"sql": sql, "result": result_text, "duration_ms": sql_ms},
                ))
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result_text,
                })
                tool_results.append({"sql": sql, "result": result_text[:1000]})
                if result_text.startswith("SQL ERROR:"):
                    self.consecutive_errors += 1
                else:
                    self.consecutive_errors = 0
            else:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": f"Unknown tool: {func['name']}",
                })

        for tr in tool_results:
            self.llm_calls.append({
                "agent": self.role,
                "type": "tool_call",
                "duration_ms": call_ms,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "sql": tr["sql"],
                "tool_result": tr["result"],
            })

        # Error guardrails
        if self.consecutive_errors >= self.config.max_consecutive_errors:
            self.messages.append({
                "role": "user",
                "content": (
                    f"You have hit {self.consecutive_errors} consecutive SQL errors. "
                    "Stop trying SQL and return your final JSON answer NOW "
                    "with whatever findings you have so far. If you have no findings, "
                    "state that the analysis could not be completed and explain why."
                ),
            })
        elif self.consecutive_errors >= 2:
            self.messages.append({
                "role": "user",
                "content": (
                    f"You have hit {self.consecutive_errors} consecutive SQL errors. "
                    "The function you are trying likely does not exist in DuckDB. "
                    "STOP retrying the same approach. Rewrite your analysis using "
                    "only basic SQL math and aggregates (AVG, STDDEV_POP, CORR, etc)."
                ),
            })

        return None

    @staticmethod
    def format_results(col_names: list[str], rows: list) -> str:
        """Format query results as a readable table string."""
        if not rows:
            return "(no rows returned)"
        header = " | ".join(col_names)
        lines = [header, "-" * len(header)]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)

    @staticmethod
    def execute_sql(session_db, sql: str) -> str:
        """Execute SQL against session DB and return formatted results."""
        try:
            result = session_db.execute(sql)
            rows = result.fetchall()
            description = result.description
            col_names = [d[0] for d in description] if description else []
            return Worker.format_results(col_names, rows)
        except Exception as e:
            return f"SQL ERROR: {e}"

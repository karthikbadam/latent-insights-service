"""
Worker agent — executes one analytical step via SQL.

Manages its own message history, retry logic, tool-use loop, and event emission.
Works with any dataset — no domain-specific assumptions.
"""

import json
import logging
import re
import time

from latent_insights.agents.base import Agent
from latent_insights.api.feed import FeedEntry, parse_llm_response
from latent_insights.core.llm import LLMClient
from latent_insights.core.parsing import detect_degeneration, parse_worker_response
from latent_insights.core.queue import Queue
from latent_insights.models import StreamEvent, WorkerResult

logger = logging.getLogger(__name__)


def _extract_tool_sql(tool_calls: list[dict] | None) -> str:
    """Join the SQL strings the model emitted across its tool_calls for a turn.

    Used to populate the `response` field of an `llm_call` record when the
    model only emitted tool_calls (empty content). Malformed JSON in a single
    arg block is tolerated: that call is skipped but others still land.
    """
    if not tool_calls:
        return ""
    sqls: list[str] = []
    for tc in tool_calls:
        func = tc.get("function") or {}
        if func.get("name") != "run_sql":
            continue
        try:
            args = json.loads(func.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        sql = args.get("sql")
        if sql:
            sqls.append(sql)
    return "\n\n".join(sqls)


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

    SYNTHESIZE_PROMPT = """\
You are the synthesis writer for an analytical thread. The thread has
already run several SQL-based investigation steps and is ready for a
final summary.

You MUST NOT run any new queries. You have no tool available. Your job
is to condense the findings below into a single narrative.

## Dataset schema

{schema_summary}

## Thread history (previous steps and their results)

{thread_history}

## Coordinator's synthesis instruction

{worker_instruction}

## Output format

Return JSON (no tool call):

{{
  "summary": "3-6 sentences. Lead with the headline finding. Include key \
numbers, caveats (NULL rates, small subgroups, confounds), and any \
limitations. Do NOT invent numbers — only use what the steps above \
produced. If the evidence is thin, say so.",
  "view_requested": null
}}
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
        self.event_counter: int = 0
        self.llm_calls: list[dict] = []
        # Step context — stamped onto every SSE event this worker emits so
        # each event is self-contained and the UI can group by step without
        # tracking cross-event state.
        self.step_number: int = 0

    def _emit_feed(self, *, event_type: str, entry_id: str, message: str, **fields):
        """Build a FeedEntry and dispatch it as a StreamEvent.

        The worker doesn't own ``store`` — its per-step ``llm_call`` /
        ``tool_call`` records are batched on ``self.llm_calls`` and flushed
        by the runner at step end. This helper handles SSE emission only,
        pulling ``feed_index`` from the shared session counter.
        """
        entry = FeedEntry(
            id=entry_id,
            feed_index=self.queue.next_feed_index(self.session_id),
            event_type=event_type,
            thread_id=self.thread_id,
            timestamp=time.time(),
            message=message,
            **fields,
        )
        entry_data = entry.model_dump(exclude_none=True)
        self.queue.append_feed(self.session_id, entry_data)
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type=event_type,
            message=message,
            data=entry_data,
            timestamp=entry.timestamp,
        ))

    @property
    def role(self) -> str:
        return "worker"

    def start(
        self,
        instruction: str,
        thread_views: str = "(none)",
        step_number: int = 0,
        move: str = "",
        thread_history: str = "",
    ):
        """Initialize worker state for a new step.

        ``thread_history`` is only consulted for ``SYNTHESIZE`` moves,
        where the worker has no tool and must write the final summary
        from what previous steps already found. For non-SYNTHESIZE moves
        the history passes through the coordinator, not the worker.
        """
        self.instruction = instruction
        self.current_model = self.model
        self.consecutive_errors = 0
        self.attempts = 0
        self.event_counter = 0
        self.llm_calls = []
        self.step_number = step_number
        self.current_move = move

        if move == "SYNTHESIZE":
            prompt = self.SYNTHESIZE_PROMPT.format(
                schema_summary=self.schema_summary,
                thread_history=thread_history or "(no prior steps recorded)",
                worker_instruction=instruction,
            )
            user_turn = (
                "Write the synthesis now. Do not run queries. Base the "
                "summary only on the thread history above."
            )
        else:
            prompt = self.SYSTEM_PROMPT.format(
                schema_summary=self.schema_summary,
                thread_views=thread_views,
                worker_instruction=instruction,
            )
            user_turn = "Execute this analysis and return results."

        self.messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_turn},
        ]

    def call(self) -> tuple:
        """Single LLM call. Returns (response, call_ms). Raises APITimeoutError on timeout."""
        self.attempts += 1
        if self.attempts > 50:
            raise ValueError("Worker exceeded 50 LLM turns without producing a result")

        if self.consecutive_errors >= self.config.max_worker_retries:
            self.current_model = self.fallback_model

        # SYNTHESIZE is a terminal summary move: the worker should NOT run
        # new queries, only condense what previous steps have already found.
        # Withholding the tool forces the model to return a final JSON
        # summary instead of looping on ``run_sql``.
        tools = None if self.current_move == "SYNTHESIZE" else [self.RUN_SQL_TOOL]

        t0 = time.monotonic()
        response = self.llm.call(
            model=self.current_model,
            messages=self.messages,
            role=self.role,
            temperature=0.0,
            tools=tools,
            timeout=self.config.llm_timeout,
        )
        call_ms = round((time.monotonic() - t0) * 1000)

        has_tools = bool(response.tool_calls)
        # `response` on an llm_call is the model's decision for this turn:
        # either its text (JSON final answer or intermediate reasoning) or,
        # when it's only emitting tool calls, the SQL it decided to run.
        response_text = response.content or ""
        if not response_text and has_tools:
            response_text = _extract_tool_sql(response.tool_calls)

        response_text_parsed, response_tables = parse_llm_response(response_text)
        preview = (response_text_parsed or response_text or "").strip()
        self.event_counter += 1
        self._emit_feed(
            event_type="llm_call",
            entry_id=f"ev:{self.thread_id}:{self.step_number}:{self.event_counter}",
            message=preview,
            full_message=response_text_parsed or response_text or "",
            agent=self.role,
            model=self.current_model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=call_ms,
            has_tool_calls=has_tools,
            response=response_text,
            response_text=response_text_parsed,
            response_tables=response_tables,
            step_number=self.step_number,
            move=self.current_move,
        )

        return response, call_ms

    def handle_response(self, response, call_ms: int) -> WorkerResult | None:
        """Process worker LLM response. Returns WorkerResult when done, None if another call needed.

        Convenience method that runs tool calls inline on the current thread.
        For continuation-passing execution where each SQL runs as its own
        pool task, use ``prepare_tool_calls`` / ``record_tool_result`` /
        ``apply_error_guardrails`` / ``handle_final`` directly.
        """
        if response.tool_calls:
            return self._handle_tool_calls(response, call_ms)
        return self._handle_final(response, call_ms)

    # ------------------------------------------------------------------
    # Granular API for continuation-passing scheduling.
    # Split the three responsibilities of ``_handle_tool_calls`` so the
    # caller can schedule each SQL execution on the pool independently,
    # releasing the pool slot between calls.
    # ------------------------------------------------------------------

    def prepare_tool_calls(self, response, call_ms: int) -> list[dict]:
        """Record the LLM turn and return SQL tasks to execute.

        Appends the assistant message with tool_calls, records the
        ``llm_call`` log entry, and extracts runnable SQL. Malformed
        ``run_sql`` arguments or unknown tool names are handled inline
        (tool message appended, error counter bumped) and omitted from
        the returned task list. Does NOT execute any SQL.

        Returns: list of ``{"tool_call_id": str, "sql": str}`` to be run.
        """
        assistant_msg = {"role": "assistant", "content": response.content or None}
        assistant_msg["tool_calls"] = response.tool_calls
        self.messages.append(assistant_msg)

        # One llm_call record per LLM turn (this method is invoked once per turn).
        self.llm_calls.append({
            "agent": self.role,
            "type": "llm_call",
            "duration_ms": call_ms,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "response": _extract_tool_sql(response.tool_calls),
        })

        tasks: list[dict] = []
        for tool_call in response.tool_calls:
            func = tool_call["function"]
            if func["name"] != "run_sql":
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": f"Unknown tool: {func['name']}",
                })
                continue
            try:
                args = json.loads(func["arguments"])
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Worker tool_call arguments malformed JSON: {e} "
                    f"(raw: {func['arguments'][:200]!r})"
                )
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": (
                        f"TOOL CALL ERROR: your `run_sql` arguments were "
                        f"malformed JSON ({e}). Reissue the tool call with "
                        f"valid JSON of the form "
                        f'{{"sql": "SELECT ..."}}. Keep the SQL on a single '
                        f"line and properly escape any quotes."
                    ),
                })
                self.consecutive_errors += 1
                continue

            sql = args.get("sql", "")
            tasks.append({"tool_call_id": tool_call["id"], "sql": sql})

        return tasks

    def record_tool_result(
        self,
        tool_call_id: str,
        sql: str,
        tool_result: str,
        sql_ms: int,
    ):
        """Record the outcome of a single SQL execution.

        Appends the ``tool`` message with the result, emits the
        ``tool_call`` SSE event, records the ``tool_call`` log entry, and
        updates the consecutive-error counter based on whether the result
        was an error.
        """
        logger.info(f"Worker executing SQL: {sql[:200]}")
        self.event_counter += 1
        self._emit_feed(
            event_type="tool_call",
            entry_id=f"ev:{self.thread_id}:{self.step_number}:{self.event_counter}",
            message=sql,
            full_message=sql,
            agent=self.role,
            sql=sql,
            tool_result=tool_result,
            duration_ms=sql_ms,
            step_number=self.step_number,
            move=self.current_move,
        )
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_result,
        })
        self.llm_calls.append({
            "agent": self.role,
            "type": "tool_call",
            "sql": sql,
            "tool_result": tool_result,
            "duration_ms": sql_ms,
        })
        if tool_result.startswith("SQL ERROR:"):
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

    def apply_error_guardrails(self):
        """After all tool results for a turn, append nudge messages if
        consecutive-error thresholds are exceeded. Mirrors the error
        handling that lived at the bottom of ``_handle_tool_calls``.
        """
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

    def handle_final(self, response, call_ms: int) -> WorkerResult | None:
        """Final-answer path (no tool_calls). Public alias of ``_handle_final``."""
        return self._handle_final(response, call_ms)

    def _handle_final(self, response, call_ms: int) -> WorkerResult | None:
        """Worker returned a final text response (no tool calls)."""
        self.llm_calls.append({
            "agent": self.role,
            "type": "llm_call",
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
                try:
                    args = json.loads(func["arguments"])
                except json.JSONDecodeError as e:
                    # Malformed tool_call arguments (e.g. unterminated string
                    # from a truncated response). Don't crash — respond with a
                    # `tool` message so the conversation stays balanced and
                    # nudge the model to reissue with valid JSON.
                    logger.warning(
                        f"Worker tool_call arguments malformed JSON: {e} "
                        f"(raw: {func['arguments'][:200]!r})"
                    )
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": (
                            f"TOOL CALL ERROR: your `run_sql` arguments were "
                            f"malformed JSON ({e}). Reissue the tool call with "
                            f"valid JSON of the form "
                            f'{{"sql": "SELECT ..."}}. Keep the SQL on a single '
                            f"line and properly escape any quotes."
                        ),
                    })
                    self.consecutive_errors += 1
                    continue
                sql = args.get("sql", "")
                logger.info(f"Worker executing SQL: {sql[:200]}")
                t_sql = time.monotonic()
                result_text = self.execute_sql(self.session_db, sql)
                sql_ms = round((time.monotonic() - t_sql) * 1000)

                self.event_counter += 1
                self._emit_feed(
                    event_type="tool_call",
                    entry_id=f"ev:{self.thread_id}:{self.step_number}:{self.event_counter}",
                    message=sql,
                    full_message=sql,
                    agent=self.role,
                    sql=sql,
                    tool_result=result_text,
                    duration_ms=sql_ms,
                    step_number=self.step_number,
                    move=self.current_move,
                )
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result_text,
                })
                tool_results.append({"sql": sql, "result": result_text, "sql_ms": sql_ms})
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

        # One llm_call record per LLM turn (this function is invoked once per
        # turn). Its `response` is the SQL the model decided to run, so the
        # trace/REST shows the same content as the SSE llm_call event for this
        # turn.
        self.llm_calls.append({
            "agent": self.role,
            "type": "llm_call",
            "duration_ms": call_ms,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "response": _extract_tool_sql(response.tool_calls),
        })
        # One tool_call record per executed SQL. Tool fields only — no LLM
        # metadata; that lives on the sibling llm_call above.
        for tr in tool_results:
            self.llm_calls.append({
                "agent": self.role,
                "type": "tool_call",
                "sql": tr["sql"],
                "tool_result": tr["result"],
                "duration_ms": tr["sql_ms"],
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

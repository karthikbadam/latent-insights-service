"""
Worker agent — executes one analytical step via SQL.

Stateless. Receives instruction, generates and executes SQL via tool use, summarizes.
Works with any dataset — no domain-specific assumptions.
"""

import json
import logging
import time

from app.core.llm import LLMClient
from app.core.parsing import parse_worker_response
from app.models import WorkerResult

logger = logging.getLogger(__name__)

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
5. Summarize for a non-technical reader. Lead with finding, not method.

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


def _format_results(col_names: list[str], rows: list) -> str:
    """Format query results as a readable table string."""
    if not rows:
        return "(no rows returned)"
    header = " | ".join(col_names)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)


def _execute_sql(session_db, sql: str) -> str:
    """Execute SQL against session DB and return formatted results."""
    try:
        result = session_db.execute(sql)
        rows = result.fetchall()
        description = result.description
        col_names = [d[0] for d in description] if description else []
        return _format_results(col_names, rows)
    except Exception as e:
        return f"SQL ERROR: {e}"


async def run_worker(
    llm: LLMClient,
    model: str,
    fallback_model: str,
    worker_instruction: str,
    schema_summary: str,
    session_db,
    thread_views: str = "(none)",
    cache_ttl_hours: int = 24,
    max_retries: int = 3,
) -> WorkerResult:
    """
    Run one worker step using tool-use loop.

    The LLM calls run_sql to execute queries against the session DB,
    then returns a final JSON response with summary and findings.
    """
    prompt = SYSTEM_PROMPT.format(
        schema_summary=schema_summary,
        thread_views=thread_views,
        worker_instruction=worker_instruction,
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Execute this analysis and return results."},
    ]

    current_model = model
    attempts = 0
    max_turns = 50
    llm_calls = []

    while True:
        attempts += 1
        if attempts > max_turns:
            raise ValueError(f"Worker exceeded {max_turns} LLM turns without producing a result")
        if attempts > max_retries:
            current_model = fallback_model

        t0 = time.monotonic()
        response = await llm.call(
            model=current_model,
            messages=messages,
            role="worker",
            temperature=0.0,
            tools=[RUN_SQL_TOOL],
            cache_ttl_hours=cache_ttl_hours
        )
        call_ms = round((time.monotonic() - t0) * 1000)

        # If no tool calls, LLM returned its final answer
        if not response.tool_calls:
            llm_calls.append({
                "agent": "worker",
                "type": "response",
                "duration_ms": call_ms,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "response": response.content[:500] if response.content else "",
            })
            if not response.content or not response.content.strip():
                logger.warning("Worker returned empty response, requesting JSON output")
                messages.append({"role": "assistant", "content": response.content or ""})
                messages.append({
                    "role": "user",
                    "content": "Your response was empty. Please provide your final answer as JSON matching the output format specified in the system prompt.",
                })
                continue
            try:
                result = parse_worker_response(response.content)
            except ValueError:
                # LLM returned text instead of JSON — ask it to reformat
                logger.warning("Worker returned non-JSON response, requesting reformat")
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": "Your response must be valid JSON matching the output format specified in the system prompt. Please reformat your answer as JSON.",
                })
                continue
            result.llm_calls = llm_calls
            return result

        # Process tool calls
        # Append the assistant message with tool calls
        assistant_msg = {"role": "assistant", "content": response.content or None}
        assistant_msg["tool_calls"] = response.tool_calls
        messages.append(assistant_msg)

        tool_results = []
        for tool_call in response.tool_calls:
            func = tool_call["function"]
            if func["name"] == "run_sql":
                args = json.loads(func["arguments"])
                sql = args.get("sql", "")
                logger.info(f"Worker executing SQL: {sql[:200]}")
                result_text = _execute_sql(session_db, sql)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result_text,
                })
                tool_results.append({"sql": sql, "result": result_text[:1000]})
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": f"Unknown tool: {func['name']}",
                })

        for tr in tool_results:
            llm_calls.append({
                "agent": "worker",
                "type": "tool_call",
                "duration_ms": call_ms,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "sql": tr["sql"],
                "tool_result": tr["result"],
            })

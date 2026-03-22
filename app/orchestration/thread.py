"""Thread state machine — coordinator-worker cycle via futures.

Each step (coordinator LLM call, worker LLM call, SQL execution) is submitted
as a future to the pool. Pool threads are released between steps so other
analytical threads can make progress.
"""

import json
import logging
import time

from openai import APITimeoutError

from app.agents.coordinator import run_coordinator
from app.agents.worker import RUN_SQL_TOOL, SYSTEM_PROMPT as WORKER_SYSTEM_PROMPT, _execute_sql
from app.core.parsing import detect_degeneration, parse_worker_response
from app.models import (
    CoordinatorStatus,
    StreamEvent,
    ThreadStatus,
    WorkerResult,
)
from app.orchestration.context import ThreadContext

logger = logging.getLogger(__name__)


def _get_thread_views(session_db, thread_id: str) -> str:
    """List existing views for this thread."""
    try:
        rows = session_db.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_type = 'VIEW' AND table_name LIKE ?
        """, [f"thread_{thread_id}_%"]).fetchall()
        if rows:
            return "\n".join(r[0] for r in rows)
    except Exception:
        pass
    return "(none)"


# --- Public API ---


def start_thread(ctx: ThreadContext):
    """Kick off the thread state machine. Non-blocking — returns immediately."""
    _start_step(ctx)


def resume_thread(ctx: ThreadContext):
    """Resume a stuck thread. Human messages should already be set on ctx."""
    if not ctx.trace_store.get_spans(ctx.thread.id):
        ctx.trace_store.load_trace(ctx.thread.id, ctx.thread.session_id)

    ctx.state.update_thread_status(ctx.thread.id, ThreadStatus.RUNNING)
    start_thread(ctx)


# --- State machine: coordinator ---


def _start_step(ctx: ThreadContext):
    """Begin a new coordinator-worker step by scheduling the coordinator call."""
    ctx.step_number = len(ctx.trace_store.get_step_spans(ctx.thread.id)) + 1
    ctx.step_start = time.monotonic()
    ctx.step_span = ctx.trace_store.start_span(
        trace_id=ctx.thread.id,
        name=f"step_{ctx.step_number}",
        kind="step",
    )

    # Emit thread_start on first step
    if ctx.step_number == 1:
        ctx.queue.emit(StreamEvent(
            session_id=ctx.thread.session_id,
            thread_id=ctx.thread.id,
            event_type="thread_start",
            message=ctx.thread.seed_question,
            data={
                "seed_question": ctx.thread.seed_question,
                "motivation": ctx.thread.motivation,
                "entry_point": ctx.thread.entry_point,
            },
        ))

    future = ctx.queue.schedule(
        fn=_coordinator_call,
        args=(ctx,),
        task_id=f"coord-{ctx.tid}-{ctx.step_number}",
        session_id=ctx.thread.session_id,
        thread_id=ctx.thread.id,
        description=f"Coordinator step {ctx.step_number}: {ctx.thread.seed_question[:60]}",
    )
    future.add_done_callback(lambda f: _safe_callback(_handle_coordinator_done, ctx, f))


def _coordinator_call(ctx: ThreadContext):
    """Run coordinator LLM call. Executes on pool thread."""
    thread_history = ctx.trace_store.format_thread_history(
        ctx.thread.id, ctx.human_messages, running_summary=ctx.thread.running_summary,
    )
    ctx.thread_views = _get_thread_views(ctx.session_db, ctx.thread.id)

    t0 = time.monotonic()
    decision, log = run_coordinator(
        llm=ctx.llm,
        model=ctx.config.models.coordinator,
        seed_question=ctx.thread.seed_question,
        motivation=ctx.thread.motivation,
        entry_point=ctx.thread.entry_point,
        schema_summary=ctx.schema_summary,
        thread_history=thread_history,
        temperature=ctx.config.temperatures.coordinator,
        queue=ctx.queue,
        session_id=ctx.thread.session_id,
        thread_id=ctx.thread.id,
    )
    coordinator_ms = round((time.monotonic() - t0) * 1000)
    log["duration_ms"] = coordinator_ms
    return decision, log, coordinator_ms


def _handle_coordinator_done(ctx: ThreadContext, future):
    """Process coordinator result and schedule worker or finalize."""
    decision, coordinator_log, coordinator_ms = future.result()

    ctx.decision = decision
    ctx.coordinator_ms = coordinator_ms

    ctx.trace_store.add_event(ctx.step_span, "coordinator", {
        "model": ctx.config.models.coordinator,
        "duration_ms": coordinator_ms,
        "status": decision.status.value,
        "next_move": decision.next_move.value,
        "assessment": decision.assessment,
        "rationale": decision.rationale,
    })

    logger.info(
        f"Thread {ctx.thread.id} coordinator: {decision.status.value} "
        f"-> {decision.next_move.value} ({coordinator_ms}ms)"
    )

    if decision.status == CoordinatorStatus.STUCK:
        _finalize_stuck(ctx)
        return

    # Initialize worker state
    ctx.worker_instruction = decision.worker_instruction or ""
    ctx.current_model = ctx.config.models.worker
    ctx.consecutive_errors = 0
    ctx.attempts = 0
    ctx.llm_calls = []

    prompt = WORKER_SYSTEM_PROMPT.format(
        schema_summary=ctx.schema_summary,
        thread_views=ctx.thread_views,
        worker_instruction=ctx.worker_instruction,
    )
    ctx.worker_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Execute this analysis and return results."},
    ]

    ctx.queue.emit(StreamEvent(
        session_id=ctx.thread.session_id,
        thread_id=ctx.thread.id,
        event_type="step_start",
        message=ctx.worker_instruction,
        data={
            "move": decision.next_move.value,
            "step_number": ctx.step_number,
            "instruction": ctx.worker_instruction,
        },
    ))

    _schedule_worker_call(ctx)


# --- State machine: worker ---


def _schedule_worker_call(ctx: ThreadContext):
    """Submit the next worker LLM call as a future."""
    future = ctx.queue.schedule(
        fn=_worker_call,
        args=(ctx,),
        task_id=f"worker-{ctx.tid}-{ctx.step_number}-{ctx.attempts}",
        session_id=ctx.thread.session_id,
        thread_id=ctx.thread.id,
        description=f"Worker call {ctx.attempts}: {ctx.worker_instruction[:60]}",
    )
    future.add_done_callback(lambda f: _safe_callback(_handle_worker_done, ctx, f))


def _worker_call(ctx: ThreadContext):
    """Single worker LLM call. Executes on pool thread."""
    ctx.attempts += 1
    if ctx.attempts > 50:
        raise ValueError("Worker exceeded 50 LLM turns without producing a result")

    if ctx.consecutive_errors >= ctx.config.max_worker_retries:
        ctx.current_model = ctx.config.models.worker_fallback

    t0 = time.monotonic()
    response = ctx.llm.call(
        model=ctx.current_model,
        messages=ctx.worker_messages,
        role="worker",
        temperature=0.0,
        tools=[RUN_SQL_TOOL],
        timeout=ctx.config.llm_timeout,
    )
    call_ms = round((time.monotonic() - t0) * 1000)

    has_tools = bool(response.tool_calls)
    ctx.queue.emit(StreamEvent(
        session_id=ctx.thread.session_id,
        thread_id=ctx.thread.id,
        event_type="llm_call",
        message=f"Worker {'executing SQL' if has_tools else 'summarizing'} ({call_ms}ms)",
        data={
            "role": "worker",
            "model": ctx.current_model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "duration_ms": call_ms,
            "has_tool_calls": has_tools,
        },
    ))

    return response, call_ms


def _handle_worker_done(ctx: ThreadContext, future):
    """Process worker LLM response — tool calls or final answer."""
    try:
        response, call_ms = future.result()
    except APITimeoutError:
        logger.warning(f"Worker LLM call timed out for thread {ctx.thread.id}")
        ctx.consecutive_errors += 1
        ctx.worker_messages.append({
            "role": "user",
            "content": "Your previous response timed out. Simplify your approach and respond more concisely.",
        })
        _schedule_worker_call(ctx)
        return

    if not response.tool_calls:
        _handle_worker_final(ctx, response, call_ms)
    else:
        _handle_worker_tool_calls(ctx, response, call_ms)


def _handle_worker_final(ctx: ThreadContext, response, call_ms: int):
    """Worker returned a final text response (no tool calls)."""
    ctx.llm_calls.append({
        "agent": "worker",
        "type": "response",
        "duration_ms": call_ms,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "response": response.content if response.content else "",
    })

    if not response.content or not response.content.strip():
        logger.warning("Worker returned empty response, requesting JSON output")
        ctx.worker_messages.append({"role": "assistant", "content": response.content or ""})
        ctx.worker_messages.append({
            "role": "user",
            "content": "Your response was empty. Please provide your final answer as JSON matching the output format specified in the system prompt.",
        })
        _schedule_worker_call(ctx)
        return

    # Check for LLM degeneration (token loops like "pull pull pull...")
    if detect_degeneration(response.content):
        logger.warning(f"Worker output degeneration detected for thread {ctx.thread.id}")
        ctx.worker_messages.append({"role": "assistant", "content": response.content})
        ctx.worker_messages.append({
            "role": "user",
            "content": "Your output contained repeated/degenerate text. Please provide a concise, clean JSON response with your findings so far.",
        })
        _schedule_worker_call(ctx)
        return

    try:
        worker_result = parse_worker_response(response.content)
    except (ValueError, json.JSONDecodeError):
        logger.warning("Worker returned non-JSON response, requesting reformat")
        ctx.worker_messages.append({"role": "assistant", "content": response.content})
        ctx.worker_messages.append({
            "role": "user",
            "content": "Your response must be valid JSON matching the output format specified in the system prompt. Please reformat your answer as JSON.",
        })
        _schedule_worker_call(ctx)
        return

    worker_result.llm_calls = ctx.llm_calls
    _complete_step(ctx, worker_result)


def _handle_worker_tool_calls(ctx: ThreadContext, response, call_ms: int):
    """Worker wants to execute SQL tools — run inline (fast) then schedule next call."""
    assistant_msg = {"role": "assistant", "content": response.content or None}
    assistant_msg["tool_calls"] = response.tool_calls
    ctx.worker_messages.append(assistant_msg)

    tool_results = []
    for tool_call in response.tool_calls:
        func = tool_call["function"]
        if func["name"] == "run_sql":
            args = json.loads(func["arguments"])
            sql = args.get("sql", "")
            logger.info(f"Worker executing SQL: {sql[:200]}")
            t_sql = time.monotonic()
            result_text = _execute_sql(ctx.session_db, sql)
            sql_ms = round((time.monotonic() - t_sql) * 1000)

            ctx.queue.emit(StreamEvent(
                session_id=ctx.thread.session_id,
                thread_id=ctx.thread.id,
                event_type="tool_call",
                message=sql,
                data={"sql": sql, "result": result_text, "duration_ms": sql_ms},
            ))
            ctx.worker_messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result_text,
            })
            tool_results.append({"sql": sql, "result": result_text[:1000]})
            if result_text.startswith("SQL ERROR:"):
                ctx.consecutive_errors += 1
            else:
                ctx.consecutive_errors = 0
        else:
            ctx.worker_messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": f"Unknown tool: {func['name']}",
            })

    for tr in tool_results:
        ctx.llm_calls.append({
            "agent": "worker",
            "type": "tool_call",
            "duration_ms": call_ms,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "sql": tr["sql"],
            "tool_result": tr["result"],
        })

    # Error guardrails
    if ctx.consecutive_errors >= ctx.config.max_consecutive_errors:
        ctx.worker_messages.append({
            "role": "user",
            "content": (
                f"You have hit {ctx.consecutive_errors} consecutive SQL errors. "
                "Stop trying SQL and return your final JSON answer NOW "
                "with whatever findings you have so far. If you have no findings, "
                "state that the analysis could not be completed and explain why."
            ),
        })
    elif ctx.consecutive_errors >= 2:
        ctx.worker_messages.append({
            "role": "user",
            "content": (
                f"You have hit {ctx.consecutive_errors} consecutive SQL errors. "
                "The function you are trying likely does not exist in DuckDB. "
                "STOP retrying the same approach. Rewrite your analysis using "
                "only basic SQL math and aggregates (AVG, STDDEV_POP, CORR, etc)."
            ),
        })

    _schedule_worker_call(ctx)


# --- Step completion and finalization ---


def _complete_step(ctx: ThreadContext, worker_result: WorkerResult):
    """A worker step finished. Record it and decide next action."""
    step_ms = round((time.monotonic() - ctx.step_start) * 1000)
    worker_ms = step_ms - ctx.coordinator_ms

    ctx.trace_store.add_event(ctx.step_span, "worker", {
        "model": ctx.config.models.worker,
        "duration_ms": worker_ms,
        "result_preview": worker_result.result[:200],
    })
    if worker_result.llm_calls:
        for call in worker_result.llm_calls:
            ctx.trace_store.add_event(ctx.step_span, "llm_call", call)

    ctx.step_span.attributes.update({
        "move": ctx.decision.next_move.value,
        "instruction": ctx.decision.worker_instruction,
        "result": worker_result.result,
        "coordinator_ms": ctx.coordinator_ms,
        "worker_ms": worker_ms,
    })
    ctx.trace_store.end_span(ctx.step_span)

    logger.info(
        f"Thread {ctx.thread.id} step {ctx.step_number} "
        f"({ctx.decision.next_move.value}): "
        f"coordinator={ctx.coordinator_ms}ms worker={worker_ms}ms "
        f"total={step_ms}ms"
    )

    ctx.queue.emit(StreamEvent(
        session_id=ctx.thread.session_id,
        thread_id=ctx.thread.id,
        event_type="step_complete",
        message=worker_result.result,
        data={
            "step_number": ctx.step_number,
            "move": ctx.decision.next_move.value,
            "result": worker_result.result,
        },
    ))

    if ctx.step_number > 1 and ctx.step_number % 5 == 0:
        _maybe_summarize(ctx)

    # Move repetition guard
    ctx.move_history.append(ctx.decision.next_move.value)
    max_same = ctx.config.max_repeated_moves
    if (
        len(ctx.move_history) >= max_same
        and len(set(ctx.move_history[-max_same:])) == 1
        and ctx.decision.status != CoordinatorStatus.DONE
    ):
        move_name = ctx.move_history[-1]
        logger.warning(
            f"Thread {ctx.thread.id} repeated {move_name} {max_same} times — forcing STUCK"
        )
        ctx.decision.status = CoordinatorStatus.STUCK
        ctx.decision.question_for_human = (
            f"Thread has repeated {move_name} {max_same} times without progress. "
            "What direction should the analysis take?"
        )
        ctx.decision.context = f"Last {max_same} moves: {', '.join(ctx.move_history[-max_same:])}"
        _finalize_stuck(ctx)
        return

    if ctx.decision.status == CoordinatorStatus.DONE:
        _finalize_complete(ctx, worker_result)
    else:
        ctx.human_messages = []
        _start_step(ctx)


def _finalize_complete(ctx: ThreadContext, worker_result: WorkerResult):
    """Thread finished successfully."""
    thread_elapsed = round(time.monotonic() - ctx.thread_start, 2)
    logger.info(
        f"Thread {ctx.thread.id} complete: "
        f"{ctx.step_number} steps in {thread_elapsed}s"
    )

    ctx.trace_store.flush_to_file(ctx.thread.id, ctx.thread.session_id)
    ctx.trace_store.clear_trace(ctx.thread.id)
    ctx.state.update_thread_status(
        ctx.thread.id, ThreadStatus.COMPLETE,
        summary=worker_result.result,
    )
    ctx.state.dump_session(ctx.thread.session_id)

    ctx.queue.emit(StreamEvent(
        session_id=ctx.thread.session_id,
        thread_id=ctx.thread.id,
        event_type="thread_complete",
        message=worker_result.result,
        data={
            "summary": worker_result.result,
            "total_seconds": thread_elapsed,
            "step_count": ctx.step_number,
        },
    ))

    _cleanup(ctx)


def _finalize_stuck(ctx: ThreadContext):
    """Thread is stuck and needs human input."""
    decision = ctx.decision
    ctx.step_span.attributes.update({
        "move": decision.next_move.value,
        "instruction": decision.question_for_human or "",
        "result": f"STUCK: {decision.context or ''}",
    })
    ctx.trace_store.end_span(ctx.step_span, status="stuck")

    ctx.trace_store.flush_to_file(ctx.thread.id, ctx.thread.session_id)
    ctx.trace_store.clear_trace(ctx.thread.id)
    ctx.state.update_thread_status(ctx.thread.id, ThreadStatus.WAITING)
    ctx.state.dump_session(ctx.thread.session_id)

    ctx.queue.emit(StreamEvent(
        session_id=ctx.thread.session_id,
        thread_id=ctx.thread.id,
        event_type="thread_waiting",
        message=decision.question_for_human or "Thread needs guidance.",
        data={
            "question": decision.question_for_human,
            "context": decision.context,
        },
    ))

    _cleanup(ctx)


def _finalize_error(ctx: ThreadContext, error: Exception):
    """Thread hit an error — ask the human for help instead of dying."""
    error_msg = f"{type(error).__name__}: {error}"
    logger.error(
        f"Thread {ctx.thread.id} error: {error_msg}",
        exc_info=True,
    )
    try:
        if ctx.step_span:
            ctx.step_span.attributes.update({
                "move": ctx.decision.next_move.value if ctx.decision else "ERROR",
                "instruction": (ctx.decision.worker_instruction or "") if ctx.decision else "",
                "result": f"Error: {error_msg}",
                "error": error_msg,
            })
            ctx.trace_store.end_span(ctx.step_span, status="error")
        ctx.trace_store.flush_to_file(ctx.thread.id, ctx.thread.session_id)
        ctx.trace_store.clear_trace(ctx.thread.id)
    except Exception:
        pass

    ctx.state.update_thread_status(ctx.thread.id, ThreadStatus.WAITING)
    ctx.state.dump_session(ctx.thread.session_id)

    ctx.queue.emit(StreamEvent(
        session_id=ctx.thread.session_id,
        thread_id=ctx.thread.id,
        event_type="thread_waiting",
        message=f"Thread encountered an error: {error_msg}. How should it proceed?",
        data={
            "question": f"Thread encountered an error: {error_msg}. How should it proceed?",
            "context": error_msg,
        },
    ))

    _cleanup(ctx)


# --- Helpers ---


def _safe_callback(handler, ctx: ThreadContext, future):
    """Wrap a done callback to catch exceptions and route to finalize_error."""
    try:
        handler(ctx, future)
    except Exception as e:
        logger.error(f"Thread {ctx.thread.id} callback error: {e}", exc_info=True)
        try:
            _finalize_error(ctx, e)
        except Exception:
            logger.error(f"Thread {ctx.thread.id} failed to finalize after error", exc_info=True)
            ctx.done_event.set()


def _cleanup(ctx: ThreadContext):
    """Close DB and signal completion."""
    try:
        ctx.session_db.close()
    except Exception:
        pass
    ctx.done_event.set()


def _maybe_summarize(ctx: ThreadContext):
    """Summarize thread history if it's getting long."""
    try:
        summary = ctx.trace_store.summarize_history(
            trace_id=ctx.thread.id,
            llm=ctx.llm,
            model=ctx.config.models.coordinator,
            seed_question=ctx.thread.seed_question,
        )
        if summary and not detect_degeneration(summary):
            ctx.state.update_thread_running_summary(ctx.thread.id, summary)
            ctx.thread.running_summary = summary
            logger.info(f"Thread {ctx.thread.id} history summarized")
        elif summary:
            logger.warning(f"Thread {ctx.thread.id} summary discarded — degenerate output detected")
    except Exception as e:
        logger.warning(f"History summarization failed: {e}")

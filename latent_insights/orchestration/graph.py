"""
LangGraph-based thread state machine.

Replaces the hand-rolled callback chain in the original ThreadRunner with a
declarative StateGraph. Each node is a plain Python function wrapping the
existing agent classes. Edges encode the routing logic (guards, stuck detection,
move repetition) that was previously scattered across callbacks.
"""

import logging
import time
from dataclasses import asdict
from datetime import timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from latent_insights.agents.coordinator import Coordinator
from latent_insights.agents.worker import Worker
from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMClient
from latent_insights.core.parsing import detect_degeneration
from latent_insights.core.queue import Queue
from latent_insights.core.state import StateStore
from latent_insights.core.tracing import TraceStore
from latent_insights.models import (
    CoordinatorDecision,
    CoordinatorStatus,
    MoveType,
    StreamEvent,
    ThreadStatus,
    WorkerResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema — all data flowing through the graph
# ---------------------------------------------------------------------------

class ThreadState(TypedDict, total=False):
    # Identity (set once at start)
    session_id: str
    thread_id: str
    seed_question: str
    motivation: str
    entry_point: str
    schema_summary: str

    # Step tracking (updated each iteration)
    step_number: int
    move_history: list[str]
    human_messages: list[str]

    # Current step results (written by nodes)
    decision: dict | None
    worker_result: dict | None
    coordinator_ms: int
    worker_ms: int
    thread_views: str

    # Control
    status: str  # "running" | "stuck" | "complete" | "error"
    error: str | None
    error_count: int
    thread_start: float
    # Why the thread is entering a terminal waiting state. One of:
    #   "coordinator_stuck"  — coordinator returned STUCK (genuine human-needed)
    #   "repeated_moves"     — move repetition guard tripped
    #   "retry_exhausted"    — transient LLM error survived all retries
    #   "unexpected_error"   — any other raised exception
    wait_reason: str

    # Config
    max_repeated_moves: int


# ---------------------------------------------------------------------------
# Node functions — each wraps an existing agent
# ---------------------------------------------------------------------------

def make_coordinator_node(
    coordinator: Coordinator,
    trace_store: TraceStore,
    queue: Queue,
    state_store: StateStore,
):
    """Create the coordinator node function."""

    def coordinator_node(state: ThreadState) -> dict:
        thread_id = state["thread_id"]
        session_id = state["session_id"]
        step_number = state.get("step_number", 0) + 1

        # Drain any pending messages injected mid-run via the interrupt API
        injected = state_store.drain_pending_messages(thread_id)
        human_messages = list(state.get("human_messages", []))
        if injected:
            human_messages.extend(injected)
            logger.info(f"Thread {thread_id[:8]} received {len(injected)} injected message(s)")

        # Format history from trace spans
        running_summary = None
        thread_history = trace_store.format_thread_history(
            thread_id,
            human_messages,
            running_summary=running_summary,
        )

        # Emit thread_start on first step. Stamp with the thread's creation
        # time (earliest guaranteed moment for this thread) and step_number=0
        # so any FE ordering by (step_number, timestamp) places START before
        # the first step_start even when the coordinator LLM call is fast.
        if step_number == 1:
            thread_obj = state_store.get_thread(thread_id)
            if thread_obj:
                # created_at is a naive datetime semantically in UTC; treat
                # it as such so the Unix timestamp matches time.time().
                created_ts = thread_obj.created_at.replace(
                    tzinfo=timezone.utc
                ).timestamp()
            else:
                created_ts = time.time()
            queue.emit(StreamEvent(
                session_id=session_id,
                thread_id=thread_id,
                event_type="thread_start",
                message=state["seed_question"],
                data={
                    "seed_question": state["seed_question"],
                    "motivation": state.get("motivation", ""),
                    "entry_point": state.get("entry_point", ""),
                    "step_number": 0,
                },
                timestamp=created_ts,
            ))

        # Start trace span
        span = trace_store.start_span(
            trace_id=thread_id, name=f"step_{step_number}", kind="step",
        )

        # Record any human messages consumed by this step as span events so
        # they appear in the step's `events` timeline alongside llm_call
        # and tool_call. Entries carry their own timestamp (captured when
        # the user posted the message) so they sort before the coordinator
        # LLM call that was influenced by them.
        for msg in human_messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                target = msg.get("target", "thread")
                ts = msg.get("timestamp")
            else:
                # Legacy bare-string entries from older persisted state.
                content = str(msg)
                target = "thread"
                ts = None
            trace_store.add_event(
                span,
                "human_message",
                {"content": content, "target": target},
                timestamp=ts,
            )

        t0 = time.monotonic()
        decision, coord_log = coordinator.call(
            seed_question=state["seed_question"],
            motivation=state.get("motivation", ""),
            entry_point=state.get("entry_point", ""),
            schema_summary=state["schema_summary"],
            thread_history=thread_history,
        )
        coordinator_ms = round((time.monotonic() - t0) * 1000)
        coord_log["duration_ms"] = coordinator_ms

        trace_store.add_event(span, "llm_call", {
            "agent": "coordinator",
            "model": coord_log["model"],
            "duration_ms": coordinator_ms,
            "input_tokens": coord_log.get("input_tokens"),
            "output_tokens": coord_log.get("output_tokens"),
            "response": coord_log.get("response"),
        })

        logger.info(
            f"Thread {thread_id} coordinator: {decision.status.value} "
            f"-> {decision.next_move.value} ({coordinator_ms}ms)"
        )

        # Early stuck override
        if decision.status == CoordinatorStatus.STUCK and step_number <= 2:
            logger.warning(f"Thread {thread_id[:8]} STUCK on step {step_number} — overriding to FORAGE")
            decision.status = CoordinatorStatus.CONTINUE
            decision.next_move = MoveType.FORAGE
            decision.worker_instruction = (
                f"Try a different exploratory approach to answer: {state['seed_question']}"
            )

        # Emit the coordinator llm_call SSE *after* the early-stuck override so
        # the event's `move` matches the step's final move (what step_start
        # emits below). Every thread-scoped event carries step_number + move so
        # the UI can group events by step without maintaining state.
        final_move = decision.next_move.value
        queue.emit(StreamEvent(
            session_id=session_id,
            thread_id=thread_id,
            event_type="llm_call",
            message=f"Coordinator deciding next move ({coordinator_ms}ms)",
            data={
                "agent": "coordinator",
                "model": coord_log["model"],
                "input_tokens": coord_log.get("input_tokens"),
                "output_tokens": coord_log.get("output_tokens"),
                "duration_ms": coordinator_ms,
                "response": coord_log.get("response", ""),
                "step_number": step_number,
                "move": final_move,
            },
        ))

        queue.emit(StreamEvent(
            session_id=session_id,
            thread_id=thread_id,
            event_type="step_start",
            message=decision.worker_instruction or "",
            data={
                "move": final_move,
                "step_number": step_number,
                "instruction": decision.worker_instruction or "",
                # The coordinator has already committed to this move (including
                # the early-stuck override at step <= 2). The UI can render it
                # authoritatively rather than waiting for step_complete.
                "provisional": False,
            },
        ))

        return {
            "decision": asdict(decision),
            "coordinator_ms": coordinator_ms,
            "step_number": step_number,
        }

    return coordinator_node


def make_worker_node(
    worker: Worker,
    trace_store: TraceStore,
    queue: Queue,
    session_db: Any,
):
    """Create the worker node function."""

    def _get_thread_views(thread_id: str) -> str:
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

    def worker_node(state: ThreadState) -> dict:
        thread_id = state["thread_id"]
        session_id = state["session_id"]
        decision_dict = state["decision"]
        step_number = state["step_number"]

        decision = CoordinatorDecision(
            assessment=decision_dict["assessment"],
            next_move=MoveType(decision_dict["next_move"]),
            rationale=decision_dict["rationale"],
            status=CoordinatorStatus(decision_dict["status"]),
            worker_instruction=decision_dict.get("worker_instruction"),
            question_for_human=decision_dict.get("question_for_human"),
            context=decision_dict.get("context"),
        )

        thread_views = _get_thread_views(thread_id)

        # Initialize worker for this step — step_number and move are stamped
        # onto every SSE event the worker emits so each event is
        # self-contained (no client-side stitching required).
        worker.start(
            instruction=decision.worker_instruction or "",
            thread_views=thread_views,
            step_number=step_number,
            move=decision.next_move.value,
        )

        # Worker tool-use loop. Transient LLM errors (connection, 429, 5xx,
        # timeouts) are retried inside LLMClient.call with exponential
        # backoff. If an exception propagates here, retries were exhausted
        # or the error is non-transient — let it bubble up to the graph-
        # level handler in thread.py, which emits thread_waiting with a
        # reason derived from the exception type.
        step_start = time.monotonic()
        result = None
        while result is None:
            response, call_ms = worker.call()
            result = worker.handle_response(response, call_ms)

        worker_ms = round((time.monotonic() - step_start) * 1000)

        # Record in trace
        spans = trace_store.get_step_spans(thread_id)
        if spans:
            span = spans[-1]
            if result.llm_calls:
                for call in result.llm_calls:
                    # Worker tags each record with its event vocabulary
                    # ("llm_call" for LLM turns, "tool_call" for SQL execs).
                    # Trust the producer; the record's own `type` is the
                    # event name on the trace.
                    trace_store.add_event(span, call["type"], call)
            span.attributes.update({
                "move": decision.next_move.value,
                "instruction": decision.worker_instruction,
                "result": result.result,
                "coordinator_ms": state.get("coordinator_ms", 0),
                "worker_ms": worker_ms,
            })
            trace_store.end_span(span)

        logger.info(
            f"Thread {thread_id} step {step_number} ({decision.next_move.value}): "
            f"coordinator={state.get('coordinator_ms', 0)}ms worker={worker_ms}ms"
        )

        coordinator_ms = state.get("coordinator_ms", 0)
        queue.emit(StreamEvent(
            session_id=session_id,
            thread_id=thread_id,
            event_type="step_complete",
            message=result.result,
            data={
                "step_number": step_number,
                "move": decision.next_move.value,
                "instruction": decision.worker_instruction or "",
                "result": result.result,
                "duration_ms": coordinator_ms + worker_ms,
            },
        ))

        # Update move history
        move_history = list(state.get("move_history", []))
        move_history.append(decision.next_move.value)

        return {
            "worker_result": asdict(result),
            "worker_ms": worker_ms,
            "move_history": move_history,
            "human_messages": [],  # clear after use
        }

    return worker_node


def make_finalize_complete_node(
    state_store: StateStore,
    trace_store: TraceStore,
    queue: Queue,
):
    """Create the finalize-complete node."""

    def finalize_complete_node(state: ThreadState) -> dict:
        thread_id = state["thread_id"]
        session_id = state["session_id"]
        worker_result = state.get("worker_result", {})
        summary = worker_result.get("result", "")

        thread_elapsed = round(time.monotonic() - state.get("thread_start", time.monotonic()), 2)
        logger.info(
            f"Thread {thread_id} complete: {state['step_number']} steps in {thread_elapsed}s"
        )

        trace_store.flush_to_file(thread_id, session_id)
        trace_store.clear_trace(thread_id)
        state_store.update_thread_status(thread_id, ThreadStatus.COMPLETE, summary=summary)
        state_store.dump_session(session_id)

        queue.emit(StreamEvent(
            session_id=session_id,
            thread_id=thread_id,
            event_type="thread_complete",
            message=summary,
            data={
                "summary": summary,
                "total_seconds": thread_elapsed,
                "total_ms": round(thread_elapsed * 1000),
                "step_count": state["step_number"],
                # Terminal marker: FE uses this to render a COMPLETE node in
                # the DAG viz without maintaining client-side state. Paired
                # with step_number so the terminal lands after the last step.
                "is_terminal": True,
                "step_number": state["step_number"] + 1,
            },
        ))
        return {"status": "complete"}

    return finalize_complete_node


def emit_thread_waiting(
    state_store: StateStore,
    trace_store: TraceStore,
    queue: Queue,
    *,
    session_id: str,
    thread_id: str,
    reason: str,
    question: str,
    context: str | None = None,
    error: str | None = None,
    span_status: str = "stuck",
) -> None:
    """Shared path for moving a thread into WAITING and emitting the SSE.

    Used by both the in-graph ``finalize_stuck`` node and the top-level
    error handler in ``thread.py``. Centralising it guarantees the payload
    shape is identical whichever path the thread took to get here.
    """
    # End current span if still open
    spans = trace_store.get_step_spans(thread_id)
    step_count = len(spans)
    if spans and spans[-1].end_time is None:
        span = spans[-1]
        span.attributes.setdefault("result", f"{reason}: {context or error or ''}")
        if error:
            span.attributes["error"] = error
        trace_store.end_span(span, status=span_status)

    trace_store.flush_to_file(thread_id, session_id)
    trace_store.clear_trace(thread_id)

    state_store.update_thread_status(
        thread_id, ThreadStatus.WAITING, error=error or reason,
    )
    state_store.dump_session(session_id)

    thread = state_store.get_thread(thread_id)
    running_summary = thread.running_summary if thread else None

    queue.emit(StreamEvent(
        session_id=session_id,
        thread_id=thread_id,
        event_type="thread_waiting",
        message=question,
        data={
            "reason": reason,
            "question": question,
            "context": context,
            "error": error,
            "running_summary": running_summary,
            # Terminal marker for the WAITING branch of the DAG viz. Paired
            # with step_number so the terminal lands after the last step.
            "is_terminal": True,
            "step_number": step_count + 1,
        },
    ))


def make_finalize_stuck_node(
    state_store: StateStore,
    trace_store: TraceStore,
    queue: Queue,
):
    """Create the finalize-stuck node — reached when the coordinator
    returns STUCK (post-step-2) or the move-repetition guard trips.
    """

    def finalize_stuck_node(state: ThreadState) -> dict:
        thread_id = state["thread_id"]
        session_id = state["session_id"]
        decision = state.get("decision") or {}
        reason = state.get("wait_reason") or "coordinator_stuck"

        question = decision.get("question_for_human") or (
            "Thread repeated the same move too many times — needs guidance."
            if reason == "repeated_moves"
            else "Thread needs guidance."
        )
        emit_thread_waiting(
            state_store, trace_store, queue,
            session_id=session_id,
            thread_id=thread_id,
            reason=reason,
            question=question,
            context=decision.get("context"),
        )
        return {"status": "stuck"}

    return finalize_stuck_node


def make_summarize_node(
    state_store: StateStore,
    trace_store: TraceStore,
    llm: LLMClient,
    config: AppConfig,
):
    """Create the optional summarize node (runs every 5 steps)."""

    def summarize_node(state: ThreadState) -> dict:
        thread_id = state["thread_id"]
        try:
            summary = trace_store.summarize_history(
                trace_id=thread_id,
                llm=llm,
                model=config.models.coordinator,
                seed_question=state["seed_question"],
            )
            if summary and not detect_degeneration(summary):
                state_store.update_thread_running_summary(thread_id, summary)
                logger.info(f"Thread {thread_id} history summarized")
            elif summary:
                logger.warning(f"Thread {thread_id} summary discarded — degenerate output")
        except Exception as e:
            logger.warning(f"History summarization failed: {e}")
        return {}

    return summarize_node


# ---------------------------------------------------------------------------
# Routing functions — encode the guard logic
# ---------------------------------------------------------------------------

def route_after_coordinator(state: ThreadState) -> str:
    """Route coordinator output to worker or finalize_stuck."""
    decision = state.get("decision") or {}
    status = decision.get("status", "CONTINUE")

    if status == "STUCK":
        # Early stuck was already overridden in the node for step <= 2.
        # Reaching here with STUCK means step > 2 — the coordinator genuinely
        # decided it needs human input.
        state["wait_reason"] = "coordinator_stuck"
        return "finalize_stuck"

    return "worker"


def route_after_worker(state: ThreadState) -> str:
    """Route worker output: loop back, complete, stuck, or summarize.

    Side effect: mutates ``state["wait_reason"]`` when routing to
    ``finalize_stuck`` so the finalize node knows which reason to emit.
    This is a pragmatic shortcut — LangGraph routing functions are
    supposed to be pure, but the alternative (adding a separate node
    just to set the reason) adds more code than it saves.
    """
    decision = state.get("decision") or {}
    move_history = state.get("move_history", [])
    max_same = state.get("max_repeated_moves", 10)
    step_number = state.get("step_number", 0)

    # Move repetition guard
    if (
        len(move_history) >= max_same
        and len(set(move_history[-max_same:])) == 1
        and decision.get("status") != "DONE"
    ):
        move_name = move_history[-1]
        logger.warning(
            f"Thread {state['thread_id']} repeated {move_name} {max_same} times — forcing STUCK"
        )
        state["wait_reason"] = "repeated_moves"
        return "finalize_stuck"

    if decision.get("status") == "DONE":
        return "finalize_complete"

    # Summarize every 5 steps
    if step_number > 1 and step_number % 5 == 0:
        return "summarize"

    return "coordinator"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_thread_graph(
    coordinator: Coordinator,
    worker: Worker,
    llm: LLMClient,
    session_db: Any,
    queue: Queue,
    state_store: StateStore,
    trace_store: TraceStore,
    config: AppConfig,
) -> StateGraph:
    """
    Build the LangGraph StateGraph for a single thread's analysis lifecycle.

    Graph structure:
        coordinator → [worker | finalize_stuck]
        worker → [coordinator | finalize_complete | finalize_stuck | summarize]
        summarize → coordinator
        finalize_complete → END
        finalize_stuck → END

    Unhandled exceptions raised from any node propagate to
    ``ThreadRunner._on_graph_done`` which emits ``thread_waiting`` via the
    same shared helper the stuck path uses.
    """
    graph = StateGraph(ThreadState)

    # Register nodes
    graph.add_node("coordinator", make_coordinator_node(coordinator, trace_store, queue, state_store))
    graph.add_node("worker", make_worker_node(worker, trace_store, queue, session_db))
    graph.add_node("finalize_complete", make_finalize_complete_node(state_store, trace_store, queue))
    graph.add_node("finalize_stuck", make_finalize_stuck_node(state_store, trace_store, queue))
    graph.add_node("summarize", make_summarize_node(state_store, trace_store, llm, config))

    # Entry point
    graph.set_entry_point("coordinator")

    # Edges
    graph.add_conditional_edges("coordinator", route_after_coordinator, {
        "worker": "worker",
        "finalize_stuck": "finalize_stuck",
    })
    graph.add_conditional_edges("worker", route_after_worker, {
        "coordinator": "coordinator",
        "finalize_complete": "finalize_complete",
        "finalize_stuck": "finalize_stuck",
        "summarize": "summarize",
    })
    graph.add_edge("summarize", "coordinator")
    graph.add_edge("finalize_complete", END)
    graph.add_edge("finalize_stuck", END)

    return graph

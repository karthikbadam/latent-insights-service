"""
Recorder — dual-write helper that records step events and emits SSE in one call.

Every public method writes to ``InvestigationStore`` (the persistent record)
**and** emits the matching ``StreamEvent`` through ``Queue`` (the SSE stream).
This guarantees the two representations cannot drift.

Event dicts stored on steps use the flat ``api.schemas.StepEvent`` shape, so
the on-disk snapshot matches the API response byte-for-byte.
"""

import time

from latent_insights.core.queue import Queue
from latent_insights.core.store import InvestigationStore, Step
from latent_insights.models import StreamEvent


class Recorder:
    """Dual-write: step events + SSE emission."""

    def __init__(
        self,
        store: InvestigationStore,
        queue: Queue,
        session_id: str,
        thread_id: str,
    ):
        self.store = store
        self.queue = queue
        self.session_id = session_id
        self.thread_id = thread_id

    # --- Thread lifecycle ---

    def thread_start(
        self,
        seed_question: str,
        motivation: str,
        entry_point: str,
        timestamp: float | None = None,
    ):
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="thread_start",
            message=seed_question,
            data={
                "seed_question": seed_question,
                "motivation": motivation,
                "entry_point": entry_point,
                "step_number": 0,
            },
            timestamp=timestamp or time.time(),
        ))

    def thread_resumed(self, from_step: int, human_messages: list):
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="thread_resumed",
            message="",
            data={
                "from_step": from_step,
                "human_messages": human_messages,
            },
        ))

    def thread_complete(self, summary: str, total_ms: int, step_count: int):
        from latent_insights.models import ThreadStatus

        self.store.update_thread_status(
            self.thread_id, ThreadStatus.COMPLETE, summary=summary,
        )
        self.store.save_session(self.session_id)

        total_seconds = round(total_ms / 1000, 2)
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="thread_complete",
            message=summary,
            data={
                "summary": summary,
                "total_seconds": total_seconds,
                "total_ms": total_ms,
                "step_count": step_count,
                "is_terminal": True,
                "step_number": step_count + 1,
            },
        ))

    def thread_waiting(
        self,
        reason: str,
        question: str,
        context: str | None = None,
        error: str | None = None,
        step_count: int = 0,
        *,
        span_status: str = "stuck",
    ):
        from latent_insights.models import ThreadStatus

        # End any open in-progress step on the thread so its row persists.
        steps = self.store.get_steps(self.thread_id)
        if steps and steps[-1].end_time is None:
            step = steps[-1]
            if not step.result:
                step.result = f"{reason}: {context or error or ''}"
            self.store.end_step(step, status=span_status)

        self.store.update_thread_status(
            self.thread_id, ThreadStatus.WAITING, error=error or reason,
        )
        self.store.save_session(self.session_id)

        thread = self.store.get_thread(self.thread_id)
        running_summary = thread.running_summary if thread else None

        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="thread_waiting",
            message=question,
            data={
                "reason": reason,
                "question": question,
                "context": context,
                "error": error,
                "running_summary": running_summary,
                "is_terminal": True,
                "step_number": step_count + 1,
            },
        ))

    # --- Step lifecycle (SSE only — step rows are managed directly on
    # the store by the runner) ---

    def step_start(self, step_number: int, move: str, instruction: str):
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="step_start",
            message=instruction,
            data={
                "move": move,
                "step_number": step_number,
                "instruction": instruction,
                "provisional": False,
            },
        ))

    def step_complete(
        self,
        step_number: int,
        move: str,
        instruction: str,
        result: str,
        duration_ms: int,
    ):
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="step_complete",
            message=result,
            data={
                "step_number": step_number,
                "move": move,
                "instruction": instruction,
                "result": result,
                "duration_ms": duration_ms,
            },
        ))

    # --- Events within a step (written as flat StepEvent dicts) ----------

    def llm_call(
        self,
        step: Step,
        *,
        step_number: int,
        move: str,
        agent: str,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_ms: int = 0,
        response: str = "",
        has_tool_calls: bool = False,
    ):
        """Record an LLM call on the step and emit the SSE event."""
        self.store.add_event(step, {
            "type": "llm_call",
            "agent": agent,
            "model": model,
            "duration_ms": duration_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response": response,
        })
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="llm_call",
            message=f"{agent.capitalize()} {'executing SQL' if has_tool_calls else 'deciding'} ({duration_ms}ms)",
            data={
                "agent": agent,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_ms": duration_ms,
                "response": response,
                "step_number": step_number,
                "move": move,
                **({"has_tool_calls": has_tool_calls} if agent == "worker" else {}),
            },
        ))

    def tool_call(
        self,
        step: Step,
        *,
        step_number: int,
        move: str,
        sql: str,
        tool_result: str,
        duration_ms: int = 0,
    ):
        """Record a SQL tool call on the step and emit the SSE event."""
        self.store.add_event(step, {
            "type": "tool_call",
            "agent": "worker",
            "sql": sql,
            "tool_result": tool_result,
            "duration_ms": duration_ms,
        })
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="tool_call",
            message=sql,
            data={
                "agent": "worker",
                "sql": sql,
                "tool_result": tool_result,
                "duration_ms": duration_ms,
                "step_number": step_number,
                "move": move,
            },
        ))

    def human_message(
        self,
        step: Step,
        *,
        step_number: int,
        move: str = "",
        content: str,
        target: str = "thread",
        timestamp: float | None = None,
    ):
        """Record a human message on the step and emit the SSE event.

        The SSE event carries ``step_number`` (the step that will
        consume the message) plus ``move`` when known, so the UI can
        slot it into the timeline the same way as sibling ``llm_call`` /
        ``tool_call`` events. ``move`` is empty when the message is
        recorded at the top of a step before the coordinator has picked
        a move — the UI renders those as "pre-step" human input.
        """
        self.store.add_event(
            step,
            {
                "type": "human_message",
                "content": content,
                "target": target,
            },
            timestamp=timestamp,
        )
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="human_message",
            message=content,
            data={
                "content": content,
                "target": target,
                "step_number": step_number,
                "move": move,
            },
            timestamp=timestamp or time.time(),
        ))

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

    def thread_resumed(self, from_step: int):
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="thread_resumed",
            message="",
            data={
                "from_step": from_step,
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
                # ``summary`` is the REST-snapshot name (ThreadResponse.summary).
                # ``result`` mirrors step_complete's payload so a UI that
                # renders every event's "content" via a common ``data.result``
                # field picks up the thread's final finding the same way it
                # picks up each step's result.
                "summary": summary,
                "result": summary,
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
        """Transition a thread into the terminal WAITING state.

        Commits a dedicated ``WAITING_FOR_HUMAN`` step whose ``result`` is
        the question the coordinator wants answered (and ``instruction``
        carries any accompanying context). The ``thread_waiting`` SSE
        event is kept as a thin terminal marker — its payload no longer
        duplicates the question/context/error, which now live on the
        waiting step and are reachable via the REST snapshot or the
        preceding ``step_complete`` SSE.
        """
        from latent_insights.models import MoveType, ThreadStatus

        # End any in-flight analytical step so its row persists with the
        # failure context. The caller passes ``span_status`` — "stuck" for
        # coordinator-declared stuck / repeated-moves, "error" for raised
        # exceptions.
        steps = self.store.get_steps(self.thread_id)
        if steps and steps[-1].end_time is None:
            step = steps[-1]
            if not step.result:
                step.result = f"{reason}: {context or error or ''}"
            self.store.end_step(step, status=span_status)

        # Commit the WAITING_FOR_HUMAN step. This is a real row in the
        # thread's timeline — UIs render it like any other step.
        waiting_step = self.store.start_step(self.thread_id)
        waiting_step.move = MoveType.WAITING_FOR_HUMAN.value
        waiting_step.instruction = context or ""
        waiting_step.result = question
        self.store.end_step(waiting_step, status=span_status)

        self.store.update_thread_status(
            self.thread_id, ThreadStatus.WAITING, error=error or reason,
        )
        self.store.save_session(self.session_id)

        thread = self.store.get_thread(self.thread_id)
        running_summary = thread.running_summary if thread else None

        # Emit step_start + step_complete for the waiting step so live
        # SSE consumers see it land in the timeline before the terminal
        # thread_waiting marker.
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="step_start",
            message=question,
            data={
                "move": MoveType.WAITING_FOR_HUMAN.value,
                "step_number": waiting_step.step_number,
                "instruction": context or "",
                "provisional": False,
                "assessment": "",
                "rationale": "",
                "status": "",
            },
        ))
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="step_complete",
            message=question,
            data={
                "step_number": waiting_step.step_number,
                "move": MoveType.WAITING_FOR_HUMAN.value,
                "instruction": context or "",
                "result": question,
                "duration_ms": waiting_step.duration_ms or 0,
            },
        ))

        # Thin terminal marker. The question/context/error are on the
        # WAITING_FOR_HUMAN step; no need to duplicate them here.
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="thread_waiting",
            message=question,
            data={
                "reason": reason,
                "running_summary": running_summary,
                "is_terminal": True,
                "step_number": waiting_step.step_number,
            },
        ))

    # --- Step lifecycle (SSE only — step rows are managed directly on
    # the store by the runner) ---

    def step_start(
        self,
        step_number: int,
        move: str,
        instruction: str,
        *,
        assessment: str = "",
        rationale: str = "",
        status: str = "",
    ):
        """Emit ``step_start`` with the coordinator's full decision.

        ``assessment`` (what the coordinator thinks of the investigation
        state), ``rationale`` (why this move now), and ``status``
        (``CONTINUE`` / ``STUCK`` / ``DONE``) are the structured
        decision fields the UI needs to render "Coordinator decided X
        because Y" without parsing the raw LLM JSON out of the sibling
        ``llm_call`` event's ``response`` text.
        """
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
                "assessment": assessment,
                "rationale": rationale,
                "status": status,
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

    # --- Mixed-initiative: human input as its own step --------------------

    def human_input_step(
        self,
        content: str,
        target: str = "thread",
        timestamp: float | None = None,
    ) -> Step:
        """Commit a ``HUMAN_INPUT`` step carrying the human's guidance.

        Every path that previously recorded a ``human_message`` event on
        a step now calls this instead — human input is a step in its
        own right, visible in the timeline alongside analytical moves.
        The ``target`` ("thread" | "session") is preserved on the step's
        instruction so the UI can badge session-broadcast messages.
        """
        from latent_insights.models import MoveType

        step = self.store.start_step(self.thread_id)
        step.move = MoveType.HUMAN_INPUT.value
        step.instruction = target  # "thread" or "session"
        step.result = content
        if timestamp is not None:
            step.start_time = timestamp
        self.store.end_step(step, status="ok")
        self.store.save_session(self.session_id)

        # Emit step_start + step_complete so a live SSE consumer sees
        # the human step the same way it sees any other step.
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="step_start",
            message=content,
            data={
                "move": MoveType.HUMAN_INPUT.value,
                "step_number": step.step_number,
                "instruction": target,
                "provisional": False,
                "assessment": "",
                "rationale": "",
                "status": "",
            },
            timestamp=timestamp or time.time(),
        ))
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
            event_type="step_complete",
            message=content,
            data={
                "step_number": step.step_number,
                "move": MoveType.HUMAN_INPUT.value,
                "instruction": target,
                "result": content,
                "duration_ms": 0,
            },
            timestamp=timestamp or time.time(),
        ))
        return step

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

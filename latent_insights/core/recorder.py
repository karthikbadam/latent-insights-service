"""
Recorder — dual-write helper that records step events and emits SSE in one call.

Every public method writes to ``InvestigationStore`` (the persistent record)
**and** emits the matching ``StreamEvent`` through ``Queue`` (the SSE stream).
This guarantees the two representations cannot drift.

Each emission carries a complete ``FeedEntry`` as its payload — the SSE
``data:`` line and the snapshot-derived feed entries share one render-ready
shape. ``feed_index`` is pulled from a per-session monotonic counter on
``Queue`` so events have a strict total order across the session.

A Recorder can be thread-scoped (``thread_id`` set) or session-scoped
(``thread_id=""``) — the session-scoped form is used by
``SessionFlow``/``patterns`` to emit ``schema_summary_ready`` /
``session_ready`` / ``scout_done`` / ``synthesis_start`` rows.
"""

import time

from latent_insights.api.feed import (
    FeedEntry,
    format_schema_summary,
    parse_llm_response,
)
from latent_insights.core.queue import Queue
from latent_insights.core.store import InvestigationStore, Step
from latent_insights.models import StreamEvent


class Recorder:
    """Dual-write: step events + SSE emission as render-ready FeedEntries."""

    def __init__(
        self,
        store: InvestigationStore,
        queue: Queue,
        session_id: str,
        thread_id: str = "",
    ):
        self.store = store
        self.queue = queue
        self.session_id = session_id
        self.thread_id = thread_id

    # --- Shared emission helper ---

    def _emit(
        self,
        *,
        event_type: str,
        entry_id: str,
        message: str,
        thread_id: str | None = None,
        timestamp: float | None = None,
        **fields,
    ) -> FeedEntry:
        tid = thread_id if thread_id is not None else self.thread_id
        ts = timestamp if timestamp is not None else time.time()
        entry = FeedEntry(
            id=entry_id,
            feed_index=self.queue.next_feed_index(self.session_id),
            event_type=event_type,
            thread_id=tid,
            timestamp=ts,
            message=message,
            **fields,
        )
        self.queue.emit(StreamEvent(
            session_id=self.session_id,
            thread_id=tid,
            event_type=event_type,
            message=message,
            data=entry.model_dump(exclude_none=True),
            timestamp=ts,
        ))
        return entry

    # --- Session-level events ---

    def schema_summary_ready(self, schema_summary: str, dataset_path: str | None = None):
        self._emit(
            event_type="schema_summary_ready",
            entry_id=f"schema:{self.session_id}",
            thread_id="",
            message="Dataset profiled.",
            schema_summary=schema_summary,
            schema_summary_markdown=format_schema_summary(schema_summary),
            dataset_path=dataset_path,
        )

    def session_ready(self, *, question_source: str, dataset_path: str | None = None):
        self._emit(
            event_type="session_ready",
            entry_id=f"session:{self.session_id}:ready",
            thread_id="",
            message="Session profiled. Waiting for human questions.",
            question_source=question_source,
            dataset_path=dataset_path,
        )

    def scout_done(
        self,
        *,
        scout_questions: list[dict],
        message: str | None = None,
    ):
        self._emit(
            event_type="scout_done",
            entry_id=f"scout:{self.session_id}",
            thread_id="",
            message=message or f"Scout found {len(scout_questions)} questions",
            scout_questions=list(scout_questions),
            question_count=len(scout_questions),
        )

    def synthesis_start(
        self,
        *,
        synthesis_thread_id: str,
        source_threads: list[str],
        finding_count: int,
    ):
        self._emit(
            event_type="synthesis_start",
            entry_id=f"synthesis:{self.session_id}:{synthesis_thread_id}",
            thread_id=synthesis_thread_id,
            message=f"Synthesizing {finding_count} thread findings",
            source_threads=list(source_threads),
            synthesis_thread=synthesis_thread_id,
        )

    # --- Thread lifecycle ---

    def thread_start(
        self,
        seed_question: str,
        motivation: str,
        entry_point: str,
        timestamp: float | None = None,
    ):
        self._emit(
            event_type="thread_start",
            entry_id=f"thread:{self.thread_id}:start",
            timestamp=timestamp,
            message=seed_question,
            full_message=seed_question,
            seed_question=seed_question,
            motivation=motivation,
            entry_point=entry_point,
            step_number=0,
        )

    def thread_resumed(self, from_step: int):
        self._emit(
            event_type="thread_resumed",
            entry_id=f"thread:{self.thread_id}:resumed:{from_step}",
            message="",
            from_step=from_step,
        )

    def thread_complete(self, summary: str, total_ms: int, step_count: int):
        from latent_insights.models import ThreadStatus

        self.store.update_thread_status(
            self.thread_id, ThreadStatus.COMPLETE, summary=summary,
        )
        self.store.save_session(self.session_id)

        total_seconds = round(total_ms / 1000, 2)
        self._emit(
            event_type="thread_complete",
            entry_id=f"thread:{self.thread_id}:complete",
            message=summary,
            full_message=summary,
            summary=summary,
            result=summary,
            total_seconds=total_seconds,
            total_ms=total_ms,
            step_count=step_count,
            is_terminal=True,
            step_number=step_count + 1,
            thread_status=ThreadStatus.COMPLETE.value,
        )

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
        """Terminal WAITING state.

        Closes any in-flight analytical step, commits a
        ``WAITING_FOR_HUMAN`` step (question on ``result``, context on
        ``instruction``) to the store for history, then emits a single
        ``thread_waiting`` FeedEntry. The WAITING_FOR_HUMAN step itself
        does not produce step_start/step_complete events — the
        ``thread_waiting`` row covers it.
        """
        from latent_insights.models import MoveType, ThreadStatus

        # End any in-flight analytical step so its row persists.
        steps = self.store.get_steps(self.thread_id)
        if steps and steps[-1].end_time is None:
            step = steps[-1]
            if not step.result:
                step.result = f"{reason}: {context or error or ''}"
            self.store.end_step(step, status=span_status)

        # Commit the WAITING_FOR_HUMAN step for history (no SSE for it).
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

        self._emit(
            event_type="thread_waiting",
            entry_id=f"thread:{self.thread_id}:waiting",
            message=question,
            full_message=question,
            reason=reason,
            running_summary=running_summary,
            is_terminal=True,
            step_number=waiting_step.step_number,
            thread_status=ThreadStatus.WAITING.value,
        )

    # --- Step lifecycle ---

    def step_start(
        self,
        step_number: int,
        move: str,
        instruction: str,
        *,
        assessment: str = "",
        rationale: str = "",
        status: str = "",
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_ms: int | None = None,
    ):
        """Emit ``step_start`` with the coordinator's full decision.

        Carries the coordinator's LLM call metrics directly so a separate
        ``llm_call`` row for the coordinator is unnecessary — the feed
        shows the structured assessment/rationale/instruction as the
        first row of the step, with SQL ``tool_call`` rows immediately
        after.
        """
        # ``message`` is the collapsed-row preview. Only set it to the
        # assessment — leaving it empty when no assessment is present
        # keeps the row a labels-only line. The full instruction is
        # served via the dedicated ``instruction`` field for the
        # expanded view; ``full_message`` is intentionally unset so
        # the frontend's generic preview-text fallback never lifts the
        # long instruction into the row header.
        self._emit(
            event_type="step_start",
            entry_id=f"step:{self.thread_id}:{step_number}:start",
            message=assessment,
            move=move,
            agent="coordinator",
            step_number=step_number,
            instruction=instruction,
            assessment=assessment,
            rationale=rationale,
            status=status,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )

    def step_complete(
        self,
        step_number: int,
        move: str,
        instruction: str,
        result: str,
        duration_ms: int,
    ):
        self._emit(
            event_type="step_complete",
            entry_id=f"step:{self.thread_id}:{step_number}:complete",
            message=result,
            full_message=result,
            step_number=step_number,
            move=move,
            instruction=instruction,
            result=result,
            duration_ms=duration_ms,
        )

    # --- Mixed-initiative: human input as its own step ---

    def human_input_step(
        self,
        content: str,
        target: str = "thread",
        timestamp: float | None = None,
    ) -> Step:
        """Commit a ``HUMAN_INPUT`` step + emit a single ``human_message`` row.

        ``target`` goes on the step's ``instruction``; the message content
        goes on ``result``. Zero duration. A single ``human_message``
        FeedEntry is emitted (not separate step_start/step_complete) so
        the wire matches what ``session_to_feed`` produces from the saved
        snapshot.
        """
        from latent_insights.models import MoveType

        step = self.store.start_step(self.thread_id)
        step.move = MoveType.HUMAN_INPUT.value
        step.instruction = target
        step.result = content
        if timestamp is not None:
            step.start_time = timestamp
        self.store.end_step(step, status="ok")
        self.store.save_session(self.session_id)

        self._emit(
            event_type="human_message",
            entry_id=f"human:{self.thread_id}:{step.step_number}",
            timestamp=timestamp,
            message=content,
            full_message=content,
            content=content,
            target=target,
            step_number=step.step_number,
            move=MoveType.HUMAN_INPUT.value,
        )
        return step

    # --- Events within a step (written as flat StepEvent dicts) ---

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
        """Record an LLM call on the step and emit the SSE FeedEntry."""
        self.store.add_event(step, {
            "type": "llm_call",
            "agent": agent,
            "model": model,
            "duration_ms": duration_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response": response,
        })
        ev_idx = len(step.events) - 1
        response_text, response_tables = parse_llm_response(response)
        label = f"{agent.capitalize()} {'executing SQL' if has_tool_calls else 'deciding'} ({duration_ms}ms)"
        extras = {}
        if agent == "worker":
            extras["has_tool_calls"] = has_tool_calls
        self._emit(
            event_type="llm_call",
            entry_id=f"ev:{self.thread_id}:{step_number}:{ev_idx}",
            message=label,
            full_message=response_text or response or label,
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            response=response,
            response_text=response_text,
            response_tables=response_tables,
            step_number=step_number,
            move=move,
            **extras,
        )

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
        """Record a SQL tool call on the step and emit the SSE FeedEntry."""
        self.store.add_event(step, {
            "type": "tool_call",
            "agent": "worker",
            "sql": sql,
            "tool_result": tool_result,
            "duration_ms": duration_ms,
        })
        ev_idx = len(step.events) - 1
        self._emit(
            event_type="tool_call",
            entry_id=f"ev:{self.thread_id}:{step_number}:{ev_idx}",
            message=sql,
            full_message=sql,
            agent="worker",
            sql=sql,
            tool_result=tool_result,
            duration_ms=duration_ms,
            step_number=step_number,
            move=move,
        )

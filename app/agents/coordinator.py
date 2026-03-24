"""
Coordinator agent — the judge. Picks the next analytical move for a thread.

One coordinator per thread. Receives history, decides next action.
Works with any dataset — no domain-specific assumptions.
"""

import json
import logging
import time

from app.agents.base import Agent
from app.core.llm import LLMClient
from app.core.parsing import parse_coordinator_response
from app.core.queue import Queue
from app.models import CoordinatorDecision, CoordinatorStatus, MoveType, StreamEvent

logger = logging.getLogger(__name__)


class Coordinator(Agent):
    """Thread judge — evaluates worker results and decides the next analytical move."""

    SYSTEM_PROMPT = """\
You are a thread coordinator for a data analysis system. You guide a
single analytical thread from question to insight.

You do NOT run queries. You evaluate worker results and decide what to
investigate next. Think, judge, direct — don't compute.

## Your thread

Question: {seed_question}
Motivation: {motivation}
Suggested entry point: {entry_point}

## Dataset context

{schema_summary}

## Sensemaking moves

**SCOPE** — Define data slice. Narrow to relevant subset. May create view.
**FORAGE** — Exploratory analysis. Distributions, correlations, groups, outliers.
**FRAME** — Propose tentative insight/hypothesis. State as testable claim.
**INTERROGATE** — Stress-test the frame. Contradictions, subgroups, confounds.
**SYNTHESIZE** — Thread conclusion. Finding, confidence, limitations.

No fixed order. Let data guide you.

## Thread history

{thread_history}

## Your decision

Return JSON:

{{
  "assessment": "Current state of investigation (2-3 sentences)",
  "next_move": "SCOPE | FORAGE | FRAME | INTERROGATE | SYNTHESIZE",
  "rationale": "Why this move now (1-2 sentences)",
  "worker_instruction": "Specific instruction for worker. Columns, filters, expected output.",
  "status": "CONTINUE | STUCK | DONE"
}}

When STUCK, replace worker_instruction with question_for_human and context.
When DONE, worker_instruction should be a SYNTHESIZE producing the final summary.

### Quality standards
- Never report stats without context. Push toward "so what?"
- Make hypotheses falsifiable.
- If something surprises you, say so.
- Do not ask the worker to run ML models, regressions, or clustering. DuckDB is a SQL engine — instruct the worker to use aggregates, grouping, correlations, and arithmetic.
"""

    def __init__(
        self,
        llm: LLMClient,
        model: str,
        temperature: float = 0.3,
        queue: Queue | None = None,
        session_id: str = "",
        thread_id: str = "",
    ):
        super().__init__(llm, model)
        self.temperature = temperature
        self.queue = queue
        self.session_id = session_id
        self.thread_id = thread_id

    @property
    def role(self) -> str:
        return "coordinator"

    def call(
        self,
        seed_question: str,
        motivation: str,
        entry_point: str,
        schema_summary: str,
        thread_history: str,
    ) -> tuple[CoordinatorDecision, dict]:
        """Run one coordinator step and return (decision, llm_call_log)."""
        prompt = self.SYSTEM_PROMPT.format(
            seed_question=seed_question,
            motivation=motivation,
            entry_point=entry_point,
            schema_summary=schema_summary,
            thread_history=thread_history,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Based on the thread history, decide your next move."},
        ]

        t0 = time.monotonic()
        response = self.llm.call(
            model=self.model,
            messages=messages,
            role=self.role,
            temperature=self.temperature,
        )
        call_ms = round((time.monotonic() - t0) * 1000)

        if self.queue:
            self.queue.emit(StreamEvent(
                session_id=self.session_id, thread_id=self.thread_id,
                event_type="llm_call",
                message=f"Coordinator deciding next move ({call_ms}ms)",
                data={"role": self.role, "model": self.model,
                      "input_tokens": response.input_tokens,
                      "output_tokens": response.output_tokens,
                      "duration_ms": call_ms},
            ))

        if not response.content or not response.content.strip():
            logger.warning("Coordinator returned empty response, retrying once")
            response = self.llm.call(
                model=self.model,
                messages=messages,
                role=self.role,
                temperature=self.temperature,
            )
            if not response.content or not response.content.strip():
                raise ValueError("Coordinator returned empty response twice")

        # Parse with retry on JSON errors
        try:
            decision = parse_coordinator_response(response.content)
        except (ValueError, json.JSONDecodeError):
            logger.warning("Coordinator returned non-JSON response, retrying with nudge")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": "Your response must be valid JSON matching the format specified. Please reformat your answer as JSON.",
            })
            response = self.llm.call(
                model=self.model,
                messages=messages,
                role=self.role,
                temperature=self.temperature,
            )
            decision = parse_coordinator_response(response.content)

        # Validate consistency
        if decision.status == CoordinatorStatus.STUCK:
            decision.next_move = MoveType.STUCK

        if decision.status == CoordinatorStatus.DONE and decision.next_move != MoveType.SYNTHESIZE:
            logger.warning("Coordinator returned DONE without SYNTHESIZE — correcting")
            decision.next_move = MoveType.SYNTHESIZE

        if decision.status == CoordinatorStatus.STUCK and not decision.question_for_human:
            logger.warning("Coordinator returned STUCK without question — adding default")
            decision.question_for_human = "I need guidance on how to proceed with this analysis."

        llm_log = {
            "agent": self.role,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "response": response.content if response.content else "",
        }

        return decision, llm_log

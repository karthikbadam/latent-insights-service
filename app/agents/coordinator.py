"""
Coordinator agent — the judge. Picks the next analytical move for a thread.

One coordinator per thread. Receives history, decides next action.
Works with any dataset — no domain-specific assumptions.
"""

import logging
import time

from app.core.llm import LLMClient
from app.core.parsing import parse_coordinator_response
from app.core.queue import Queue
from app.models import CoordinatorDecision, CoordinatorStatus, MoveType, StreamEvent

logger = logging.getLogger(__name__)

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


def run_coordinator(
    llm: LLMClient,
    model: str,
    seed_question: str,
    motivation: str,
    entry_point: str,
    schema_summary: str,
    thread_history: str,
    temperature: float = 0.3,
    queue: Queue | None = None,
    session_id: str = "",
    thread_id: str = "",
) -> tuple[CoordinatorDecision, dict]:
    """Run one coordinator step and return (decision, llm_call_log)."""
    prompt = SYSTEM_PROMPT.format(
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
    response = llm.call(
        model=model,
        messages=messages,
        role="coordinator",
        temperature=temperature,
    )
    call_ms = round((time.monotonic() - t0) * 1000)

    if queue:
        queue.emit(StreamEvent(
            session_id=session_id, thread_id=thread_id,
            event_type="llm_call",
            message=f"Coordinator: {seed_question[:120]} ({call_ms}ms)",
            data={"role": "coordinator", "model": model,
                  "input_tokens": response.input_tokens,
                  "output_tokens": response.output_tokens,
                  "duration_ms": call_ms},
        ))

    if not response.content or not response.content.strip():
        logger.warning("Coordinator returned empty response, retrying once")
        response = llm.call(
            model=model,
            messages=messages,
            role="coordinator",
            temperature=temperature,
        )
        if not response.content or not response.content.strip():
            raise ValueError("Coordinator returned empty response twice")

    decision = parse_coordinator_response(response.content)

    # Validate consistency
    if decision.status == CoordinatorStatus.DONE and decision.next_move != MoveType.SYNTHESIZE:
        logger.warning("Coordinator returned DONE without SYNTHESIZE — correcting")
        decision.next_move = MoveType.SYNTHESIZE

    if decision.status == CoordinatorStatus.STUCK and not decision.question_for_human:
        logger.warning("Coordinator returned STUCK without question — adding default")
        decision.question_for_human = "I need guidance on how to proceed with this analysis."

    llm_log = {
        "agent": "coordinator",
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "response": response.content[:500] if response.content else "",
    }

    return decision, llm_log

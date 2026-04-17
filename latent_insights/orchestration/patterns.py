"""
Pre-built flow patterns — factory functions for different thread topologies.

- fan_out_with_synthesis: parallel threads with post-hoc synthesis
- PATTERN_REGISTRY: API enumeration of available patterns
"""

import logging
from threading import Lock
from typing import Any

from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMClient
from latent_insights.core.queue import Queue
from latent_insights.core.store import InvestigationStore
from latent_insights.models import StreamEvent

logger = logging.getLogger(__name__)


def fan_out_with_synthesis(
    questions: list[str],
    session_id: str,
    config: AppConfig,
    llm: LLMClient,
    db: Any,
    queue: Queue,
    store: InvestigationStore,
    schema_summary: str,
):
    """
    Run N independent analysis threads, then synthesize their findings.

    The synthesis kicks off from the last-to-finish analysis runner's
    ``on_done`` callback — no pool worker is held blocked on a wait.
    Each analysis runner decrements a shared counter when it terminates
    (complete / waiting / error); whichever one takes the counter to
    zero is responsible for scheduling the synthesis.

    Returns list of thread IDs (analysis threads only; the synthesis
    thread is created later inside the callback).
    """
    from latent_insights.orchestration.runner import ThreadRunner

    thread_ids: list[str] = []
    runners: list[ThreadRunner] = []

    # Shared counter + lock so the last-to-finish runner triggers the
    # synthesis exactly once. The counter lives in a list so the
    # callback can mutate it from the pool thread that fires it.
    remaining = [len(questions)]
    remaining_lock = Lock()
    synthesis_started = [False]

    def _on_analysis_done():
        with remaining_lock:
            remaining[0] -= 1
            is_last = remaining[0] == 0 and not synthesis_started[0]
            if is_last:
                synthesis_started[0] = True
        if is_last:
            _start_synthesis()

    def _start_synthesis():
        findings = []
        for tid in thread_ids:
            t = store.get_thread(tid)
            if t and t.summary:
                findings.append(f"**{t.seed_question}**\n{t.summary}")

        if not findings:
            logger.warning(
                f"Fan-out synthesis: no findings from {len(thread_ids)} threads"
            )
            return

        synthesis_question = (
            "Synthesize the following parallel analyses into a unified summary. "
            "Identify connections, contradictions, and overarching patterns.\n\n"
            + "\n\n---\n\n".join(findings)
        )
        synth_thread = store.create_thread(
            session_id, synthesis_question, "Fan-out synthesis", "",
        )
        synth_db = db.open_session_connection(session_id)
        synth_runner = ThreadRunner(
            config=config, llm=llm, session_db=synth_db, queue=queue,
            store=store, thread=synth_thread,
            schema_summary=schema_summary,
        )
        synth_runner.start()

        queue.emit(StreamEvent(
            session_id=session_id,
            thread_id=synth_thread.id,
            event_type="synthesis_start",
            message=f"Synthesizing {len(findings)} thread findings",
            data={
                "source_threads": list(thread_ids),
                "synthesis_thread": synth_thread.id,
            },
        ))

    # Spawn analysis threads with the on_done hook wired up.
    for q in questions:
        thread = store.create_thread(session_id, q, "", "")
        thread_db = db.open_session_connection(session_id)
        runner = ThreadRunner(
            config=config, llm=llm, session_db=thread_db, queue=queue,
            store=store, thread=thread,
            schema_summary=schema_summary,
            on_done=_on_analysis_done,
        )
        runner.start()
        runners.append(runner)
        thread_ids.append(thread.id)

    return thread_ids


# ---------------------------------------------------------------------------
# Pattern registry — for API enumeration
# ---------------------------------------------------------------------------

PATTERN_REGISTRY = {
    "coordinator_worker": {
        "name": "coordinator_worker",
        "description": "Standard coordinator-worker analysis cycle with move guards",
        "input_schema": {
            "question": {"type": "string", "required": True},
            "motivation": {"type": "string", "required": False, "default": ""},
            "max_steps": {"type": "integer", "required": False, "default": 50},
        },
    },
    "fan_out": {
        "name": "fan_out",
        "description": "Run N parallel analysis threads, then collect results",
        "input_schema": {
            "questions": {"type": "array", "items": "string", "required": True},
        },
    },
    "human_in_the_loop": {
        "name": "human_in_the_loop",
        "description": "Coordinator-worker with human approval before each step",
        "input_schema": {
            "question": {"type": "string", "required": True},
        },
    },
}

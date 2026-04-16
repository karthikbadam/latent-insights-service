"""
Pre-built flow patterns — factory functions for different thread topologies.

- fan_out_with_synthesis: parallel threads with post-hoc synthesis
- PATTERN_REGISTRY: API enumeration of available patterns
"""

import logging
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

    1. Spawns N coordinator-worker threads (one per question).
    2. Waits for all to complete.
    3. Creates a synthesis thread that combines all findings.

    Returns list of thread IDs (analysis threads + synthesis thread).
    """
    from latent_insights.orchestration.loop import ThreadLoop

    # Spawn analysis threads
    runners = []
    thread_ids = []
    for q in questions:
        thread = store.create_thread(session_id, q, "", "")
        thread_db = db.open_session_connection(session_id)
        loop = ThreadLoop(
            config=config, llm=llm, session_db=thread_db, queue=queue,
            store=store, thread=thread,
            schema_summary=schema_summary,
        )
        loop.start()
        runners.append(loop)
        thread_ids.append(thread.id)

    # Schedule synthesis after all threads complete
    def _wait_and_synthesize():
        for runner in runners:
            runner.done_event.wait(timeout=config.llm_timeout * 60)

        # Collect findings from completed threads
        findings = []
        for tid in thread_ids:
            t = store.get_thread(tid)
            if t and t.summary:
                findings.append(f"**{t.seed_question}**\n{t.summary}")

        if not findings:
            logger.warning(f"Fan-out synthesis: no findings from {len(thread_ids)} threads")
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
        synth_loop = ThreadLoop(
            config=config, llm=llm, session_db=synth_db, queue=queue,
            store=store, thread=synth_thread,
            schema_summary=schema_summary,
        )
        synth_loop.start()
        thread_ids.append(synth_thread.id)

        queue.emit(StreamEvent(
            session_id=session_id,
            thread_id=synth_thread.id,
            event_type="synthesis_start",
            message=f"Synthesizing {len(findings)} thread findings",
            data={
                "source_threads": thread_ids[:-1],
                "synthesis_thread": synth_thread.id,
            },
        ))

    queue.schedule(
        fn=_wait_and_synthesize,
        args=(),
        task_id=f"fanout-synth-{session_id[:8]}",
        session_id=session_id,
        description=f"Fan-out synthesis: {len(questions)} threads",
    )

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

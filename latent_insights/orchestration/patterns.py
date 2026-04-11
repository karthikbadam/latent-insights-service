"""
Pre-built flow patterns — factory functions that return compiled LangGraph graphs.

Each pattern wires agents into a specific agentic topology:
- coordinator_worker_cycle: the standard analysis loop
- fan_out_analysis: parallel threads with independent cycles
- sequential_chain: linear pipeline of steps
- human_in_the_loop_cycle: coordinator-worker with interrupt for human review
"""

import logging
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from latent_insights.agents.coordinator import Coordinator
from latent_insights.agents.worker import Worker
from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMClient
from latent_insights.core.queue import Queue
from latent_insights.core.state import StateStore
from latent_insights.core.tracing import TraceStore
from latent_insights.models import StreamEvent, ThreadStatus
from latent_insights.orchestration.graph import ThreadState, build_thread_graph

logger = logging.getLogger(__name__)


def coordinator_worker_cycle(
    coordinator: Coordinator,
    worker: Worker,
    llm: LLMClient,
    session_db: Any,
    queue: Queue,
    state_store: StateStore,
    trace_store: TraceStore,
    config: AppConfig,
):
    """
    Standard coordinator→worker analysis loop.

    This is the direct replacement for the original ThreadRunner state machine.
    Returns a compiled graph ready to invoke.
    """
    graph = build_thread_graph(
        coordinator=coordinator,
        worker=worker,
        llm=llm,
        session_db=session_db,
        queue=queue,
        state_store=state_store,
        trace_store=trace_store,
        config=config,
    )
    return graph.compile()


def human_in_the_loop_step(
    config: AppConfig,
    llm: LLMClient,
    session_db: Any,
    queue: Queue,
    state_store: StateStore,
    trace_store: TraceStore,
    thread: Any,
    schema_summary: str,
):
    """
    Run ONE coordinator→worker step, then pause for human approval.

    Unlike the standard cycle which loops until DONE/STUCK, this runs
    exactly one step and then sets the thread to WAITING. The human
    reviews the result and sends a message to continue (or redirect).

    Each call to start/resume runs one step then pauses again.
    """
    from latent_insights.orchestration.thread import ThreadRunner

    class StepAndPauseRunner(ThreadRunner):
        """ThreadRunner variant that pauses after each step."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Override: set max_repeated_moves to 1 so the graph runs
            # exactly one coordinator→worker cycle, then we intercept
            # in _on_graph_done to set WAITING instead of looping.

        def _run_graph(self):
            """Run the graph, then force WAITING status regardless of outcome."""
            self._graph.invoke(self._initial_state)

        def _on_graph_done(self, future):
            """After one full graph run, pause the thread for human review."""
            try:
                future.result()
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"HITL thread {self.thread.id} error: {error_msg}", exc_info=True)

            thread_obj = state_store.get_thread(self.thread.id)
            if thread_obj and thread_obj.status == ThreadStatus.COMPLETE:
                # Thread naturally completed (DONE) — emit as normal, don't override
                pass
            elif thread_obj and thread_obj.status == ThreadStatus.RUNNING:
                # Step finished but thread would loop — pause for human review
                state_store.update_thread_status(self.thread.id, ThreadStatus.WAITING)
                state_store.dump_session(self.thread.session_id)

                # Get the last step result for the human to review
                spans = trace_store.get_step_spans(self.thread.id)
                last_result = ""
                last_move = ""
                if spans:
                    last = spans[-1]
                    last_result = last.attributes.get("result", "")
                    last_move = last.attributes.get("move", "")

                queue.emit(StreamEvent(
                    session_id=self.thread.session_id,
                    thread_id=self.thread.id,
                    event_type="thread_waiting",
                    message=f"Step complete ({last_move}). Review and send a message to continue.",
                    data={
                        "pattern": "human_in_the_loop",
                        "last_move": last_move,
                        "last_result": last_result,
                        "step_number": len(spans),
                    },
                ))

            try:
                self.session_db.close()
            except Exception:
                pass
            self.done_event.set()

    # Build with max_repeated_moves=1 so the graph naturally exits after 1 cycle
    hitl_config = config.with_overrides({"max_repeated_moves": 1})

    runner = StepAndPauseRunner(
        config=hitl_config,
        llm=llm,
        session_db=session_db,
        queue=queue,
        state=state_store,
        trace_store=trace_store,
        thread=thread,
        schema_summary=schema_summary,
    )
    return runner


def sequential_chain(
    steps: list[tuple[str, Callable]],
):
    """
    Linear pipeline: step1 → step2 → ... → END.

    Each step is a (name, function) pair. Functions receive and return dicts.

    Example:
        chain = sequential_chain([
            ("profile", lambda s: {"schema": profiler.call(db, table)}),
            ("scout", lambda s: {"questions": scout.call(s["schema"], ...)}),
        ])
        result = chain.invoke({})
    """
    graph = StateGraph(dict)
    names = [name for name, _ in steps]

    for name, fn in steps:
        graph.add_node(name, fn)

    graph.set_entry_point(names[0])

    for i in range(len(names) - 1):
        graph.add_edge(names[i], names[i + 1])

    graph.add_edge(names[-1], END)
    return graph.compile()


def fan_out_with_synthesis(
    questions: list[str],
    session_id: str,
    config: AppConfig,
    llm: LLMClient,
    db: Any,
    queue: Queue,
    state_store: StateStore,
    trace_store: TraceStore,
    schema_summary: str,
):
    """
    Run N independent analysis threads, then synthesize their findings.

    1. Spawns N coordinator-worker threads (one per question).
    2. Waits for all to complete.
    3. Creates a synthesis thread that combines all findings.

    Returns list of thread IDs (analysis threads + synthesis thread).
    """
    from latent_insights.orchestration.thread import ThreadRunner

    # Spawn analysis threads
    runners = []
    thread_ids = []
    for q in questions:
        thread = state_store.create_thread(session_id, q, "", "")
        thread_db = db.open_session_connection(session_id)
        runner = ThreadRunner(
            config=config, llm=llm, session_db=thread_db, queue=queue,
            state=state_store, trace_store=trace_store, thread=thread,
            schema_summary=schema_summary,
        )
        runner.start()
        runners.append(runner)
        thread_ids.append(thread.id)

    # Schedule synthesis after all threads complete
    def _wait_and_synthesize():
        for runner in runners:
            runner.done_event.wait(timeout=config.llm_timeout * 60)

        # Collect findings from completed threads
        findings = []
        for tid in thread_ids:
            t = state_store.get_thread(tid)
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
        synth_thread = state_store.create_thread(
            session_id, synthesis_question, "Fan-out synthesis", "",
        )
        synth_db = db.open_session_connection(session_id)
        synth_runner = ThreadRunner(
            config=config, llm=llm, session_db=synth_db, queue=queue,
            state=state_store, trace_store=trace_store, thread=synth_thread,
            schema_summary=schema_summary,
        )
        synth_runner.start()
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
    "sequential_chain": {
        "name": "sequential_chain",
        "description": "Linear pipeline of analysis steps",
        "input_schema": {
            "steps": {"type": "array", "items": {"instruction": "string"}, "required": True},
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

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


def human_in_the_loop_cycle(
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
    Coordinator→worker loop that pauses before each coordinator step
    for human approval. Uses LangGraph's interrupt_before mechanism.
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
    return graph.compile(interrupt_before=["coordinator"])


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


def fan_out_analysis(
    questions: list[dict],
    create_runner_fn: Callable,
):
    """
    Run N independent analysis threads in parallel.

    This doesn't create a single graph — it creates N independent graphs
    (one per question) and returns them for the caller to invoke via Queue.

    Args:
        questions: List of dicts with 'question', 'motivation', 'entry_point'.
        create_runner_fn: Factory that creates a ThreadRunner for a question.

    Returns:
        List of ThreadRunner instances ready to start.
    """
    runners = [create_runner_fn(q) for q in questions]
    return runners


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

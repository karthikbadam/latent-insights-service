"""Thread state machine — coordinator-worker cycle via LangGraph.

The ThreadRunner delegates to a LangGraph StateGraph that encodes the
coordinator→worker loop, move guards, stuck detection, and finalization
as declarative nodes and conditional edges. See orchestration/graph.py
for the graph definition.

Public API (start/resume) is unchanged from the original futures-based
implementation, so SessionFlow and API routes work without modification.
"""

import logging
import time
from threading import Event
from typing import Any

from latent_insights.agents.coordinator import Coordinator
from latent_insights.agents.worker import Worker
from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMClient
from latent_insights.core.queue import Queue
from latent_insights.core.state import StateStore
from latent_insights.core.tracing import TraceStore
from latent_insights.models import Thread, ThreadStatus
from latent_insights.orchestration.graph import ThreadState, build_thread_graph

logger = logging.getLogger(__name__)


class ThreadRunner:
    """Drives one analytical thread through its coordinator-worker lifecycle.

    Internally uses a LangGraph StateGraph. The graph handles:
    - Coordinator LLM calls and move selection
    - Worker SQL tool-use loop
    - Move repetition guards
    - Early stuck override
    - History summarization
    - Finalization (complete / stuck / error)

    All event emission and state persistence happen inside graph nodes.
    """

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        session_db: Any,
        queue: Queue,
        state: StateStore,
        trace_store: TraceStore,
        thread: Thread,
        schema_summary: str,
        human_messages: list[str] | None = None,
    ):
        self.config = config
        self.queue = queue
        self.state = state
        self.trace_store = trace_store
        self.thread = thread
        self.session_db = session_db
        self.human_messages = human_messages or []

        # Completion signaling (preserved for callers that wait on it)
        self.done_event: Event = Event()

        # Build agents
        self.coordinator = Coordinator(
            llm=llm,
            model=config.models.coordinator,
            temperature=config.temperatures.coordinator,
            queue=queue,
            session_id=thread.session_id,
            thread_id=thread.id,
        )
        self.worker = Worker(
            llm=llm,
            model=config.models.worker,
            fallback_model=config.models.worker_fallback,
            schema_summary=schema_summary,
            session_db=session_db,
            config=config,
            queue=queue,
            session_id=thread.session_id,
            thread_id=thread.id,
        )

        # Build the LangGraph graph
        graph_def = build_thread_graph(
            coordinator=self.coordinator,
            worker=self.worker,
            llm=llm,
            session_db=session_db,
            queue=queue,
            state_store=state,
            trace_store=trace_store,
            config=config,
        )
        self._graph = graph_def.compile()

        # Initial state for the graph
        self._initial_state: ThreadState = {
            "session_id": thread.session_id,
            "thread_id": thread.id,
            "seed_question": thread.seed_question,
            "motivation": thread.motivation,
            "entry_point": thread.entry_point,
            "schema_summary": schema_summary,
            "step_number": 0,
            "move_history": [],
            "human_messages": self.human_messages,
            "decision": None,
            "worker_result": None,
            "coordinator_ms": 0,
            "worker_ms": 0,
            "thread_views": "(none)",
            "status": "running",
            "error": None,
            "error_count": 0,
            "thread_start": time.monotonic(),
            "max_repeated_moves": config.max_repeated_moves,
        }

    @property
    def tid(self) -> str:
        return self.thread.id[:8]

    # --- Public API (unchanged from original) ---

    def start(self):
        """Kick off the thread state machine. Non-blocking — returns immediately."""
        future = self.queue.schedule(
            fn=self._run_graph,
            args=(),
            task_id=f"thread-{self.tid}",
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            description=f"Thread: {self.thread.seed_question[:60]}",
        )
        future.add_done_callback(lambda f: self._on_graph_done(f))

    def resume(self, human_messages: list[str] | None = None):
        """Resume a stuck thread. Human messages guide the next step."""
        if human_messages:
            self.human_messages = human_messages
            self._initial_state["human_messages"] = human_messages

        if not self.trace_store.get_spans(self.thread.id):
            self.trace_store.load_trace(self.thread.id, self.thread.session_id)

        # Reset step counter from existing trace spans
        existing_steps = len(self.trace_store.get_step_spans(self.thread.id))
        self._initial_state["step_number"] = existing_steps
        self._initial_state["thread_start"] = time.monotonic()

        self.state.update_thread_status(self.thread.id, ThreadStatus.RUNNING)
        self.start()

    # --- Internal ---

    def _run_graph(self):
        """Execute the LangGraph graph synchronously. Runs on a pool thread."""
        self._graph.invoke(self._initial_state)

    def _on_graph_done(self, future):
        """Handle graph completion or failure."""
        try:
            future.result()
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"Thread {self.thread.id} graph error: {error_msg}", exc_info=True)

            # Finalize as error if the graph itself raised
            try:
                self.state.update_thread_status(self.thread.id, ThreadStatus.WAITING)
                self.state.dump_session(self.thread.session_id)
                from latent_insights.models import StreamEvent
                self.queue.emit(StreamEvent(
                    session_id=self.thread.session_id,
                    thread_id=self.thread.id,
                    event_type="thread_waiting",
                    message=f"Thread encountered an error: {error_msg}",
                    data={"question": f"Error: {error_msg}", "context": error_msg},
                ))
            except Exception:
                logger.error(f"Thread {self.thread.id} failed to finalize after graph error", exc_info=True)
        finally:
            try:
                self.session_db.close()
            except Exception:
                pass
            self.done_event.set()

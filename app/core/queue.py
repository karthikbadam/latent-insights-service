"""
Queue — centralized task scheduling, event dispatch, and concurrency.

All coordination goes through here. Single place to reason about
what's running, what's waiting, and what events are flowing.
"""

import logging
import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from app.models import StreamEvent

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Metadata about a running task."""

    future: Future
    session_id: str
    thread_id: str | None = None
    description: str = ""
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        return round(time.monotonic() - self.started_at, 2)


class Queue:
    """
    Central coordinator for thread-pool tasks and events.

    Responsibilities:
    - Schedule and track tasks (thread runs, scout, profiler)
    - Dispatch events to SSE listeners
    """

    def __init__(self, max_workers: int = 16):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TaskInfo] = {}
        self._event_queues: dict[str, list[queue.Queue]] = {}

    # --- Task management ---

    def schedule(
        self,
        fn,
        args: tuple = (),
        task_id: str = "",
        session_id: str = "",
        thread_id: str | None = None,
        description: str = "",
    ) -> Future:
        """Schedule a sync function on the thread pool and track it."""
        future = self._executor.submit(fn, *args)
        self._tasks[task_id] = TaskInfo(
            future=future,
            session_id=session_id,
            thread_id=thread_id,
            description=description,
        )
        future.add_done_callback(lambda f: self._on_task_done(task_id, f))
        logger.info(f"Scheduled task: {task_id} ({description})")
        return future

    def _on_task_done(self, task_id: str, future: Future):
        info = self._tasks.pop(task_id, None)
        elapsed = info.elapsed_seconds if info else 0
        if future.cancelled():
            logger.info(f"Task cancelled: {task_id} ({elapsed}s)")
        elif future.exception():
            logger.error(f"Task failed: {task_id} ({elapsed}s) — {future.exception()}")
        else:
            logger.info(f"Task completed: {task_id} ({elapsed}s)")

    def get_active_tasks(self, session_id: str | None = None) -> list[TaskInfo]:
        """List active tasks, optionally filtered by session."""
        tasks = list(self._tasks.values())
        if session_id:
            tasks = [t for t in tasks if t.session_id == session_id]
        return tasks

    def cancel_session(self, session_id: str):
        """Cancel all tasks for a session."""
        for task_id, info in list(self._tasks.items()):
            if session_id == "*" or info.session_id == session_id:
                info.future.cancel()
                logger.info(f"Cancelled task: {task_id}")

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=wait)

    # --- Event bus ---

    def subscribe(self, session_id: str) -> queue.Queue:
        """Subscribe to events for a session. Returns a queue to read from."""
        if session_id not in self._event_queues:
            self._event_queues[session_id] = []
        q = queue.Queue()
        self._event_queues[session_id].append(q)
        logger.debug(f"SSE subscriber added for session {session_id}")
        return q

    def unsubscribe(self, session_id: str, q: queue.Queue):
        """Remove a subscriber."""
        if session_id in self._event_queues:
            self._event_queues[session_id] = [
                existing for existing in self._event_queues[session_id] if existing is not q
            ]

    def emit(self, event: StreamEvent):
        """Dispatch an event to all subscribers for the session."""
        queues = self._event_queues.get(event.session_id, [])
        for q in queues:
            q.put(event)
        logger.debug(
            f"Event emitted: {event.event_type} thread={event.thread_id} "
            f"→ {len(queues)} subscribers"
        )

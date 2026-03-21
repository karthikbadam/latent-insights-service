"""
Queue — centralized task scheduling, event dispatch, and concurrency.

All async coordination goes through here. Single place to reason about
what's running, what's waiting, and what events are flowing.
"""

import asyncio
import logging
from dataclasses import dataclass

from app.models import ThreadEvent

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Metadata about a running task."""

    task: asyncio.Task
    session_id: str
    thread_id: str | None = None
    description: str = ""


class Queue:
    """
    Central coordinator for async tasks and events.

    Responsibilities:
    - Schedule and track async tasks (thread runs, scout, profiler)
    - Dispatch events to SSE listeners
    - Serialize DuckDB writes via asyncio.Lock
    """

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._event_queues: dict[str, list[asyncio.Queue]] = {}
        self._db_write_lock = asyncio.Lock()

    # --- Task management ---

    def schedule(
        self,
        coro,
        task_id: str,
        session_id: str,
        thread_id: str | None = None,
        description: str = "",
    ) -> asyncio.Task:
        """Schedule an async task and track it."""
        task = asyncio.create_task(coro, name=task_id)
        self._tasks[task_id] = TaskInfo(
            task=task,
            session_id=session_id,
            thread_id=thread_id,
            description=description,
        )
        task.add_done_callback(lambda t: self._on_task_done(task_id, t))
        logger.info(f"Scheduled task: {task_id} ({description})")
        return task

    def _on_task_done(self, task_id: str, task: asyncio.Task):
        self._tasks.pop(task_id, None)
        if task.cancelled():
            logger.info(f"Task cancelled: {task_id}")
        elif task.exception():
            logger.error(f"Task failed: {task_id} — {task.exception()}")
        else:
            logger.info(f"Task completed: {task_id}")

    def get_active_tasks(self, session_id: str | None = None) -> list[TaskInfo]:
        """List active tasks, optionally filtered by session."""
        tasks = list(self._tasks.values())
        if session_id:
            tasks = [t for t in tasks if t.session_id == session_id]
        return tasks

    async def cancel_session(self, session_id: str):
        """Cancel all tasks for a session."""
        for task_id, info in list(self._tasks.items()):
            if info.session_id == session_id:
                info.task.cancel()
                logger.info(f"Cancelled task: {task_id}")

    # --- Event bus ---

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Subscribe to events for a session. Returns a queue to read from."""
        if session_id not in self._event_queues:
            self._event_queues[session_id] = []
        q = asyncio.Queue()
        self._event_queues[session_id].append(q)
        logger.debug(f"SSE subscriber added for session {session_id}")
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue):
        """Remove a subscriber."""
        if session_id in self._event_queues:
            self._event_queues[session_id] = [
                existing for existing in self._event_queues[session_id] if existing is not q
            ]

    async def emit(self, event: ThreadEvent):
        """Dispatch an event to all subscribers for the session."""
        queues = self._event_queues.get(event.session_id, [])
        for q in queues:
            await q.put(event)
        logger.debug(
            f"Event emitted: {event.event_type} thread={event.thread_id} "
            f"→ {len(queues)} subscribers"
        )

    # --- DB write serialization ---

    @property
    def db_write_lock(self) -> asyncio.Lock:
        """Use this lock for all DuckDB write operations."""
        return self._db_write_lock

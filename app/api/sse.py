"""
SSE — Server-Sent Events for real-time thread updates.
"""

import asyncio
import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.core.queue import Queue

router = APIRouter()

# Will be set by main.py on startup
queue: Queue | None = None


@router.get("/sessions/{session_id}/events")
async def session_events(session_id: str):
    """
    SSE endpoint — streams thread events as human-readable messages.

    Events:
    - step: analytical step progress/completion
    - thinking: coordinator is reasoning
    - waiting: thread needs human input
    - complete: thread finished with a finding
    - error: thread encountered an error
    - scout_done: scout finished, questions available
    """

    async def event_generator():
        q = queue.subscribe(session_id)
        try:
            while True:
                event = await q.get()
                yield {
                    "event": event.event_type,
                    "data": json.dumps({
                        "thread_id": event.thread_id,
                        "message": event.message,
                        "timestamp": event.timestamp,
                        **event.data,
                    }),
                }
        except asyncio.CancelledError:
            queue.unsubscribe(session_id, q)

    return EventSourceResponse(event_generator())

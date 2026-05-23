"""
SSE — Server-Sent Events for real-time thread updates.

Each ``data:`` line is a complete ``FeedEntry`` (see ``api/feed.py``)
serialized as JSON, so subscribers can append straight to their feed
without reshaping. The ``event:`` header carries the entry's
``event_type``.
"""

import asyncio
import json
import queue as stdlib_queue

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


@router.get("/sessions/{session_id}/events")
async def session_events(session_id: str, request: Request):
    """Stream the session's FeedEntries as SSE."""

    queue = request.app.state.queue

    async def event_generator():
        q = queue.subscribe(session_id)
        try:
            loop = asyncio.get_running_loop()
            while True:
                try:
                    event = await loop.run_in_executor(
                        None, q.get, True, 30.0,
                    )
                except stdlib_queue.Empty:
                    yield {"comment": "keepalive"}
                    continue
                yield {
                    "event": event.event_type,
                    "data": json.dumps(event.data),
                }
        except asyncio.CancelledError:
            queue.unsubscribe(session_id, q)

    return EventSourceResponse(event_generator())

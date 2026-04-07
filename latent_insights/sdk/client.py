"""
Python SDK for the latent-insights-service API.

Provides typed access to sessions, threads, patterns, streaming,
and graph state inspection.

Usage:
    from latent_insights.sdk import LatentInsightsClient

    client = LatentInsightsClient("http://localhost:8000")
    session = client.create_session(file_path="data/earthquakes.csv")
    for event in session.stream():
        print(f"[{event['event_type']}] {event['message']}")
"""

import json
import time
from typing import Iterator

import httpx


class LatentInsightsClient:
    """HTTP client for the latent-insights-service REST API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(base_url=f"{self._base}/api", timeout=timeout)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --- Sessions ---

    def create_session(
        self,
        file_path: str | None = None,
        dataset_path: str | None = None,
        config: dict | None = None,
    ) -> "SessionHandle":
        """Create analysis session from CSV file or dataset path."""
        if file_path:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.split("/")[-1], f)}
                data = {}
                if config:
                    data["config"] = json.dumps(config)
                resp = self._http.post("/sessions", files=files, data=data)
        elif dataset_path:
            resp = self._http.post("/sessions", params={"dataset_path": dataset_path})
        else:
            raise ValueError("Provide either file_path or dataset_path")
        resp.raise_for_status()
        body = resp.json()
        return SessionHandle(self, body["session_id"], body)

    def get_session(self, session_id: str) -> dict:
        """Get full session state with threads and steps."""
        resp = self._http.get(f"/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json()

    def list_sessions(self) -> list[dict]:
        """List all sessions with metadata."""
        resp = self._http.get("/sessions")
        resp.raise_for_status()
        return resp.json()

    def continue_session(self, session_id: str) -> dict:
        """Resume stuck threads and scout new questions."""
        resp = self._http.post(f"/sessions/{session_id}/continue")
        resp.raise_for_status()
        return resp.json()

    # --- Threads ---

    def create_thread(self, session_id: str, question: str, motivation: str = "") -> dict:
        """Create a user-initiated analysis thread."""
        resp = self._http.post(
            f"/sessions/{session_id}/threads",
            json={"question": question, "motivation": motivation},
        )
        resp.raise_for_status()
        return resp.json()

    def get_thread(self, thread_id: str) -> dict:
        """Get a single thread with its steps."""
        resp = self._http.get(f"/threads/{thread_id}")
        resp.raise_for_status()
        return resp.json()

    def post_message(self, thread_id: str, content: str) -> dict:
        """Post a human message to a stuck thread, resuming it."""
        resp = self._http.post(f"/threads/{thread_id}/messages", json={"content": content})
        resp.raise_for_status()
        return resp.json()

    # --- Patterns ---

    def list_patterns(self) -> list[dict]:
        """List available agentic patterns."""
        resp = self._http.get("/patterns")
        resp.raise_for_status()
        return resp.json()

    def run_pattern(self, session_id: str, pattern: str, inputs: dict | None = None) -> dict:
        """Run a named pattern for a session."""
        resp = self._http.post(
            f"/sessions/{session_id}/patterns/{pattern}",
            json={"inputs": inputs or {}},
        )
        resp.raise_for_status()
        return resp.json()

    # --- Graph State ---

    def get_graph_state(self, thread_id: str) -> dict:
        """Inspect the LangGraph state for a thread."""
        resp = self._http.get(f"/threads/{thread_id}/graph-state")
        resp.raise_for_status()
        return resp.json()

    # --- Streaming ---

    def stream_events(self, session_id: str) -> Iterator[dict]:
        """Connect to SSE stream. Yields parsed event dicts."""
        url = f"{self._base}/api/sessions/{session_id}/events"
        with httpx.stream("GET", url, timeout=None) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

    # --- System ---

    def stats(self) -> dict:
        """Get system stats."""
        resp = self._http.get("/system/stats")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Health check."""
        resp = httpx.get(f"{self._base}/health", timeout=5.0)
        resp.raise_for_status()
        return resp.json()


class SessionHandle:
    """Convenience wrapper around a session."""

    def __init__(self, client: LatentInsightsClient, session_id: str, data: dict):
        self._client = client
        self._session_id = session_id
        self._data = data

    @property
    def id(self) -> str:
        return self._session_id

    @property
    def data(self) -> dict:
        return self._data

    def refresh(self) -> dict:
        """Re-fetch session data from server."""
        self._data = self._client.get_session(self._session_id)
        return self._data

    @property
    def threads(self) -> list[dict]:
        """Get threads (refreshes from server)."""
        return self.refresh().get("threads", [])

    def ask(self, question: str, motivation: str = "") -> dict:
        """Create a thread with this question."""
        return self._client.create_thread(self._session_id, question, motivation)

    def run_pattern(self, pattern: str, inputs: dict | None = None) -> dict:
        """Run a named pattern for this session."""
        return self._client.run_pattern(self._session_id, pattern, inputs)

    def stream(self) -> Iterator[dict]:
        """Stream all events for this session."""
        return self._client.stream_events(self._session_id)

    def continue_(self) -> dict:
        """Resume stuck threads and scout new questions."""
        return self._client.continue_session(self._session_id)

    def wait(self, timeout: float = 600) -> dict:
        """Block until all threads complete by consuming SSE stream."""
        deadline = time.monotonic() + timeout
        for event in self.stream():
            if event.get("event_type") in ("thread_complete", "thread_waiting"):
                data = self.refresh()
                active = [t for t in data.get("threads", []) if t["status"] == "running"]
                if not active:
                    return data
            if time.monotonic() > deadline:
                raise TimeoutError(f"Session {self.id} did not complete within {timeout}s")
        return self.refresh()

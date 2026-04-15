# Error handling & human message changes

This doc describes backend changes on the `claude/error-handling-retries`
branch. It's written for the frontend team (so they know what to consume)
and for anyone reviewing the code changes.

There are three related improvements, plus a matching simplification pass
that removes redundant error-handling paths.

---

## 1. Transient LLM errors are retried automatically

### What changed

`LLMClient.call` now transparently retries transient failures with
exponential backoff before raising:

| Exception                  | Retried?                                         |
|----------------------------|--------------------------------------------------|
| `APIConnectionError`       | Yes                                              |
| `APITimeoutError`          | Yes                                              |
| `APIStatusError` with 408, 409, 425, 429, 500, 502, 503, 504 | Yes |
| `APIStatusError` with 400, 401, 403, 404 (auth/bad request)  | No  |
| Any other exception        | No                                               |

Defaults: 3 retries, backoff 1s → 2s → 4s. Tunable via
`LLMClient(max_transient_retries=..., transient_backoff_base=...)`.

### Why

Previously a single transient API error (rate limit, 503, connection
reset) propagated straight through and marked the whole thread
`waiting`. Transient errors now heal themselves.

### Removed

- `LLMClient.call_with_retry` — dead code, replaced by the retry inside
  `call()`. Any callers would have to migrate, but there were none.
- `Worker.handle_timeout` + the `"timeout" in type_name` string check
  in `graph.py::worker_node` — timeouts are now handled uniformly by
  the transient-retry layer.

---

## 2. `thread_waiting` now explains why

### What changed

Every `thread_waiting` SSE event (and `Thread.error` in the snapshot)
now carries a `reason` that tells the UI why the thread stopped:

| Reason              | Meaning                                                              |
|---------------------|----------------------------------------------------------------------|
| `coordinator_stuck` | The coordinator explicitly returned STUCK (step > 2). Genuine human-needed. |
| `repeated_moves`    | The move repetition guard tripped — the coordinator was looping.     |
| `retry_exhausted`   | LLM transient errors survived all retries. Usually resolves by itself; user can re-send the last message to retry. |
| `unexpected_error`  | A non-transient error raised somewhere (bug or misconfiguration).    |

### SSE payload

```json
{
  "event": "thread_waiting",
  "data": {
    "thread_id": "t-abc",
    "reason": "retry_exhausted",
    "question": "The LLM provider was unreachable after multiple retries. Send a message when you want the thread to try again.",
    "context": "APIConnectionError: ...",
    "error": "APIConnectionError: ...",
    "running_summary": "So far found that..."
  }
}
```

### Snapshot change

`ThreadResponse.error` is now populated on every waiting transition
(previously only set sporadically). Reasons like `retry_exhausted` and
`coordinator_stuck` appear there so the UI can render a meaningful
header without special-casing the SSE reason field.

### UI guidance

- `retry_exhausted`: show a **"Retry"** button. Posting any message to
  the thread re-runs it (the LLM provider may already be healthy again).
- `coordinator_stuck`: show the **`question`** and **`context`** from
  the coordinator's last decision. Prompt the user to reply with
  guidance.
- `repeated_moves`: same UX as `coordinator_stuck` but with a different
  header like "Analysis got stuck in a loop."
- `unexpected_error`: show the `error` text and a "Contact support"
  suggestion; this shouldn't happen in practice.

### Removed

- The dead `finalize_error_node` node and its factory `make_finalize_error_node`
  — it was defined but never wired into the graph. All errors now route
  through `ThreadRunner._on_graph_done`, which uses the same
  `emit_thread_waiting` helper as the in-graph stuck path for payload
  consistency.

---

## 3. Human messages are step events (no separate audit list)

### What changed

Human messages posted via `POST /api/threads/{id}/messages` or
`POST /api/sessions/{id}/messages` are now recorded as events on the
coordinator's step span, alongside `llm_call` and `tool_call`:

```json
// GET /api/sessions/{id} → threads[i].steps[j].events[]
[
  {
    "type": "human_message",
    "timestamp": 1744725127.1234,
    "content": "focus on cohort A",
    "target": "thread"
  },
  {
    "type": "llm_call",
    "timestamp": 1744725128.5,
    "agent": "coordinator",
    "model": "...",
    "response": "{\"next_move\": \"FORAGE\", ...}",
    "duration_ms": 1200
  },
  {
    "type": "tool_call",
    "timestamp": 1744725131.2,
    "agent": "worker",
    "sql": "SELECT ...",
    "tool_result": "...",
    "duration_ms": 234
  }
]
```

### `StepEvent` schema additions

Two optional fields, populated only when `type == "human_message"`:

| Field     | Type       | Description                                             |
|-----------|------------|---------------------------------------------------------|
| `content` | string     | The message text the user posted.                       |
| `target`  | `"thread"` \| `"session"` | Whether it was sent to this thread directly or broadcast to the session. |

### Timestamp ordering

`timestamp` is captured **at post time** (when the user clicked send),
not when the coordinator happened to drain the queue. This keeps events
in causal order: the human message sorts before the coordinator
`llm_call` that read it.

### What the UI should do

Treat `human_message` events like any other entry in `step.events` —
render them chronologically. Because they're inline, you get the right
visualization "for free": a user intervention appears in-place in the
step that responded to it.

No separate audit endpoint, no separate SSE event type for retrieval.
The existing `message_injected` SSE (fired at post time, for live
feedback) stays unchanged.

### Removed

- `Session.human_messages` — the per-session audit list I briefly added
  in an earlier iteration. Redundant: the same data is in the step
  event timeline.
- `Thread.human_messages` — same.
- `SessionResponse.human_messages`, `ThreadResponse.human_messages` —
  removed.
- `StateStore.push_human_message`, `push_thread_human_message`,
  `push_session_human_message` — removed.

`StateStore.push_pending_message(thread_id, content, target="thread")`
is now the single way to queue a human message; its entries carry the
timestamp set at queue time. `drain_pending_messages` returns
`list[dict]` (was `list[str]`) with `{content, target, timestamp}` per
entry.

---

## Summary of file changes

| File                                                    | Change                                                             |
|---------------------------------------------------------|--------------------------------------------------------------------|
| `latent_insights/core/llm.py`                           | Added transient retry; exported `is_transient_llm_error`; deleted `call_with_retry`. |
| `latent_insights/core/state.py`                         | Pending messages carry `target` + `timestamp`; `drain_*` returns dicts; removed three `*_human_message` helpers. |
| `latent_insights/core/tracing.py`                       | `add_event` accepts an explicit `timestamp`; `format_thread_history` tolerates dict entries. |
| `latent_insights/models.py`                             | Dropped `Session.human_messages` and `Thread.human_messages`.      |
| `latent_insights/orchestration/graph.py`                | Coordinator records each drained human message as a `human_message` span event; added `emit_thread_waiting` helper + `wait_reason` in state; deleted `make_finalize_error_node`; removed the timeout-name-check in `worker_node`. |
| `latent_insights/orchestration/thread.py`               | `_on_graph_done` routes through `emit_thread_waiting` with a classified `reason`. |
| `latent_insights/agents/worker.py`                      | Removed `handle_timeout`.                                          |
| `latent_insights/api/schemas.py`                        | `StepEvent` gains optional `content` + `target`; dropped `*.human_messages` fields on responses. |
| `latent_insights/api/routes.py`                         | `/threads/{id}/messages` and `/sessions/{id}/messages` pass `target` through `push_pending_message`; no longer write separate audit records. |
| `tests/test_llm.py`                                     | New tests for transient retry, non-retry, and classifier.          |
| `tests/test_integration.py`                             | New tests for each `reason` value and for human-message interleaving. |
| `tests/test_interrupts_and_config.py`                   | Updated pending-message tests for the new dict shape.              |

---

## Quick migration checklist (frontend)

- [ ] Render `step.events[]` entries where `type === "human_message"`
      using `content` + `target`. They're already in the existing list.
- [ ] Read `thread_waiting.data.reason` (and/or `thread.error`) to
      distinguish retry-exhausted vs coordinator-stuck vs repeated-moves
      vs unexpected-error — tailor the UI prompt accordingly.
- [ ] No longer fetch any separate "human messages" list from the API;
      use `step.events[].type === "human_message"` everywhere.
- [ ] `message_injected` SSE still fires for live feedback at post
      time — keep consuming it for the "message delivered" toast.

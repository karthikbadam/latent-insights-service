# SSE v2 — UI Migration Guide

## Overview

The service's SSE stream (`GET /api/sessions/{id}/events`) and REST snapshot
(`GET /api/sessions/{id}`) now expose the same shapes. Previously the UI had
to guess field names, defer rendering, or wait for a refresh to see things
that existed in the snapshot but not in the stream — that's fixed.

All field names on SSE events now match the REST `StepEvent` / `StepResponse`
/ `ThreadResponse` / `SessionResponse` schemas. A UI that subscribes to the
SSE stream can build the same view the snapshot provides, incrementally, with
no remapping.

**This is a breaking change for the SSE consumer.** There is no dual-write
shim. Coordinate the UI update with the backend deploy.

---

## Breaking changes (renames)

### 1. `llm_call.role` → `llm_call.agent`

Applies to both coordinator and worker LLM calls.

**Before**
```json
{
  "event": "llm_call",
  "data": {
    "thread_id": "t-abc",
    "role": "coordinator",
    "model": "gpt-oss-20b",
    "duration_ms": 1523
  }
}
```

**After**
```json
{
  "event": "llm_call",
  "data": {
    "thread_id": "t-abc",
    "agent": "coordinator",
    "model": "gpt-oss-20b",
    "duration_ms": 1523,
    "input_tokens": 450,
    "output_tokens": 120,
    "response": "{\"assessment\": \"...\", \"next_move\": \"FORAGE\"}"
  }
}
```

### 2. `tool_call.result` → `tool_call.tool_result`

Also adds `agent: "worker"` so `StepEvent.agent` is always populated.

**Before**
```json
{
  "event": "tool_call",
  "data": {
    "thread_id": "t-abc",
    "sql": "SELECT borough, AVG(pm25) FROM dataset GROUP BY 1",
    "result": "borough | avg_pm25\n...",
    "duration_ms": 234
  }
}
```

**After**
```json
{
  "event": "tool_call",
  "data": {
    "thread_id": "t-abc",
    "agent": "worker",
    "sql": "SELECT borough, AVG(pm25) FROM dataset GROUP BY 1",
    "tool_result": "borough | avg_pm25\n...",
    "duration_ms": 234
  }
}
```

### 3. `session_ready.schema_summary` removed

Full schema now arrives via the new `schema_summary_ready` event (see below),
which fires on **both** scout and human question paths — not just human mode.
The `session_ready` event still fires in `question_source="human"` mode but
no longer carries a truncated schema.

---

## New event types

### `schema_summary_ready`

Fires once per session, immediately after profiling completes, before any
thread starts. Session-scoped (`thread_id` is empty string).

```json
{
  "event": "schema_summary_ready",
  "data": {
    "thread_id": "",
    "message": "Dataset profiled.",
    "schema_summary": "## Schema\n\nThe dataset has 12 columns... (full text)"
  }
}
```

**UI action:** on receipt, populate `session.schema_summary` in the reducer
and render the dataset panel. Remove the code path that fetches the snapshot
just to display the schema.

---

## New fields on existing events

All additive — safe to ignore on old UI builds, but the UI should consume
them to match snapshot fidelity.

| Event             | New field           | Notes                                                     |
|-------------------|---------------------|-----------------------------------------------------------|
| `step_start`      | `provisional: false`| The move is authoritative; stop ignoring it.              |
| `step_complete`   | `instruction`       | The coordinator's instruction to the worker for this step.|
| `step_complete`   | `duration_ms`       | Integer milliseconds, matches `StepResponse.duration_ms`. |
| `llm_call`        | `response`          | Full response text (matches `StepEvent.response`).        |
| `llm_call`        | `step_number`       | Which step this call belongs to.                          |
| `llm_call`        | `move`              | The step's move (e.g. `FORAGE`). Same value as `step_start.move`. |
| `tool_call`       | `agent: "worker"`   | Always set.                                               |
| `tool_call`       | `step_number`       | Which step this call belongs to.                          |
| `tool_call`       | `move`              | The step's move.                                          |
| `thread_waiting`  | `running_summary`   | Thread's accumulated findings so far (or `null`).         |
| `thread_waiting`  | `error`             | Non-null only on the error path.                          |
| `thread_complete` | `total_ms`          | Integer ms. `total_seconds` (float) still present for bwc.|

### Why `step_number` and `move` on every event

Previously the UI had to remember "the last `step_start` said step 3,
move FORAGE" and attribute any subsequent `llm_call` / `tool_call` to that
step. That stitching breaks under SSE reconnects, out-of-order delivery, or
when the UI mounts mid-stream.

With `step_number` and `move` on every thread-scoped event, each event is
self-contained:

```json
{
  "event": "tool_call",
  "data": {
    "thread_id": "t-abc",
    "step_number": 3,
    "move": "FORAGE",
    "agent": "worker",
    "sql": "SELECT ...",
    "tool_result": "...",
    "duration_ms": 234
  }
}
```

The UI can key events by `(thread_id, step_number)` directly and drop any
per-thread "current step" tracker.

**Consistency guarantee:** the `move` on a coordinator `llm_call` always
matches the `step_start.move` that follows it, even when the early-stuck
override fires on steps 1–2. The emission is deferred until after the
override runs, so both events see the same final move.

---

## Config change: `seed_threads` is now honored

The `POST /api/sessions` endpoint accepts a `config` form field (JSON string).
`seed_threads` is now a valid key inside that JSON:

```json
{
  "seed_threads": 2
}
```

**Semantics:** when set, `seed_threads` caps **both**:
- the number of questions the scout agent generates
  (`num_scout_seed_questions`), and
- the total number of spawned threads (`max_threads`).

If a request sends both `seed_threads` and `max_threads`, `max_threads` wins
for the thread cap (but scout question count still follows `seed_threads`).

**UI cleanup:** the UI currently sends `seed_threads` three ways (form field,
`config` JSON, query param) as a workaround. After this change, only the
`config` JSON path is honored — drop the other two.

Example POST:
```
POST /api/sessions
Content-Type: multipart/form-data

file=<csv>
config={"seed_threads": 2, "question_source": "scout"}
```

---

## Field reference: snapshot ↔ SSE

Every field the snapshot exposes, and which SSE event carries its
streaming equivalent:

| Snapshot path                               | SSE source                                                       |
|---------------------------------------------|------------------------------------------------------------------|
| `session.schema_summary`                    | `schema_summary_ready.schema_summary`                            |
| `session.scout_questions[]`                 | `scout_done.questions[]`                                         |
| `thread.seed_question`                      | `thread_start.seed_question`                                     |
| `thread.motivation`                         | `thread_start.motivation`                                        |
| `thread.summary`                            | `thread_complete.summary`                                        |
| `thread.running_summary`                    | `thread_waiting.running_summary`                                 |
| `thread.error`                              | `thread_waiting.error`                                           |
| `thread.status`                             | derived from `thread_start` / `thread_complete` / `thread_waiting` |
| `step.move`                                 | `step_start.move` (authoritative) and `step_complete.move`       |
| `step.instruction`                          | `step_start.instruction` and `step_complete.instruction`         |
| `step.result`                               | `step_complete.result`                                           |
| `step.duration_ms`                          | `step_complete.duration_ms`                                      |
| `step.events[].type`                        | `llm_call` or `tool_call` event type                             |
| `step.events[].agent`                       | `llm_call.agent` / `tool_call.agent`                             |
| `step.events[].model`                       | `llm_call.model`                                                 |
| `step.events[].input_tokens`                | `llm_call.input_tokens`                                          |
| `step.events[].output_tokens`               | `llm_call.output_tokens`                                         |
| `step.events[].duration_ms`                 | `llm_call.duration_ms` / `tool_call.duration_ms`                 |
| `step.events[].sql`                         | `tool_call.sql`                                                  |
| `step.events[].tool_result`                 | `tool_call.tool_result`                                          |
| `step.events[].response`                    | `llm_call.response`                                              |
| (grouping key) `step.step_number` / `step.move` | every thread-scoped event carries `step_number` + `move`     |

---

## Migration checklist

- [ ] Rename SSE handler field reads: `role` → `agent` on `llm_call`,
      `result` → `tool_result` on `tool_call`.
- [ ] Add an `agent` field reader on `tool_call` (was absent).
- [ ] Start consuming `llm_call.response` — this is where the LLM
      assessment / JSON blob lives. No more parsing from `message`.
- [ ] Stop ignoring `step_start.move` — it's authoritative. Drop any
      "wait for step_complete to learn the move" logic.
- [ ] Populate `step.instruction` from either `step_start` or
      `step_complete` (both carry it now).
- [ ] Populate `step.duration_ms` from `step_complete.duration_ms`.
- [ ] Group `llm_call` / `tool_call` events by `(thread_id, step_number)`
      directly — drop any "current step per thread" tracker that inferred
      the step from the most recent `step_start`.
- [ ] Handle the new `schema_summary_ready` event and remove the
      post-refresh schema fetch.
- [ ] Read `thread.running_summary` from `thread_waiting.running_summary`
      on live streams (stop showing an empty expanded view during the wait
      state).
- [ ] Drop the `seed_threads` form field and `?seed_threads=` query param
      from the upload call; keep only the `config` JSON field.
- [ ] Update the session-upload code path so the UI no longer tries to
      read `schema_summary` from `session_ready` (it has been removed).

---

## Unit contracts

- `duration_ms`, `total_ms`, `input_tokens`, `output_tokens` are always
  integers.
- `duration_ms` and `total_ms` are always milliseconds.
- `total_seconds` on `thread_complete` is retained as a float (legacy); new
  UI should prefer `total_ms`.
- Session-scoped events (`schema_summary_ready`, `scout_done`,
  `session_ready`, `message_injected` when broadcast) have
  `thread_id: ""` (empty string). Thread-scoped events always set a non-empty
  `thread_id`.

---

## Not changed

- `scout_done` payload shape is unchanged.
- `thread_start` still carries `seed_question`, `motivation`, `entry_point`.
- `message_injected` is unchanged.
- `synthesis_start` is unchanged.
- Keepalive comments (`: keepalive`) from the SSE framing are unchanged.

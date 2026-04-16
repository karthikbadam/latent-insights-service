# Latent Insights

Parallel-agent sensemaking for collaborative data analysis.
For any uploaded dataset, the system discovers questions, spawns analytical threads, executes LLM orchestrated tool calls, and builds insights with you.

The agent is designed to follow steps from a sensemaking process such as foraging for evidence, framing the hypothesis, investigating the data, and synthesizing the results.

## Install

```bash
pip install latent-insights
```

## Quick start

### OpenRouter (default)

```bash
export LLM_API_KEY=<your-key>
```

### Ollama (local, free)

```bash
export LLM_PROVIDER=ollama
```

Override individual models if needed:

```bash
export LLM_PROVIDER=ollama
export MODEL_WORKER=gemma3:4b
```

### As a library

```python
from latent_insights import AppConfig
from latent_insights.core.llm import LLMClient
from latent_insights.core.queue import Queue
from latent_insights.core.store import InvestigationStore
from latent_insights.db.connection import Database
from latent_insights.orchestration.session import SessionFlow

config = AppConfig.from_env()
llm = LLMClient(
    api_key=config.llm_api_key,
    base_url=config.llm_base_url,
    app_name=config.app_name,
    app_url=config.app_url,
)
db = Database(data_dir=config.data_dir)
queue = Queue()
store = InvestigationStore(data_dir=config.data_dir)

flow = SessionFlow(config, llm, db, queue, store)
session = store.create_session("path/to/data.csv")
flow.create(session.id, "path/to/data.csv")
```

## Development

```bash
git clone https://github.com/karthikbadam/latent-insights.git
cd latent-insights
uv sync --extra dev

# Run dev server with hot reload
uv run uvicorn latent_insights.main:app --reload

# Run tests
uv run pytest                    # all tests
uv run pytest -m "not live"      # skip API-calling tests
uv run ruff check .
```

## Architecture

```
POST /api/sessions (upload CSV)
         │
         ▼
┌─────────────-────┐
│     Session      │
│  Profiler → Scout│──── schema summary + seed questions
└────────┬─────-───┘
         │ spawns N threads
         ▼
┌─────────────-────┐   ┌────────────-─────┐   ┌─────────────-────┐
│    Thread 1      │   │    Thread 2      │   │    Thread N      │
│                  │   │                  │   │                  │
│  ┌────────────┐  │   │  ┌────────────┐  │   │  ┌────────────┐  │
│  │Coordinator │◄─┤   │  │Coordinator │◄─┤   │  │Coordinator │◄─┤
│  │  (judge)   │  │   │  │  (judge)   │  │   │  │  (judge)   │  │
│  └─────┬──────┘  │   │  └─────┬──────┘  │   │  └─────┬──────┘  │
│        │ decide  │   │        │         │   │        │         │
│        ▼         │   │        ▼         │   │        ▼         │
│  ┌────────────┐  │   │  ┌────────────┐  │   │  ┌────────────┐  │
│  │   Worker   │  │   │  │   Worker   │  │   │  │   Worker   │  │
│  │  LLM+SQL   │  │   │  │  LLM+SQL   │  │   │  │  LLM+SQL   │  │
│  └────────────┘  │   │  └────────────┘  │   │  └────────────┘  │
│                  │   │                  │   │                  │
│  Steps: SCOPE    │   │  Steps: FORAGE   │   │  Steps: FRAME    │
│  → FORAGE        │   │  → INTERROGATE   │   │  → INTERROGATE   │
│  → FRAME         │   │  → SYNTHESIZE    │   │  → SYNTHESIZE    │
│  → INTERROGATE   │   │  ✓ DONE          │   │  ? STUCK (human) │
│  → SYNTHESIZE    │   │                  │   │                  │
│  ✓ DONE          │   │                  │   │                  │
└────────────────-─┘   └───────────────-──┘   └───────────────-──┘
         │                    │                       │
         └────────────────────┴───────────────────────┘
                              │
                    GET /api/sessions/{id}/events (SSE)
                    ← schema_summary_ready, scout_done,
                      thread_start, step_start, llm_call,
                      tool_call, human_message, step_complete,
                      thread_complete, thread_waiting
```

### Sensemaking moves


| Move            | Purpose                                                      |
| --------------- | ------------------------------------------------------------ |
| **SCOPE**       | Define data slice, narrow to relevant subset                 |
| **FORAGE**      | Exploratory analysis — distributions, correlations, outliers |
| **FRAME**       | Propose tentative hypothesis as testable claim               |
| **INTERROGATE** | Stress-test the frame — contradictions, confounds            |
| **SYNTHESIZE**  | Terminal summary — writes the final finding from prior steps. The worker receives no SQL tool on this move; it can only condense what earlier steps found. |


The coordinator picks moves freely based on data — no fixed order.

### State management

Three components own all per-thread state:

| Component | File | Role |
| --- | --- | --- |
| **`InvestigationStore`** | `core/store.py` | In-memory store for `Session`, `Thread`, and `Step` records plus pending human messages. Persists one JSON file per session at `data/sessions/{id}.json` whose shape matches the `SessionResponse` REST snapshot — saved files and live snapshots are byte-for-byte equivalent. |
| **`ThreadRunner`** | `orchestration/runner.py` | Drives one thread through its coordinator→worker lifecycle in continuation-passing style. Each LLM call, each SQL tool call, and the periodic history summarizer is submitted as a separate task to the shared `Queue`, chained via `future.add_done_callback`. Pool workers are released between calls, so N active threads can share a pool of `max_workers` without one thread starving another. |
| **`Recorder`** | `core/recorder.py` | Dual-write helper: every event the runner emits goes through one method (`recorder.llm_call`, `recorder.tool_call`, `recorder.human_message`, `recorder.step_start`, `recorder.step_complete`, `recorder.thread_complete`, `recorder.thread_waiting`) that records it on the current step **and** emits the matching `StreamEvent` over SSE. Span events and SSE frames cannot drift. |

Supporting pieces:

- **`Queue`** (`core/queue.py`) — thread pool for scheduling, plus the session→subscriber registry for SSE.
- **`Coordinator`** and **`Worker`** (`agents/`) — agent classes called by the runner; the worker exposes a granular API (`prepare_tool_calls`, `record_tool_result`, `apply_error_guardrails`, `handle_final`) so each SQL exec can be its own pool task.

The `Step` dataclass is flat and mirrors `api.schemas.StepResponse` directly (`move`, `instruction`, `result`, `view_created`, `events[]`, `start_time`, `end_time`). Its `events[]` are flat `StepEvent` dicts (`type`, `timestamp`, plus type-specific fields like `sql` or `response`) — the same shape clients consume from REST and SSE.

### Resilience behaviors

- **Transient LLM errors** (connection resets, 429, 5xx, timeouts) retry inside `LLMClient.call` with exponential backoff. If retries exhaust, the thread enters `thread_waiting` with `reason=retry_exhausted`.
- **Context-length errors** (400 with "maximum context length" in the body) are caught by the runner, which closes the current step with a self-describing result — *"Context overflow: this step's prompt exceeded the model's context window. The next move should narrow the data…"* — and lets the coordinator read that result through `format_thread_history` on the next step to pick a simpler move. After `max_context_recoveries` (default 2) attempts in a thread, it falls through to `thread_waiting` with `reason=context_exhausted`.
- **Move repetition guard** — if the coordinator picks the same move `max_repeated_moves` (default 10) steps in a row without `DONE`, the thread enters `thread_waiting` with `reason=repeated_moves`.
- **Early STUCK override** — if the coordinator returns `STUCK` on step 1 or 2, the runner overrides it to `FORAGE` to give the thread a chance to find something before asking for help.
- **Periodic summarization** — every 5 steps the runner schedules a summarizer LLM call that condenses the history into a `running_summary`, which prepends future coordinator prompts.

## API


| Endpoint                           | Description                                                   |
| ---------------------------------- | ------------------------------------------------------------- |
| `GET /health`                      | Health check                                                  |
| `POST /api/sessions`               | Create session (upload CSV + profile + scout + spawn threads) |
| `GET /api/sessions`                | List all sessions with metadata and thread counts             |
| `GET /api/sessions/{id}`           | Full session state with threads, steps, and step events       |
| `GET /api/sessions/{id}/saved`     | Previously saved session snapshot from `data/sessions/{id}.json` |
| `POST /api/sessions/{id}/threads`  | Create custom thread with a question                          |
| `POST /api/sessions/{id}/continue` | Resume stuck threads + scout new questions                    |
| `POST /api/sessions/{id}/messages` | Broadcast a message to all running/waiting threads            |
| `GET /api/threads/{id}`            | Get single thread with steps and events                       |
| `POST /api/threads/{id}/messages`  | Reply to or interrupt a thread                                |
| `GET /api/sessions/{id}/events`    | SSE event stream (see event list below)                       |
| `GET /api/patterns`                | List available agentic patterns                               |
| `POST /api/sessions/{id}/patterns/{name}` | Run a named pattern (coordinator_worker, fan_out, human_in_the_loop) |
| `GET /api/threads/{id}/graph-state` | Debug view — step count, move history, current status        |
| `GET /api/system/stats`            | Session and thread counts                                     |


### SSE events

`GET /api/sessions/{id}/events` streams JSON events. Field names and
semantics align with the REST snapshot (`StepEvent` / `StepResponse` /
`ThreadResponse` / `SessionResponse`) so both sources can populate the same
client-side view model.

| Event                  | Scope    | Key fields                                                              |
| ---------------------- | -------- | ----------------------------------------------------------------------- |
| `schema_summary_ready` | session  | `schema_summary` (full profiler output)                                 |
| `scout_done`           | session  | `question_count`, `questions[]`                                         |
| `session_ready`        | session  | `question_source` (emitted only in `human` mode)                        |
| `message_injected`     | session/thread | `content`, `target`, `injected_threads`, `resumed_threads`        |
| `thread_start`         | thread   | `seed_question`, `motivation`, `entry_point`, `step_number: 0`          |
| `thread_resumed`       | thread   | `from_step`, `human_messages`                                           |
| `step_start`           | thread   | `move`, `step_number`, `instruction`, `provisional: false`              |
| `llm_call`             | thread   | `agent`, `model`, `input_tokens`, `output_tokens`, `duration_ms`, `response`, `step_number`, `move` |
| `tool_call`            | thread   | `agent`, `sql`, `tool_result`, `duration_ms`, `step_number`, `move`     |
| `human_message`        | thread   | `content`, `target`, `step_number`, `move`                              |
| `step_complete`        | thread   | `step_number`, `move`, `instruction`, `result`, `duration_ms`           |
| `thread_complete`      | thread   | `summary`, `total_ms`, `total_seconds`, `step_count`, `is_terminal: true` |
| `thread_waiting`       | thread   | `reason` (`coordinator_stuck` \| `repeated_moves` \| `retry_exhausted` \| `context_exhausted` \| `unexpected_error`), `question`, `context`, `running_summary`, `error`, `is_terminal: true` |
| `synthesis_start`      | thread   | `source_threads`, `synthesis_thread` (fan-out pattern only)             |

All `duration_ms` and `total_ms` values are integer milliseconds.
Session-scoped events use `thread_id: ""`.

Every thread-scoped event carries `step_number` and `move` so each event is
self-contained and the UI can group events into steps without maintaining
cross-event state. Exceptions: `thread_start` (step_number is 0, move is absent)
and `thread_complete` / `thread_waiting` (step-independent terminal events that
carry `is_terminal: true`).


### Per-session config

`POST /api/sessions` accepts optional per-session overrides via a `config` object. All fields are optional — omitted fields use server defaults from environment variables.

```bash
curl -X POST http://localhost:8000/api/sessions \
  -F "file=@data/samples/cars.csv" \
  -F 'config={"seed_threads": 3, "initial_questions": ["What factors most influence fuel efficiency?", "Are there regional differences in car specifications?"]}'
```

`seed_threads` is the simplest way to bound the analysis: it caps both the
number of questions the scout generates and the total spawned thread count.
For finer control, set `num_scout_seed_questions` and `max_threads`
independently.

Available config fields:


| Field                      | Type       | Description                               |
| -------------------------- | ---------- | ----------------------------------------- |
| `model_profiler`           | `string`   | Model for dataset profiling               |
| `model_scout`              | `string`   | Model for question discovery              |
| `model_coordinator`        | `string`   | Model for thread coordination             |
| `model_worker`             | `string`   | Model for SQL analysis                    |
| `model_worker_fallback`    | `string`   | Fallback model after worker retries       |
| `temp_profiler`            | `float`    | Temperature for profiler                  |
| `temp_scout`               | `float`    | Temperature for scout                     |
| `temp_coordinator`         | `float`    | Temperature for coordinator               |
| `temp_worker`              | `float`    | Temperature for worker                    |
| `max_threads`              | `int`      | Cap on total threads spawned              |
| `seed_threads`             | `int`      | Shortcut: caps both `num_scout_seed_questions` and `max_threads` |
| `max_worker_retries`       | `int`      | Worker retries before fallback model      |
| `max_consecutive_errors`   | `int`      | SQL errors before forcing summary         |
| `max_repeated_moves`       | `int`      | Repeated coordinator moves before abort   |
| `llm_timeout`              | `float`    | LLM call timeout in seconds               |
| `num_scout_seed_questions` | `int`      | Number of questions scout should discover |
| `initial_questions`        | `string[]` | Seed questions to start alongside scout   |
| `question_source`          | `string`   | `scout` (default), `human`, or `both`     |
| `scout_context`            | `string`   | Free-text guidance to steer scout         |
| `default_pattern`          | `string`   | `coordinator_worker` (default), `fan_out`, or `human_in_the_loop` |


## Publishing to PyPI

```bash
# Build the package
uv build

# Publish (requires PyPI API token)
uv publish

# Or test with TestPyPI first
uv publish --publish-url https://test.pypi.org/legacy/
```

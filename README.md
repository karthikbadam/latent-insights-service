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
from latent_insights.core.state import StateStore
from latent_insights.core.tracing import TraceStore
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
state = StateStore(data_dir=config.data_dir)
trace = TraceStore(data_dir=config.data_dir)

flow = SessionFlow(config, llm, db, queue, state, trace)
flow.create(session_id, "path/to/data.csv")
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
                      tool_call, step_complete,
                      thread_complete, thread_waiting
```

### Sensemaking moves


| Move            | Purpose                                                      |
| --------------- | ------------------------------------------------------------ |
| **SCOPE**       | Define data slice, narrow to relevant subset                 |
| **FORAGE**      | Exploratory analysis — distributions, correlations, outliers |
| **FRAME**       | Propose tentative hypothesis as testable claim               |
| **INTERROGATE** | Stress-test the frame — contradictions, confounds            |
| **SYNTHESIZE**  | Thread conclusion — finding, confidence, limitations         |


The coordinator picks moves freely based on data — no fixed order.

## API


| Endpoint                           | Description                                                   |
| ---------------------------------- | ------------------------------------------------------------- |
| `GET /health`                      | Health check                                                  |
| `POST /api/sessions`               | Create session (upload CSV + profile + scout + spawn threads) |
| `GET /api/sessions`                | List all sessions with metadata and thread counts             |
| `GET /api/sessions/{id}`           | Full session state with threads, steps, and step events       |
| `POST /api/sessions/{id}/threads`  | Create custom thread with a question                          |
| `POST /api/sessions/{id}/continue` | Resume stuck threads + scout new questions                    |
| `POST /api/sessions/{id}/messages` | Broadcast a message to all running/waiting threads            |
| `GET /api/threads/{id}`            | Get single thread with steps and events                       |
| `POST /api/threads/{id}/messages`  | Reply to or interrupt a thread                                |
| `GET /api/sessions/{id}/events`    | SSE event stream (see event list below)                       |
| `GET /api/patterns`                | List available agentic patterns                               |
| `POST /api/sessions/{id}/patterns/{name}` | Run a named pattern (coordinator_worker, fan_out, human_in_the_loop) |
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
| `thread_start`         | thread   | `seed_question`, `motivation`, `entry_point`                            |
| `step_start`           | thread   | `move`, `step_number`, `instruction`, `provisional: false`              |
| `llm_call`             | thread   | `agent`, `model`, `input_tokens`, `output_tokens`, `duration_ms`, `response`, `step_number`, `move` |
| `tool_call`            | thread   | `agent`, `sql`, `tool_result`, `duration_ms`, `step_number`, `move`     |
| `step_complete`        | thread   | `step_number`, `move`, `instruction`, `result`, `duration_ms`           |
| `thread_complete`      | thread   | `summary`, `total_ms`, `total_seconds`, `step_count`                    |
| `thread_waiting`       | thread   | `question`, `context`, `running_summary`, `error`                       |

All `duration_ms` and `total_ms` values are integer milliseconds.
Session-scoped events use `thread_id: ""`.

Every thread-scoped event carries `step_number` and `move` (the same values
that appear on `step_start` / `step_complete`), so each event is
self-contained and the UI can group events into steps without maintaining
cross-event state. Exceptions: `thread_start` (no step context yet) and
`thread_complete` / `thread_waiting` (step-independent terminal events).

Frontend migrating from the previous SSE schema: see
[docs/sse-migration-v2.md](docs/sse-migration-v2.md) for a field-by-field
rename guide and migration checklist.


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

## Docs

- [docs/SPEC.md](docs/SPEC.md) — architecture spec
- [docs/PROMPTS.md](docs/PROMPTS.md) — agent prompt designs


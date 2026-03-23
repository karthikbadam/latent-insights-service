# Latent Insights

Parallel-agent sensemaking for collaborative data analysis.
For any uploaded dataset, the system discovers questions, spawns analytical threads, executes LLM orchestrated tool calls, and builds insights with you.

The agent is designed to follow steps from a sensemaking process such as foraging for evidence, framing the hypothesis, investigating the data, and synthesizing the results. 

## Quick start

### OpenRouter (default)

```bash
uv sync --extra dev
export LLM_PROVIDER=openrouter
export LLM_API_KEY=<your-key>
uv run uvicorn app.main:app --reload
```

### Ollama (local, free)

```bash
ollama pull gpt-oss:20b
export LLM_PROVIDER=ollama
uv run uvicorn app.main:app --reload
```

Override individual models if needed:

```bash
export LLM_PROVIDER=ollama
export MODEL_WORKER=gemma3:4b
uv run uvicorn app.main:app --reload
```

## Tests

```bash
uv run pytest                    # all tests
uv run pytest -m "not live"      # skip API-calling tests
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
                    ← llm_call, tool_call, step, complete
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


| Endpoint                          | Description                                                   |
| --------------------------------- | ------------------------------------------------------------- |
| `GET /health`                     | Health check                                                  |
| `POST /api/sessions`              | Create session (upload CSV + profile + scout + spawn threads) |
| `GET /api/sessions`               | List all sessions with metadata and thread counts             |
| `GET /api/sessions/{id}`          | Full session state with threads and steps                     |
| `POST /api/sessions/{id}/threads` | Create custom thread with a question                          |
| `POST /api/sessions/{id}/continue`| Resume stuck threads + scout new questions                    |
| `GET /api/threads/{id}`           | Get single thread with steps and events                       |
| `POST /api/threads/{id}/messages` | Reply to stuck thread, resuming it                            |
| `GET /api/sessions/{id}/events`   | SSE event stream (llm_call, tool_call, step, complete)        |
| `GET /api/system/stats`           | Session and thread counts                                     |

### Per-session config

`POST /api/sessions` accepts optional per-session overrides via a `config` object. All fields are optional — omitted fields use server defaults from environment variables.

```bash
curl -X POST http://localhost:8000/api/sessions \
  -F file=@data.csv \
  -F config='{"model_worker": "google/gemini-2.5-flash", "max_threads": 5}'
```

Available config fields:

| Field                    | Type       | Description                                |
| ------------------------ | ---------- | ------------------------------------------ |
| `model_profiler`         | `string`   | Model for dataset profiling                |
| `model_scout`            | `string`   | Model for question discovery               |
| `model_coordinator`      | `string`   | Model for thread coordination              |
| `model_worker`           | `string`   | Model for SQL analysis                     |
| `model_worker_fallback`  | `string`   | Fallback model after worker retries        |
| `temp_profiler`          | `float`    | Temperature for profiler                   |
| `temp_scout`             | `float`    | Temperature for scout                      |
| `temp_coordinator`       | `float`    | Temperature for coordinator                |
| `temp_worker`            | `float`    | Temperature for worker                     |
| `max_threads`            | `int`      | Cap on total threads spawned               |
| `max_worker_retries`     | `int`      | Worker retries before fallback model       |
| `max_consecutive_errors` | `int`      | SQL errors before forcing summary          |
| `max_repeated_moves`     | `int`      | Repeated coordinator moves before abort    |
| `llm_timeout`            | `float`    | LLM call timeout in seconds               |
| `num_scout_seed_questions`| `int`     | Number of questions scout should discover  |
| `initial_questions`      | `string[]` | Seed questions to start alongside scout    |


## Docs

- [docs/SPEC.md](docs/SPEC.md) — architecture spec
- [docs/PROMPTS.md](docs/PROMPTS.md) — agent prompt designs


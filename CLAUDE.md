# CLAUDE.md — Latent Insights

## What this project is

Latent Insights is a parallel-agent sensemaking tool for data analysis. A user uploads a dataset (starting with NASA's exoplanet catalog). The system automatically discovers interesting questions, spawns parallel analytical threads, and builds insights collaboratively with the human.

Each thread follows a sensemaking loop (scope → forage → frame → interrogate → synthesize) guided by a coordinator agent that acts as a judge. Workers execute SQL against DuckDB. When a thread gets stuck, it asks the human for help. The frontend is a feed of thread cards updating in real time via SSE.

## Project structure

```
latent-insights/
├── latent_insights/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — COMPLETE, working
│   ├── config.py             # All settings via env vars — COMPLETE
│   ├── models.py             # Dataclasses for engine — COMPLETE
│   ├── core/
│   │   ├── llm.py            # LLM client (OpenRouter/Ollama) + cache — COMPLETE, working
│   │   ├── parsing.py        # LLM JSON → dataclasses — COMPLETE
│   │   ├── queue.py          # Task scheduling + event bus — COMPLETE
│   │   └── errors.py         # Custom exceptions — COMPLETE
│   ├── api/
│   │   ├── routes.py         # REST endpoints — SKELETON, needs impl
│   │   ├── schemas.py        # Pydantic for API boundary — COMPLETE
│   │   └── sse.py            # SSE streaming — SKELETON, mostly done
│   ├── agents/
│   │   ├── profiler.py       # Dataset profiler — SKELETON
│   │   ├── scout.py          # Question discovery — SKELETON
│   │   ├── coordinator.py    # Thread judge — SKELETON
│   │   └── worker.py         # SQL execution — SKELETON
│   └── db/
│       ├── connection.py     # DuckDB connection mgmt — COMPLETE
│       ├── schema.py         # Table definitions — COMPLETE
│       ├── queries.py        # Typed query functions — COMPLETE
│       └── mcp.py            # DuckDB MCP extension — COMPLETE
├── tests/
│   ├── conftest.py           # Shared fixtures
│   └── fixtures/             # Sample data + canned LLM responses
├── docs/
│   ├── SPEC.md               # Full product & architecture spec
│   └── PROMPTS.md            # Agent prompt designs
├── pyproject.toml
├── Dockerfile
└── railway.toml
```

## Tech stack

- **Python 3.12+** with uv for package management
- **FastAPI** + uvicorn for the API
- **DuckDB** for all storage (session state, cache, dataset analysis)
- **OpenRouter** or **Ollama** for LLM calls (OpenAI-compatible SDK)
- **SSE** (sse-starlette) for real-time frontend updates
- **pytest** + pytest-asyncio for testing

## Key architectural decisions

- **Dataclasses for engine, Pydantic only at API boundaries.** Internal data flows use `latent_insights/models.py` dataclasses. Pydantic is in `latent_insights/api/schemas.py` only.
- **All LLM calls go through `latent_insights/core/llm.py`.** Never import openai directly. The LLMClient handles caching, retries, and provider headers.
- **DuckDB is the single data store.** Session state, thread history, LLM cache, and the actual dataset — all in DuckDB. No Redis, no Postgres.
- **Coordinator never touches data.** It only reasons over worker summaries. Workers generate and execute SQL.
- **No artificial limits.** No hard caps on steps per thread, no forced move sequences. The coordinator runs until it's DONE or STUCK (and asks the human).
- **Models are configurable.** All model IDs in `latent_insights/config.py`, loaded from env vars. Swap freely between Haiku, Sonnet, DeepSeek, free models.

## Running the project

```bash
# Install dependencies
uv sync

# Set API key (or use LLM_PROVIDER=ollama for local)
export LLM_API_KEY=<your-openrouter-key>

# Run dev server
uv run uvicorn latent_insights.main:app --reload

# Run tests
uv run pytest
uv run pytest -m "not live"  # skip API-calling tests
```

## Development approach

Build and test each agent independently before composing them:

1. `core/llm.py` — already done, verify with tests
2. `core/parsing.py` — already done, test with edge cases
3. `db/` layer — already done, test round-trips
4. `agents/profiler.py` — first agent to flesh out
5. `agents/worker.py` — most critical, SQL generation + execution
6. `agents/coordinator.py` — the judge loop
7. `agents/scout.py` — question discovery
8. `core/queue.py` + orchestration — compose the thread loop
9. `api/routes.py` + `api/sse.py` — HTTP layer last

## Code style

- `__init__.py` in all package directories (required for pip distribution)
- Type hints everywhere
- Logging via `logging.getLogger(__name__)`
- f-strings preferred
- No comments explaining obvious code, but TODO comments for unfinished work
- ruff for formatting: `uv run ruff check . --fix`

## Environment variables

```
LLM_PROVIDER=              # "openrouter" (default) or "ollama"
LLM_API_KEY=               # Required for openrouter; auto-set for ollama
LLM_BASE_URL=              # Auto-set per provider; override if needed
MODEL_PROFILER=            # Default depends on provider
MODEL_SCOUT=               # Default depends on provider
MODEL_COORDINATOR=         # Default depends on provider
MODEL_WORKER=              # Default depends on provider
MODEL_WORKER_FALLBACK=     # Default depends on provider
TEMP_PROFILER=             # Default: 0.0
TEMP_SCOUT=                # Default: 0.7
TEMP_COORDINATOR=          # Default: 0.3
TEMP_WORKER=               # Default: 0.0
DATA_DIR=                  # Default: data
PORT=                      # Default: 8000
```

## Commits

- Commit regularly as work progresses; don't batch many changes into one commit.
- Commit message format: `<type>: <description>`
  - `type` is one of: `add`, `remove`, `fix`
  - `description` is at most 5 words
- Examples:
  - `add: job retry endpoint`
  - `fix: token refresh race condition`
  - `remove: unused websocket handler`

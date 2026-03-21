# Latent Insights — Product & Architecture Spec

## Vision

A parallel-agent sensemaking tool where LLM agents collaboratively analyze data with a human. Not a chatbot — a feed of concurrent analytical threads that discover, investigate, and synthesize insights from uploaded datasets.

The interaction model is closer to a Facebook feed than a chat interface: threads of analysis update in real time, the user can expand any thread to see its reasoning, redirect it with a message, or spawn new threads from questions they care about.

## Demo dataset

NASA Exoplanet Archive — Planetary Systems Composite Parameters table.
- ~6,100 confirmed exoplanets
- Download CSV via TAP API: `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv`
- Column definitions: `https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html`

Key columns: `pl_name`, `pl_orbper`, `pl_orbsmax`, `pl_orbeccen`, `pl_rade`, `pl_bmasse`, `pl_eqt`, `pl_insol`, `hostname`, `st_teff`, `st_rad`, `st_mass`, `st_spectype`, `discoverymethod`, `disc_year`, `disc_facility`, `ra`, `dec`, `sy_dist`.

## System flow

```
User uploads dataset
    ↓
Profiler agent runs → produces schema_summary (stored in session)
    ↓
Scout agent runs → discovers 7-10 interesting questions
    ↓
Session manager spawns threads for top 5 questions
User sees scout thread + 5 analytical threads in feed
    ↓
Each thread runs independently:
    Coordinator (judge) picks next move → Worker executes SQL → result fed back → loop
    ↓
Thread states:
    RUNNING  → coordinator loop active, steps appearing in feed
    WAITING  → stuck, question surfaced to user, waiting for input
    COMPLETE → finding synthesized, thread done
    ↓
User can:
    - Watch threads update in real time (SSE)
    - Expand a thread to see all steps
    - Reply to a WAITING thread to unstick it
    - Create new threads from remaining scout questions or their own question
```

## Agent architecture

### Profiler (runs once per session)
- Input: raw dataset in DuckDB
- Output: markdown schema summary (column profiles, NULL rates, notable patterns)
- Model: Haiku (cheap, mechanical)
- Cached: yes (same dataset = same profile forever)

### Scout (runs once per session)  
- Input: schema_summary
- Output: 7-10 ranked questions with motivation and entry_point
- Model: Sonnet (needs creativity and domain reasoning)
- Cached: no (want variation)
- Key design: questions must have "stakes" — not descriptive ("what's the distribution") but investigative ("are we missing Earth-like planets because of how we look?")

### Coordinator (one per thread, loops)
- Input: seed_question, schema_summary, thread_history
- Output: CoordinatorDecision (next_move, worker_instruction, status)
- Model: Haiku
- Cached: no (depends on evolving history)
- Sensemaking moves: SCOPE, FORAGE, FRAME, INTERROGATE, SYNTHESIZE
- No fixed order — coordinator chooses based on what's been found
- Three statuses: CONTINUE (keep going), STUCK (ask human), DONE (synthesize)
- No artificial step limits

### Worker (stateless, called by coordinator)
- Input: worker_instruction, schema_summary, available views
- Output: WorkerResult (SQL queries, summary, details)
- Model: Haiku with Sonnet fallback on SQL errors
- Cached: yes (same instruction + schema = same SQL)
- Executes SQL against session's DuckDB instance
- Returns narrative summary, not raw numbers

## Data layer

All DuckDB:
- **Main DB** (`data/main.duckdb`): sessions, threads, steps, llm_cache tables
- **Session DBs** (`data/session_{id}.duckdb`): uploaded dataset + thread views

Tables:
- `sessions`: id, dataset_path, schema_summary, scout_output, created_at
- `threads`: id, session_id, seed_question, motivation, entry_point, status, timestamps
- `steps`: id, thread_id, step_number, move, instruction, result_summary, result_details, view_created, created_at
- `llm_cache`: cache_key (SHA256), model, role, response (JSON), token counts, created_at, ttl_hours

Thread views: workers can create filtered views scoped to their thread (`thread_{id}_{name}`), published via DuckDB MCP extension if available.

## LLM integration

All calls through OpenRouter (OpenAI-compatible API):
- Single `LLMClient` class in `app/core/llm.py`
- Cache backed by DuckDB `llm_cache` table
- Cache key = SHA256(model + messages + tools + temperature)
- Only deterministic calls cached (temperature = 0)
- Retry with model fallback on failure

## API design

### REST endpoints
- `POST /api/sessions` — upload dataset, returns session_id
- `GET /api/sessions/{id}` — session state + scout questions
- `GET /api/sessions/{id}/threads` — all threads with status and latest step
- `POST /api/sessions/{id}/threads` — create user-initiated thread
- `GET /api/threads/{id}` — full thread detail with all steps
- `POST /api/threads/{id}/messages` — human reply to stuck thread

### SSE
- `GET /api/sessions/{id}/events` — real-time event stream
- Event types: `step_completed`, `thread_waiting`, `thread_complete`, `scout_done`

## Concurrency model

- asyncio-based, single process
- Each thread is an async task managed by `core/queue.py`
- DuckDB writes serialized via asyncio.Lock (single-writer constraint)
- DuckDB reads are concurrent (safe)
- Event bus: in-memory dict of session_id → list[asyncio.Queue]

## Deployment

- **Backend**: Railway (Hobby plan, ~$5/month)
- **Frontend**: GitHub Pages (existing React/TS site, new page)
- DuckDB file on Railway persistent volume
- Dockerfile for Railway deployment

## Cost estimates

- Per session (scout + 5 threads × ~8 steps): ~$0.07 at Haiku pricing
- Railway hosting: ~$5-8/month
- Swap to DeepSeek or free models for development: near zero

## Theoretical grounding

The analytical loop is informed by:
- **Shneiderman's Visual Information Seeking Mantra**: overview first, zoom and filter, details on demand → maps to SCOPE and FORAGE moves
- **Pirolli & Card's Sensemaking Loop**: foraging/synthesis cycle → the overall thread lifecycle
- **Klein's Data-Frame Model**: frame a hypothesis, seek confirming/disconfirming evidence → FRAME and INTERROGATE moves
- **Bertin's Levels of Reading**: elementary → intermediate → comprehensive → the natural narrative arc within each thread

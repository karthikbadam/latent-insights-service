# Latent Insights

Parallel-agent sensemaking tool for collaborative data analysis. Upload any dataset — the system discovers questions, spawns analytical threads, and builds insights with you.

## Quick start

### OpenRouter (default)

```bash
uv sync --extra dev
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

## API

| Endpoint | Description |
|---|---|
| `POST /api/datasets/upload` | Upload CSV |
| `GET /api/datasets` | List datasets |
| `POST /api/sessions` | Create session (upload + profile + scout + spawn threads) |
| `GET /api/sessions/{id}` | Session state |
| `GET /api/sessions/{id}/threads` | List threads |
| `POST /api/sessions/{id}/threads` | Create custom thread |
| `GET /api/threads/{id}` | Thread detail with steps |
| `POST /api/threads/{id}/messages` | Reply to stuck thread |
| `GET /api/sessions/{id}/events` | SSE event stream |
| `GET /api/system/tasks` | Active tasks |
| `GET /api/system/stats` | System stats |
| `GET /api/system/cache` | LLM cache stats |

## Docs

- [docs/SPEC.md](docs/SPEC.md) — architecture spec
- [docs/PROMPTS.md](docs/PROMPTS.md) — agent prompt designs

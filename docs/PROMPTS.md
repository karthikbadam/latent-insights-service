# Latent Insights — Prompt, Tool & Infrastructure Design

## Overview

Four prompts in the system, executed in this order:

1. **Profiler prompt** — runs once on upload, produces dataset schema context
2. **Scout prompt** — runs once per session, discovers questions worth asking
3. **Coordinator prompt** — runs per thread, orchestrates the sensemaking loop
4. **Worker prompt** — stateless, executes one analytical step via DuckDB MCP

---

## LLM provider: OpenRouter

All LLM calls route through OpenRouter (`https://openrouter.ai/api/v1`).
OpenAI-compatible — use the standard `openai` Python SDK.

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = await client.chat.completions.create(
    model="anthropic/claude-3.5-haiku",
    messages=[...],
    extra_headers={
        "HTTP-Referer": "https://yoursite.github.io",
        "X-Title": "Latent Insights",
    },
)
```

### Model selection (configurable per role)

| Role              | Default model                | Reasoning                         |
|-------------------|------------------------------|-----------------------------------|
| Profiler          | `anthropic/claude-3.5-haiku` | Mechanical, structured output     |
| Scout             | `anthropic/claude-sonnet-4`  | Needs creativity + domain reason  |
| Coordinator       | `anthropic/claude-3.5-haiku` | Structured decisions, small ctx   |
| Worker            | `anthropic/claude-3.5-haiku` | SQL generation, structured output |
| Worker (fallback) | `anthropic/claude-sonnet-4`  | Retry on SQL errors               |

For prototyping, swap to `deepseek/deepseek-chat-v3-0324` (~$0.27/M in,
$1.10/M out) or `openrouter/free` for zero-cost testing. Model IDs stored
in config, not hardcoded:

```python
MODEL_CONFIG = {
    "profiler":         "anthropic/claude-3.5-haiku",
    "scout":            "anthropic/claude-sonnet-4",
    "coordinator":      "anthropic/claude-3.5-haiku",
    "worker":           "anthropic/claude-3.5-haiku",
    "worker_fallback":  "anthropic/claude-sonnet-4",
}
```

---

## LLM call cache

Every deterministic LLM call is cached in DuckDB to avoid redundant API
calls. Saves cost during development (restart and re-run often) and in
production (parallel threads may ask similar questions about same data).

### Cache table

```sql
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key     VARCHAR PRIMARY KEY,
    model         VARCHAR NOT NULL,
    role          VARCHAR NOT NULL,
    response      JSON NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    created_at    TIMESTAMP DEFAULT current_timestamp,
    ttl_hours     INTEGER DEFAULT 24
);
```

### Cache key computation

```python
import hashlib, json

def compute_cache_key(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
) -> str:
    """Deterministic cache key from the full request signature."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools or [],
        "temperature": temperature,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()
```

### Cache-aware wrapper

```python
async def cached_llm_call(
    db,                          # DuckDB connection
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    role: str,
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    ttl_hours: int = 24,
) -> dict:
    # Skip cache for non-deterministic calls
    if temperature > 0:
        return await _raw_llm_call(client, model, messages, tools, temperature)

    cache_key = compute_cache_key(model, messages, tools, temperature)

    # Check cache
    try:
        row = db.execute("""
            SELECT response FROM llm_cache
            WHERE cache_key = ?
              AND created_at > current_timestamp - INTERVAL (? || ' hours')
        """, [cache_key, ttl_hours]).fetchone()
        if row:
            return json.loads(row[0])
    except Exception:
        pass

    # Cache miss
    response = await _raw_llm_call(client, model, messages, tools, temperature)

    # Store
    usage = response.get("usage", {})
    db.execute("""
        INSERT OR REPLACE INTO llm_cache
        (cache_key, model, role, response, input_tokens, output_tokens, ttl_hours)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        cache_key, model, role, json.dumps(response),
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        ttl_hours,
    ])

    return response
```

### Cache behavior per role

| Role        | Temp | Cached? | TTL    | Why                                  |
|-------------|------|---------|--------|--------------------------------------|
| Profiler    | 0.0  | Yes     | 1 year | Same dataset = same profile always   |
| Scout       | 0.7  | No      | —      | Want creative variation each session  |
| Coordinator | 0.3  | No      | —      | Depends on evolving thread history    |
| Worker      | 0.0  | Yes     | 24h    | Same instruction + schema = same SQL |

### Cache observability

```sql
SELECT role,
       COUNT(*) as cached_calls,
       SUM(input_tokens) as input_tokens_saved,
       SUM(output_tokens) as output_tokens_saved
FROM llm_cache
GROUP BY role;
```

---

## Tool layer: DuckDB MCP

The worker interacts with DuckDB via the `duckdb_mcp` community extension
in server mode. DuckDB exposes query/describe/list tools natively through
MCP — no custom tool definitions needed.

### Setup (on session creation)

```python
import duckdb

def init_session_db(dataset_path: str, session_id: str):
    db = duckdb.connect(f"data/{session_id}.duckdb")

    db.execute("INSTALL duckdb_mcp FROM community")
    db.execute("LOAD duckdb_mcp")

    db.execute(f"""
        CREATE TABLE IF NOT EXISTS dataset AS
        SELECT * FROM read_csv_auto('{dataset_path}')
    """)

    db.execute("SELECT mcp_server_start('stdio')")
    db.execute("""
        SELECT mcp_publish_table('dataset', 'data://tables/dataset', 'json')
    """)

    return db
```

### Tools exposed by DuckDB MCP

Discovered by worker LLM via MCP protocol (not hardcoded in prompt):

- **query** — Read-only SQL execution, returns JSON
- **describe** — Schema for a table or view
- **list_resources** — All available tables and views
- **export** — Query results in various formats

v2.0 enforces read-only by default — destructive SQL blocked at DuckDB.

### Thread view creation

```python
def create_thread_view(db, thread_id: str, view_name: str, sql: str):
    full_name = f"thread_{thread_id}_{view_name}"
    db.execute(f"CREATE OR REPLACE VIEW {full_name} AS {sql}")
    db.execute(f"""
        SELECT mcp_publish_table(
            '{full_name}', 'data://views/{full_name}', 'json'
        )
    """)
```

---

## Prompt 0: Profiler

Runs once per session after upload. Produces compact schema summary
injected into every subsequent prompt as `{schema_summary}`.

**Model:** `MODEL_CONFIG["profiler"]` | **Temp:** 0.0 | **Cached:** Yes

```
You are a dataset profiler. Examine a dataset and produce a concise
schema summary that other analysts will use as their reference.

You are connected to a DuckDB database via MCP. The dataset is in the
`dataset` table. Use the available query and describe tools.

Produce a schema summary in exactly this format:

## Dataset summary
- **Rows:** <count>
- **Columns:** <count>

## Column profiles
For each column, one line:
<column_name> | <type> | <non-null count>/<total> | <summary>

Where <summary> is:
- Numeric: min, max, mean, median, stddev
- Categorical (< 20 unique): all values with counts
- Categorical (≥ 20 unique): top 5 values + total unique count
- Date: min, max, span
- Text: avg length, sample

## Notable patterns
3-5 observations about data quality, skew, sparsity, or structure
that an analyst should know. Examples:
- "82% of pl_orbeccen values are NULL — eccentricity analysis limited"
- "disc_year ranges 1992-2025 but 68% post-2014 (Kepler/TESS era)"
- "pl_bmasse spans 8 orders of magnitude — log scale needed"

Be precise. No filler.
```

---

## Prompt 1: Scout

Runs once per session after profiler. Discovers questions that seed
analytical threads.

**Model:** `MODEL_CONFIG["scout"]` | **Temp:** 0.7 | **Cached:** No

```
You are an analytical scout. You are looking at a dataset for the first
time. Your job is not to answer questions — it is to discover the most
interesting questions this data could answer.

You are connected to a DuckDB database via MCP. The dataset is in the
`dataset` table. Use the available tools to explore.

Here is the dataset schema:
{schema_summary}

## Your approach

1. Run exploratory queries to understand distributions, correlations,
   outliers, and gaps. Look for tensions, surprises, and asymmetries.

2. For each interesting pattern, ask: "Why might this be?" and "What
   would it mean if this weren't true?"

3. Think about what's NOT in the data. Missing values, underrepresented
   categories, and measurement limitations often point to the most
   revealing questions.

## What makes a good question

Good questions have stakes — they imply something we assume might be
wrong, a pattern has an explanation worth finding, or two unrelated
things are connected.

- BAD: "What is the distribution of discovery methods?"
  (Descriptive, no stakes)

- GOOD: "Are we systematically missing Earth-like planets because of
  how we look for them?"
  (Implies catalog bias, leads to detection method analysis)

- BAD: "What are the most common stellar types hosting planets?"
  (Descriptive)

- GOOD: "Do planets form differently around red dwarfs vs Sun-like
  stars, or do they just look different because of detection bias?"
  (Separates physics from observational artifact)

- BAD: "Which planets have the highest eccentricity?"
  (Lookup, not analysis)

- GOOD: "Why do some systems have wildly eccentric orbits while others
  are circular — can we predict which from the host star?"
  (Connects dynamics to stellar properties, testable)

## Domain context (exoplanet-specific)

- Transit detection biased toward large, short-period planets. Radial
  velocity toward massive planets. Direct imaging finds wide-orbit
  giants. Each method has blind spots.

- Habitable zone depends on stellar luminosity. For M-dwarfs very close
  in — raising tidal locking and stellar flare questions.

- Planet radius vs mass reveals composition. Rocky planets cluster below
  ~1.6 Earth radii ("radius gap"). Above → thick atmospheres.

- Multi-planet system spacing and resonances encode dynamical history.

- Kepler (2009-2018) and TESS (2018-present) dominate discovery counts,
  creating temporal and observational biases.

## Output format

Return JSON:

{
  "exploration_notes": "Narrative of what you found (2-3 paragraphs).
    Shown to user as scout thread content.",
  "questions": [
    {
      "question": "Complete sentence question",
      "motivation": "Why interesting (1-2 sentences)",
      "entry_point": "Specific first analytical step",
      "difficulty": "simple | moderate | deep"
    }
  ]
}

Return 7-10 questions ranked by likelihood of non-obvious insights.
Mix: 2-3 simple, 3-4 moderate, 2-3 deep.
```

---

## Prompt 2: Coordinator

Core loop. One per thread. Receives history, decides next move.

**Model:** `MODEL_CONFIG["coordinator"]` | **Temp:** 0.3 | **Cached:** No

```
You are a thread coordinator for a data analysis system. You guide a
single analytical thread from question to insight.

You do NOT run queries. You evaluate worker results and decide what to
investigate next. Think, judge, direct — don't compute.

## Your thread

Question: {seed_question}
Motivation: {motivation}
Suggested entry point: {entry_point}

## Dataset context

{schema_summary}

## Sensemaking moves

**SCOPE** — Define data slice. Narrow to relevant subset. May create view.

**FORAGE** — Exploratory analysis. Distributions, correlations, groups,
  outliers. Cast wide net.

**FRAME** — Propose tentative insight/hypothesis. State as testable claim.

**INTERROGATE** — Stress-test the frame. Contradictions, subgroups,
  confounds, edge cases.

**SYNTHESIZE** — Thread conclusion. Finding, confidence, limitations,
  what would change your mind.

No fixed order. Forage twice before framing. Interrogate then reframe.
Let data guide you.

## Thread history

{thread_history}

## Your decision

Return JSON:

{
  "assessment": "Current state of investigation (2-3 sentences)",
  "next_move": "SCOPE | FORAGE | FRAME | INTERROGATE | SYNTHESIZE",
  "rationale": "Why this move now (1-2 sentences)",
  "worker_instruction": "Specific instruction for worker. Columns,
    filters, expected output. Precise enough for SQL translation.",
  "status": "CONTINUE | STUCK | DONE"
}

### When STUCK

Signs: consecutive forages return similar results, frame flip-flops,
data lacks resolution, need domain expertise to interpret pattern.

Replace worker_instruction with:
{
  "status": "STUCK",
  "question_for_human": "Specific question — 'I found X pattern, is
    this a known detection artifact or real physical effect?'",
  "context": "What found and why stuck (2-3 sentences)"
}

### When DONE

Worker instruction should be SYNTHESIZE producing final summary.

### Quality standards

- Never report stats without context. "76% transit" is trivia.
  "Transit dominates → sample biased toward short periods → patterns
  may reflect how we look, not what exists" is insight.

- Always ask "so what?" — if no answer, not worth reporting.

- Make hypotheses falsifiable.

- If something surprises you, say so. Surprise is signal.
```

### Thread history format

```
Step 1 [SCOPE]:
  Instruction: "Filter to planets with mass and radius measurements"
  Result: View with 1,847 rows (30%). RV overrepresented (42% vs 18%).
  Assessment: Good subset, need to account for method bias.

Step 2 [FORAGE]:
  Instruction: "Compare mass-radius by discovery method"
  Result: Transit clusters 1-4 Re. RV spread 10-1000 Me. Imaging >100Me.
  Assessment: Clear method-dependent selection effects.

[Human input]: "Looked at radius gap by stellar type?"
```

---

## Prompt 3: Worker

Stateless. One instruction → SQL → summary.

**Model:** `MODEL_CONFIG["worker"]` | **Temp:** 0.0 | **Cached:** Yes
**On SQL error:** Retry with `MODEL_CONFIG["worker_fallback"]`

```
You are a data analysis worker. You receive an analytical instruction
and execute it against a DuckDB database.

You are connected to DuckDB via MCP. Use the available tools to query,
describe, and list resources.

## Dataset schema

{schema_summary}

## Available thread views

{thread_views}

## Your instruction

{worker_instruction}

## How to work

1. Plan your query — columns, filters, aggregations.

2. Execute SQL via MCP tools. DuckDB supports CTEs, window functions,
   PERCENTILE_CONT, QUALIFY, HISTOGRAM(), APPROX_QUANTILE, etc.

3. If asked to create a filtered subset, include view definition in
   your response — the backend will create and publish it.

4. Summarize for a non-technical reader. Lead with finding, not method.

## Output format

Return JSON:

{
  "queries_executed": [
    {
      "purpose": "What this query checks",
      "sql": "The SQL",
      "key_results": "Important numbers or patterns"
    }
  ],
  "summary": "2-4 sentence narrative. Lead with most interesting
    finding. Specific numbers, explained.",
  "details": "Methodology notes, NULL caveats, secondary findings.",
  "view_requested": {"name": "...", "sql": "..."} or null
}

## Rules

- Check NULL rates before computing stats.
- Comparing groups → absolute numbers AND effect size.
  "Median 1.8 Re for M-dwarf vs 3.4 Re for G-dwarf (N=412 vs 891)"
- Log scale thinking for mass, radius, period, distance.
- Ambiguous instruction → reasonable interpretation + note assumption.
```

---

## Cost estimates

### Per thread (8 steps)

| Calls          | Count | Input tokens | Output tokens |
|----------------|-------|-------------|---------------|
| Coordinator    | 8     | ~9,600      | ~2,400        |
| Worker         | 8     | ~6,400      | ~3,200        |
| **Total**      | 16    | ~16,000     | ~5,600        |

At Haiku via OpenRouter: **~$0.01/thread**

### Per session

| Component      | Est. cost |
|----------------|-----------|
| Profiler       | ~$0.002   |
| Scout (Sonnet) | ~$0.02    |
| 5 threads      | ~$0.05    |
| **Total**      | **~$0.07**|

With cache: lower during development. With DeepSeek/free: near zero.

---

## Config summary

```python
import os

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_CONFIG = {
    "profiler":         "anthropic/claude-3.5-haiku",
    "scout":            "anthropic/claude-sonnet-4",
    "coordinator":      "anthropic/claude-3.5-haiku",
    "worker":           "anthropic/claude-3.5-haiku",
    "worker_fallback":  "anthropic/claude-sonnet-4",
}

TEMPERATURE_CONFIG = {
    "profiler":    0.0,
    "scout":       0.7,
    "coordinator": 0.3,
    "worker":      0.0,
}

CACHE_TTL_HOURS = {
    "profiler": 8760,
    "worker":   24,
}

DUCKDB_DATA_DIR = "data/"
DATASET_TABLE_NAME = "dataset"
```

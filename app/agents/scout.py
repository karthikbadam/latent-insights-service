"""
Scout agent — discovers interesting questions from the dataset.

Runs once per session after profiler. Output seeds the analytical threads.
Works with any dataset — no domain-specific assumptions.
"""

import logging

from app.core.llm import LLMClient
from app.core.parsing import parse_scout_response
from app.models import ScoutOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an analytical scout. You are looking at a dataset for the first
time. Your job is not to answer questions — it is to discover the most
interesting questions this data could answer.

## Your approach

1. Examine the schema and any exploration results provided. Look for
   distributions, correlations, outliers, and gaps. Don't just describe
   columns — look for tensions, surprises, and asymmetries.

2. For each interesting pattern, ask: "Why might this be?" and "What
   would it mean if this weren't true?"

3. Think about what's NOT in the data. Missing values, underrepresented
   categories, and measurement limitations often point to revealing questions.

## What makes a good question

Good questions have stakes — they imply something we assume might be wrong,
a pattern has an explanation worth finding, or two unrelated things connect.

- BAD: "What is the distribution of X?" (descriptive, no stakes)
- GOOD: "Are we systematically missing a category of X because of how we measure?"
- BAD: "What are the most common values of Y?" (bar chart)
- GOOD: "Do patterns in Y change when controlling for Z, or is it confounded?"

## Output format

Return JSON:

{
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
"""


def _run_exploratory_queries(session_db, table_name: str) -> str:
    """Run generic exploratory queries to give the scout real data context."""
    tbl = f'"{table_name}"'
    explorations = []

    # 1. Basic shape
    try:
        row_count = session_db.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        cols = session_db.execute(f"DESCRIBE {tbl}").fetchall()
        explorations.append(f"### Shape\n{row_count} rows, {len(cols)} columns")
    except Exception as e:
        explorations.append(f"### Shape\nERROR: {e}")
        return "\n\n".join(explorations)

    # 2. Per-column summary: type, null rate, distinct count
    try:
        col_summaries = []
        for col_name, col_type, *_ in cols:
            try:
                stats = session_db.execute(f"""
                    SELECT COUNT("{col_name}") as non_null,
                           COUNT(DISTINCT "{col_name}") as distinct_ct
                    FROM {tbl}
                """).fetchone()
                null_ct = row_count - stats[0]
                col_summaries.append(
                    f"{col_name} | {col_type} | {null_ct} nulls | {stats[1]} distinct"
                )
            except Exception:
                col_summaries.append(f"{col_name} | {col_type} | (stats failed)")
        explorations.append("### Column overview\n" + "\n".join(col_summaries))
    except Exception as e:
        explorations.append(f"### Column overview\nERROR: {e}")

    # 3. Low-cardinality columns: value distributions
    try:
        for col_name, col_type, *_ in cols:
            try:
                distinct = session_db.execute(
                    f'SELECT COUNT(DISTINCT "{col_name}") FROM {tbl} WHERE "{col_name}" IS NOT NULL'
                ).fetchone()[0]
                if 2 <= distinct <= 15:
                    rows = session_db.execute(f"""
                        SELECT "{col_name}", COUNT(*) as cnt
                        FROM {tbl} WHERE "{col_name}" IS NOT NULL
                        GROUP BY "{col_name}" ORDER BY cnt DESC
                    """).fetchall()
                    dist = "; ".join(f"{v}: {c}" for v, c in rows)
                    explorations.append(f"### {col_name} distribution\n{dist}")
            except Exception:
                pass
    except Exception:
        pass

    # 4. Numeric column basic stats
    try:
        numeric_types = {"BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "SMALLINT", "REAL"}
        for col_name, col_type, *_ in cols:
            base_type = col_type.split("(")[0].upper().strip()
            if base_type in numeric_types:
                try:
                    stats = session_db.execute(f"""
                        SELECT MIN("{col_name}"), MAX("{col_name}"),
                               AVG("{col_name}"), MEDIAN("{col_name}")
                        FROM {tbl} WHERE "{col_name}" IS NOT NULL
                    """).fetchone()
                    explorations.append(
                        f"### {col_name} stats\n"
                        f"min={stats[0]}, max={stats[1]}, "
                        f"mean={stats[2]:.4g}, median={stats[3]}"
                    )
                except Exception:
                    pass
    except Exception:
        pass

    return "\n\n".join(explorations)


def run_scout(
    llm: LLMClient,
    model: str,
    schema_summary: str,
    table_name: str = "dataset",
    session_db=None,
) -> ScoutOutput:
    """Run scout and return discovered questions."""
    exploration_data = ""
    if session_db is not None:
        exploration_data = _run_exploratory_queries(session_db, table_name)

    user_content = f"Here is the dataset schema:\n\n{schema_summary}\n\n"
    if exploration_data:
        user_content += (
            f"Here are results from exploratory queries:\n\n{exploration_data}\n\n"
        )
    user_content += (
        "Based on the schema and these exploration results, discover the most "
        "interesting questions this data could answer."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = llm.call(
        model=model,
        messages=messages,
        role="scout",
        temperature=0.7,
    )

    return parse_scout_response(response.content)

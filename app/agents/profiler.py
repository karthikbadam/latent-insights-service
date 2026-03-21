"""
Profiler agent — examines a dataset, produces schema summary.

Runs once per session. Output is injected into every other agent's prompt.
Works with any dataset — no domain-specific assumptions.
"""

import logging

from app.core.llm import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a dataset profiler. Examine a dataset and produce a concise
schema summary that other analysts will use as their reference.

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
- Categorical (>= 20 unique): top 5 values + total unique count
- Date: min, max, span
- Text: avg length, sample

## Notable patterns
3-5 observations about data quality, skew, sparsity, or structure
that an analyst should know. Be precise. No filler.
"""

NUMERIC_TYPES = frozenset({
    "BIGINT", "INTEGER", "SMALLINT", "TINYINT", "HUGEINT",
    "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC",
    "UBIGINT", "UINTEGER", "USMALLINT", "UTINYINT",
})


def _is_numeric(col_type: str) -> bool:
    base = col_type.split("(")[0].upper().strip()
    return base in NUMERIC_TYPES


def _gather_column_stats(session_db, columns_info: list) -> str:
    """Run SQL to gather per-column statistics. Works on any dataset."""
    row_count = session_db.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    if row_count == 0:
        return "(empty dataset)"

    stats_parts = []

    for col_name, col_type, *_ in columns_info:
        try:
            non_null = session_db.execute(
                f'SELECT COUNT("{col_name}") FROM dataset'
            ).fetchone()[0]
            null_pct = ((row_count - non_null) / row_count) * 100

            if _is_numeric(col_type):
                if non_null == 0:
                    summary = "all NULL"
                else:
                    stats = session_db.execute(f"""
                        SELECT MIN("{col_name}"), MAX("{col_name}"),
                               AVG("{col_name}"), MEDIAN("{col_name}"),
                               STDDEV("{col_name}")
                        FROM dataset WHERE "{col_name}" IS NOT NULL
                    """).fetchone()
                    parts = []
                    parts.append(f"min={stats[0]}")
                    parts.append(f"max={stats[1]}")
                    if stats[2] is not None:
                        parts.append(f"mean={stats[2]:.4g}")
                    if stats[3] is not None:
                        parts.append(f"median={stats[3]}")
                    if stats[4] is not None:
                        parts.append(f"stddev={stats[4]:.4g}")
                    summary = ", ".join(parts)
            else:
                unique_count = session_db.execute(
                    f'SELECT COUNT(DISTINCT "{col_name}") FROM dataset WHERE "{col_name}" IS NOT NULL'
                ).fetchone()[0]

                if unique_count == 0:
                    summary = "all NULL"
                elif unique_count <= 20:
                    value_counts = session_db.execute(f"""
                        SELECT "{col_name}", COUNT(*) as cnt
                        FROM dataset WHERE "{col_name}" IS NOT NULL
                        GROUP BY "{col_name}" ORDER BY cnt DESC
                    """).fetchall()
                    summary = "; ".join(f"{v}: {c}" for v, c in value_counts)
                else:
                    top = session_db.execute(f"""
                        SELECT "{col_name}", COUNT(*) as cnt
                        FROM dataset WHERE "{col_name}" IS NOT NULL
                        GROUP BY "{col_name}" ORDER BY cnt DESC LIMIT 5
                    """).fetchall()
                    summary = f"{unique_count} unique. Top: " + "; ".join(
                        f"{v}: {c}" for v, c in top
                    )

            stats_parts.append(
                f"{col_name} | {col_type} | {non_null}/{row_count} ({100 - null_pct:.0f}%) | {summary}"
            )
        except Exception as e:
            logger.warning(f"Stats failed for column {col_name}: {e}")
            stats_parts.append(f"{col_name} | {col_type} | (stats unavailable: {e})")

    return "\n".join(stats_parts)


async def run_profiler(
    llm: LLMClient,
    model: str,
    session_db,
    cache_ttl_hours: int = 8760,
) -> str:
    """Run profiler and return schema summary as markdown string."""
    info = session_db.execute("DESCRIBE dataset").fetchall()
    row_count = session_db.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    col_count = len(info)
    column_stats = _gather_column_stats(session_db, info)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"The dataset has {row_count} rows and {col_count} columns.\n\n"
            f"Here are the detailed column statistics I've gathered:\n\n{column_stats}\n\n"
            "Using these stats, produce the schema summary in the format specified. "
            "Add your observations in the Notable Patterns section."
        )},
    ]

    response = await llm.call(
        model=model,
        messages=messages,
        role="profiler",
        temperature=0.0,
        cache_ttl_hours=cache_ttl_hours,
    )

    return response.content

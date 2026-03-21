"""
Typed query functions — raw SQL in, dataclasses out.

All DB reads/writes go through here. No raw SQL anywhere else.
"""

import json
import uuid
from datetime import datetime

import duckdb

from app.models import MoveType, Session, Step, Thread, ThreadStatus


def generate_id() -> str:
    return uuid.uuid4().hex[:12]


# --- Sessions ---


def create_session(db: duckdb.DuckDBPyConnection, dataset_path: str, table_name: str = "dataset") -> Session:
    session = Session(id=generate_id(), dataset_path=dataset_path, table_name=table_name)
    db.execute(
        "INSERT INTO sessions (id, dataset_path, table_name) VALUES (?, ?, ?)",
        [session.id, session.dataset_path, session.table_name],
    )
    return session


def get_session(db: duckdb.DuckDBPyConnection, session_id: str) -> Session | None:
    row = db.execute(
        "SELECT id, dataset_path, table_name, schema_summary, scout_output, created_at FROM sessions WHERE id = ?",
        [session_id],
    ).fetchone()
    if not row:
        return None
    return Session(
        id=row[0],
        dataset_path=row[1],
        table_name=row[2],
        schema_summary=row[3],
        scout_output=json.loads(row[4]) if row[4] else None,
        created_at=row[5],
    )


def update_session_table_name(db: duckdb.DuckDBPyConnection, session_id: str, table_name: str):
    db.execute(
        "UPDATE sessions SET table_name = ? WHERE id = ?",
        [table_name, session_id],
    )


def update_session_schema(db: duckdb.DuckDBPyConnection, session_id: str, schema_summary: str):
    db.execute(
        "UPDATE sessions SET schema_summary = ? WHERE id = ?",
        [schema_summary, session_id],
    )


def update_session_scout(db: duckdb.DuckDBPyConnection, session_id: str, scout_output: dict):
    db.execute(
        "UPDATE sessions SET scout_output = ? WHERE id = ?",
        [json.dumps(scout_output), session_id],
    )


# --- Threads ---


def create_thread(
    db: duckdb.DuckDBPyConnection,
    session_id: str,
    seed_question: str,
    motivation: str = "",
    entry_point: str = "",
) -> Thread:
    thread = Thread(
        id=generate_id(),
        session_id=session_id,
        seed_question=seed_question,
        motivation=motivation,
        entry_point=entry_point,
    )
    db.execute(
        """INSERT INTO threads (id, session_id, seed_question, motivation, entry_point, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [thread.id, thread.session_id, thread.seed_question,
         thread.motivation, thread.entry_point, thread.status.value],
    )
    return thread


def get_threads(db: duckdb.DuckDBPyConnection, session_id: str) -> list[Thread]:
    rows = db.execute(
        """SELECT id, session_id, seed_question, motivation, entry_point,
                  status, summary, error, created_at, updated_at
           FROM threads WHERE session_id = ? ORDER BY created_at""",
        [session_id],
    ).fetchall()
    return [
        Thread(
            id=r[0], session_id=r[1], seed_question=r[2],
            motivation=r[3], entry_point=r[4],
            status=ThreadStatus(r[5]), summary=r[6], error=r[7],
            created_at=r[8], updated_at=r[9],
        )
        for r in rows
    ]


def get_thread(db: duckdb.DuckDBPyConnection, thread_id: str) -> Thread | None:
    row = db.execute(
        """SELECT id, session_id, seed_question, motivation, entry_point,
                  status, summary, error, created_at, updated_at
           FROM threads WHERE id = ?""",
        [thread_id],
    ).fetchone()
    if not row:
        return None
    return Thread(
        id=row[0], session_id=row[1], seed_question=row[2],
        motivation=row[3], entry_point=row[4],
        status=ThreadStatus(row[5]), summary=row[6], error=row[7],
        created_at=row[8], updated_at=row[9],
    )


def update_thread_status(
    db: duckdb.DuckDBPyConnection,
    thread_id: str,
    status: ThreadStatus,
    summary: str | None = None,
    error: str | None = None,
):
    db.execute(
        "UPDATE threads SET status = ?, summary = COALESCE(?, summary), error = COALESCE(?, error), updated_at = ? WHERE id = ?",
        [status.value, summary, error, datetime.utcnow(), thread_id],
    )


# --- Steps ---


def append_step(
    db: duckdb.DuckDBPyConnection,
    thread_id: str,
    move: MoveType,
    instruction: str,
    result: str,
    view_created: str | None = None,
    duration_ms: int | None = None,
    llm_calls: list[dict] | None = None,
) -> Step:
    # Get next step number
    row = db.execute(
        "SELECT COALESCE(MAX(step_number), 0) FROM steps WHERE thread_id = ?",
        [thread_id],
    ).fetchone()
    step_number = row[0] + 1

    step = Step(
        id=generate_id(),
        thread_id=thread_id,
        step_number=step_number,
        move=move,
        instruction=instruction,
        result=result,
        view_created=view_created,
        duration_ms=duration_ms,
        llm_calls=llm_calls,
    )
    db.execute(
        """INSERT INTO steps (id, thread_id, step_number, move, instruction,
                             result, view_created, duration_ms, llm_calls)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [step.id, step.thread_id, step.step_number, step.move.value,
         step.instruction, step.result, step.view_created, step.duration_ms,
         json.dumps(step.llm_calls) if step.llm_calls else None],
    )
    # Touch thread updated_at
    db.execute(
        "UPDATE threads SET updated_at = ? WHERE id = ?",
        [datetime.utcnow(), thread_id],
    )
    return step


def get_steps(db: duckdb.DuckDBPyConnection, thread_id: str) -> list[Step]:
    rows = db.execute(
        """SELECT id, thread_id, step_number, move, instruction,
                  result, view_created, duration_ms, llm_calls, created_at
           FROM steps WHERE thread_id = ? ORDER BY step_number""",
        [thread_id],
    ).fetchall()
    return [
        Step(
            id=r[0], thread_id=r[1], step_number=r[2],
            move=MoveType(r[3]), instruction=r[4],
            result=r[5], view_created=r[6], duration_ms=r[7],
            llm_calls=json.loads(r[8]) if r[8] else None,
            created_at=r[9],
        )
        for r in rows
    ]


def format_thread_history(steps: list[Step], human_messages: list[str] | None = None) -> str:
    """Format step history for injection into coordinator prompt."""
    parts = []
    for step in steps:
        parts.append(
            f"Step {step.step_number} [{step.move.value}]:\n"
            f"  Instruction: \"{step.instruction}\"\n"
            f"  Result: {step.result}"
        )
    if human_messages:
        for msg in human_messages:
            parts.append(f"[Human input]: \"{msg}\"")
    return "\n\n".join(parts) if parts else "(No steps yet — this is the first move)"

"""
Schema — CREATE TABLE statements for main database.
"""

import duckdb


def create_tables(db: duckdb.DuckDBPyConnection):
    """Create all tables if they don't exist."""

    db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              VARCHAR PRIMARY KEY,
            dataset_path    VARCHAR NOT NULL,
            table_name      VARCHAR NOT NULL DEFAULT 'dataset',
            schema_summary  VARCHAR,
            scout_output    JSON,
            created_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            id              VARCHAR PRIMARY KEY,
            session_id      VARCHAR NOT NULL,
            seed_question   VARCHAR NOT NULL,
            motivation      VARCHAR,
            entry_point     VARCHAR,
            status          VARCHAR DEFAULT 'running',
            summary         VARCHAR,
            error           VARCHAR,
            created_at      TIMESTAMP DEFAULT current_timestamp,
            updated_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            id              VARCHAR PRIMARY KEY,
            thread_id       VARCHAR NOT NULL,
            step_number     INTEGER NOT NULL,
            move            VARCHAR NOT NULL,
            instruction     VARCHAR NOT NULL,
            result          VARCHAR,
            view_created    VARCHAR,
            duration_ms     INTEGER,
            llm_calls       JSON,
            created_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key       VARCHAR PRIMARY KEY,
            model           VARCHAR NOT NULL,
            role            VARCHAR NOT NULL,
            response        JSON NOT NULL,
            input_tokens    INTEGER DEFAULT 0,
            output_tokens   INTEGER DEFAULT 0,
            created_at      TIMESTAMP DEFAULT current_timestamp,
            ttl_hours       INTEGER DEFAULT 24
        )
    """)

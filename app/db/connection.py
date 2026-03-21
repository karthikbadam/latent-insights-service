"""
DuckDB connection management.

Single connection per session with write lock managed by Queue.
"""

import logging
import os

import duckdb

from app.db.schema import create_tables

logger = logging.getLogger(__name__)


class Database:
    """
    Manages DuckDB connections.

    For the hobby project: one main DB for session/thread state + cache,
    and one DB per session for dataset analysis.
    """

    def __init__(self, data_dir: str = "data"):
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._main_db: duckdb.DuckDBPyConnection | None = None

    def get_main_db(self) -> duckdb.DuckDBPyConnection:
        """Get or create the main database (state + cache)."""
        if self._main_db is None:
            path = os.path.join(self._data_dir, "main.duckdb")
            self._main_db = duckdb.connect(path)
            create_tables(self._main_db)
            logger.info(f"Main DB initialized: {path}")
        return self._main_db

    def create_session_db(self, session_id: str, dataset_path: str) -> duckdb.DuckDBPyConnection:
        """Create a per-session DB with the uploaded dataset loaded."""
        path = os.path.join(self._data_dir, f"session_{session_id}.duckdb")
        db = duckdb.connect(path)
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS dataset AS
            SELECT * FROM read_csv_auto('{dataset_path}')
        """)
        logger.info(f"Session DB created: {path} ({dataset_path})")
        return db

    def close(self):
        if self._main_db:
            self._main_db.close()
            self._main_db = None

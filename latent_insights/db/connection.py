"""
DuckDB connection management.

DuckDB is used only as a read-only dataset query engine.
One DB file per session with the uploaded dataset loaded.
"""

import logging
import os
import re

import duckdb

logger = logging.getLogger(__name__)


def table_name_from_path(dataset_path: str) -> str:
    """Derive a valid DuckDB table name from a file path.

    'data/uploads/exoplanets-nasa.csv' -> 'exoplanets_nasa'
    """
    stem = os.path.splitext(os.path.basename(dataset_path))[0]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", stem).strip("_").lower()
    if name and name[0].isdigit():
        name = f"t_{name}"
    return name or "data"


class Database:
    """
    Manages per-session DuckDB connections for dataset analysis.
    """

    def __init__(self, data_dir: str = "data"):
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def create_session_db(
        self, session_id: str, dataset_path: str, table_name: str | None = None,
    ) -> tuple[duckdb.DuckDBPyConnection, str]:
        """Create a per-session DB with the uploaded dataset loaded.

        Returns (connection, table_name).
        """
        if table_name is None:
            table_name = table_name_from_path(dataset_path)
        path = os.path.join(self._data_dir, f"session_{session_id}.duckdb")
        db = duckdb.connect(path)
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" AS
            SELECT * FROM read_csv_auto('{dataset_path}')
        """)
        logger.info(f"Session DB created: {path} table={table_name} ({dataset_path})")
        return db, table_name

    def open_session_connection(self, session_id: str) -> duckdb.DuckDBPyConnection:
        """Open a new read-only connection to an existing session DB."""
        path = os.path.join(self._data_dir, f"session_{session_id}.duckdb")
        db = duckdb.connect(path, read_only=True)
        logger.debug(f"Opened read connection: {path}")
        return db

    def close(self):
        pass

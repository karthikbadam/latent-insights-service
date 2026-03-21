"""
DuckDB MCP extension setup.

Manages the MCP server mode for exposing tables/views to worker agents.
This is optional — workers can also use direct SQL via the connection.
"""

import logging

import duckdb

logger = logging.getLogger(__name__)


def setup_mcp_server(db: duckdb.DuckDBPyConnection):
    """Install and start MCP server on a DuckDB connection."""
    try:
        db.execute("INSTALL duckdb_mcp FROM community")
        db.execute("LOAD duckdb_mcp")
        db.execute("SELECT mcp_server_start('stdio')")
        logger.info("DuckDB MCP server started")
    except Exception as e:
        logger.warning(f"MCP setup failed (will use direct SQL): {e}")


def publish_table(db: duckdb.DuckDBPyConnection, table_name: str):
    """Publish a table as an MCP resource."""
    try:
        db.execute(f"""
            SELECT mcp_publish_table('{table_name}', 'data://tables/{table_name}', 'json')
        """)
    except Exception as e:
        logger.warning(f"MCP publish failed for {table_name}: {e}")


def publish_view(db: duckdb.DuckDBPyConnection, view_name: str):
    """Publish a view as an MCP resource."""
    try:
        db.execute(f"""
            SELECT mcp_publish_table('{view_name}', 'data://views/{view_name}', 'json')
        """)
    except Exception as e:
        logger.warning(f"MCP publish failed for {view_name}: {e}")


def create_thread_view(
    db: duckdb.DuckDBPyConnection,
    thread_id: str,
    view_name: str,
    sql: str,
):
    """Create a thread-scoped view and publish it via MCP."""
    full_name = f"thread_{thread_id}_{view_name}"
    db.execute(f"CREATE OR REPLACE VIEW {full_name} AS {sql}")
    publish_view(db, full_name)
    logger.info(f"Thread view created: {full_name}")
    return full_name

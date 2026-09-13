"""
src/database/connection.py
===========================
Database connection management for SignalScope.

Default backend: SQLite (local file, zero infrastructure).
The `DatabaseManager` class provides a clean abstraction so the rest of
the application never constructs SQL directly, and so the backend can be
replaced with PostgreSQL without touching callers.

Image binary data is NEVER stored here.
Only image hashes, paths, metadata, and analysis results are persisted.

Design notes:
  - Uses Python stdlib `sqlite3` only — no heavyweight ORM dependency.
  - All queries use parameterised placeholders (%s / ?) to prevent SQL injection.
  - Transactions are explicit: use `db.begin()` / `db.commit()` / `db.rollback()`
    or the context manager `with db.transaction():`.
  - Thread-safe: each call to `get_db()` returns a thread-local connection.
"""
from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

logger = logging.getLogger("signalscope.database")

# Thread-local storage for connections
_local = threading.local()

# Default DB path (can be overridden in config or tests)
_DEFAULT_DB_PATH: Optional[Path] = None


def set_db_path(path: Path) -> None:
    """Override the database path (used in tests and config-driven init)."""
    global _DEFAULT_DB_PATH
    _DEFAULT_DB_PATH = Path(path)
    # Clear any existing thread-local connection
    if hasattr(_local, "connection"):
        try:
            _local.connection.close()
        except Exception:
            pass
        del _local.connection


def _resolve_db_path() -> Path:
    """Resolve the database path from config or default."""
    if _DEFAULT_DB_PATH is not None:
        return _DEFAULT_DB_PATH
    # Try project config
    try:
        from src.utils import load_config
        cfg = load_config()
        db_path = cfg.get("database", {}).get("path", "data/signalscope.db")
    except Exception:
        db_path = "data/signalscope.db"

    # Resolve relative to project root
    project_root = Path(__file__).resolve().parents[2]
    return project_root / db_path


class DatabaseManager:
    """
    Lightweight SQLite database manager.

    Usage:
        db = DatabaseManager(db_path)
        db.init()
        with db.transaction():
            db.execute("INSERT INTO ...", (val,))

    Image binary data is NEVER stored — only hashes, paths, and results.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._connection: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """Open (or return existing) SQLite connection."""
        if self._connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                timeout=30,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent reads
            conn.execute("PRAGMA foreign_keys=ON")    # Enforce FK constraints
            conn.execute("PRAGMA synchronous=NORMAL")
            self._connection = conn
            logger.debug(f"Database connected: {self.db_path}")
        return self._connection

    def close(self) -> None:
        """Close the connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def init(self) -> None:
        """Initialize schema (creates tables if they don't exist)."""
        from src.database.schema import init_db
        init_db(self)

    def execute(
        self,
        sql: str,
        params: Iterable[Any] = (),
    ) -> sqlite3.Cursor:
        """Execute a single parameterised SQL statement."""
        conn = self.connect()
        try:
            return conn.execute(sql, params)
        except sqlite3.Error as e:
            logger.error(f"SQL error: {e}\nSQL: {sql}\nParams: {params}")
            raise

    def executemany(self, sql: str, params_list: list[Iterable[Any]]) -> None:
        """Execute a parameterised statement for multiple rows."""
        conn = self.connect()
        conn.executemany(sql, params_list)

    def fetchone(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        return self.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.execute(sql, params).fetchall()

    def last_insert_id(self) -> int:
        row = self.fetchone("SELECT last_insert_rowid() AS id")
        return row["id"] if row else -1

    def commit(self) -> None:
        if self._connection:
            self._connection.commit()

    def rollback(self) -> None:
        if self._connection:
            self._connection.rollback()

    @contextlib.contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Context manager for explicit transactions."""
        try:
            yield
            self.commit()
        except Exception:
            self.rollback()
            raise

    def __enter__(self) -> "DatabaseManager":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# Module-level singleton (one per process, thread-safe for SQLite WAL mode)
_db_instance: Optional[DatabaseManager] = None
_db_lock = threading.Lock()


def get_db() -> DatabaseManager:
    """Return the global DatabaseManager singleton, creating it if needed."""
    global _db_instance
    with _db_lock:
        if _db_instance is None or _db_instance._connection is None:
            path = _resolve_db_path()
            _db_instance = DatabaseManager(path)
            _db_instance.connect()
        return _db_instance

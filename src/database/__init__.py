"""
src/database/__init__.py
========================
SignalScope database layer.

Provides:
  - SQLite-backed persistence (dev/SIH default)
  - Clean abstraction so PostgreSQL can be dropped in later
  - No image binary storage — only hashes, paths, metadata, results

Usage:
    from src.database import get_db, init_db
    db = get_db()
    db.init()
"""
from src.database.connection import get_db, DatabaseManager
from src.database.schema import init_db, SCHEMA_VERSION
from src.database.repositories import (
    insert_calibration_record,
    get_calibration_record,
    update_analysis_reliability,
)

__all__ = [
    "get_db", "init_db", "DatabaseManager", "SCHEMA_VERSION",
    "insert_calibration_record", "get_calibration_record",
    "update_analysis_reliability",
]

"""
Logs every API prediction to SQLite for drift monitoring.
Designed to never crash the API — all errors are silently swallowed.
"""
import sqlite3
import os
from datetime import datetime

HERE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "predictions.db")

_CREATE = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    status          TEXT,
    result_code     TEXT,
    is_valid        INTEGER,
    processing_ms   REAL,
    appearance_label TEXT,
    gender          TEXT,
    source          TEXT DEFAULT 'api'
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);
"""


def _conn():
    os.makedirs(HERE, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript(_CREATE)
    return c


def log(status: str, result_code: str, is_valid: bool,
        processing_ms: float, appearance_label: str = "",
        gender: str = "", source: str = "api") -> None:
    try:
        c = _conn()
        c.execute(
            "INSERT INTO predictions "
            "(timestamp,status,result_code,is_valid,processing_ms,appearance_label,gender,source) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(),
             status, result_code, int(bool(is_valid)),
             processing_ms, appearance_label, gender, source)
        )
        c.commit()
        c.close()
    except Exception:
        pass


def recent(n: int = 200):
    """Return last n predictions as list of dicts."""
    try:
        c = _conn()
        rows = c.execute(
            "SELECT timestamp,status,result_code,is_valid,processing_ms "
            "FROM predictions ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        c.close()
        return [
            {"timestamp": r[0], "status": r[1], "result_code": r[2],
             "is_valid": bool(r[3]), "processing_ms": r[4]}
            for r in rows
        ]
    except Exception:
        return []


def total_count() -> int:
    try:
        c = _conn()
        n = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        c.close()
        return n
    except Exception:
        return 0

"""Persistent history of analyses, backed by SQLite.

Kept deliberately small: one table, no ORM, no migrations framework. The full
result JSON is stored alongside indexed summary columns so the list view is a
cheap query and re-opening a past run needs no re-scoring.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from ..config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id           TEXT PRIMARY KEY,
    created_at   REAL NOT NULL,
    label        TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL DEFAULT 'paste',
    preview      TEXT NOT NULL DEFAULT '',
    words        INTEGER NOT NULL DEFAULT 0,
    score        REAL NOT NULL DEFAULT 0,
    probability  REAL NOT NULL DEFAULT 0,
    threshold    REAL NOT NULL DEFAULT 0,
    flagged      INTEGER NOT NULL DEFAULT 0,
    verdict      TEXT NOT NULL DEFAULT '',
    stratum      TEXT NOT NULL DEFAULT 'overall',
    fpr          TEXT NOT NULL DEFAULT 'fpr_0.01',
    payload      TEXT NOT NULL,
    text         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_flagged ON analyses(flagged);
"""


class HistoryStore:
    """Thread-safe wrapper over a single SQLite file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or settings.history_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # ----------------------------------------------------------------- write
    def add(
        self,
        result: dict[str, Any],
        text: str,
        label: str = "",
        source: str = "paste",
        keep_text: bool = True,
    ) -> str:
        decision = result["decision"]
        stats = result.get("statistics", {})
        record_id = uuid.uuid4().hex[:16]
        preview = " ".join(text.split())[:280]

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    id, created_at, label, source, preview, words, score,
                    probability, threshold, flagged, verdict, stratum, fpr,
                    payload, text
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record_id,
                    time.time(),
                    label or preview[:60],
                    source,
                    preview,
                    int(stats.get("words", 0)),
                    float(decision["score"]),
                    float(decision["probability"]),
                    float(decision["threshold"]),
                    1 if decision["flagged"] else 0,
                    decision["verdict"],
                    decision["stratum"],
                    decision["fpr"],
                    json.dumps(result, separators=(",", ":")),
                    text if keep_text else "",
                ),
            )
            self._trim(conn)
        return record_id

    def _trim(self, conn: sqlite3.Connection) -> None:
        """Keep the table bounded so the file cannot grow without limit."""
        conn.execute(
            """
            DELETE FROM analyses WHERE id IN (
                SELECT id FROM analyses ORDER BY created_at DESC LIMIT -1 OFFSET ?
            )
            """,
            (settings.history_limit,),
        )

    # ------------------------------------------------------------------ read
    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
        flagged_only: bool = False,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = []
        if query:
            clauses.append("(preview LIKE ? OR label LIKE ?)")
            params += [f"%{query}%", f"%{query}%"]
        if flagged_only:
            clauses.append("flagged = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        conn = self._connect()
        total = conn.execute(
            f"SELECT COUNT(*) FROM analyses {where}", params
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT id, created_at, label, source, preview, words, score,
                   probability, threshold, flagged, verdict, stratum, fpr
            FROM analyses {where}
            ORDER BY created_at DESC LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
        return {
            "total": total,
            "items": [dict(r) for r in rows],
            "limit": limit,
            "offset": offset,
        }

    def get(self, record_id: str) -> dict[str, Any] | None:
        row = self._connect().execute(
            "SELECT * FROM analyses WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["payload"] = json.loads(record["payload"])
        return record

    def stats(self) -> dict[str, Any]:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(flagged), 0) AS flagged,
                   COALESCE(AVG(probability), 0) AS avg_probability,
                   COALESCE(SUM(words), 0) AS words
            FROM analyses
            """
        ).fetchone()
        by_verdict = conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM analyses GROUP BY verdict"
        ).fetchall()
        return {
            **dict(row),
            "by_verdict": {r["verdict"]: r["n"] for r in by_verdict},
        }

    # ---------------------------------------------------------------- delete
    def delete(self, record_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM analyses WHERE id = ?", (record_id,))
        return cur.rowcount > 0

    def clear(self) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM analyses")
        return cur.rowcount

    def export_rows(self) -> Iterable[dict[str, Any]]:
        for row in self._connect().execute(
            "SELECT id, created_at, label, source, words, score, probability, "
            "threshold, flagged, verdict, stratum, fpr, preview "
            "FROM analyses ORDER BY created_at DESC"
        ):
            yield dict(row)


history = HistoryStore()

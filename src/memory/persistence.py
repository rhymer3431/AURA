from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteMemoryPersistence:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
                    payload_json TEXT NOT NULL
                )
                """
            )

    def save_snapshot(self, kind: str, payload: dict[str, Any]) -> int:
        self.initialize()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO memory_snapshots(kind, payload_json) VALUES (?, ?)",
                (str(kind), json.dumps(payload, ensure_ascii=True)),
            )
            return int(cursor.lastrowid)

    def load_latest_snapshot(self, kind: str) -> dict[str, Any] | None:
        self.initialize()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload_json FROM memory_snapshots WHERE kind = ? ORDER BY snapshot_id DESC LIMIT 1",
                (str(kind),),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row[0]))

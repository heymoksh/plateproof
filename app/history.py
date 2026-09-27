"""Local SQLite record of previously analysed images, used for reuse detection.

Only hashes and a few identifiers are stored, never the image itself.
Near-duplicate search is a linear scan, which is fine for thousands of
records; a larger deployment would add a BK-tree or LSH index.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from .hashing import from_hex, hamming

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      TEXT NOT NULL,
    order_id     TEXT,
    customer_id  TEXT,
    filename     TEXT,
    sha256       TEXT NOT NULL,
    phash        TEXT NOT NULL,
    dhash        TEXT NOT NULL,
    submitted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submissions_sha256 ON submissions (sha256);
"""


class HistoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Upgrade databases created by earlier versions (claim_id -> order_id)."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(submissions)")}
        if "claim_id" in columns and "order_id" not in columns:
            conn.execute("ALTER TABLE submissions RENAME COLUMN claim_id TO order_id")
        if "customer_id" not in columns:
            conn.execute("ALTER TABLE submissions ADD COLUMN customer_id TEXT")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]

    def find_matches(
        self,
        sha256: str,
        hashes: list[tuple[int, int, str]],
        max_phash: int,
        max_dhash: int,
        include_near: bool = True,
        limit: int = 10,
    ) -> list[dict]:
        """Return previous submissions that are exact or perceptual matches.

        `hashes` is a list of (phash, dhash, label) variants of the query image,
        e.g. the image itself and its mirror image.
        """
        matches: list[dict] = []
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM submissions ORDER BY id DESC").fetchall()
        for row in rows:
            record = {k: row[k] for k in ("case_id", "order_id", "customer_id", "filename", "submitted_at")}
            if row["sha256"] == sha256:
                matches.append({**record, "match": "exact", "phash_distance": 0, "dhash_distance": 0})
                continue
            if not include_near:
                continue
            stored_p, stored_d = from_hex(row["phash"]), from_hex(row["dhash"])
            best = None
            for p, d, label in hashes:
                dp, dd = hamming(p, stored_p), hamming(d, stored_d)
                if dp <= max_phash and dd <= max_dhash and (best is None or dp + dd < best[0] + best[1]):
                    best = (dp, dd, label)
            if best:
                matches.append({**record, "match": best[2], "phash_distance": best[0], "dhash_distance": best[1]})
        order = {"exact": 0, "near": 1, "mirrored": 2}
        matches.sort(key=lambda m: (order.get(m["match"], 3), m["phash_distance"]))
        return matches[:limit]

    def exists(self, sha256: str, order_id: str | None) -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT 1 FROM submissions WHERE sha256 = ? AND order_id IS ? LIMIT 1", (sha256, order_id)
            ).fetchone()
        return row is not None

    def add(self, case_id: str, order_id: str | None, filename: str | None,
            sha256: str, phash: str, dhash: str, submitted_at: str, customer_id: str | None = None) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO submissions (case_id, order_id, customer_id, filename, sha256, phash, dhash, submitted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (case_id, order_id, customer_id, filename, sha256, phash, dhash, submitted_at),
            )

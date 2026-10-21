from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Scope:
    user: str
    permissions: tuple[str, ...]
    version: str
    entity: str
    date: str

    def key(self) -> str:
        values = (self.user, self.version, self.entity, self.date)
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("Scope fields must be nonempty strings")
        if not isinstance(self.permissions, tuple) or any(
            not isinstance(p, str) or not p.strip() for p in self.permissions
        ):
            raise ValueError("Permissions must be a tuple of nonempty strings")
        return json.dumps([*values, sorted(set(self.permissions))])


@dataclass(frozen=True)
class Hit:
    answer: str
    kind: str
    score: float
    query: str


class Cache:
    """One SQLite connection per instance; transactions coordinate independent instances."""

    def __init__(
        self,
        path: str = ":memory:",
        *,
        model: str,
        dimensions: int,
        threshold: float = 0.9,
        clock: Callable[[], float] = time.time,
    ):
        if not model or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("A model revision and positive dimensions are required")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be finite and between zero and one")
        self.model, self.dimensions, self.threshold = model, dimensions, threshold
        self.clock = clock
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False, isolation_level=None)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS epoch (id INTEGER PRIMARY KEY, n INTEGER NOT NULL);"
            "INSERT OR IGNORE INTO epoch VALUES (1, 0);"
            "CREATE TABLE IF NOT EXISTS entries ("
            "scope TEXT, user TEXT, version TEXT, model TEXT, dimensions INTEGER,"
            "query TEXT, answer TEXT, vector TEXT, expires REAL,"
            "PRIMARY KEY(scope, model, dimensions, query));"
        )

    def close(self) -> None:
        with self.lock:
            self.db.close()

    def _vector(self, vector: Iterable[float]) -> tuple[float, ...]:
        result = tuple(float(x) for x in vector)
        if len(result) != self.dimensions or any(not math.isfinite(x) for x in result):
            raise ValueError("Embedding dimension or finite-value violation")
        norm = math.sqrt(sum(x * x for x in result))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Embedding must have a finite nonzero norm")
        return tuple(x / norm for x in result)

    @staticmethod
    def _query(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonempty string")

    def ticket(self) -> int:
        """Capture before generation; invalidation rejects all older in-flight writes."""
        with self.lock:
            return self.db.execute("SELECT n FROM epoch WHERE id=1").fetchone()[0]

    def put(
        self, scope: Scope, query: str, answer: str, vector: Iterable[float],
        *, ttl: float, ticket: int,
    ) -> bool:
        key = scope.key()
        self._query(query)
        encoded = json.dumps(self._vector(vector))
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must be a nonempty string")
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("TTL must be finite and positive")
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                if ticket != self.ticket():
                    self.db.execute("ROLLBACK")
                    return False
                self.db.execute(
                    "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, scope.user, scope.version, self.model, self.dimensions,
                     query, answer, encoded, self.clock() + ttl),
                )
                self.db.execute("COMMIT")
                return True
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def lookup(
        self, scope: Scope, query: str, vector: Iterable[float] | None = None,
    ) -> Hit | None:
        key = scope.key()
        self._query(query)
        normalized = None if vector is None else self._vector(vector)
        with self.lock:
            rows = self.db.execute(
                "SELECT query, answer, vector FROM entries WHERE scope=? AND model=? "
                "AND dimensions=? AND expires>? ORDER BY query",
                (key, self.model, self.dimensions, self.clock()),
            ).fetchall()
        for saved_query, answer, _ in rows:
            if saved_query == query:
                return Hit(answer, "exact", 1.0, saved_query)
        if normalized is None:
            return None
        best = None
        for saved_query, answer, encoded in rows:
            score = sum(a * b for a, b in zip(normalized, json.loads(encoded), strict=True))
            if score >= self.threshold and (best is None or score > best.score):
                best = Hit(answer, "semantic", score, saved_query)
        return best

    def invalidate(self, *, user: str | None = None, version: str | None = None) -> int:
        if user is None and version is None:
            raise ValueError("Select a user or version explicitly")
        clauses, values = [], []
        for field, value in (("user", user), ("version", version)):
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError("Invalid invalidation selector")
                clauses.append(f"{field}=?")
                values.append(value)
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                count = self.db.execute(
                    "DELETE FROM entries WHERE " + " AND ".join(clauses), values
                ).rowcount
                self.db.execute("UPDATE epoch SET n=n+1 WHERE id=1")
                self.db.execute("COMMIT")
                return count
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def prune(self) -> int:
        with self.lock:
            return self.db.execute("DELETE FROM entries WHERE expires<=?", (self.clock(),)).rowcount

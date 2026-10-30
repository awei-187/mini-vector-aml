"""AML-only durable request ledger and atomic ingestion; no benchmark runner imports."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3

from .memory import SQLiteVectorStore, _now, _scope_sql
from .vector_scan import VectorPool

# Decoded float64 pools for the users being ingested concurrently. The service is
# the database's only writer (see _DatabaseLease), so pools stay valid until this
# process changes them and are extended in place after each committed Add.
VECTOR_CACHE_BYTES = 1536 * 1024 * 1024


class DatabaseInUse(RuntimeError):
    pass


class _DatabaseLease:
    """One server process per database, including on Windows."""

    def __init__(self, path: Path):
        self.handle = path.open("a+b")
        try:
            self.handle.seek(0, 2)
            if self.handle.tell() == 0:
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise DatabaseInUse("AML database is already open by another service") from None

    def close(self):
        self.handle.close()  # OS releases the advisory lock, also after process death.


class _DeferredConnection:
    """Keep legacy store context managers from committing an outer transaction."""

    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def commit(self):
        pass


class AMLStore(SQLiteVectorStore):
    def __init__(self, path: str, identity: dict, *, vector_cache_bytes: int = VECTOR_CACHE_BYTES):
        self.identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        db_path = Path(path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lease = _DatabaseLease(db_path.with_name(db_path.name + ".lock"))
        self.db = None
        self._pools = OrderedDict()
        self._scope_versions = {}
        self._pending = []
        self._atomic = False
        self._pool_bytes = vector_cache_bytes
        try:
            # Inspect before SQLiteVectorStore can change an existing database.
            if db_path.exists() and db_path.stat().st_size:
                with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as existing:
                    try:
                        row = existing.execute("SELECT identity FROM aml_config WHERE singleton=1").fetchone()
                    except sqlite3.Error:
                        raise ValueError("Refusing a database without an AML configuration marker") from None
                    stored_identity = json.loads(row[0]) if row is not None else {}
                    if "extraction_prompt" in identity:
                        # A missing field in an older database means the original prompt.
                        stored_identity.setdefault("extraction_prompt", "original")
                    normalized = json.dumps(stored_identity, sort_keys=True, separators=(",", ":"))
                    if row is None or normalized != self.identity:
                        raise ValueError("AML database configuration does not match this service")
            super().__init__(str(db_path))
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS aml_config (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    identity TEXT NOT NULL, dimension INTEGER
                );
                CREATE TABLE IF NOT EXISTS aml_requests (
                    request_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL,
                    response TEXT NOT NULL, completed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS aml_context (
                    session_scope TEXT PRIMARY KEY, messages TEXT NOT NULL
                );
            """)
            self.db.execute("INSERT OR IGNORE INTO aml_config(singleton, identity) VALUES (1, ?)",
                            (self.identity,))
            self.db.commit()
        except BaseException:
            if self.db is not None:
                self.db.close()
            self.lease.close()
            raise

    @contextmanager
    def atomic(self):
        # Hold the store lock only for the write phase; provider calls run before it.
        with self.lock:
            connection = self.db
            connection.execute("BEGIN IMMEDIATE")
            self.db = _DeferredConnection(connection)
            self._atomic = True
            try:
                yield
                connection.commit()
            except BaseException:
                connection.rollback()
                self._pending.clear()
                raise
            finally:
                self.db = connection
                self._atomic = False
            self._apply_pending()

    def check_dimension(self, dimension: int, *, persist: bool = False):
        """Reject a changed dimension; record the first one only together with a write.

        Provider-call validation (persist=False) never writes, so a rolled-back or
        read-only request leaves the database unconfigured.
        """
        with self.lock, self.db:
            current = self.db.execute("SELECT dimension FROM aml_config WHERE singleton=1").fetchone()[0]
            if current is None:
                if persist:
                    self.db.execute("UPDATE aml_config SET dimension=? WHERE singleton=1", (dimension,))
            elif current != dimension:
                raise ValueError("Embedding dimension changed for this AML database")

    def save(self, memory_id, text, vector, metadata, queries=()):
        with self.lock:
            self.check_dimension(len(vector), persist=True)
            super().save(memory_id, text, vector, metadata, queries)
            self._pending.append((metadata, memory_id, vector, [v for _, v in queries]))
            if not self._atomic:
                self._apply_pending()

    @staticmethod
    def _matches(scope_items, metadata):
        return all(metadata.get(name) == value for name, value in scope_items)

    def _apply_pending(self):
        """Extend cached pools of matching scopes with committed rows, in commit order."""
        pending, self._pending = self._pending, []
        for metadata, memory_id, vector, query_vectors in pending:
            for scope_items in self._scope_versions:
                if self._matches(scope_items, metadata):
                    self._scope_versions[scope_items] += 1
            for key in list(self._pools):
                scope_items, questions = key
                if not self._matches(scope_items, metadata):
                    continue
                rows = ([(memory_id, v) for v in query_vectors] if questions else [(memory_id, vector)])
                if not rows:
                    continue
                try:
                    self._pools[key].extend(rows)
                except Exception:
                    del self._pools[key]  # Rebuilt from SQLite on next use.
        self._trim_pools()

    def _trim_pools(self, keep=None):
        """Evict least recently used pools until the byte budget holds, sparing keep."""
        total = sum(pool.nbytes for pool in self._pools.values())
        for key in list(self._pools):
            if total <= self._pool_bytes:
                break
            if key != keep:
                total -= self._pools.pop(key).nbytes

    def _vector_pool(self, scope, *, questions):
        """Return a cached pool, or decode one from SQLite without holding the lock.

        Decoding thousands of JSON vectors takes seconds for a large user; doing it
        outside the lock keeps other users' reads and commits flowing. The result is
        cached only if no commit touched this scope meanwhile; otherwise it is used
        once as a consistent snapshot of the rows fetched.
        """
        key = (tuple(sorted(scope.items())), questions)
        clause, args = _scope_sql(scope)
        if questions:
            sql = f"""SELECT q.memory_id,q.embedding FROM memory_queries q
                JOIN memories m ON m.id=q.memory_id WHERE {clause} ORDER BY m.rowid,q.position"""
        else:
            sql = f"SELECT id,embedding FROM memories WHERE {clause} ORDER BY rowid"
        with self.lock:
            if self._atomic:
                # Uncommitted rows: never cache them, and keep the transaction's lock.
                return VectorPool(self.db.execute(sql, args), backend=self.vector_backend)
            if self._pool_bytes > 0 and key in self._pools:
                self._pools.move_to_end(key)
                return self._pools[key]
            version = self._scope_versions.setdefault(key[0], 0)
            rows = self.db.execute(sql, args).fetchall()
        pool = VectorPool(((row[0], row[1]) for row in rows), backend=self.vector_backend)
        with self.lock:
            if key in self._pools:
                return self._pools[key]  # Another thread finished first; share its pool.
            if (self._scope_versions.get(key[0]) == version and 0 < pool.nbytes <= self._pool_bytes):
                self._pools[key] = pool
                self._trim_pools(keep=key)
        return pool

    def semantic(self, vector, scope, limit):
        pool = self._vector_pool(scope, questions=False)
        with self.lock:
            ranked = pool.rank(vector, limit)
            payloads = self.get_many(mid for mid, _ in ranked)
        return [(payloads[mid], score) for mid, score in ranked if mid in payloads]

    def query_semantic(self, vector, scope, limit):
        pool = self._vector_pool(scope, questions=True)
        with self.lock:
            ranked = pool.rank(vector, limit)
            payloads = self.get_many(mid for mid, _ in ranked)
        return [(payloads[mid], score) for mid, score in ranked if mid in payloads]

    def _invalidate_pools(self):
        self._pools.clear()
        for scope_items in self._scope_versions:
            self._scope_versions[scope_items] += 1

    def update(self, *args, **kwargs):
        with self.lock:
            try:
                return super().update(*args, **kwargs)
            finally:
                self._invalidate_pools()

    def delete(self, *args, **kwargs):
        with self.lock:
            try:
                return super().delete(*args, **kwargs)
            finally:
                self._invalidate_pools()

    def replace_queries(self, *args, **kwargs):
        with self.lock:
            try:
                return super().replace_queries(*args, **kwargs)
            finally:
                self._invalidate_pools()

    def save_messages(self, messages, session):
        # Keep the same ten-message context as MiniVector, retaining source timestamps.
        previous = self.last_messages(session)
        recent = (previous + messages)[-10:]
        super().save_messages(messages, session)
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO aml_context VALUES (?, ?)",
                            (session, json.dumps(recent, ensure_ascii=False)))

    def last_messages(self, session):
        with self.lock:
            row = self.db.execute("SELECT messages FROM aml_context WHERE session_scope=?", (session,)).fetchone()
        return json.loads(row[0]) if row else []

    def request(self, request_id):
        with self.lock:
            return self.db.execute("SELECT payload_hash, response FROM aml_requests WHERE request_id=?",
                                   (request_id,)).fetchone()

    def complete(self, request_id, payload_hash, response):
        self.db.execute("INSERT INTO aml_requests VALUES (?, ?, ?, ?)",
                        (request_id, payload_hash, json.dumps(response, ensure_ascii=False), _now()))

    def close(self):
        try:
            with self.lock:
                self._pools.clear()
                super().close()
        finally:
            self.lease.close()

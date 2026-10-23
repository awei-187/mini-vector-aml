"""AML-only durable request ledger and atomic ingestion; no benchmark runner imports."""
from __future__ import annotations

from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3

from .memory import SQLiteVectorStore, _now


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
    def __init__(self, path: str, identity: dict):
        self.identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        db_path = Path(path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lease = _DatabaseLease(db_path.with_name(db_path.name + ".lock"))
        self.db = None
        try:
            # Inspect before SQLiteVectorStore can change an existing database.
            if db_path.exists() and db_path.stat().st_size:
                with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as existing:
                    try:
                        row = existing.execute("SELECT identity FROM aml_config WHERE singleton=1").fetchone()
                    except sqlite3.Error:
                        raise ValueError("Refusing a database without an AML configuration marker") from None
                    if row is None or row[0] != self.identity:
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
        with self.lock:
            connection = self.db
            connection.execute("BEGIN IMMEDIATE")
            self.db = _DeferredConnection(connection)
            try:
                yield
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                self.db = connection

    def check_dimension(self, dimension: int):
        with self.lock, self.db:
            current = self.db.execute("SELECT dimension FROM aml_config WHERE singleton=1").fetchone()[0]
            if current is None:
                self.db.execute("UPDATE aml_config SET dimension=? WHERE singleton=1", (dimension,))
            elif current != dimension:
                raise ValueError("Embedding dimension changed for this AML database")

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
        return self.db.execute("SELECT payload_hash, response FROM aml_requests WHERE request_id=?",
                               (request_id,)).fetchone()

    def complete(self, request_id, payload_hash, response):
        self.db.execute("INSERT INTO aml_requests VALUES (?, ?, ?, ?)",
                        (request_id, payload_hash, json.dumps(response, ensure_ascii=False), _now()))

    def close(self):
        try:
            super().close()
        finally:
            self.lease.close()

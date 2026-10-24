from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from .config import setting
from .prompts import MEMORY_EXTRACTION_PROMPT, build_extraction_input
from .retrieval import memory_queries, rerank_channel, rerank_fused
from .vector_scan import VectorPool, np


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope(filters: dict[str, Any] | None) -> dict[str, str]:
    scope = {k: str(v) for k, v in (filters or {}).items()
             if k in {"user_id", "agent_id", "run_id"} and v is not None and str(v).strip()}
    if not scope:
        raise ValueError("Provide at least one scope key: user_id, agent_id, or run_id")
    return scope


def _scope_sql(scope: dict[str, str]) -> tuple[str, list[str]]:
    return " AND ".join(f"json_extract(metadata, '$.{key}') = ?" for key in scope), list(scope.values())


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimension does not match stored vectors")
    denom = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
    return max(0.0, min(1.0, sum(x*y for x, y in zip(a, b)) / denom)) if denom else 0.0


def _clean_fts_query(query: str) -> str:
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


_TRACE_METADATA_FIELDS = {
    "source_id", "source_ids", "source_dialogue_ids", "source_memory_ids",
    "source_chunk_ids", "dia_id", "dia_ids", "dialogue_id", "session_id",
    "episode_id", "episode_ids", "created_at", "updated_at",
}


def _trace_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Keep provenance fields while excluding arbitrary configuration/secrets."""
    return {key: value for key, value in (metadata or {}).items() if key in _TRACE_METADATA_FIELDS}


class SQLiteVectorStore:
    """Persistent vector records in SQLite, searched by exact cosine scan + SQLite FTS5 BM25."""

    def __init__(self, path: str = "mini_vector.sqlite3", *, vector_backend: str = "auto",
                 vector_cache_bytes: int = 128*1024*1024):
        if vector_backend not in {"auto", "python", "numpy"}:
            raise ValueError("vector_backend must be auto, python or numpy")
        if vector_backend == "numpy" and np is None:
            raise RuntimeError("Install mini-vector-memory[fast] to use the numpy backend")
        if type(vector_cache_bytes) is not int or vector_cache_bytes < 0:
            raise ValueError("vector_cache_bytes must be a non-negative integer")
        self.vector_backend = vector_backend
        self.vector_cache_bytes = vector_cache_bytes
        self._vector_cache = OrderedDict()
        self._vector_cache_stamp = None
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, data TEXT NOT NULL, embedding TEXT NOT NULL,
                hash TEXT NOT NULL, metadata TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS memory_hash_idx ON memories(hash);
            CREATE INDEX IF NOT EXISTS memory_user_scope_idx ON memories(json_extract(metadata, '$.user_id'));
            CREATE INDEX IF NOT EXISTS memory_agent_scope_idx ON memories(json_extract(metadata, '$.agent_id'));
            CREATE INDEX IF NOT EXISTS memory_run_scope_idx ON memories(json_extract(metadata, '$.run_id'));
            CREATE TABLE IF NOT EXISTS history (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL,
                old_memory TEXT, new_memory TEXT, event TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS history_id_idx ON history(id, seq);
            CREATE TABLE IF NOT EXISTS messages (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, session_scope TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, name TEXT, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS message_scope_idx ON messages(session_scope, seq);
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id UNINDEXED, data);
            CREATE TABLE IF NOT EXISTS memory_queries (
                memory_id TEXT NOT NULL, position INTEGER NOT NULL,
                query TEXT NOT NULL, embedding TEXT NOT NULL,
                PRIMARY KEY (memory_id, position)
            );
        """)
        self.db.commit()

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        meta = json.loads(row["metadata"])
        promoted = {"user_id", "agent_id", "run_id", "role", "actor_id", "attributed_to"}
        reserved = promoted | {"id", "memory", "hash", "data", "created_at", "updated_at", "metadata"}
        result = {"id": row["id"], "memory": row["data"], "hash": row["hash"],
                  "created_at": meta["created_at"], "updated_at": meta["updated_at"],
                  **{k: meta[k] for k in promoted if k in meta},
                  "metadata": {k: v for k, v in meta.items() if k not in reserved}}
        return result

    def save(self, memory_id: str, text: str, vector: list[float], metadata: dict[str, Any],
             queries: list[tuple[str, list[float]]] = ()) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?)",
                            (memory_id, text, json.dumps(vector), hashlib.md5(text.encode()).hexdigest(),
                             json.dumps(metadata, ensure_ascii=False)))
            self.db.execute("INSERT INTO memory_fts(id, data) VALUES (?, ?)", (memory_id, text))
            self._replace_queries(memory_id, queries)
            self._history(memory_id, None, text, "ADD", metadata["created_at"], metadata["updated_at"])

    def _replace_queries(self, memory_id: str, queries) -> None:
        self.db.execute("DELETE FROM memory_queries WHERE memory_id=?", (memory_id,))
        self.db.executemany("INSERT INTO memory_queries VALUES (?, ?, ?, ?)",
                            [(memory_id, index, query, json.dumps(vector))
                             for index, (query, vector) in enumerate(queries)])

    def replace_queries(self, memory_id: str, queries, *, expected_text: str) -> None:
        with self.lock, self.db:
            row = self.db.execute("SELECT data FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None or row[0] != expected_text:
                raise ValueError("Memory changed while generating retrieval queries")
            self._replace_queries(memory_id, queries)

    def get_queries(self, memory_id: str) -> list[str]:
        with self.lock:
            rows = self.db.execute("SELECT query FROM memory_queries WHERE memory_id=? ORDER BY position",
                                   (memory_id,)).fetchall()
        return [row[0] for row in rows]

    def _history(self, mid: str, old: str | None, new: str | None, event: str, created: str, updated: str) -> None:
        self.db.execute("INSERT INTO history(id, old_memory, new_memory, event, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (mid, old, new, event, created, updated))

    def get(self, memory_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute("SELECT id,data,hash,metadata FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._row(row) if row else None

    def get_many(self, memory_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Read candidate payloads once, without fetching their large vectors."""
        ids = list(dict.fromkeys(memory_ids))
        found = {}
        with self.lock:
            for start in range(0, len(ids), 900):
                batch = ids[start:start+900]
                rows = self.db.execute(
                    f"SELECT id,data,hash,metadata FROM memories WHERE id IN ({','.join('?' for _ in batch)})",
                    batch).fetchall()
                found.update((row["id"], self._row(row)) for row in rows)
        return found

    def list(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        clause, args = _scope_sql(scope)
        with self.lock:
            rows = self.db.execute(f"SELECT id,data,hash,metadata FROM memories WHERE {clause} ORDER BY rowid", args).fetchall()
        return [self._row(row) for row in rows]

    def duplicate_hashes(self, scope: dict[str, str], hashes: Iterable[str]) -> set[str]:
        clause, args = _scope_sql(scope)
        wanted = list(set(hashes))
        if not wanted:
            return set()
        with self.lock:
            rows = self.db.execute(f"SELECT hash FROM memories WHERE {clause} AND hash IN ({','.join('?' for _ in wanted)})",
                                   [*args, *wanted]).fetchall()
        return {row[0] for row in rows}

    def semantic(self, vector: list[float], scope: dict[str, str], limit: int) -> list[tuple[dict[str, Any], float]]:
        with self.lock:
            ranked = self._vector_pool(scope, questions=False).rank(vector, limit)
            payloads = self.get_many(mid for mid, _ in ranked)
            return [(payloads[mid], score) for mid, score in ranked if mid in payloads]

    def query_semantic(self, vector: list[float], scope: dict[str, str], limit: int
                       ) -> list[tuple[dict[str, Any], float]]:
        """Max question cosine per memory; deduplicate before applying the limit."""
        with self.lock:
            ranked = self._vector_pool(scope, questions=True).rank(vector, limit)
            payloads = self.get_many(mid for mid, _ in ranked)
            return [(payloads[mid], score) for mid, score in ranked if mid in payloads]

    def _vector_pool(self, scope: dict[str, str], *, questions: bool) -> VectorPool:
        # total_changes covers this connection, data_version covers other
        # connections. Never retain a pool built inside a transaction: rollback
        # does not increment total_changes and must not expose uncommitted data.
        stamp = (self.db.total_changes, self.db.execute("PRAGMA data_version").fetchone()[0])
        if stamp != self._vector_cache_stamp:
            self._vector_cache.clear()
            self._vector_cache_stamp = stamp
        key = (tuple(sorted(scope.items())), questions)
        cacheable = not self.db.in_transaction and self.vector_cache_bytes > 0
        if cacheable and key in self._vector_cache:
            self._vector_cache.move_to_end(key)
            return self._vector_cache[key]
        clause, args = _scope_sql(scope)
        if questions:
            sql = f"""SELECT q.memory_id,q.embedding FROM memory_queries q
                JOIN memories m ON m.id=q.memory_id WHERE {clause} ORDER BY m.rowid,q.position"""
        else:
            sql = f"SELECT id,embedding FROM memories WHERE {clause} ORDER BY rowid"
        pool = VectorPool(self.db.execute(sql, args), backend=self.vector_backend)
        # Count compact numeric payloads. Python ID/group bookkeeping is extra.
        if cacheable and pool.nbytes <= self.vector_cache_bytes:
            while self._vector_cache and (len(self._vector_cache) >= 8 or
                    sum(p.nbytes for p in self._vector_cache.values())+pool.nbytes > self.vector_cache_bytes):
                self._vector_cache.popitem(last=False)
            self._vector_cache[key] = pool
        return pool

    def bm25(self, query: str, scope: dict[str, str], limit: int,
             research_trace: dict[str, Any] | None = None) -> dict[str, float]:
        match = _clean_fts_query(query)
        if not match:
            if research_trace is not None:
                research_trace.update(status="skipped", reason="query_has_no_fts_terms", candidates=[])
            return {}
        clause, args = _scope_sql(scope)
        details = ", m.data, m.metadata" if research_trace is not None else ""
        sql = f"SELECT f.id, -bm25(memory_fts) AS score{details} FROM memory_fts f JOIN memories m ON m.id=f.id WHERE memory_fts MATCH ? AND {clause} ORDER BY bm25(memory_fts) LIMIT ?"
        try:
            with self.lock:
                rows = self.db.execute(sql, [match, *args, limit]).fetchall()
        except sqlite3.OperationalError:
            if research_trace is not None:
                research_trace.update(status="error", error="sqlite_operational_error", candidates=[])
            return {}
        positive = [r["score"] for r in rows if r["score"] > 0]
        if not positive:
            if research_trace is not None:
                research_trace.update(status="recorded", candidates=[])
            return {}
        # FTS5 BM25 values are usually around 1e-6; normalize within this
        # result pool before applying a bounded sigmoid, preserving BM25 rank.
        best = max(positive)
        normalized = {r["id"]: 1 / (1 + math.exp(-8 * (r["score"] / best - .5)))
                      for r in rows if r["score"] > 0}
        if research_trace is not None:
            research_trace.update(status="recorded", candidates=[
                {"rank": rank, "memory_id": row["id"], "raw_score": float(row["score"]),
                 "normalized_score": normalized[row["id"]],
                 "text": row["data"], "metadata": _trace_metadata(json.loads(row["metadata"]))}
                for rank, row in enumerate((row for row in rows if row["id"] in normalized), start=1)
            ])
        return normalized

    def update(self, memory_id: str, text: str, vector: list[float] | None, metadata: dict[str, Any],
               queries: list[tuple[str, list[float]]] = ()) -> None:
        with self.lock, self.db:
            row = self.db.execute("SELECT data FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                raise ValueError(f"Memory {memory_id} not found")
            if vector is None:
                self.db.execute("UPDATE memories SET data=?, hash=?, metadata=? WHERE id=?",
                                (text, hashlib.md5(text.encode()).hexdigest(), json.dumps(metadata, ensure_ascii=False), memory_id))
            else:
                self.db.execute("UPDATE memories SET data=?, embedding=?, hash=?, metadata=? WHERE id=?",
                                (text, json.dumps(vector), hashlib.md5(text.encode()).hexdigest(),
                                 json.dumps(metadata, ensure_ascii=False), memory_id))
                self._replace_queries(memory_id, queries)
            self.db.execute("DELETE FROM memory_fts WHERE id=?", (memory_id,))
            self.db.execute("INSERT INTO memory_fts(id, data) VALUES (?, ?)", (memory_id, text))
            self._history(memory_id, row[0], text, "UPDATE", metadata["created_at"], metadata["updated_at"])

    def delete(self, memory_id: str) -> None:
        with self.lock, self.db:
            row = self.db.execute("SELECT data, metadata FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                raise ValueError(f"Memory {memory_id} not found")
            meta = json.loads(row["metadata"])
            self.db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self.db.execute("DELETE FROM memory_fts WHERE id=?", (memory_id,))
            now = _now()
            self.db.execute("DELETE FROM memory_queries WHERE memory_id=?", (memory_id,))
            self._history(memory_id, row[0], None, "DELETE", meta["created_at"], now)

    def delete_all(self, scope: dict[str, str]) -> int:
        entries = self.list(scope)
        for entry in entries:
            self.delete(entry["id"])
        return len(entries)

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute("SELECT id AS memory_id, old_memory, new_memory, event, created_at, updated_at FROM history WHERE id=? ORDER BY seq",
                                   (memory_id,)).fetchall()
        return [dict(r) for r in rows]

    def save_messages(self, messages: list[dict[str, Any]], session: str) -> None:
        now = _now()
        with self.lock, self.db:
            self.db.executemany("INSERT INTO messages(session_scope, role, content, name, created_at) VALUES (?, ?, ?, ?, ?)",
                                [(session, m["role"], m["content"], m.get("name"), now) for m in messages])
            self.db.execute("DELETE FROM messages WHERE session_scope=? AND seq NOT IN (SELECT seq FROM messages WHERE session_scope=? ORDER BY seq DESC LIMIT 10)",
                            (session, session))

    def last_messages(self, session: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute("SELECT role, content, name FROM messages WHERE session_scope=? ORDER BY seq", (session,)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._vector_cache.clear()
        self.db.close()


class MiniVector:
    """ADD-only extraction, memory-side questions and scoped channel retrieval."""

    def __init__(self, db_path: str = "mini_vector.sqlite3", *, base_url: str | None = None,
                 chat_model: str | None = None, embedding_model: str | None = None,
                 llm: Callable[[list[dict[str, str]]], str] | None = None,
                 embedder: Callable[[list[str]], list[list[float]]] | None = None,
                 store: SQLiteVectorStore | None = None,
                 memory_query_enabled: bool = True, queries_per_memory: int = 3,
                 rerank_mode: str = "llm", rerank_limit: int = 30):
        if type(memory_query_enabled) is not bool:
            raise ValueError("memory_query_enabled must be boolean")
        if type(queries_per_memory) is not int or queries_per_memory < 1:
            raise ValueError("queries_per_memory must be a positive integer")
        if rerank_mode not in {"none", "llm", "global", "evidence"}:
            raise ValueError("rerank_mode must be none, llm, global or evidence")
        if type(rerank_limit) is not int or rerank_limit < 1:
            raise ValueError("rerank_limit must be a positive integer")
        self.memory_query_enabled = memory_query_enabled
        self.queries_per_memory = queries_per_memory
        self.rerank_mode = rerank_mode
        self.rerank_limit = rerank_limit
        self.store = store or SQLiteVectorStore(db_path)
        self.llm, self.embedder = llm, embedder
        self.chat_model = chat_model or setting("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.embedding_model = embedding_model or setting("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        if llm is None or embedder is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install openai to use the default LLM and embedding clients") from exc
            self.client = OpenAI(api_key=setting("OPENAI_API_KEY", "not-needed"),
                                 base_url=base_url or setting("OPENAI_BASE_URL"))

    def _complete(self, messages: list[dict[str, str]]) -> str:
        if self.llm:
            return self.llm(messages)
        response = self.client.chat.completions.create(model=self.chat_model, messages=messages,
                                                       response_format={"type": "json_object"})
        return response.choices[0].message.content or "{}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedder:
            vectors = self.embedder(texts)
        else:
            response = self.client.embeddings.create(model=self.embedding_model, input=texts)
            vectors = [item.embedding for item in response.data]
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise ValueError("Embedding provider returned an invalid batch")
        return [[float(x) for x in vector] for vector in vectors]

    def _query_records(self, texts: list[str]) -> list[list[tuple[str, list[float]]]]:
        if not self.memory_query_enabled or not texts:
            return [[] for _ in texts]
        try:
            groups = memory_queries(self._complete, texts, self.queries_per_memory)
        except Exception as exc:
            raise RuntimeError("Memory query generation failed") from exc
        flat = [query for group in groups for query in group]
        vectors = iter(self._embed(flat))
        return [[(query, next(vectors)) for query in group] for group in groups]

    @staticmethod
    def _messages(value: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(value, str):
            value = [{"role": "user", "content": value}]
        elif isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("messages must be a string, message object, or list of message objects")
        cleaned = []
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("role"), str) or not isinstance(item.get("content"), str):
                raise ValueError("Each message needs string role and content")
            cleaned.append({k: item[k] for k in ("role", "content", "name", "timestamp") if k in item})
        if not cleaned:
            raise ValueError("messages cannot be empty")
        return cleaned

    def add(self, messages: str | dict[str, Any] | list[dict[str, Any]], *, user_id: str | None = None,
            agent_id: str | None = None, run_id: str | None = None, metadata: dict[str, Any] | None = None,
            infer: bool = True) -> dict[str, list[dict[str, Any]]]:
        scope = _scope({"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
        msgs = self._messages(messages)
        session = json.dumps(scope, sort_keys=True)
        output = []
        if not infer:
            raw = [msg for msg in msgs if msg["role"] != "system"]
            queries = self._query_records([msg["content"] for msg in raw])
            for msg, query_records in zip(raw, queries):
                text = msg["content"]
                mid = self._persist(text, self._embed([text])[0], {**(metadata or {}), **scope,
                               "role": msg["role"], **({"actor_id": msg["name"]} if msg.get("name") else {})}, query_records)
                output.append({"id": mid, "memory": text, "event": "ADD"})
            self.store.save_messages(msgs, session)
            return {"results": output}

        current = "\n".join(f'{m["role"]}: {m["content"]}' for m in msgs if m["role"] != "system")
        if not current.strip():
            self.store.save_messages(msgs, session)
            return {"results": []}
        recent = self.store.last_messages(session)
        current_vector = self._embed([current])[0]
        similar = self.store.semantic(current_vector, scope, 10)
        existing = [{"id": item["id"], "text": item["memory"]} for item, _ in similar]
        prompt = build_extraction_input(msgs, recent, existing, metadata)
        try:
            response = self._complete([{"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
                                       {"role": "user", "content": prompt}])
        except Exception as exc:
            raise RuntimeError(f"Memory extraction request failed: {exc}") from exc
        try:
            parsed = json.loads(response)
            facts = parsed.get("memory", [])
            if not isinstance(facts, list):
                raise ValueError("LLM JSON field 'memory' must be a list")
            facts = [f["text"].strip() for f in facts if isinstance(f, dict) and isinstance(f.get("text"), str) and f["text"].strip()]
        except Exception as exc:
            raise ValueError(f"Could not parse memory extraction JSON: {exc}") from exc
        # Save context even when extraction returns no facts, matching mem0's workflow.
        if facts:
            hashes = [hashlib.md5(f.encode()).hexdigest() for f in facts]
            duplicates = self.store.duplicate_hashes(scope, hashes)
            seen: set[str] = set()
            unique = []
            for fact, digest in zip(facts, hashes):
                if digest in duplicates or digest in seen:
                    continue
                seen.add(digest)
                unique.append(fact)
            queries = self._query_records(unique)
            vectors = self._embed(unique) if unique else []
            for fact, vector, query_records in zip(unique, vectors, queries):
                mid = self._persist(fact, vector, {**(metadata or {}), **scope}, query_records)
                output.append({"id": mid, "memory": fact, "event": "ADD"})
        self.store.save_messages(msgs, session)
        return {"results": output}

    def _persist(self, text: str, vector: list[float], metadata: dict[str, Any],
                 queries: list[tuple[str, list[float]]] = ()) -> str:
        now = _now()
        mid = str(uuid.uuid4())
        payload = {**metadata, "created_at": metadata.get("created_at", now), "updated_at": now}
        self.store.save(mid, text, vector, payload, queries)
        return mid

    def search(self, query: str, *, top_k: int = 20, filters: dict[str, Any] | None = None,
               threshold: float = 0.1, explain: bool = False,
               research_trace: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
        if not self.memory_query_enabled and self.rerank_mode == "none":
            return self._search_legacy(query, top_k=top_k, filters=filters, threshold=threshold,
                                       explain=explain, research_trace=research_trace)
        return self._search_channels(query, top_k=top_k, filters=filters, threshold=threshold,
                                     explain=explain, research_trace=research_trace)

    def _search_legacy(self, query: str, *, top_k: int = 20, filters: dict[str, Any] | None = None,
                       threshold: float = 0.1, explain: bool = False,
                       research_trace: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
        scope = _scope(filters)
        if not isinstance(query, str) or not query.strip() or top_k < 1 or not 0 <= threshold <= 1:
            raise ValueError("query must be non-empty, top_k positive, and threshold between 0 and 1")
        qvec = self._embed([query.strip()])[0]
        candidate_limit = max(top_k * 4, 60)
        candidates = self.store.semantic(qvec, scope, candidate_limit)
        semantic_by_id = {item["id"]: float(score) for item, score in candidates}
        bm25_trace: dict[str, Any] | None = {} if research_trace is not None else None
        bm25 = (self.store.bm25(query, scope, candidate_limit, research_trace=bm25_trace)
                if research_trace is not None else self.store.bm25(query, scope, candidate_limit))
        max_score = 2.0 if bm25 else 1.0
        scored = []
        threshold_excluded = []
        for item, semantic in candidates:
            if semantic < threshold:
                if research_trace is not None:
                    threshold_excluded.append({"memory_id": item["id"], "semantic_score": semantic})
                continue
            score = min(1.0, (semantic + bm25.get(item["id"], 0.0)) / max_score)
            result = {**item, "score": score}
            if explain:
                result["score_details"] = {"semantic_score": semantic, "bm25_score": bm25.get(item["id"], 0.0), "final_score": score}
            scored.append(result)
        scored.sort(key=lambda item: item["score"], reverse=True)
        results = scored[:top_k]
        if research_trace is not None:
            by_id = {item["id"]: item for item, _ in candidates}
            research_trace.clear()
            research_trace.update({
                "schema_version": 1,
                "engine": "mini-vector",
                "candidate_limit": candidate_limit,
                "top_k": top_k,
                "threshold": threshold,
                "score_semantics": {"semantic": "cosine similarity", "bm25": "normalized SQLite FTS5 BM25",
                                    "final": "(semantic + normalized BM25) / 2 when BM25 candidates exist; otherwise semantic"},
                "channels": {
                    "semantic": {"status": "recorded", "candidates": [
                        {"rank": rank, "memory_id": item["id"], "score": float(score),
                         "text": item["memory"], "metadata": _trace_metadata(item.get("metadata", {}))}
                        for rank, (item, score) in enumerate(candidates, start=1)
                    ]},
                    "bm25": bm25_trace or {"status": "recorded", "candidates": []},
                },
                "stages": {
                    "threshold_excluded": threshold_excluded,
                    "semantic_eligible": [item["id"] for item, score in candidates if score >= threshold],
                    "combined_order": [item["id"] for item in scored],
                    "final": [{"rank": rank, "memory_id": item["id"], "score": item["score"],
                               "score_details": item.get("score_details", {
                                   "semantic_score": semantic_by_id[item["id"]],
                                   "bm25_score": bm25.get(item["id"], 0.0),
                                   "final_score": item["score"],
                              }), "metadata": _trace_metadata(item.get("metadata", {}))}
                              for rank, item in enumerate(results, start=1)],
                },
                "source_attribution": "memory metadata as recorded; absent source IDs remain unknown",
            })
        return {"results": results}

    def _search_channels(self, query: str, *, top_k: int, filters, threshold: float,
                         explain: bool, research_trace) -> dict[str, list[dict[str, Any]]]:
        scope = _scope(filters)
        if not isinstance(query, str) or not query.strip() or top_k < 1 or not 0 <= threshold <= 1:
            raise ValueError("query must be non-empty, top_k positive, and threshold between 0 and 1")
        vector = self._embed([query.strip()])[0]
        candidate_limit = max(top_k * 4, 60)
        semantic = self.store.semantic(vector, scope, candidate_limit)
        generated = (self.store.query_semantic(vector, scope, candidate_limit)
                     if self.memory_query_enabled else [])
        bm25_trace = {} if research_trace is not None else None
        lexical_scores = self.store.bm25(query, scope, candidate_limit, research_trace=bm25_trace)
        lexical_payloads = self.store.get_many(lexical_scores)
        lexical = [(lexical_payloads[mid], score) for mid, score in lexical_scores.items()
                   if mid in lexical_payloads]
        channels = {"semantic": semantic, "bm25": lexical, "memory_query": generated}
        merged, channel_trace = {}, {}
        active = 0
        for name, candidates in channels.items():
            eligible = [(item, score) for item, score in candidates
                        if name == "bm25" or score >= threshold]
            ranked, rerank_trace = rerank_channel(
                self._complete, query, [item for item, _ in eligible],
                self.rerank_limit, self.rerank_mode == "llm")
            active += bool(ranked)
            for rank, item in enumerate(ranked, start=1):
                entry = merged.setdefault(item["id"], {"item": item, "rrf": 0.0, "ranks": {}})
                entry["rrf"] += 1 / (60 + rank)
                entry["ranks"][name] = rank
            if research_trace is not None:
                channel_trace[name] = {
                    "status": "recorded", "candidates": [
                        {"rank": rank, "memory_id": item["id"], "score": float(score),
                         "text": item["memory"], "metadata": _trace_metadata(item.get("metadata"))}
                        for rank, (item, score) in enumerate(candidates, start=1)],
                    "eligible_ids": [item["id"] for item, _ in eligible],
                    "rerank": rerank_trace,
                }
        # Equal-weight RRF of channel ranks, normalized to [0, 1]. The LLM only
        # changes the head of each channel; distinct candidates survive fusion.
        ordered = sorted(merged.values(), key=lambda entry: (-entry["rrf"], entry["item"]["id"]))
        combined_order = [entry["item"]["id"] for entry in ordered]
        global_trace = None
        if self.rerank_mode in {"global", "evidence"}:
            selected, global_trace = rerank_fused(
                self._complete, query, [entry["item"] for entry in ordered],
                self.rerank_limit, top_k, diverse=self.rerank_mode == "evidence")
            by_id = {entry["item"]["id"]: entry for entry in ordered}
            ordered = [by_id[item["id"]] for item in selected]
        semantic_scores = {item["id"]: score for item, score in semantic}
        query_scores = {item["id"]: score for item, score in generated}
        results = []
        for entry in ordered[:top_k]:
            item = entry["item"]
            score = entry["rrf"] / (active / 61)
            result = {**item, "score": min(1.0, score)}
            if explain or research_trace is not None:
                result["score_details"] = {
                    "semantic_score": semantic_scores.get(item["id"], 0.0),
                    "bm25_score": lexical_scores.get(item["id"], 0.0),
                    "memory_query_score": query_scores.get(item["id"], 0.0),
                    "channel_ranks": entry["ranks"], "rrf_score": entry["rrf"],
                    "final_score": result["score"],
                }
            results.append(result)
        if research_trace is not None:
            if bm25_trace:
                channel_trace["bm25"].update(status=bm25_trace["status"],
                                             lexical_details=bm25_trace)
            research_trace.clear()
            research_trace.update({
                "schema_version": 3 if global_trace is not None else 2,
                "engine": "mini-vector", "candidate_limit": candidate_limit,
                "top_k": top_k, "threshold": threshold,
                "memory_query_enabled": self.memory_query_enabled,
                "rerank_mode": self.rerank_mode, "rerank_limit": self.rerank_limit,
                "score_semantics": {"final": "equal-weight channel RRF (k=60), normalized by active_channels/61"},
                "channels": channel_trace,
                "stages": {"combined_order": combined_order,
                           "final": [{"rank": rank, "memory_id": item["id"], "score": item["score"],
                                      "score_details": item["score_details"],
                                      "metadata": _trace_metadata(item.get("metadata"))}
                                     for rank, item in enumerate(results, start=1)]},
                "source_attribution": "memory metadata as recorded; absent source IDs remain unknown",
            })
            if global_trace is not None:
                research_trace["global_rerank"] = global_trace
                research_trace["stages"]["selected_order"] = [entry["item"]["id"] for entry in ordered]
                research_trace["score_semantics"]["final"] += "; RRF score retained after global selection (not a rank probability)"
            if not explain:
                for item in results:
                    item.pop("score_details", None)
        return {"results": results}

    def backfill_memory_queries(self, *, filters: dict[str, Any] | None = None) -> dict[str, int]:
        """Explicitly index existing memories; skip already indexed rows on restart.

        Processes one memory at a time. Completed rows remain committed if a later
        provider call fails. Never invoked implicitly by opening or searching a DB.
        """
        scope = _scope(filters)
        if not self.memory_query_enabled:
            raise ValueError("Enable memory queries before backfilling")
        indexed = 0
        for item in self.store.list(scope):
            if self.store.get_queries(item["id"]):
                continue
            records = self._query_records([item["memory"]])[0]
            self.store.replace_queries(item["id"], records, expected_text=item["memory"])
            indexed += 1
        return {"indexed": indexed}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        return self.store.get(memory_id)

    def get_all(self, *, filters: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
        return {"results": self.store.list(_scope(filters))}

    def update(self, memory_id: str, text: str | None = None, *, metadata: dict[str, Any] | None = None) -> dict[str, str]:
        old = self.store.get(memory_id)
        if old is None:
            raise ValueError(f"Memory {memory_id} not found")
        if text is None and metadata is None:
            raise ValueError("Provide text or metadata to update")
        new_text = text if text is not None else old["memory"]
        queries = self._query_records([new_text])[0] if text is not None else []
        vector = self._embed([new_text])[0] if text is not None else None
        immutable = {"user_id", "agent_id", "run_id", "role", "actor_id", "attributed_to",
                     "id", "memory", "hash", "data", "created_at", "updated_at", "metadata"}
        new_meta = {**old.get("metadata", {}), **{k: old[k] for k in immutable & old.keys()}}
        new_meta.update({k: v for k, v in (metadata or {}).items() if k not in immutable})
        new_meta.update(created_at=old["created_at"], updated_at=_now())
        self.store.update(memory_id, new_text, vector, new_meta, queries)
        return {"message": "Memory updated successfully"}

    def delete(self, memory_id: str) -> dict[str, str]:
        self.store.delete(memory_id)
        return {"message": "Memory deleted successfully"}

    def delete_all(self, *, user_id: str | None = None, agent_id: str | None = None,
                   run_id: str | None = None) -> dict[str, int]:
        return {"deleted": self.store.delete_all(_scope({"user_id": user_id, "agent_id": agent_id, "run_id": run_id}))}

    def history(self, memory_id: str) -> dict[str, list[dict[str, Any]]]:
        return {"history": self.store.history(memory_id)}

    def close(self) -> None:
        self.store.close()

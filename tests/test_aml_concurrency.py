"""Concurrent AML ingestion: overlap across users, order within a user, cache equivalence."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error

from mini_vector.aml_config import embedding_vectors
from mini_vector.reranker import QwenReranker, RerankerConfig
from mini_vector.vector_scan import VectorPool, np

HAS_API = importlib.util.find_spec("fastapi") is not None
if HAS_API:
    from fastapi.testclient import TestClient
    from mini_vector.aml_api import create_app


class VectorPoolExtendTests(unittest.TestCase):
    def rows(self, count, dimension, seed):
        rng = random.Random(seed)
        # Repeated IDs model several generated questions owned by one memory.
        return [(f"m{index // 2}", [rng.uniform(-1, 1) for _ in range(dimension)]) for index in range(count)]

    def test_extend_matches_rebuild_for_both_backends(self):
        backends = ["python"] + (["numpy"] if np is not None else [])
        first, second = self.rows(9, 6, 1), self.rows(7, 6, 2)
        second = [("m3", second[0][1])] + [(f"n{i}", v) for i, (_, v) in enumerate(second[1:])]
        for backend in backends:
            with self.subTest(backend=backend):
                rebuilt = VectorPool([(mid, json.dumps(v)) for mid, v in first + second], backend=backend)
                extended = VectorPool([(mid, json.dumps(v)) for mid, v in first], backend=backend)
                extended.extend(second)
                self.assertEqual(extended.ids, rebuilt.ids)
                self.assertEqual(extended.groups, rebuilt.groups)
                self.assertEqual(extended.nbytes, rebuilt.nbytes)
                for query in (v for _, v in self.rows(5, 6, 3)):
                    self.assertEqual(extended.rank(query, 4), rebuilt.rank(query, 4))

    def test_extend_empty_pool_sets_dimension(self):
        pool = VectorPool([])
        self.assertEqual(pool.rank([1.0, 0.0], 3), [])
        pool.extend([("a", [1.0, 0.0]), ("b", [0.0, 1.0])])
        self.assertEqual(pool.rank([1.0, 0.0], 1), [("a", 1.0)])
        with self.assertRaisesRegex(ValueError, "dimension"):
            pool.extend([("c", [1.0])])


class GroupedQueryGenerationTests(unittest.TestCase):
    def memory(self, llm):
        from mini_vector.memory import MiniVector
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        memory = MiniVector(str(Path(folder.name) / "m.sqlite3"), llm=llm,
                            embedder=lambda texts: [[1.0, float(i + 1)] for i, _ in enumerate(texts)],
                            rerank_mode="none", query_rewrite_enabled=False)
        self.addCleanup(memory.close)
        return memory

    def test_groups_run_concurrently_and_keep_memory_order(self):
        guard, state = threading.Lock(), {"active": 0, "peak": 0, "calls": 0}

        def llm(messages):
            data = json.loads(messages[-1]["content"])
            with guard:
                state["calls"] += 1
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.05)
            with guard:
                state["active"] -= 1
            return json.dumps({"memories": [{"id": m["id"], "queries": ["Q " + m["text"]]}
                                            for m in data["memories"]]})

        texts = [f"fact {i}" for i in range(14)]
        records = self.memory(llm)._query_records(texts)
        self.assertEqual([group[0][0] for group in records], ["Q " + text for text in texts])
        self.assertEqual(state["calls"], 3)
        self.assertGreater(state["peak"], 1)

    def test_any_failed_group_fails_the_whole_generation(self):
        def llm(messages):
            data = json.loads(messages[-1]["content"])
            if data["memories"][0]["text"] == "fact 6":
                return '{"memories":[]}'
            return json.dumps({"memories": [{"id": m["id"], "queries": ["Q"]} for m in data["memories"]]})

        with self.assertRaisesRegex(RuntimeError, "Memory query generation failed"):
            self.memory(llm)._query_records([f"fact {i}" for i in range(10)])


class EmbeddingBatchTests(unittest.TestCase):
    class Client:
        def __init__(self, fail_batch=None):
            self.active = self.peak = 0
            self.guard = threading.Lock()
            self.fail_batch = fail_batch
            self.embeddings = self

        def create(self, *, input, **_kwargs):
            with self.guard:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                time.sleep(0.05)
                if self.fail_batch is not None and input[0] == f"t{self.fail_batch * 10}":
                    raise ConnectionError(f"batch {self.fail_batch}")
                from types import SimpleNamespace
                # Rows come back out of order; the helper must restore input order.
                rows = [SimpleNamespace(index=i, embedding=[float(text[1:])] + [0.0] * 1023)
                        for i, text in enumerate(input)]
                return SimpleNamespace(data=list(reversed(rows)))
            finally:
                with self.guard:
                    self.active -= 1

    def test_batches_run_concurrently_and_keep_input_order(self):
        client = self.Client()
        texts = [f"t{i}" for i in range(35)]
        vectors = embedding_vectors(client, texts)
        self.assertEqual([v[0] for v in vectors], [float(i) for i in range(35)])
        self.assertGreater(client.peak, 1)

    def test_first_failing_batch_in_input_order_is_raised(self):
        with self.assertRaisesRegex(ConnectionError, "batch 1"):
            embedding_vectors(self.Client(fail_batch=1), [f"t{i}" for i in range(30)])


class RerankerRetryTests(unittest.TestCase):
    def reranker(self):
        return QwenReranker(RerankerConfig("https://rerank.example/api", "synthetic-key", instruction="x"))

    def opener(self, outcomes):
        calls = []

        class Opener:
            def open(self, _request, timeout):
                calls.append(timeout)
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return io.BytesIO(json.dumps(outcome).encode())

        return Opener(), calls

    def http_error(self, code):
        return urllib.error.HTTPError("https://rerank.example/api", code, "error", {}, io.BytesIO(b"{}"))

    def test_transient_failures_are_retried_then_succeed(self):
        good = {"output": {"results": [{"index": 0, "relevance_score": 0.5}]}}
        opener, calls = self.opener([self.http_error(429), urllib.error.URLError("reset"), good])
        with patch("mini_vector.reranker.urllib.request.build_opener", return_value=opener), \
             patch("mini_vector.reranker.time.sleep") as sleep:
            self.assertEqual(self.reranker()("q", ["d"]), good)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleep.call_count, 2)

    def test_permanent_failure_is_not_retried(self):
        opener, calls = self.opener([self.http_error(400)])
        with patch("mini_vector.reranker.urllib.request.build_opener", return_value=opener), \
             patch("mini_vector.reranker.time.sleep"):
            with self.assertRaises(urllib.error.HTTPError):
                self.reranker()("q", ["d"])
        self.assertEqual(len(calls), 1)

    def test_retries_are_bounded(self):
        opener, calls = self.opener([self.http_error(503)] * 5)
        with patch("mini_vector.reranker.urllib.request.build_opener", return_value=opener), \
             patch("mini_vector.reranker.time.sleep"):
            with self.assertRaises(urllib.error.HTTPError):
                self.reranker()("q", ["d"])
        self.assertEqual(len(calls), 3)


@unittest.skipUnless(HAS_API, "Install mini-vector-memory[api] to run HTTP tests")
class ConcurrentAddTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "aml.sqlite3"
        self.headers = {"Authorization": "Bearer local-test-token"}
        self.guard = threading.Lock()
        self.active = {}
        self.overlap = []

    def tracked_llm(self, delay=0.0, gate=None):
        def llm(messages):
            if "Generate retrieval questions" in messages[0]["content"]:
                data = json.loads(messages[-1]["content"])
                return json.dumps({"memories": [{"id": item["id"], "queries": ["Q " + item["text"]]}
                                               for item in data["memories"]]})
            payload = json.loads(messages[-1]["content"].split("\n\n", 1)[1])
            user = payload["new_messages"][0]["content"].split(":", 1)[0]
            with self.guard:
                self.active[user] = self.active.get(user, 0) + 1
                self.overlap.append(dict(self.active))
            try:
                if gate is not None:
                    gate.wait(5)
                time.sleep(delay)
                return json.dumps({"memory": [{"text": m["content"]} for m in payload["new_messages"]]})
            finally:
                with self.guard:
                    self.active[user] -= 1
        return llm

    @staticmethod
    def embed(texts):
        return [[1.0, float(len(text) % 7) + 0.5] for text in texts]

    def app(self, llm):
        return create_app(db_path=self.path, api_token="local-test-token", llm=llm, embedder=self.embed,
                          rerank_mode="none", query_rewrite_enabled=False)

    def add(self, client, user, request_id, content, session="s1"):
        return client.post("/add", headers=self.headers, json={
            "request_id": request_id, "user_id": user, "session_id": session,
            "messages": [{"role": "user", "content": f"{user}: {content}"}]})

    def test_different_users_overlap_while_one_user_stays_serial(self):
        gate = threading.Barrier(3, timeout=5)
        with TestClient(self.app(self.tracked_llm(gate=gate))) as client, \
             ThreadPoolExecutor(max_workers=3) as pool:
            # Three users reach extraction together only if Adds overlap.
            responses = list(pool.map(lambda user: self.add(client, user, f"r-{user}", "fact"),
                                      ["alice", "bob", "carol"]))
        self.assertTrue(all(r.status_code == 200 for r in responses), [r.text for r in responses])
        self.assertEqual(max(sum(state.values()) for state in self.overlap), 3)

    def test_same_user_adds_never_overlap_and_see_previous_context(self):
        with TestClient(self.app(self.tracked_llm(delay=0.02))) as client, \
             ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(lambda i: self.add(client, "alice", f"r{i}", f"fact {i}"), range(6)))
            self.assertTrue(all(r.status_code == 200 for r in responses))
            self.assertTrue(all(state.get("alice", 0) <= 1 for state in self.overlap))
            store = client.app.state.backend.store
            self.assertEqual(store.db.execute("SELECT count(*) FROM memories").fetchone()[0], 6)
            self.assertEqual(store.db.execute("SELECT count(*) FROM aml_requests").fetchone()[0], 6)

    def test_search_is_not_blocked_by_another_users_pending_add(self):
        started, release = threading.Event(), threading.Event()
        base = self.tracked_llm()

        def llm(messages):
            if "bob:" in messages[-1]["content"]:
                started.set()
                release.wait(5)
            return base(messages)

        with TestClient(self.app(llm)) as client, ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(self.add(client, "alice", "r1", "Alice lives in Berlin").status_code, 200)
            pending = pool.submit(self.add, client, "bob", "r2", "Bob lives in Rome")
            try:
                self.assertTrue(started.wait(3))
                search = client.post("/search", headers=self.headers,
                                     json={"query": "Berlin", "user_id": "alice", "top_k": 5})
                self.assertEqual(search.status_code, 200)
                self.assertEqual(len(search.json()["data"]), 1)
                self.assertFalse(pending.done())
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=5).status_code, 200)

    def test_same_request_id_under_two_users_commits_once(self):
        gate = threading.Barrier(2, timeout=5)
        with TestClient(self.app(self.tracked_llm(gate=gate))) as client, \
             ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda user: self.add(client, user, "shared", "fact"), ["alice", "bob"]))
            self.assertEqual(sorted(r.status_code for r in responses), [200, 409])
            store = client.app.state.backend.store
            self.assertEqual(store.db.execute("SELECT count(*) FROM memories").fetchone()[0], 1)
            self.assertEqual(store.db.execute("SELECT count(*) FROM messages").fetchone()[0], 1)

    def test_cached_pools_match_a_rebuild_after_concurrent_commits_and_rollback(self):
        with TestClient(self.app(self.tracked_llm())) as client:
            store = client.app.state.backend.store
            self.assertEqual(self.add(client, "alice", "r0", "seed").status_code, 200)
            # Warm both pools for alice, then grow them from several threads.
            store.semantic([1.0, 0.5], {"user_id": "alice"}, 10)
            store.query_semantic([1.0, 0.5], {"user_id": "alice"}, 10)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: self.add(client, "alice" if i % 2 else "bob", f"r{i + 1}", f"f{i}"),
                              range(8)))
            with patch.object(store, "complete", side_effect=RuntimeError("late failure")):
                self.assertEqual(self.add(client, "alice", "rollback", "never stored").status_code, 500)
            scope = {"user_id": "alice"}
            cached = [store.semantic([1.0, 2.5], scope, 50), store.query_semantic([1.0, 2.5], scope, 50)]
            self.assertTrue(store._pools)
            store._invalidate_pools()
            rebuilt = [store.semantic([1.0, 2.5], scope, 50), store.query_semantic([1.0, 2.5], scope, 50)]
            as_ids = lambda rows: [(item["id"], score) for item, score in rows]
            self.assertEqual([as_ids(rows) for rows in cached], [as_ids(rows) for rows in rebuilt])
            self.assertEqual(len(rebuilt[0]), 5)
            self.assertNotIn("never stored", json.dumps([item["memory"] for item, _ in rebuilt[0]]))

    def test_pool_decoding_releases_the_lock_and_stale_builds_are_not_cached(self):
        import mini_vector.aml_store as aml_store
        with TestClient(self.app(self.tracked_llm())) as client:
            store = client.app.state.backend.store
            self.assertEqual(self.add(client, "alice", "r1", "first").status_code, 200)
            store._invalidate_pools()
            decoding, release = threading.Event(), threading.Event()
            real_pool = aml_store.VectorPool
            first = []

            def slow_pool(records, **kwargs):
                rows = list(records)  # Rows were fetched under the lock; decoding is not.
                if not first:
                    first.append(threading.get_ident())
                    decoding.set()
                    release.wait(5)
                return real_pool(rows, **kwargs)

            with patch.object(aml_store, "VectorPool", side_effect=slow_pool), \
                 ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(store.semantic, [1.0, 0.5], {"user_id": "alice"}, 10)
                try:
                    self.assertTrue(decoding.wait(3))
                    # A commit for the same user proceeds while the pool decodes.
                    self.assertEqual(self.add(client, "alice", "r2", "second").status_code, 200)
                finally:
                    release.set()
                snapshot = pending.result(timeout=5)
            key = ((("user_id", "alice"),), False)
            # The Add's own lookup cached a pool that r2's commit then extended; the
            # slow build, made from pre-r2 rows, must not have replaced it.
            self.assertEqual(len(store._pools[key].ids), 2)
            self.assertIn(len(snapshot), (1, 2))
            self.assertEqual(len(store.semantic([1.0, 0.5], {"user_id": "alice"}, 10)), 2)
            store._invalidate_pools()
            self.assertEqual(len(store.semantic([1.0, 0.5], {"user_id": "alice"}, 10)), 2)

    def test_search_on_empty_database_does_not_fix_the_dimension(self):
        with TestClient(self.app(self.tracked_llm())) as client:
            search = client.post("/search", headers=self.headers,
                                 json={"query": "Berlin", "user_id": "alice", "top_k": 5})
            self.assertEqual(search.json(), {"data": []})
            store = client.app.state.backend.store
            self.assertIsNone(store.db.execute("SELECT dimension FROM aml_config").fetchone()[0])
            self.assertEqual(self.add(client, "alice", "r1", "fact").status_code, 200)
            self.assertEqual(store.db.execute("SELECT dimension FROM aml_config").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()

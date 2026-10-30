"""Exact retrieval and cache isolation across writes, transactions and restarts."""
from contextlib import closing
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from mini_vector.memory import SQLiteVectorStore, _cosine
from mini_vector.vector_scan import VectorPool, np


def meta(user="a", agent="one"):
    return {"user_id": user, "agent_id": agent, "created_at": "2026-10-25", "updated_at": "2026-10-25"}


class VectorScanTests(unittest.TestCase):
    def store(self, **kwargs):
        store = SQLiteVectorStore(":memory:", **kwargs)
        self.addCleanup(store.close)
        return store

    def seed(self, store):
        for mid, vector, queries, user, agent in (
            ("z", [1., 0.], [[0., 1.], [1., 0.]], "a", "two"),
            ("a", [1., 0.], [[1., 0.]], "a", "one"),
            ("b", [-1., 0.], [[0., 0.]], "a", "one"),
            ("foreign", [1., 0.], [[1., 0.]], "b", "one"),
        ):
            store.save(mid, mid+" evidence", vector, meta(user, agent),
                       [("question "+str(i), q) for i, q in enumerate(queries)])

    def test_both_backends_match_original_scores_and_stable_ties(self):
        for backend in ("python", "numpy") if np is not None else ("python",):
            with self.subTest(backend=backend):
                store = self.store(vector_backend=backend)
                self.seed(store)
                for method in (store.semantic, store.query_semantic):
                    result = method([1., 0.], {"user_id": "a"}, 3)
                    self.assertEqual([(r["id"], s) for r, s in result], [("z", 1.), ("a", 1.), ("b", 0.)])
                    self.assertEqual([r["id"] for r, _ in method([0., 0.], {"user_id": "a"}, 3)], ["z", "a", "b"])
                    self.assertEqual(method([1., 0.], {"user_id": "absent"}, 3), [])
                    self.assertEqual([r["id"] for r, _ in method([1., 0.], {"user_id": "a", "agent_id": "one"}, 1)], ["a"])

    def test_random_and_near_tie_shortlists_equal_original_python_scan(self):
        rng = random.Random(513)
        vectors = [[rng.uniform(-1, 1) for _ in range(1536)] for _ in range(90)]
        vectors += [vectors[0][:], [v*(1+1e-14) for v in vectors[0]], [0.]*1536]
        for backend in ("python", "numpy") if np is not None else ("python",):
            pool = VectorPool([(str(i//3), json.dumps(v)) for i, v in enumerate(vectors)], backend=backend)
            for query in (vectors[0], vectors[4], [rng.random() for _ in range(1536)]):
                expected = {}
                for i, vector in enumerate(vectors):
                    mid = str(i//3)
                    score = _cosine(query, vector)
                    expected[mid] = max(expected.get(mid, score), score)
                expected = sorted(expected.items(), key=lambda pair: pair[1], reverse=True)
                for limit in (0, 1, 20, 80, -1):
                    self.assertEqual(pool.rank(query, limit), expected[:limit])

    def test_cache_reuses_decode_but_reloads_after_each_kind_of_mutation(self):
        store = self.store()
        self.seed(store)
        scope = {"user_id": "a"}
        first = store._vector_pool(scope, questions=False)
        self.assertIs(store._vector_pool(scope, questions=False), first)
        store.update("b", "b new", [1., 0.], meta(), [("new q", [1., 0.])])
        self.assertIsNot(store._vector_pool(scope, questions=False), first)
        self.assertEqual(dict((r["id"], s) for r, s in store.semantic([1., 0.], scope, 4))["b"], 1.)
        store.query_semantic([1., 0.], scope, 4)
        store.replace_queries("b", [("changed", [0., 1.])], expected_text="b new")
        self.assertEqual(dict((r["id"], s) for r, s in store.query_semantic([1., 0.], scope, 4))["b"], 0.)
        store.update("z", "z evidence", None, {**meta(), "color": "blue"})
        self.assertEqual(store.semantic([1., 0.], scope, 4)[0][0]["metadata"]["color"], "blue")
        store.delete("z")
        self.assertNotIn("z", [r["id"] for r, _ in store.semantic([1., 0.], scope, 4)])
        store.save("later", "later", [1., 0.], meta())
        self.assertIn("later", [r["id"] for r, _ in store.semantic([1., 0.], scope, 4)])

    def test_uncommitted_pool_does_not_survive_rollback_or_commit(self):
        store = self.store()
        self.seed(store)
        scope = {"user_id": "a"}
        before = store.semantic([1., 0.], scope, 4)
        store.db.execute("BEGIN IMMEDIATE")
        store.db.execute("UPDATE memories SET embedding='[0,1]' WHERE id='z'")
        self.assertEqual(dict((r["id"], s) for r, s in store.semantic([1., 0.], scope, 4))["z"], 0.)
        self.assertFalse(store._vector_cache)
        store.db.rollback()
        self.assertEqual(store.semantic([1., 0.], scope, 4), before)
        store.db.execute("BEGIN IMMEDIATE")
        store.db.execute("UPDATE memories SET embedding='[0,1]' WHERE id='z'")
        store.semantic([1., 0.], scope, 4)
        store.db.commit()
        self.assertEqual(dict((r["id"], s) for r, s in store.semantic([1., 0.], scope, 4))["z"], 0.)

    def test_external_commit_invalidates_both_channels_and_reopen_is_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"store.sqlite3")
            store = SQLiteVectorStore(path)
            try:
                self.seed(store)
                scope = {"user_id": "a"}
                store.semantic([1., 0.], scope, 4)
                store.query_semantic([1., 0.], scope, 4)
                with closing(sqlite3.connect(path)) as other:
                    other.execute("UPDATE memories SET embedding='[0,1]' WHERE id='z'")
                    other.execute("UPDATE memory_queries SET embedding='[0,1]' WHERE memory_id='z'")
                    other.commit()
                for method in (store.semantic, store.query_semantic):
                    self.assertEqual(dict((r["id"], s) for r, s in method([1., 0.], scope, 4))["z"], 0.)
                expected = store.semantic([1., 0.], scope, 4)
            finally:
                store.close()
            reopened = SQLiteVectorStore(path)
            try:
                self.assertEqual(reopened.semantic([1., 0.], scope, 4), expected)
            finally:
                reopened.close()

    def test_cache_limits_and_disabled_cache(self):
        store = self.store(vector_cache_bytes=80)
        self.seed(store)
        first = store._vector_pool({"user_id": "a", "agent_id": "one"}, questions=False)
        second = store._vector_pool({"user_id": "b"}, questions=False)
        self.assertLessEqual(sum(p.nbytes for p in store._vector_cache.values()), 80)
        self.assertNotIn(first, store._vector_cache.values())
        self.assertIn(second, store._vector_cache.values())
        store._vector_pool({"user_id": "a"}, questions=True)
        self.assertLessEqual(sum(p.nbytes for p in store._vector_cache.values()), 80)
        disabled = self.store(vector_cache_bytes=0)
        self.seed(disabled)
        disabled.semantic([1., 0.], {"user_id": "a"}, 4)
        self.assertFalse(disabled._vector_cache)

    def test_missing_numpy_uses_python_and_explicit_numpy_fails_clearly(self):
        with patch("mini_vector.vector_scan.np", None):
            pool = VectorPool([("a", "[1,0]")])
            self.assertEqual(pool.backend, "python")
            self.assertEqual(pool.rank([1., 0.], 1), [("a", 1.)])
        with patch("mini_vector.memory.np", None):
            with self.assertRaisesRegex(RuntimeError, "\[fast\]"):
                SQLiteVectorStore(":memory:", vector_backend="numpy")

    def test_dimension_mismatch_raises_and_get_many_has_no_vectors(self):
        store = self.store()
        self.seed(store)
        for method in (store.semantic, store.query_semantic):
            with self.assertRaisesRegex(ValueError, "dimension"):
                method([1.], {"user_id": "a"}, 4)
        self.assertEqual(store.get_many(["z", "z", "absent"]), {"z": store.get("z")})
        self.assertNotIn("embedding", store.get("z"))
        with self.assertRaisesRegex(ValueError, "dimension"):
            VectorPool([("a", "[1,0]"), ("b", "[1]")])

    def test_scope_query_uses_index_and_keeps_original_insertion_order(self):
        store = self.store()
        self.seed(store)
        plan = [row[3] for row in store.db.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM memories WHERE json_extract(metadata, '$.user_id')=? ORDER BY rowid", ("a",))]
        self.assertTrue(any("memory_user_scope_idx" in text for text in plan))
        self.assertEqual([r["id"] for r in store.list({"user_id": "a"})], ["z", "a", "b"])


if __name__ == "__main__":
    unittest.main()

"""Retrieval regressions with real SQLite, synthetic vectors and injected LLMs."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from mini_vector import MiniVector
from mini_vector.retrieval import memory_queries, rerank_channel


class Models:
    def __init__(self):
        self.calls = []
        self.reverse = True

    def __call__(self, messages):
        system = messages[0]["content"]
        if "Generate retrieval questions" in system:
            data = json.loads(messages[-1]["content"])
            self.calls.append("queries")
            return json.dumps({"memories": [{"id": item["id"], "queries": [
                "What is the commuting method?", "How does the person commute?"
            ]} for item in data["memories"]]})
        if "Rank candidate memories" in system:
            data = json.loads(messages[-1]["content"])
            self.calls.append(("rerank", data))
            ids = [item["id"] for item in data["candidates"]]
            return json.dumps({"ids": list(reversed(ids)) if self.reverse else ids})
        self.calls.append("extract")
        return '{"memory":[{"text":"Alice cycles."},{"text":"Alice cycles."}]}'


class EnhancedRetrievalTests(unittest.TestCase):
    def memory(self, **kwargs):
        models = kwargs.pop("llm", Models())
        memory = MiniVector(db_path=kwargs.pop("db_path", ":memory:"), llm=models,
                            embedder=kwargs.pop("embedder", lambda texts: [[1., 0.] for _ in texts]),
                            **kwargs)
        self.addCleanup(memory.close)
        return memory, models

    def test_memory_question_route_returns_original_fact_despite_orthogonal_fact_vector(self):
        def embed(texts):
            return [[0., 1.] if "commut" in text.lower() else [1., 0.] for text in texts]

        memory, models = self.memory(embedder=embed)
        added = memory.add("I cycle.", user_id="alice")
        mid = added["results"][0]["id"]
        self.assertEqual(len(added["results"]), 1)
        self.assertEqual(len(memory.store.get_queries(mid)), 2)
        trace = {}
        found = memory.search("commuting method", filters={"user_id": "alice"}, research_trace=trace)
        self.assertEqual([item["memory"] for item in found["results"]], ["Alice cycles."])
        self.assertEqual(trace["channels"]["semantic"]["eligible_ids"], [])
        self.assertEqual(trace["channels"]["memory_query"]["eligible_ids"], [mid])
        self.assertEqual(trace["channels"]["bm25"]["eligible_ids"], [])
        self.assertNotIn("score_details", found["results"][0])
        self.assertEqual(memory.search("commuting", filters={"user_id": "bob"})["results"], [])
        self.assertEqual(models.calls, ["extract", "queries"])
        memory.add("I cycle again.", user_id="alice")
        self.assertEqual(models.calls, ["extract", "queries", "extract"])

    def test_question_pool_deduplicates_before_limit_and_never_crosses_scope(self):
        memory, _ = self.memory()
        for user, text in (("a", "first"), ("a", "second"), ("b", "foreign")):
            memory.add(text, user_id=user, infer=False)
        ranked = memory.store.query_semantic([1., 0.], {"user_id": "a"}, 2)
        self.assertEqual({item["memory"] for item, _ in ranked}, {"first", "second"})
        self.assertEqual(len({item["id"] for item, _ in ranked}), 2)

    def test_each_channel_reranks_before_rrf_and_preserves_evidence_and_top_k(self):
        memory, models = self.memory(rerank_limit=2)
        for text in ("commute alpha", "commute beta", "commute gamma"):
            memory.add(text, user_id="a", infer=False)
        trace = {}
        result = memory.search("commute", filters={"user_id": "a"}, top_k=2,
                               explain=True, research_trace=trace)["results"]
        self.assertEqual(len(result), 2)
        self.assertEqual(len([call for call in models.calls if isinstance(call, tuple)]), 3)
        for channel in trace["channels"].values():
            original = channel["eligible_ids"]
            self.assertEqual(channel["rerank"]["status"], "succeeded")
            self.assertEqual(channel["rerank"]["output_ids"], original[1::-1] + original[2:])
            self.assertEqual(set(channel["rerank"]["output_ids"]), set(original))
        self.assertEqual([row["id"] for row in result], [row["memory_id"] for row in trace["stages"]["final"]])
        for item in result:
            ranks = item["score_details"]["channel_ranks"]
            self.assertAlmostEqual(item["score"], sum(1 / (60 + rank) for rank in ranks.values()) / (3 / 61))
            self.assertEqual(item["memory"], memory.get(item["id"])["memory"])

    def test_lexical_only_candidate_survives_semantic_threshold(self):
        def embed(texts):
            return [[0., 1.] if text == "needle evidence" else [1., 0.] for text in texts]

        memory, _ = self.memory(memory_query_enabled=False, embedder=embed, rerank_mode="llm")
        memory.add("other material", user_id="a", infer=False)
        target = memory.add("needle evidence", user_id="a", infer=False)["results"][0]["id"]
        result = memory.search("needle", filters={"user_id": "a"}, top_k=2)["results"]
        self.assertIn(target, [item["id"] for item in result])

    def test_update_metadata_keeps_queries_text_update_replaces_and_delete_cleans(self):
        memory, models = self.memory()
        mid = memory.add("old text", user_id="a", infer=False)["results"][0]["id"]
        before = memory.store.get_queries(mid)
        models.calls.clear()
        memory.update(mid, metadata={"color": "blue"})
        self.assertEqual(models.calls, [])
        self.assertEqual(memory.store.get_queries(mid), before)

        def new_queries(_messages):
            return '{"memories":[{"id":"m01","queries":["Where is the new place?"]}]}'
        memory.llm = new_queries
        memory.update(mid, "new text")
        self.assertEqual(memory.store.get_queries(mid), ["Where is the new place?"])
        memory.delete(mid)
        self.assertEqual(memory.store.get_queries(mid), [])
        self.assertEqual(memory.store.query_semantic([1., 0.], {"user_id": "a"}, 10), [])
        self.assertEqual([row["event"] for row in memory.history(mid)["history"]],
                         ["ADD", "UPDATE", "UPDATE", "DELETE"])

    def test_disabled_generation_text_update_removes_stale_question_index(self):
        memory, _ = self.memory()
        mid = memory.add("old fact", user_id="a", infer=False)["results"][0]["id"]
        memory.memory_query_enabled = False
        memory.update(mid, "new fact")
        self.assertEqual(memory.store.get_queries(mid), [])

    def test_bad_generation_never_persists_fact_or_mutates_updated_fact(self):
        memory, _ = self.memory()
        mid = memory.add("preserved", user_id="a", infer=False)["results"][0]["id"]
        old_queries = memory.store.get_queries(mid)
        memory.llm = lambda _: "not JSON"
        with self.assertRaisesRegex(RuntimeError, "query generation"):
            memory.add("new fact", user_id="a", infer=False)
        with self.assertRaisesRegex(RuntimeError, "query generation"):
            memory.update(mid, "changed")
        self.assertEqual([item["memory"] for item in memory.get_all(filters={"user_id": "a"})["results"]], ["preserved"])
        self.assertEqual(memory.store.get_queries(mid), old_queries)

    def test_old_database_reopen_explicit_backfill_and_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "old.sqlite3")
            old = MiniVector(db_path=path, llm=Models(), embedder=lambda texts: [[1., 0.] for _ in texts],
                             memory_query_enabled=False, rerank_mode="none")
            mid = old.add("Alice cycles.", user_id="a", infer=False)["results"][0]["id"]
            old.add("Private", user_id="b", infer=False)
            old.close()
            # Simulate an actual pre-upgrade database without the new side table.
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("DROP TABLE memory_queries")
                connection.commit()
            current = MiniVector(db_path=path, llm=Models(), embedder=lambda texts: [[1., 0.] for _ in texts])
            try:
                self.assertEqual(current.store.get_queries(mid), [])
                self.assertEqual(current.backfill_memory_queries(filters={"user_id": "a"}), {"indexed": 1})
                self.assertEqual(current.backfill_memory_queries(filters={"user_id": "a"}), {"indexed": 0})
                foreign = current.get_all(filters={"user_id": "b"})["results"][0]
                self.assertEqual(current.store.get_queries(foreign["id"]), [])
                self.assertEqual(len(current.history(mid)["history"]), 1)
            finally:
                current.close()
            reopened = MiniVector(db_path=path, llm=Models(), embedder=lambda texts: [[1., 0.] for _ in texts])
            try:
                self.assertEqual(len(reopened.store.get_queries(mid)), 2)
                self.assertEqual(reopened.get(mid)["memory"], "Alice cycles.")
            finally:
                reopened.close()

    def test_invalid_configuration_is_rejected_before_creating_database(self):
        for kwargs in ({"memory_query_enabled": "yes"}, {"queries_per_memory": 0},
                       {"queries_per_memory": True}, {"rerank_mode": "unknown"}, {"rerank_limit": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MiniVector(db_path=":memory:", llm=Models(), embedder=lambda _: [], **kwargs)


class StructuredOutputTests(unittest.TestCase):
    def test_invalid_question_arrays_and_id_mapping_are_rejected(self):
        for response in ({}, [], {"memories": []},
                         {"memories": [{"id": "m99", "queries": ["q"]}]},
                         *({"memories": [{"id": "m01", "queries": queries}]}
                           for queries in ([], "q", [" "], [1], ["q", "q"], ["q", "x", "y", "z"]))):
            with self.subTest(response=response), self.assertRaises(ValueError):
                memory_queries(lambda _: json.dumps(response), ["fact"], 3)

    def test_malformed_or_failed_rerank_preserves_entire_channel_and_sanitizes_error(self):
        candidates = [{"id": str(i), "memory": f"fact {i}"} for i in range(4)]
        for response in ({}, {"ids": []}, {"ids": ["m01"]}, {"ids": ["m01", "m01"]},
                         {"ids": ["m01", "m99"]}, {"ids": ["m01", 2]}, "invalid JSON"):
            def complete(_):
                return response if isinstance(response, str) else json.dumps(response)
            with self.subTest(response=response):
                result, trace = rerank_channel(complete, "query", candidates, 2, True)
                self.assertEqual(result, candidates)
                self.assertEqual(trace["status"], "fallback")
        def fail(_):
            raise ConnectionError("synthetic-private-provider-detail")
        result, trace = rerank_channel(fail, "query", candidates, 2, True)
        self.assertEqual(result, candidates)
        self.assertEqual(trace["error_type"], "ConnectionError")
        self.assertNotIn("synthetic-private-provider-detail", json.dumps(trace))

    def test_empty_and_singleton_channels_skip_model(self):
        def fail(_):
            raise AssertionError("No call expected")
        for candidates in ([], [{"id": "one", "memory": "fact"}]):
            ranked, trace = rerank_channel(fail, "query", candidates, 30, True)
            self.assertEqual(ranked, candidates)
            self.assertEqual(trace["api_calls"], 0)


if __name__ == "__main__":
    unittest.main()

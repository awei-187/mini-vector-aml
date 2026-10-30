"""Decomposition rewrite contract, search_path dispatch and multi-channel RRF math."""
import json
import unittest

from legacy_helpers import legacy_memory as MiniVector
from mini_vector.query_rewrite import QUERY_REWRITE_PROMPT, rewrite_query


class FakeReranker:
    def __init__(self, score_for):
        self.config = type("Cfg", (), {"public": lambda self: {"model": "fake"}})()
        self._score_for = score_for

    def __call__(self, query, documents):
        values = self._score_for(documents)
        return {"output": {"results": [{"index": i, "relevance_score": s}
                                       for i, s in enumerate(values)]}}


def keyword_embed(texts):
    table = {"alpha": [1., 0., 0.], "beta": [0., 1., 0.], "gamma": [0., 0., 1.]}
    return [next((v for k, v in table.items() if k in text), [0., 0., 0.]) for text in texts]


class RewriteContractTests(unittest.TestCase):
    def complete(self, value):
        return lambda _: value if isinstance(value, str) else json.dumps(value)

    def test_valid_responses_keep_order_dedupe_and_truncate(self):
        rows = [({"queries": ["one"]}, ["one"], False),
                ({"queries": [" b ", "b", "a"]}, ["b", "a"], False),
                ({"queries": ["a", "b", "c", "d"]}, ["a", "b", "c"], True),
                ({"queries": ["x"], "extra": 1}, ["x"], False)]
        for value, expected, truncated in rows:
            with self.subTest(value=value):
                subqueries, info = rewrite_query(self.complete(value), "question", 3)
                self.assertEqual(subqueries, expected)
                self.assertEqual(info["status"], "ok")
                self.assertEqual(info["truncated"], truncated)
                self.assertEqual(info["subqueries"], expected)

    def test_structural_failures_fall_back_with_reason_and_no_subqueries(self):
        rows = [("not json", "JSONDecodeError"), ('["a"]', "invalid_response"),
                ('{"nope": 1}', "invalid_queries"), ('{"queries": "x"}', "invalid_queries"),
                ('{"queries": []}', "empty_queries"), ('{"queries": [" ", ""]}', "empty_queries"),
                ('{"queries": [1, 2]}', "empty_queries")]
        for value, reason in rows:
            with self.subTest(value=value):
                subqueries, info = rewrite_query(self.complete(value), "question", 3)
                self.assertEqual((subqueries, info["status"], info["reason"]), ([], "fallback", reason))

    def test_provider_exception_fails_closed_without_error_text(self):
        def broken(_):
            raise ConnectionError("secret provider body")
        subqueries, info = rewrite_query(broken, "question", 3)
        self.assertEqual((subqueries, info["status"], info["reason"]), ([], "fallback", "ConnectionError"))
        self.assertNotIn("secret", json.dumps(info))

    def test_limit_is_respected_for_smaller_limits(self):
        subqueries, info = rewrite_query(self.complete({"queries": ["a", "b", "c"]}), "q", 1)
        self.assertEqual((subqueries, info["truncated"]), (["a"], True))

    def test_frozen_prompt_matches_plan_section_3(self):
        self.assertTrue(QUERY_REWRITE_PROMPT.startswith("You rewrite a single user question"))
        self.assertIn('{"queries": ["...", ...]}', QUERY_REWRITE_PROMPT)
        self.assertTrue(QUERY_REWRITE_PROMPT.rstrip().endswith("outside the JSON object."))


class SearchPathAndRewriteTests(unittest.TestCase):
    def memory(self, *, llm=None, **kwargs):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return keyword_embed(texts)

        def default_llm(messages):
            self.assertEqual(messages[0]["content"], QUERY_REWRITE_PROMPT)
            self.assertEqual(messages[1]["content"], json.dumps({"question": "alpha"}))
            return json.dumps({"queries": ["beta", "gamma"]})
        memory = MiniVector(db_path=":memory:", llm=llm or default_llm, embedder=embed, **kwargs)
        self.addCleanup(memory.close)
        memory._embed_calls = calls
        for text in ("alpha fact", "beta fact", "gamma fact"):
            memory.add(text, user_id="a", infer=False)
        calls.clear()
        return memory

    def test_rewrite_adds_four_channels_with_exact_rrf_scores_and_one_embedding_batch(self):
        memory = self.memory(memory_query_enabled=False, rerank_mode="none",
                             search_path="channels", query_rewrite_enabled=True)
        trace = {}
        found = memory.search("alpha", filters={"user_id": "a"}, explain=True,
                              research_trace=trace)["results"]
        self.assertEqual(memory._embed_calls, [["alpha", "beta", "gamma"]])
        self.assertEqual(trace["schema_version"], 5)
        self.assertEqual(trace["query_rewrite"], {"enabled": True, "status": "ok",
                                                  "subqueries": ["beta", "gamma"], "truncated": False})
        self.assertEqual(trace["search_path"], "channels")
        self.assertEqual(sorted(trace["channels"]),
                         ["bm25", "memory_query", "rewrite_1", "rewrite_1_bm25",
                          "rewrite_2", "rewrite_2_bm25", "semantic"])
        scores = {row["memory"]: row["score"] for row in found}
        # Six active channels; the question and each subquery each contribute two
        # rank-1 channels (semantic + bm25), so every fact scores (2/61)/(6/61).
        self.assertAlmostEqual(scores["alpha fact"], (2 / 61) / (6 / 61))
        self.assertAlmostEqual(scores["beta fact"], (2 / 61) / (6 / 61))
        self.assertAlmostEqual(scores["gamma fact"], (2 / 61) / (6 / 61))
        ranks = {row["memory"]: sorted(row["score_details"]["channel_ranks"]) for row in found}
        self.assertEqual(ranks["alpha fact"], ["bm25", "semantic"])
        self.assertEqual(ranks["beta fact"], ["rewrite_1", "rewrite_1_bm25"])
        self.assertEqual(ranks["gamma fact"], ["rewrite_2", "rewrite_2_bm25"])

    def test_rewrite_fallback_keeps_original_search_and_records_reason(self):
        def broken(_):
            raise RuntimeError("provider down")
        memory = self.memory(llm=broken, memory_query_enabled=False, rerank_mode="none",
                             search_path="channels", query_rewrite_enabled=True)
        trace = {}
        found = memory.search("alpha", filters={"user_id": "a"}, research_trace=trace)["results"]
        self.assertEqual([row["memory"] for row in found], ["alpha fact"])
        self.assertEqual(memory._embed_calls, [["alpha"]])
        self.assertEqual(trace["schema_version"], 5)
        self.assertEqual(trace["query_rewrite"]["status"], "fallback")
        self.assertEqual(trace["query_rewrite"]["reason"], "RuntimeError")
        self.assertEqual(trace["query_rewrite"]["subqueries"], [])
        self.assertEqual(sorted(trace["channels"]), ["bm25", "memory_query", "semantic"])

    def test_empty_subqueries_response_falls_back(self):
        memory = self.memory(llm=lambda _: '{"queries": []}', memory_query_enabled=False,
                             rerank_mode="none", search_path="channels", query_rewrite_enabled=True)
        trace = {}
        memory.search("alpha", filters={"user_id": "a"}, research_trace=trace)
        self.assertEqual(trace["query_rewrite"]["status"], "fallback")
        self.assertEqual(trace["query_rewrite"]["reason"], "empty_queries")

    def test_legacy_path_never_rewrites_and_stays_byte_identical(self):
        memory = self.memory(llm=lambda _: (_ for _ in ()).throw(AssertionError("no rewrite on legacy")),
                             memory_query_enabled=False, rerank_mode="none",
                             query_rewrite_enabled=True)
        trace = {}
        memory.search("alpha", filters={"user_id": "a"}, research_trace=trace)
        self.assertEqual(trace["schema_version"], 1)
        self.assertNotIn("search_path", trace)
        self.assertNotIn("query_rewrite", trace)

    def test_channels_path_gives_two_channel_rrf_while_auto_keeps_legacy(self):
        for path, version in (("auto", 1), ("channels", 2)):
            memory = self.memory(memory_query_enabled=False, rerank_mode="none", search_path=path)
            trace = {}
            found = memory.search("alpha", filters={"user_id": "a"}, research_trace=trace)["results"]
            self.assertEqual(trace["schema_version"], version)
            if path == "channels":
                self.assertEqual(trace["search_path"], "channels")
                # semantic rank 1 + bm25 rank 1 over two active channels.
                self.assertAlmostEqual(found[0]["score"], min(1.0, (2 / 61) / (2 / 61)))
            else:
                self.assertNotIn("search_path", trace)
                # Legacy semantics: (semantic 1.0 + normalized bm25 sigmoid(4)) / 2.
                legacy_bm25 = 1 / (1 + pow(2.718281828459045, -4))
                self.assertAlmostEqual(found[0]["score"], (1.0 + legacy_bm25) / 2)

    def test_qwen_rerank_works_over_rewrite_channels(self):
        # Score by document text so the outcome is independent of UUID tie order.
        def scores(documents):
            return [0.9 if "gamma" in doc else 0.1 for doc in documents]
        memory = self.memory(memory_query_enabled=False, rerank_mode="qwen",
                             query_rewrite_enabled=True, reranker=FakeReranker(scores))
        trace = {}
        found = memory.search("alpha", filters={"user_id": "a"}, research_trace=trace)["results"]
        self.assertEqual(trace["query_rewrite"]["status"], "ok")
        self.assertEqual(trace["qwen_rerank"]["status"], "succeeded")
        self.assertEqual(trace["schema_version"], 5)
        self.assertEqual([row["memory"] for row in found][0], "gamma fact")

    def test_disabled_rewrite_makes_no_retrieval_llm_call(self):
        memory = self.memory(llm=lambda _: (_ for _ in ()).throw(AssertionError("no llm calls")),
                             memory_query_enabled=False, rerank_mode="none")
        found = memory.search("alpha", filters={"user_id": "a"})["results"]
        self.assertEqual([row["memory"] for row in found], ["alpha fact"])

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"search_path": "legacy"}, {"search_path": 1},
                       {"query_rewrite_enabled": "yes"}, {"query_rewrite_limit": 0},
                       {"query_rewrite_limit": 4}, {"query_rewrite_limit": True},
                       {"query_rewrite_limit": 2.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MiniVector(db_path=":memory:", llm=lambda _: "{}", embedder=keyword_embed, **kwargs)


if __name__ == "__main__":
    unittest.main()

"""Fused selection contracts, evidence preservation and isolated actual search."""
import json
import unittest

from mini_vector import MiniVector
from mini_vector.retrieval import rerank_fused


class GlobalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"id": str(i), "memory": f"evidence {i}",
                            "metadata": {"session_date": "2023-01-01", "private_field": "excluded"}}
                           for i in range(6)]

    def test_partial_selection_is_valid_and_every_unselected_fact_survives(self):
        captured = []
        def model(messages):
            captured.append(json.loads(messages[-1]["content"]))
            return '{"ids":["m03","m01"]}'
        result, trace = rerank_fused(model, "query", self.candidates, 4, 2, diverse=True)
        self.assertEqual([row["id"] for row in result], ["2", "0", "1", "3", "4", "5"])
        self.assertEqual(trace["status"], "succeeded")
        self.assertEqual(trace["selected_ids"], ["2", "0"])
        self.assertEqual(trace["api_calls"], 1)
        self.assertEqual(captured[0]["max_select"], 2)
        self.assertEqual(captured[0]["query"], "query")
        self.assertNotIn("private_field", json.dumps(captured))
        self.assertEqual(captured[0]["candidates"][0]["observation_date"], "2023-01-01")
        self.assertEqual(result[0], self.candidates[2])

    def test_invalid_or_provider_failure_falls_back_without_dropping_or_rewriting(self):
        for payload in ([], {}, {"ids": []}, {"ids": ["m01", "m01"]},
                        {"ids": ["m99"]}, {"ids": [1]},
                        {"ids": ["m01", "m02", "m03"]}, "malformed"):
            def model(_):
                return payload if isinstance(payload, str) else json.dumps(payload)
            with self.subTest(payload=payload):
                ranked, trace = rerank_fused(model, "query", self.candidates, 4, 2)
                self.assertEqual(ranked, self.candidates)
                self.assertEqual(trace["status"], "fallback")
                self.assertEqual(trace["selected_ids"], [])
        def fail(_):
            raise ConnectionError("private-provider-detail")
        _, trace = rerank_fused(fail, "query", self.candidates, 4, 2)
        self.assertNotIn("private-provider-detail", json.dumps(trace))

    def test_empty_and_singleton_skip_calls(self):
        def forbidden(_):
            self.fail("No API call expected")
        for rows in ([], self.candidates[:1]):
            found, trace = rerank_fused(forbidden, "query", rows, 60, 20)
            self.assertEqual(found, rows)
            self.assertEqual(trace["api_calls"], 0)

    def test_search_calls_once_after_fusion_and_keeps_scope_and_scores(self):
        for mode in ("global", "evidence"):
            calls = []
            def model(messages):
                calls.append(messages)
                return '{"ids":["m03"]}'
            with self.subTest(mode=mode):
                memory = MiniVector(db_path=":memory:", llm=model, embedder=lambda texts: [[1., 0.] for _ in texts],
                                    memory_query_enabled=False, rerank_mode=mode, rerank_limit=3)
                try:
                    for i in range(5):
                        memory.add(f"evidence {i}", user_id="a", infer=False)
                    memory.add("private evidence", user_id="b", infer=False)
                    trace = {}
                    result = memory.search("evidence", top_k=2, filters={"user_id": "a"},
                                           explain=True, research_trace=trace)["results"]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(trace["schema_version"], 3)
                    self.assertEqual(result[0]["id"], trace["stages"]["combined_order"][2])
                    self.assertEqual(result[1]["id"], trace["stages"]["combined_order"][0])
                    self.assertEqual(trace["global_rerank"]["diverse"], mode == "evidence")
                    self.assertEqual(len(trace["global_rerank"]["output_ids"]), 5)
                    self.assertNotIn("private evidence", json.dumps(calls))
                    self.assertTrue(all(channel["rerank"]["api_calls"] == 0 for channel in trace["channels"].values()))
                    for row in result:
                        self.assertEqual(row["score"], row["score_details"]["final_score"])
                finally:
                    memory.close()

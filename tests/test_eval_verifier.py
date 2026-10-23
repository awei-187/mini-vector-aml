"""Independent enhanced-trace verification using actual SQLite search output."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

from mini_vector import MiniVector

spec = importlib.util.spec_from_file_location("enhanced_eval_verifier", Path(__file__).resolve().parents[1] / "evals/verify_results.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class EnhancedVerifierTests(unittest.TestCase):
    def fixture(self, *, fallback=False, memory_queries=True):
        def llm(messages):
            data = json.loads(messages[-1]["content"])
            if "Generate retrieval questions" in messages[0]["content"]:
                return json.dumps({"memories": [{"id": row["id"], "queries": [row["text"] + "?"]}
                                               for row in data["memories"]]})
            return json.dumps({"ids": [] if fallback else [row["id"] for row in reversed(data["candidates"])]})

        memory = MiniVector(db_path=":memory:", llm=llm, embedder=lambda texts: [[1., 0.] for _ in texts],
                            memory_query_enabled=memory_queries, rerank_limit=2)
        self.addCleanup(memory.close)
        for word in ("alpha", "beta", "gamma", "delta"):
            memory.add("evidence " + word, user_id="test", infer=False)
        trace = {}
        found = memory.search("evidence", top_k=2, filters={"user_id": "test"},
                              explain=True, research_trace=trace)["results"]
        exported = {row["id"]: {"id": row["id"], "data": row["data"], "embedding": json.loads(row["embedding"])}
                    for row in memory.store.db.execute("SELECT id,data,embedding FROM memories")}
        questions = [{"memory_id": mid, "position": pos, "query": query, "embedding": json.loads(vector)}
                     for mid, pos, query, vector in memory.store.db.execute(
                         "SELECT memory_id,position,query,embedding FROM memory_queries")]
        return trace, found, [1., 0.], questions, exported

    def test_real_three_channel_trace_and_rerank_tail_pass(self):
        trace, found, vector, questions, exported = self.fixture()
        self.assertEqual(verifier.verify_enhanced_ranking(trace, found, vector, questions, exported),
                         {"semantic": "succeeded", "bm25": "succeeded", "memory_query": "succeeded"})

    def test_wrong_question_vectors_and_fusion_scores_are_rejected(self):
        trace, found, vector, questions, exported = self.fixture()
        broken = copy.deepcopy(questions)
        for row in broken:
            row["embedding"] = [0., 1.]
        with self.assertRaises(AssertionError):
            verifier.verify_enhanced_ranking(trace, found, vector, broken, exported)
        broken = copy.deepcopy(found)
        broken[0]["score"] = 0.
        with self.assertRaises(AssertionError):
            verifier.verify_enhanced_ranking(trace, broken, vector, questions, exported)

    def test_channel_fallback_and_disabled_question_channel_pass(self):
        for enabled in (True, False):
            with self.subTest(memory_queries=enabled):
                trace, found, vector, questions, exported = self.fixture(fallback=True, memory_queries=enabled)
                states = verifier.verify_enhanced_ranking(trace, found, vector, questions, exported)
                self.assertEqual(states["semantic"], "fallback")
                self.assertEqual(states["memory_query"], "fallback" if enabled else "skipped")


if __name__ == "__main__":
    unittest.main()

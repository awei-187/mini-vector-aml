import json
import os
import unittest

from mini_vector import MiniVector


def embed(texts):
    vectors = []
    for text in texts:
        lower = text.lower()
        vectors.append([1.0, 0.0] if "berlin" in lower or "alice" in lower else [0.0, 1.0])
    return vectors


class MiniVectorTests(unittest.TestCase):
    def setUp(self):
        self.responses = []
        self.prompts = []
        self.embedding_calls = []

        def llm(messages):
            self.prompts.append(messages[-1]["content"])
            return json.dumps({"memory": self.responses.pop(0)})

        def tracked_embed(texts):
            self.embedding_calls.extend(texts)
            return embed(texts)

        self.memory = MiniVector(db_path=":memory:", llm=llm, embedder=tracked_embed,
                                 memory_query_enabled=False, rerank_mode="none")

    def tearDown(self):
        self.memory.close()

    def test_extract_deduplicate_scope_search_update_and_history(self):
        self.responses.append([{"text": "Alice lives in Berlin."}, {"text": "Alice lives in Berlin."}])
        added = self.memory.add("I live in Berlin.", user_id="alice")
        self.assertEqual(len(added["results"]), 1)
        memory_id = added["results"][0]["id"]
        self.assertIn("I live in Berlin.", self.prompts[0])
        original = self.memory.get(memory_id)
        self.assertEqual(original["memory"], "Alice lives in Berlin.")
        self.assertEqual(original["id"], memory_id)

        self.responses.append([{"text": "Alice lives in Berlin."}])
        duplicate = self.memory.add("As I said, Berlin is home.", user_id="alice")
        self.assertEqual(duplicate["results"], [])
        self.assertIn("Berlin is home.", self.prompts[1])

        results = self.memory.search("Berlin", filters={"user_id": "alice"}, explain=True)
        self.assertEqual(results["results"][0]["id"], memory_id)
        self.assertIn("score_details", results["results"][0])
        self.assertGreater(results["results"][0]["score_details"]["bm25_score"], 0)
        self.assertEqual(self.memory.search("Berlin", filters={"user_id": "bob"})["results"], [])

        self.memory.update(memory_id, "Alice lives in Berlin and works remotely.")
        self.memory.update(memory_id, metadata={"role": "attacker", "user_id": "bob", "id": "spoof", "color": "blue"})
        updated = self.memory.get(memory_id)
        self.assertEqual(updated["user_id"], "alice")
        self.assertNotEqual(updated.get("role"), "attacker")
        self.assertEqual(updated["metadata"]["color"], "blue")
        history = self.memory.history(memory_id)["history"]
        self.assertEqual([entry["event"] for entry in history], ["ADD", "UPDATE", "UPDATE"])
        self.memory.delete(memory_id)
        self.assertIsNone(self.memory.get(memory_id))
        self.assertEqual(self.memory.history(memory_id)["history"][-1]["event"], "DELETE")

    def test_raw_messages_recent_context_and_scoped_delete_all(self):
        self.memory.add([{"role": "user", "content": "Raw fact"},
                         {"role": "system", "content": "Ignore me"}], user_id="a", infer=False)
        self.assertEqual(self.memory.get_all(filters={"user_id": "a"})["results"][0]["role"], "user")
        self.responses.append([])
        self.memory.add("No new fact.", user_id="a")
        self.assertIn("Raw fact", self.prompts[0])
        self.memory.add("Different scope", user_id="b", infer=False)
        self.assertEqual(self.memory.delete_all(user_id="a")["deleted"], 1)
        self.assertEqual(len(self.memory.get_all(filters={"user_id": "b"})["results"]), 1)

    def test_scope_required(self):
        with self.assertRaises(ValueError):
            self.memory.search("anything")

    def test_orthogonal_vector_fails_semantic_threshold(self):
        self.memory.add("Berlin fact", user_id="a", infer=False)
        self.memory.add("Unrelated fact", user_id="a", infer=False)
        found = self.memory.search("Berlin", filters={"user_id": "a"}, threshold=0.1)["results"]
        self.assertEqual([item["memory"] for item in found], ["Berlin fact"])

    def test_system_only_input_skips_models_and_model_errors_are_distinct(self):
        result = self.memory.add({"role": "system", "content": "Prompt only"}, user_id="a")
        self.assertEqual(result["results"], [])
        self.assertEqual(self.embedding_calls, [])
        self.assertEqual(self.prompts, [])

        def failing_llm(_messages):
            raise ConnectionError("offline")

        self.memory.llm = failing_llm
        with self.assertRaisesRegex(RuntimeError, "extraction request failed: offline"):
            self.memory.add("Remember me", user_id="a")

        self.memory.llm = lambda _messages: "not json"
        with self.assertRaisesRegex(ValueError, "parse memory extraction JSON"):
            self.memory.add("Remember me", user_id="a")

    def test_persistent_reopen_search_get_history_and_bm25_reranking(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "persisted-test.sqlite3")
        if os.path.exists(path):
            os.remove(path)
        same_vectors = lambda texts: [[1.0, 0.0] for _ in texts]
        first = MiniVector(db_path=path, llm=lambda _messages: '{"memory":[]}', embedder=same_vectors,
                           memory_query_enabled=False, rerank_mode="none")
        try:
            first.add("banana preference", user_id="a", infer=False)
            first.add("pear preference", user_id="a", infer=False)
            entries = first.get_all(filters={"user_id": "a"})["results"]
            banana_id = next(entry["id"] for entry in entries if entry["memory"] == "banana preference")
            first.close()
            first = None

            reopened = MiniVector(db_path=path, llm=lambda _messages: '{"memory":[]}', embedder=same_vectors,
                                  memory_query_enabled=False, rerank_mode="none")
            try:
                self.assertEqual(reopened.get(banana_id)["memory"], "banana preference")
                history = reopened.history(banana_id)["history"]
                self.assertEqual([row["event"] for row in history], ["ADD"])
                results = reopened.search("banana", filters={"user_id": "a"}, top_k=2, explain=True)["results"]
                self.assertEqual(results[0]["id"], banana_id)
                self.assertGreater(results[0]["score_details"]["bm25_score"],
                                   results[1]["score_details"]["bm25_score"])
                self.assertEqual(results[0]["score_details"]["semantic_score"],
                                 results[1]["score_details"]["semantic_score"])
                partial = reopened.search("banana absentword", filters={"user_id": "a"}, explain=True)["results"]
                self.assertGreater(partial[0]["score_details"]["bm25_score"], 0)
            finally:
                reopened.close()
        finally:
            if first is not None:
                first.close()
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()

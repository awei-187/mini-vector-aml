"""Offline coverage of the supplied system prompt and its runtime context."""
from datetime import datetime, timezone
import hashlib
import json
import unittest

from mini_vector import MiniVector
from mini_vector.prompts import MEMORY_EXTRACTION_PROMPT, build_extraction_input, observation_date


class ExtractionPromptTests(unittest.TestCase):
    def test_supplied_text_is_preserved_verbatim_and_routed_to_fact_extraction(self):
        # Digest of the user's 2026-10-23 attachment after universal newline decoding.
        self.assertEqual(hashlib.sha256(MEMORY_EXTRACTION_PROMPT.encode()).hexdigest(),
                         "3824de576bfdbf82a162eec166e4d055013616c5bc726059b9706400d683e9d7")
        calls = []

        def llm(messages):
            calls.append(messages)
            return json.dumps({"memory": [{"id": "0", "text": "Alice likes cycling.",
                                            "attributed_to": "user", "linked_memory_ids": []}]})

        memory = MiniVector(db_path=":memory:", llm=llm,
                            embedder=lambda texts: [[1., 0.] for _ in texts],
                            memory_query_enabled=False, rerank_mode="none")
        self.addCleanup(memory.close)
        old = memory.add({"role": "user", "name": "Alice", "content": "Alice lives in Berlin."},
                         user_id="alice", infer=False)["results"][0]
        current = [{"role": "user", "content": "I like cycling.", "timestamp": 1704067200000}]
        added = memory.add(current, user_id="alice")["results"][0]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], {"role": "system", "content": MEMORY_EXTRACTION_PROMPT})
        payload = json.loads(calls[0][-1]["content"].split("\n\n", 1)[1])
        self.assertEqual(payload["new_messages"], current)
        self.assertEqual(payload["recent_messages"][0]["content"], "Alice lives in Berlin.")
        self.assertEqual(payload["existing_memories"], [{"id": old["id"], "text": old["memory"]}])
        self.assertEqual(payload["summary"], "")
        self.assertEqual(payload["recently_extracted_memories"], [])
        self.assertEqual(payload["observation_date"], "2024-01-01")
        self.assertEqual(payload["current_date"], datetime.now(timezone.utc).date().isoformat())
        self.assertNotEqual(added["id"], "0")
        self.assertEqual(memory.get(added["id"])["memory"], "Alice likes cycling.")

    def test_explicit_observation_date_and_original_locomo_session_date(self):
        messages = [{"role": "user", "content": "Yesterday", "timestamp": 1704067200000}]
        self.assertEqual(observation_date(messages, {"observation_date": "2023-05-08"}), "2023-05-08")
        self.assertEqual(observation_date([], {"session_date": "1:56 pm on 8 May, 2023"}), "2023-05-08")

    def test_missing_source_date_does_not_use_ingestion_or_current_date(self):
        prompt = build_extraction_input([{"role": "user", "content": "I went yesterday."}], [], [],
                                        {"created_at": "2026-10-23T10:00:00Z"})
        payload = json.loads(prompt.split("\n\n", 1)[1])
        self.assertTrue(payload["observation_date"].startswith("UNKNOWN:"))
        self.assertNotEqual(payload["observation_date"], payload["current_date"])

    def test_mixed_source_dates_are_not_collapsed_to_one_anchor(self):
        messages = [{"role": "user", "content": "first", "timestamp": 1704067200000},
                    {"role": "assistant", "content": "second", "timestamp": 1704153600000}]
        anchor = observation_date(messages, None)
        self.assertIn("Multiple source dates", anchor)
        self.assertIn("2024-01-01", anchor)
        self.assertIn("2024-01-02", anchor)
        payload = json.loads(build_extraction_input(messages, [], [], None).split("\n\n", 1)[1])
        self.assertEqual(payload["new_messages"], messages)

    def test_metadata_source_timestamps_and_invalid_dates(self):
        self.assertEqual(observation_date([], {"source_timestamps": [None, 1704067200000]}), "2024-01-01")
        for invalid in (True, float("nan"), float("inf"), "not a date", 10**400):
            with self.subTest(invalid_type=type(invalid).__name__):
                self.assertTrue(observation_date([{"timestamp": invalid}], None).startswith("UNKNOWN:"))


if __name__ == "__main__":
    unittest.main()

import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from mini_vector import MiniVector
from mini_vector.aml_api import AMLBackend, create_app
from mini_vector.prompts import MEMORY_EXTRACTION_PROMPT
from mini_vector.prompts_time_v2 import MEMORY_EXTRACTION_PROMPT_TIME_V2


class ExtractionPromptChoiceTests(unittest.TestCase):
    def test_default_original_and_explicit_v2_only_change_system_prompt(self):
        requests = []
        def llm(messages):
            requests.append(messages)
            return '{"memory": [{"text": "Ada bought a cello last year."}]}'
        for choice, expected in ((None, MEMORY_EXTRACTION_PROMPT), ("time_v2", MEMORY_EXTRACTION_PROMPT_TIME_V2)):
            options = {} if choice is None else {"extraction_prompt": choice}
            memory = MiniVector(":memory:", llm=llm, embedder=lambda texts: [[1., 0.] for _ in texts],
                                memory_query_enabled=False, rerank_mode="none", **options)
            try:
                memory.add({"role": "user", "content": "I bought a cello last year.", "timestamp": "2023-06-01"}, user_id="a")
                self.assertEqual(requests[-1][0]["content"], expected)
            finally:
                memory.close()
        self.assertEqual(requests[0][1], requests[1][1])
        path = Path(__file__).resolve().parents[1] / "mini_vector/prompts.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), "5d8a3b5dc31ec842e12e0f4737402360a8fcaace876a7b1ed2c4bde9f1785e17")

    def test_invalid_value_is_rejected_before_any_provider(self):
        with self.assertRaisesRegex(ValueError, "extraction_prompt"):
            MiniVector(extraction_prompt="bogus")
        with self.assertRaisesRegex(ValueError, "extraction_prompt"):
            create_app(extraction_prompt="bogus", api_token="test")

    def test_aml_database_identity_distinguishes_prompt_and_preserves_legacy_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory.sqlite3"
            def backend(prompt):
                return AMLBackend(str(path), llm=lambda messages: '{"memory": []}',
                                  embedder=lambda texts: [[1., 0.] for _ in texts], rerank_mode="none",
                                  memory_query_enabled=False, extraction_prompt=prompt)
            backend("original").close()
            with self.assertRaisesRegex(ValueError, "configuration"):
                backend("time_v2")
            with closing(sqlite3.connect(path)) as connection:
                identity = json.loads(connection.execute("SELECT identity FROM aml_config").fetchone()[0])
                self.assertEqual(identity.pop("extraction_prompt"), "original")
                connection.execute("UPDATE aml_config SET identity=?", (json.dumps(identity, sort_keys=True, separators=(",", ":")),))
                connection.commit()
            backend("original").close()
            with self.assertRaisesRegex(ValueError, "configuration"):
                backend("time_v2")


if __name__ == "__main__":
    unittest.main()

import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from mini_vector.aml_api import AMLBackend, SearchRequest, create_app
from mini_vector.aml_time import date_prefix


class AmlSearchTimeFormatTests(unittest.TestCase):
    def test_default_and_invalid_choice(self):
        self.assertEqual(inspect.signature(create_app).parameters["search_time_format"].default, "as_is")
        self.assertEqual(inspect.signature(AMLBackend).parameters["search_time_format"].default, "as_is")
        with self.assertRaisesRegex(ValueError, "search_time_format"):
            create_app(api_token="test", search_time_format="created_at_observed")

    def test_rendering_reuses_source_prefix_without_changing_other_fields_or_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "memory.sqlite3")
            metadata_values = [{"source_timestamps": [1683504000000]},
                               {"source_timestamps": [1683504000000, 1683590400000]}, {},
                               {"session_date": "1:56 pm on 8 May, 2023"}]
            for choice in ("as_is", "date_prefix"):
                backend = AMLBackend(path, llm=lambda _: '{"memory": []}', embedder=lambda texts: [[1., 0.] for _ in texts],
                                     rerank_mode="none", memory_query_enabled=False, search_time_format=choice)
                try:
                    for metadata in metadata_values:
                        evidence = {"id": "one", "memory": "Went hiking last week.", "score": .7,
                                    "created_at": "2026-10-01T10:00:00Z", "metadata": metadata}
                        before = copy.deepcopy(evidence)
                        backend.memory.search = Mock(return_value={"results": [evidence]})
                        found = backend.search(SearchRequest(query="hiking", user_id="a", top_k=20))["data"][0]
                        self.assertEqual(found["content"], (date_prefix(metadata) if choice == "date_prefix" else "") + evidence["memory"])
                        self.assertEqual(found["created_at"], evidence["created_at"])
                        self.assertEqual((found["id"], found["score"]), ("one", .7))
                        self.assertEqual(evidence, before)
                finally:
                    backend.close()


if __name__ == "__main__":
    unittest.main()

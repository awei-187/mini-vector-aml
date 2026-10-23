"""Corpus consistency must ignore object-key order but reject content changes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("optimization_runner", Path(__file__).resolve().parents[1] / "evals/run_optimization.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CorpusComparisonTests(unittest.TestCase):
    def fixture(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        corpus, child = Path(folder.name) / "corpus", Path(folder.name) / "child"
        for root in (corpus, child):
            root.mkdir()
            (root / "memory_snapshot.json").write_text(json.dumps({"count": 1, "memories": [{"id": "a", "memory": "fact"}]}))
            (root / "memories_embeddings.json").write_text('{"memories":[[1,0]]}')
            (root / "memory_queries_embeddings.json").write_text('{"queries":[[0,1]]}')
        return corpus, child

    def test_object_key_order_is_not_a_corpus_change(self):
        corpus, child = self.fixture()
        (child / "memory_snapshot.json").write_text('{"memories":[{"memory":"fact","id":"a"}],"count":1}')
        runner.verify_corpus_exports(corpus, child)

    def test_fact_or_vector_changes_are_rejected(self):
        for filename, changed in (("memory_snapshot.json", '{"count":1,"memories":[{"id":"a","memory":"changed"}]}'),
                                  ("memories_embeddings.json", '{"memories":[[0,1]]}'),
                                  ("memory_queries_embeddings.json", '{"queries":[[1,0]]}')):
            with self.subTest(filename=filename):
                corpus, child = self.fixture()
                (child / filename).write_text(changed)
                with self.assertRaises(AssertionError):
                    runner.verify_corpus_exports(corpus, child)

    def test_missing_exports_are_rejected(self):
        corpus, child = self.fixture()
        (child / "memory_snapshot.json").unlink()
        (corpus / "memory_snapshot.json").unlink()
        with self.assertRaises(AssertionError):
            runner.verify_corpus_exports(corpus, child)

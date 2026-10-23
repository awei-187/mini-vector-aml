"""Benchmark safety checks; these use no live model services."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mini_vector import MiniVector

spec = importlib.util.spec_from_file_location("mini_locomo", Path(__file__).resolve().parents[1] / "evals/run_locomo.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AtomicIngestionTests(unittest.TestCase):
    def test_checkpoint_survives_transient_windows_replace_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run_status.json"
            path.write_text('{"status":"running"}', encoding="utf-8")
            original_replace = runner.reference.os.replace
            attempts = []

            def replace(source, destination):
                attempts.append((source, destination))
                if len(attempts) == 1:
                    raise PermissionError("Windows sharing violation")
                return original_replace(source, destination)

            with patch.object(runner.reference.os, "replace", side_effect=replace), patch.object(runner.time, "sleep"):
                runner.write_json(path, {"status": "complete", "questions": 152})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "complete", "questions": 152})
            self.assertEqual(len(attempts), 2)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_interruption_rolls_back_all_records_then_resume_commits_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "memory.sqlite3")
            memory = MiniVector(db_path=path, llm=lambda _: json.dumps({"memory": [
                {"text": "Alice lives in Berlin."}, {"text": "Alice likes cycling."}]}),
                embedder=lambda texts: [[1., 0.] for _ in texts],
                memory_query_enabled=False, rerank_mode="none")
            memory.store.db.execute("CREATE TABLE benchmark_ingestion(dialogue_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
            memory.store.db.commit()
            turn = {"role": "user", "content": "Alice lives in Berlin and likes cycling.",
                    "session_key": "session_1", "session_date": "2023-01-01", "dialogue_id": "D1:1", "speaker": "Alice"}
            scope = {"user_id": "test", "run_id": "test"}
            with patch.object(memory.store, "save_messages", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    runner.atomic_ingest(memory, turn, scope)
            for table in ("memories", "memory_fts", "history", "messages", "benchmark_ingestion"):
                self.assertEqual(memory.store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
            record = runner.atomic_ingest(memory, turn, scope)
            self.assertEqual(len(record["result"]["results"]), 2)
            memory.close()
            reopened = MiniVector(db_path=path, llm=lambda _: "{}", embedder=lambda texts: [[1., 0.] for _ in texts])
            self.assertEqual(len(reopened.get_all(filters=scope)["results"]), 2)
            self.assertEqual(reopened.store.db.execute("SELECT count(*) FROM benchmark_ingestion").fetchone()[0], 1)
            self.assertEqual(reopened.store.db.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
            reopened.close()

    def test_answer_context_does_not_include_judge_gold_fields(self):
        context = runner.reference.build_context([{"memory": "Alice lives in Berlin.",
                                                  "metadata": {"session_date": "2023-01-01"}}])
        self.assertIn("Alice lives in Berlin.", context)
        self.assertNotIn("gold_answer", context)


if __name__ == "__main__":
    unittest.main()

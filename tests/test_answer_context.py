import json
import unittest

from mini_vector.answer_context import answer_messages, build_context
from mini_vector.source_context import SourceIndex


class AnswerContextTests(unittest.TestCase):
    def test_observation_is_explicit_and_gold_has_no_input_slot(self):
        context = build_context([{"memory": "Ada travelled last year.", "metadata": {"session_date": "2024-06-11"}}])
        self.assertIn("observation_time: 2024-06-11", context)
        self.assertNotIn("event_time:", context)
        payload = json.loads(answer_messages("When?", context)[-1]["content"])
        self.assertEqual(set(payload), {"context", "question"})
        with self.assertRaises(TypeError):
            answer_messages("When?", context, gold="DO_NOT_SEND")

    def test_source_windows_remain_verbatim_and_never_cross_sessions(self):
        turns = [{"dialogue_id": str(i), "session_key": "a" if i < 3 else "b", "speaker": "Ada",
                  "session_date": "2024-06-11", "content": text}
                 for i, text in enumerate(("Hello!", "I read a book yesterday.", "The title is Aurora.", "Unrelated private note."))]
        index = SourceIndex(turns)
        self.addCleanup(index.close)
        windows, trace = index.select("What book did Ada read?", [], max_windows=1, radius=1)
        self.assertEqual(windows[0]["anchor"], "1")
        self.assertEqual([message["text"] for message in windows[0]["messages"]], [turn["content"] for turn in turns[:3]])
        self.assertNotIn("3", trace["selected_source_ids"])
        self.assertIn("SOURCE_MESSAGES", build_context([], windows))

    def test_overlap_budget_unknown_origins_and_untrusted_query(self):
        turns = [{"dialogue_id": str(i), "session_key": "a", "speaker": "Ada", "session_date": "2024-06-11",
                  "content": "evidence " + str(i)} for i in range(5)]
        index = SourceIndex(turns)
        self.addCleanup(index.close)
        windows, trace = index.select('evidence OR " * ); DROP TABLE source_fts;',
                                     [{"metadata": {"source_dialogue_ids": ["unknown", "2"]}}], max_chars=40)
        self.assertEqual(len(trace["selected_source_ids"]), len(set(trace["selected_source_ids"])))
        self.assertLessEqual(trace["source_text_chars"], 40)
        self.assertTrue(all(message["text"] == turns[int(message["source_id"])]["content"]
                            for window in windows for message in window["messages"]))
        self.assertNotIn("unknown", trace["origin_ids"])

    def test_duplicate_ids_and_invalid_budgets_fail(self):
        turn = {"dialogue_id": "1", "session_key": "a", "speaker": "Ada", "session_date": "2024-06-11", "content": "text"}
        with self.assertRaises(ValueError):
            SourceIndex([turn, turn])
        index = SourceIndex([turn])
        self.addCleanup(index.close)
        with self.assertRaises(ValueError):
            index.select("query", [], max_windows=0)

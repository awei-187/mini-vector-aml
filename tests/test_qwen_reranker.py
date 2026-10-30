"""Real retrieval with injected scoring, plus dedicated API/config contracts."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from legacy_helpers import legacy_memory as MiniVector
from mini_vector.reranker import (DEFAULT_INSTRUCTION, SUPPORT_INSTRUCTION, QwenReranker,
    RerankerConfig, parse_scores, qwen_reranker_config, rerank_qwen)


def response(scores):
    return {"request_id": "synthetic", "output": {"results": [
        {"index": i, "relevance_score": s} for i, s in reversed(list(enumerate(scores)))]}}


class QwenRerankerTests(unittest.TestCase):
    def test_independent_payload_has_only_query_texts_model_and_instruction(self):
        received = []
        cfg = RerankerConfig("https://example.test/rerank", "private-key", "qwen3-rerank", SUPPORT_INSTRUCTION)
        client = QwenReranker(cfg, transport=lambda body: received.append(body) or response([.2, .9]))
        candidates = [{"id": "a", "memory": "one", "metadata": {"gold": "NEVER SEND"}},
                      {"id": "b", "memory": "two", "source": "NEVER SEND"}]
        original = deepcopy(candidates)
        ranked, trace = rerank_qwen(client, "query", candidates, 60)
        self.assertEqual(received, [{"model": "qwen3-rerank", "input": {
            "query": "query", "documents": ["one", "two"]}, "parameters": {
            "top_n": 2, "return_documents": False, "instruct": SUPPORT_INSTRUCTION}}])
        self.assertEqual([c["id"] for c in ranked], ["b", "a"])
        self.assertEqual(candidates, original)
        self.assertEqual(trace["status"], "succeeded")
        self.assertNotIn("private-key", repr(cfg) + json.dumps(trace))

    def test_scores_validate_complete_index_mapping_and_stable_ties_with_tail(self):
        items = [{"id": str(i), "memory": str(i)} for i in range(5)]
        ranked, trace = rerank_qwen(lambda q, d: response([.1, .8, .8]), "q", items, 3)
        self.assertEqual([r["id"] for r in ranked], ["1", "2", "0", "3", "4"])
        self.assertEqual(len(trace["scores"]), 3)
        self.assertEqual(parse_scores(response([0., 1.]), 2), [0., 1.])

    def test_invalid_protocol_and_provider_errors_fall_back_without_error_text(self):
        cases = [response([.5]), response([True, .4]), response([float("nan"), .4]),
                 response([float("inf"), .4]), response([1.1, .4]), {"error": "secret"},
                 {"output": {"results": [{"index": 0, "relevance_score": .2},
                                         {"index": 0, "relevance_score": .3}]}},
                 {"output": {"results": [{"index": True, "relevance_score": .2},
                                         {"index": 0, "relevance_score": .3}]}}]
        items = [{"id": "a", "memory": "one"}, {"id": "b", "memory": "two"}]
        def broken(*_):
            raise RuntimeError("secret credentials and provider body")
        for result in cases + [None]:
            with self.subTest(result=result):
                client = broken if result is None else lambda q, d: result
                ranked, trace = rerank_qwen(client, "q", items, 60)
                self.assertEqual(ranked, items)
                self.assertEqual(trace["status"], "fallback")
                self.assertEqual(trace["scores"], [])
                self.assertNotIn("secret", json.dumps(trace))

    def test_empty_and_singleton_never_call_provider(self):
        def forbidden(*_):
            self.fail("Unnecessary API call")
        for items in ([], [{"id": "a", "memory": "one"}]):
            ranked, trace = rerank_qwen(forbidden, "q", items, 60)
            self.assertEqual(ranked, items)
            self.assertEqual(trace["api_calls"], 0)

    def test_configuration_is_separate_explicit_and_does_not_modify_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "config.env"
            env.write_text("QWEN_RERANK_ENDPOINT=https://example.test/rerank\nQWEN_RERANK_MODEL=qwen3-rerank\n"
                           "QWEN_RERANK_API_KEY_ENV=TASK_RERANK_KEY\nTASK_RERANK_KEY=file-key\n"
                           "QWEN_RERANK_INSTRUCTION=file instruction\nOPENAI_CHAT_MODEL=unrelated\n")
            with patch.dict(os.environ, {"TASK_RERANK_KEY": "process-key"}, clear=True):
                before = dict(os.environ)
                cfg = qwen_reranker_config(env_file=env, instruction="explicit instruction")
                self.assertEqual(cfg.api_key, "process-key")
                self.assertEqual(cfg.instruction, "explicit instruction")
                self.assertEqual(cfg.model, "qwen3-rerank")
                self.assertEqual(dict(os.environ), before)
            with patch.dict(os.environ, {"TASK_RERANK_KEY": ""}, clear=True):
                with self.assertRaises(ValueError):
                    qwen_reranker_config(env_file=env)

    def test_invalid_configuration_rejected_before_network(self):
        for endpoint in ("http://example.test", "https://user:key@example.test", "https://example.test?key=value"):
            with self.assertRaises(ValueError):
                RerankerConfig(endpoint, "key")
        for timeout in (0, -1, True, float("nan")):
            with self.assertRaises(ValueError):
                RerankerConfig("https://example.test", "key", timeout=timeout)

    def test_none_does_not_load_reranker_settings_or_use_chat_reranking(self):
        with patch("mini_vector.memory.qwen_reranker_config", side_effect=AssertionError):
            memory = MiniVector(":memory:", llm=lambda _: self.fail("Unexpected chat"),
                embedder=lambda t: [[1., 0.] for _ in t], memory_query_enabled=False,
                rerank_mode="none", reranker_env_file="missing.env")
            self.addCleanup(memory.close)
            memory.add("one", user_id="a", infer=False)
            memory.add("two", user_id="a", infer=False)
            memory.search("one", filters={"user_id": "a"})

    def test_constructor_keeps_chat_and_reranker_models_independent(self):
        memory = MiniVector(":memory:", llm=lambda _: "{}", embedder=lambda t: [[1., 0.] for _ in t],
            chat_model="chat-model", rerank_mode="qwen", reranker_model="qwen3-rerank",
            reranker_instruction="specific evidence", reranker_endpoint="https://example.test/rerank",
            reranker_api_key="test-key")
        self.addCleanup(memory.close)
        self.assertEqual(memory.chat_model, "chat-model")
        self.assertEqual(memory.reranker.config.model, "qwen3-rerank")
        self.assertEqual(memory.reranker.config.instruction, "specific evidence")

    def test_aml_qwen_search_and_persisted_instruction_identity(self):
        from mini_vector.aml_api import AMLBackend, SearchRequest
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "aml.sqlite3")
            config = RerankerConfig("https://example.test/rerank", "test-secret")
            client = QwenReranker(config, transport=lambda body: response([.2, .9]))
            backend = AMLBackend(path, llm=lambda _: "{}", embedder=lambda t:[[1.,0.] for _ in t],
                memory_query_enabled=False, rerank_mode="qwen", reranker=client, query_rewrite_enabled=False)
            try:
                backend.memory.add("needle one", user_id="a", infer=False)
                backend.memory.add("needle two", user_id="a", infer=False)
                result = backend.search(SearchRequest(query="needle",user_id="a",top_k=1))
                self.assertEqual(len(result["data"]),1)
                self.assertEqual(set(result["data"][0]),{"id","content","score","created_at"})
            finally:backend.close()
            changed = QwenReranker(RerankerConfig(config.endpoint,config.api_key,instruction="new instruction"))
            with self.assertRaises((RuntimeError,ValueError)):
                AMLBackend(path,llm=lambda _:"{}",embedder=lambda t:[[1.,0.] for _ in t],
                    memory_query_enabled=False,rerank_mode="qwen",reranker=changed,query_rewrite_enabled=False)

    def test_qwen_scores_fused_pool_once_and_preserves_scoping_rrf_and_metadata(self):
        calls = []
        def scorer(query, documents):
            calls.append((query, documents))
            return response([i / len(documents) for i in range(len(documents))])
        memory = MiniVector(":memory:", llm=lambda _: self.fail("Chat rerank used"),
            embedder=lambda t: [[1., 0.] for _ in t], memory_query_enabled=False,
            rerank_mode="none", rerank_limit=3, reranker=scorer)
        self.addCleanup(memory.close)
        for text in ("needle one", "needle two", "needle three", "needle four"):
            memory.add(text, user_id="a", infer=False, metadata={"source_dialogue_ids": [text]})
        memory.add("needle foreign", user_id="b", infer=False)
        memory.memory_query_enabled = True  # Use three-way RRF, not the legacy two-route baseline.
        base_trace, new_trace = {}, {}
        baseline = memory.search("needle", top_k=4, filters={"user_id": "a"},
                                 explain=True, research_trace=base_trace)["results"]
        memory.rerank_mode = "qwen"
        actual = memory.search("needle", top_k=4, filters={"user_id": "a"},
                               explain=True, research_trace=new_trace)["results"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], [r["memory"] for r in baseline[:3]])
        self.assertEqual([r["id"] for r in actual], [r["id"] for r in baseline[:3]][::-1] + [baseline[3]["id"]])
        self.assertEqual(base_trace["stages"]["combined_order"], new_trace["stages"]["combined_order"])
        for row in actual:
            old = next(r for r in baseline if r["id"] == row["id"])
            self.assertEqual({k:v for k,v in row.items() if k != "reranker_score"}, old)
        self.assertIn("reranker_score", actual[0])
        self.assertNotIn("reranker_score", actual[-1])
        self.assertEqual(new_trace["schema_version"], 4)
        self.assertTrue(all(c["rerank"]["status"] == "disabled" for c in new_trace["channels"].values()))

    def test_qwen_fallback_is_identical_to_disabled_search(self):
        memory = MiniVector(":memory:", llm=lambda _: "{}", embedder=lambda t: [[1., 0.] for _ in t],
            memory_query_enabled=False, rerank_mode="none", reranker=lambda q,d: response([]))
        self.addCleanup(memory.close)
        for text in ("needle one", "needle two"):
            memory.add(text, user_id="a", infer=False)
        memory.memory_query_enabled = True
        expected = memory.search("needle", filters={"user_id": "a"}, explain=True)
        memory.rerank_mode = "qwen"
        trace = {}
        self.assertEqual(memory.search("needle", filters={"user_id": "a"}, explain=True, research_trace=trace), expected)
        self.assertEqual(trace["qwen_rerank"]["status"], "fallback")


if __name__ == "__main__":
    unittest.main()

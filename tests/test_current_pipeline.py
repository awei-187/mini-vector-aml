"""Current defaults through real scoped SQLite retrieval and injected providers."""
import json
from pathlib import Path
import tempfile
import unittest

from mini_vector import MiniVector
from mini_vector.answer_context import answer_messages, build_context, temporal_answer_messages
from mini_vector.query_rewrite import QUERY_REWRITE_PROMPT
from mini_vector.reranker import SUPPORT_INSTRUCTION, QwenReranker, RerankerConfig
from mini_vector.source_context import SourceIndex


class CurrentPipelineTests(unittest.TestCase):
    def providers(self):
        calls = {"rewrite": [], "rerank": [], "embedding": []}

        def llm(messages):
            system = messages[0]["content"]
            if "Generate retrieval questions" in system:
                facts = json.loads(messages[-1]["content"])["memories"]
                return json.dumps({"memories": [{"id": f["id"], "queries": ["Question about " + f["text"]]} for f in facts]})
            self.assertEqual(system, QUERY_REWRITE_PROMPT)
            calls["rewrite"].append(messages)
            return json.dumps({"queries": ["Quartz", "Orbital", "launch"]})

        def embed(texts):
            calls["embedding"].append(list(texts))
            return [[1., 0.] for _ in texts]

        def transport(body):
            calls["rerank"].append(body)
            docs = body["input"]["documents"]
            return {"output": {"results": [{"index": i, "relevance_score": .9 if "Quartz" in d else .1}
                                            for i, d in enumerate(docs)]}}

        reranker = QwenReranker(RerankerConfig("https://rerank.example.test/api", "synthetic-key",
                                             instruction=SUPPORT_INSTRUCTION), transport=transport)
        return llm, embed, reranker, calls

    def test_default_pipeline_preserves_scope_fuses_nine_routes_and_reranks_once(self):
        llm, embed, reranker, calls = self.providers()
        memory = MiniVector(":memory:", llm=llm, embedder=embed, reranker=reranker)
        self.addCleanup(memory.close)
        for text in ("Ada leads Quartz.", "Ada uses Orbital.", "Ada plans a launch."):
            memory.add(text, user_id="a", infer=False)
        memory.add("Foreign user's secret Quartz.", user_id="b", infer=False)
        trace = {}
        found = memory.search("What does Ada lead?", filters={"user_id": "a"}, research_trace=trace)["results"]
        self.assertEqual(found[0]["memory"], "Ada leads Quartz.")
        self.assertEqual(len(found), len({row["id"] for row in found}))
        self.assertTrue(all(row["user_id"] == "a" for row in found))
        self.assertEqual(len(trace["channels"]), 9)
        self.assertEqual((memory.rerank_mode, memory.rerank_limit, memory.query_rewrite_limit), ("qwen", 60, 3))
        self.assertEqual((len(calls["rewrite"]), len(calls["rerank"])), (1, 1))
        self.assertEqual(calls["rerank"][0]["parameters"]["instruct"], SUPPORT_INSTRUCTION)
        self.assertNotIn("Foreign", json.dumps(calls["rerank"]))
        self.assertEqual(trace["qwen_rerank"]["status"], "succeeded")
        self.assertTrue(all(channel["rerank"]["status"] == "disabled" for channel in trace["channels"].values()))

    def test_answer_preparation_keeps_source_windows_without_extra_time_rules(self):
        llm, embed, reranker, calls = self.providers()
        memory = MiniVector(":memory:", llm=llm, embedder=embed, reranker=reranker)
        self.addCleanup(memory.close)
        memory.add("Ada leads Quartz.", user_id="a", infer=False)
        turns = [{"dialogue_id": str(i), "session_key": "s", "speaker": "Ada",
                  "session_date": "2024-06-11", "content": text}
                 for i, text in enumerate(("Hello.", "I started Quartz yesterday.", "I lead the project."))]
        sources = SourceIndex(turns)
        self.addCleanup(sources.close)
        prepared = memory.prepare_answer("Who leads Quartz?", filters={"user_id": "a"}, source_index=sources)
        self.assertTrue(prepared["source_windows"])
        self.assertIn("I started Quartz yesterday.", prepared["answer_context"])
        system = prepared["messages"][0]["content"]
        for text in ("Resolve relative dates", "observation_time", "last year", "event date", "preserving the stated precision"):
            self.assertNotIn(text, system)
        self.assertNotIn("timestamps are observation times", prepared["answer_context"])
        self.assertEqual(set(json.loads(prepared["messages"][-1]["content"])), {"context", "question"})
        self.assertEqual(prepared["answer_policy"], "source_windows_no_temporal_rules")

    def test_temporal_policy_is_explicit_opt_in(self):
        self.assertNotIn("Resolve relative dates", answer_messages("q", build_context([]))[0]["content"])
        self.assertIn("Resolve relative dates", temporal_answer_messages("q", "context")[0]["content"])

    def test_default_model_and_support_instruction_are_used_without_injected_reranker(self):
        memory = MiniVector(":memory:", llm=lambda _: "{}", embedder=lambda texts: [[1., 0.] for _ in texts],
                            reranker_endpoint="https://rerank.example.test/api", reranker_api_key="synthetic-key")
        self.addCleanup(memory.close)
        self.assertEqual(memory.reranker.config.model, "qwen3-rerank")
        self.assertEqual(memory.reranker.config.instruction, SUPPORT_INSTRUCTION)

    def test_http_factory_uses_current_defaults_and_keeps_search_evidence_contract(self):
        from fastapi.testclient import TestClient
        from mini_vector.aml_api import create_app
        llm, embed, reranker, calls = self.providers()
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(db_path=Path(directory) / "http.sqlite3", api_token="synthetic-token",
                             llm=llm, embedder=embed, reranker=reranker)
            with TestClient(app) as client:
                headers = {"Authorization": "Bearer synthetic-token"}
                app.state.backend.memory.add("Ada leads Quartz.", user_id="a", infer=False)
                app.state.backend.memory.add("Ada uses Orbital.", user_id="a", infer=False)
                response = client.post("/search", headers=headers,
                                       json={"query": "What does Ada lead?", "user_id": "a", "top_k": 20})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(set(response.json()["data"][0]), {"id", "content", "score", "created_at"})
                self.assertEqual((len(calls["rewrite"]), len(calls["rerank"])), (1, 1))
                self.assertTrue(app.state.backend.memory.query_rewrite_enabled)

    def test_aml_defaults_use_same_retrieval_and_rewrite_identity_is_bound(self):
        from mini_vector.aml_api import AMLBackend, SearchRequest
        llm, embed, reranker, calls = self.providers()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "current.sqlite3")
            backend = AMLBackend(path, llm=llm, embedder=embed, reranker=reranker)
            try:
                backend.memory.add("Ada leads Quartz.", user_id="a", infer=False)
                backend.memory.add("Ada uses Orbital.", user_id="a", infer=False)
                result = backend.search(SearchRequest(query="What does Ada lead?", user_id="a", top_k=20))
                self.assertEqual(result["data"][0]["content"], "Ada leads Quartz.")
                identity = json.loads(backend.store.identity)
                self.assertTrue(identity["query_rewrite_enabled"])
                self.assertEqual(identity["query_rewrite_limit"], 3)
                self.assertEqual(identity["rerank_limit"], 60)
            finally:
                backend.close()
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                AMLBackend(path, llm=llm, embedder=embed, reranker=reranker, query_rewrite_enabled=False)

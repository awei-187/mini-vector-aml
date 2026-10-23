"""HTTP contract and durable-ingestion tests with real SQLite and injected models."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

HAS_API = importlib.util.find_spec("fastapi") is not None
if HAS_API:
    from fastapi.testclient import TestClient
    from mini_vector.aml_api import create_app
    from mini_vector.aml_store import AMLStore, DatabaseInUse


@unittest.skipUnless(HAS_API, "Install mini-vector-memory[api] to run HTTP tests")
class AMLAPITests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "aml.sqlite3"
        self.prompts = []
        self.embeddings = []
        self.headers = {"Authorization": "Bearer local-test-token"}

    def llm(self, messages):
        if "Generate retrieval questions" in messages[0]["content"]:
            data = json.loads(messages[-1]["content"])
            return json.dumps({"memories": [{"id": item["id"], "queries": ["Question about " + item["text"]]}
                                           for item in data["memories"]]})
        if "Rank candidate memories" in messages[0]["content"]:
            data = json.loads(messages[-1]["content"])
            return json.dumps({"ids": [item["id"] for item in data["candidates"]]})
        payload = json.loads(messages[-1]["content"].split("\n\n", 1)[1])
        self.prompts.append(payload)
        return json.dumps({"memory": [{"text": m["content"]} for m in payload["new_messages"]]})

    def embed(self, texts):
        self.embeddings.append(texts)
        return [[1.0, 0.5] for _ in texts]

    def app(self, **kwargs):
        return create_app(db_path=self.path, api_token="local-test-token",
                          llm=kwargs.get("llm", self.llm), embedder=kwargs.get("embedder", self.embed))

    def payload(self, request_id="r1", user_id="alice", session_id="s1", content="Alice lives in Berlin."):
        return {"request_id": request_id, "user_id": user_id, "session_id": session_id,
                "messages": [{"role": "user", "content": content, "timestamp": 1704067200000}]}

    def add(self, client, **kwargs):
        return client.post("/add", json=self.payload(**kwargs), headers=self.headers)

    def search(self, client, user_id="alice", **kwargs):
        return client.post("/search", json={"query": "Berlin", "user_id": user_id, "top_k": 100,
                                            **kwargs}, headers=self.headers)

    def test_contract_incremental_sessions_top_k_and_timestamp_context(self):
        app = self.app()
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            self.assertEqual(self.prompts, [])
            self.assertEqual(self.embeddings, [])
            payload = self.payload()
            response = self.add(client)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"success": True, **{k: payload[k] for k in
                                                                  ("request_id", "user_id", "session_id")}})
            self.assertEqual(self.prompts[0]["new_messages"][0]["timestamp"], 1704067200000)
            self.assertEqual(self.add(client, request_id="r2", session_id="s2",
                                     content="Alice likes pears.").status_code, 200)
            self.assertEqual(self.prompts[1]["recent_messages"][0]["timestamp"], 1704067200000)
            result = self.search(client, top_k=1, options=["A. Berlin", "B. Paris"])
            self.assertEqual(result.status_code, 200)
            self.assertEqual(len(result.json()["data"]), 1)
            item = result.json()["data"][0]
            self.assertEqual(set(item), {"id", "content", "score", "created_at"})
            self.assertEqual(item["content"], "Alice lives in Berlin.")
            self.assertEqual(len(self.search(client).json()["data"]), 2)
            self.assertEqual(len(self.prompts), 2)  # Only Add performs fact extraction.
            memories = app.state.backend.memory.get_all(filters={"user_id": "alice"})["results"]
            self.assertEqual({m["metadata"]["session_id"] for m in memories}, {"s1", "s2"})
            self.assertEqual(memories[0]["metadata"]["source_timestamps"], [1704067200000])

    def test_exact_user_isolation_including_whitespace(self):
        with TestClient(self.app()) as client:
            self.assertEqual(self.add(client, user_id=" alice ").status_code, 200)
            self.assertEqual(self.search(client).json(), {"data": []})
            self.assertEqual(self.search(client, "bob").json(), {"data": []})
            self.assertEqual(len(self.search(client, " alice ").json()["data"]), 1)

    def test_retry_reordering_json_keys_restart_and_conflict(self):
        with TestClient(self.app()) as client:
            first = self.add(client)
            calls = (len(self.prompts), len(self.embeddings))
            retry = client.post("/add", json=dict(reversed(list(self.payload().items()))), headers=self.headers)
            self.assertEqual(retry.json(), first.json())
            self.assertEqual((len(self.prompts), len(self.embeddings)), calls)
            original_id = self.search(client).json()["data"][0]["id"]
            for change in ({"content": "A different fact"}, {"user_id": "bob"}, {"session_id": "other"}):
                self.assertEqual(self.add(client, **change).status_code, 409)
        with TestClient(self.app()) as client:
            self.assertEqual(self.add(client).json(), first.json())
            self.assertEqual(len(self.prompts), calls[0])
            self.assertEqual(self.search(client).json()["data"][0]["id"], original_id)

    def test_late_failure_rolls_back_every_table_and_allows_retry(self):
        app = self.app()
        with TestClient(app) as client:
            store = app.state.backend.store
            with patch.object(store, "complete", side_effect=RuntimeError("private-error")):
                response = self.add(client)
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("private-error", response.text)
            for table in ("memories", "memory_fts", "memory_queries", "history", "messages", "aml_context", "aml_requests"):
                self.assertEqual(store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
            self.assertIsNone(store.db.execute("SELECT dimension FROM aml_config").fetchone()[0])
            self.assertEqual(self.add(client).status_code, 200)
            self.assertEqual(len(self.search(client).json()["data"]), 1)

    def test_mid_batch_failure_rolls_back_already_persisted_facts(self):
        app = self.app()
        with TestClient(app) as client:
            payload = self.payload()
            payload["messages"].append({"role": "assistant", "content": "Alice likes pears."})
            store = app.state.backend.store
            real_save = store.save
            calls = []

            def save(*args):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("disk failure")
                return real_save(*args)

            with patch.object(store, "save", side_effect=save):
                self.assertEqual(client.post("/add", json=payload, headers=self.headers).status_code, 500)
            self.assertEqual(store.db.execute("SELECT count(*) FROM memories").fetchone()[0], 0)
            self.assertEqual(client.post("/add", json=payload, headers=self.headers).status_code, 200)
            self.assertEqual(store.db.execute("SELECT count(*) FROM memories").fetchone()[0], 2)

    def test_parallel_duplicate_add_is_applied_once(self):
        app = self.app()
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(lambda _: self.add(client), range(6)))
            self.assertTrue(all(r.status_code == 200 for r in responses))
            self.assertEqual(len(self.prompts), 1)
            self.assertEqual(app.state.backend.store.db.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
            self.assertEqual(app.state.backend.store.db.execute("SELECT count(*) FROM aml_requests").fetchone()[0], 1)

    def test_health_remains_responsive_while_model_is_waiting(self):
        started, release = threading.Event(), threading.Event()

        def slow_llm(messages):
            started.set()
            if not release.wait(5):
                raise TimeoutError()
            return self.llm(messages)

        with TestClient(self.app(llm=slow_llm)) as client, ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.add, client)
            try:
                self.assertTrue(started.wait(3))
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertFalse(pending.done())
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=3).status_code, 200)

    def test_auth_validation_and_only_three_routes(self):
        with TestClient(self.app()) as client:
            for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Token local-test-token"}):
                self.assertEqual(client.post("/add", json=self.payload(), headers=headers).status_code, 401)
                self.assertEqual(client.post("/search", json={}, headers=headers).status_code, 401)
            for path in ("/docs", "/openapi.json", "/memories"):
                self.assertEqual(client.get(path).status_code, 404)
            for field, value in (("user_id", " "), ("request_id", 42), ("messages", []),
                                 ("messages", [{"role": "system", "content": "secret-payload"}]),
                                 ("messages", [{"role": "user", "content": "x", "timestamp": True}])):
                payload = {**self.payload(), field: value}
                response = client.post("/add", json=payload, headers=self.headers)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("secret-payload", response.text)
                self.assertNotIn('"input"', response.text)
            for value in (0, -1, True, "100"):
                self.assertEqual(self.search(client, top_k=value).status_code, 422)
            response = client.post("/add", content='{malformed secret-payload', headers=self.headers)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("secret-payload", response.text)
            self.assertEqual(self.prompts, [])

    def test_upstream_errors_are_sanitized_and_never_empty_success(self):
        for error, status in ((TimeoutError("secret-provider"), 504),
                              (ConnectionError("secret-provider"), 503),
                              (RuntimeError("secret-provider"), 500)):
            with self.subTest(status=status):
                def fail(_messages):
                    raise error
                with TestClient(self.app(llm=fail)) as client:
                    with self.assertLogs("mini_vector.aml_api", level="INFO") as logs:
                        response = self.add(client)
                    self.assertEqual(response.status_code, status)
                    self.assertNotIn("secret-provider", response.text + str(logs.output))
                    with patch.object(client.app.state.backend.memory, "search", side_effect=error):
                        search = self.search(client)
                    self.assertEqual(search.status_code, status)
                    self.assertNotIn("secret-provider", search.text)

    def test_sdk_http_and_timeout_errors_are_sanitized(self):
        import httpx
        from openai import BadRequestError, APITimeoutError
        request = httpx.Request("POST", "https://example.test/v1/embeddings")
        errors = ((BadRequestError("secret-provider", response=httpx.Response(400, request=request),
                                  body={"message": "secret-provider"}), 503),
                  (APITimeoutError(request=request), 504))
        for error, status in errors:
            def fail(_texts):
                raise error
            with TestClient(self.app(embedder=fail)) as client:
                response = self.search(client)
                self.assertEqual(response.status_code, status)
                self.assertNotIn("secret-provider", response.text)

    def test_competition_models_ignore_ordinary_library_defaults(self):
        with patch.dict("os.environ", {"OPENAI_CHAT_MODEL": "ordinary-chat",
                                       "OPENAI_EMBEDDING_MODEL": "ordinary-embedding"}):
            app = self.app()
            with TestClient(app):
                self.assertEqual(app.state.backend.memory.chat_model, "gpt-4o-mini")
                self.assertEqual(app.state.backend.memory.embedding_model, "text-embedding-v4")

    def test_live_client_routing_uses_separate_keys_and_closes_both(self):
        from mini_vector.aml_config import EmbeddingConfig
        chat, embedding = Mock(), Mock()
        chat.chat.completions.create.side_effect = lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.llm(kwargs["messages"])))])
        embedding.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[1.0] * 1024) for i, _ in enumerate(kwargs["input"])])
        settings = {"OPENAI_BASE_URL": "https://chat.example/v1", "OPENAI_API_KEY": "synthetic-chat-key"}
        with patch("mini_vector.aml_api.setting", side_effect=lambda name: settings.get(name)), \
             patch("mini_vector.aml_api.embedding_config", return_value=EmbeddingConfig(
                 "https://qwen.example/compatible-mode/v1", "synthetic-qwen-key")), \
             patch("openai.OpenAI", side_effect=[chat, embedding]) as factory:
            app = create_app(db_path=self.path, api_token="local-test-token")
            with TestClient(app) as client:
                self.assertEqual(self.add(client).status_code, 200)
                self.assertEqual(self.search(client).status_code, 200)
                self.assertEqual(factory.call_args_list[0].kwargs["api_key"], "synthetic-chat-key")
                self.assertEqual(factory.call_args_list[1].kwargs["api_key"], "synthetic-qwen-key")
                self.assertEqual(factory.call_args_list[0].kwargs["base_url"], "https://chat.example/v1")
                self.assertEqual(factory.call_args_list[1].kwargs["base_url"], "https://qwen.example/compatible-mode/v1")
                chat.embeddings.create.assert_not_called()
                embedding.chat.completions.create.assert_not_called()
            chat.close.assert_called_once()
            embedding.close.assert_called_once()

    def test_dimension_guard_and_invalid_vectors(self):
        app = self.app()
        with TestClient(app) as client:
            self.assertEqual(self.add(client).status_code, 200)
        for vectors in ([[1.0]], [[float("nan"), 1.0]], [[0.0, 0.0]]):
            with TestClient(self.app(embedder=lambda texts: vectors * len(texts))) as client:
                self.assertEqual(self.search(client).status_code, 500)

    def test_empty_extraction_is_successful_and_idempotent(self):
        app = self.app(llm=lambda _: '{"memory":[]}')
        with TestClient(app) as client:
            self.assertEqual(self.add(client).status_code, 200)
            self.assertEqual(self.add(client).status_code, 200)
            self.assertEqual(self.search(client).json(), {"data": []})
            self.assertEqual(app.state.backend.store.db.execute("SELECT count(*) FROM messages").fetchone()[0], 1)

    def test_generated_queries_commit_once_and_search_reranks_each_channel(self):
        operations = []
        def complete(messages):
            operations.append(messages[0]["content"])
            return self.llm(messages)
        app = self.app(llm=complete)
        with TestClient(app) as client:
            self.assertEqual(self.add(client).status_code, 200)
            count = len(operations)
            self.assertEqual(self.add(client).status_code, 200)
            self.assertEqual(len(operations), count)
            self.assertEqual(self.add(client, request_id="r2", content="Alice visits Berlin.").status_code, 200)
            rows = app.state.backend.store.db.execute("SELECT memory_id,query FROM memory_queries").fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(self.search(client).status_code, 200)
            self.assertEqual(sum("Rank candidate memories" in prompt for prompt in operations), 3)

    def test_generation_failure_rolls_back_ingestion_and_retry_rebuilds_queries(self):
        def complete(messages):
            if "Generate retrieval questions" in messages[0]["content"]:
                return '{"memories":[]}'
            return self.llm(messages)
        app = self.app(llm=complete)
        with TestClient(app) as client:
            self.assertEqual(self.add(client).status_code, 500)
            store = app.state.backend.store
            for table in ("memories", "memory_queries", "memory_fts", "history", "aml_context", "aml_requests"):
                self.assertEqual(store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
            app.state.backend.memory.llm = self.llm
            self.assertEqual(self.add(client).status_code, 200)
            self.assertEqual(store.db.execute("SELECT count(*) FROM memory_queries").fetchone()[0], 1)

    def test_rerank_failure_returns_original_evidence_without_exposing_provider_detail(self):
        def complete(messages):
            if "Rank candidate memories" in messages[0]["content"]:
                raise ConnectionError("synthetic-provider-secret")
            return self.llm(messages)
        with TestClient(self.app(llm=complete)) as client:
            self.assertEqual(self.add(client).status_code, 200)
            self.assertEqual(self.add(client, request_id="r2", content="Alice likes Berlin.").status_code, 200)
            with self.assertLogs("mini_vector.aml_api", level="INFO") as logs:
                response = self.search(client)
            self.assertEqual(response.status_code, 200)
            self.assertEqual({item["content"] for item in response.json()["data"]},
                             {"Alice lives in Berlin.", "Alice likes Berlin."})
            self.assertNotIn("synthetic-provider-secret", response.text + str(logs.output))

    def test_retrieval_configuration_change_cannot_reuse_aml_database(self):
        with TestClient(self.app()) as client:
            self.assertEqual(self.add(client).status_code, 200)
        changed = create_app(db_path=self.path, api_token="local-test-token", llm=self.llm,
                             embedder=self.embed, rerank_mode="none")
        with self.assertRaisesRegex(RuntimeError, "AML startup failed"):
            with TestClient(changed):
                pass
        with TestClient(self.app()) as client:
            self.assertEqual(len(self.search(client).json()["data"]), 1)

    def test_database_identity_single_process_lease_and_legacy_protection(self):
        identity = {"model": "test"}
        store = AMLStore(str(self.path), identity)
        try:
            with self.assertRaises(DatabaseInUse):
                AMLStore(str(self.path), identity)
        finally:
            store.close()
        with self.assertRaisesRegex(ValueError, "configuration"):
            AMLStore(str(self.path), {"model": "changed"})
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(json.loads(db.execute("SELECT identity FROM aml_config").fetchone()[0]), identity)
        legacy = Path(self.folder.name) / "legacy.sqlite3"
        db = sqlite3.connect(legacy)
        db.execute("CREATE TABLE preserved (value TEXT)")
        db.execute("INSERT INTO preserved VALUES ('original')")
        db.commit()
        db.close()
        original = legacy.read_bytes()
        with self.assertRaisesRegex(ValueError, "without an AML"):
            AMLStore(str(legacy), identity)
        self.assertEqual(legacy.read_bytes(), original)

    def test_token_required_before_database_is_created(self):
        with patch.dict("os.environ", {"AML_API_TOKEN": ""}), patch("mini_vector.aml_api.setting", return_value=None):
            with self.assertRaisesRegex(ValueError, "AML_API_TOKEN"):
                create_app(db_path=self.path)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()

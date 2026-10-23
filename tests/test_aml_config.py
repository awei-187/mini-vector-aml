"""Qwen routing and batching tests with synthetic credentials and no live calls."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mini_vector.aml_config import (EMBEDDING_DIMENSIONS, EmbeddingConfig, embedding_config,
                                    embedding_vectors)


class QwenConfigTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / ".env"
        self.values = {
            "QWEN_RERANK_ENDPOINT": "https://example.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            "QWEN_EMBEDDING_MODEL": "text-embedding-v4",
            "QWEN_RERANK_API_KEY_ENV": "DASHSCOPE_API_KEY",
            "DASHSCOPE_API_KEY": "synthetic-qwen-key",
            "OPENAI_API_KEY": "unrelated-key",
            "OPENAI_BASE_URL": "https://unrelated.example/v1",
        }
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("DASHSCOPE_API_KEY", None)

    def load(self, **changes):
        self.path.write_text("\n".join(f"{key}={value}" for key, value in {**self.values, **changes}.items()),
                             encoding="utf-8-sig")
        return embedding_config(self.path)

    def test_workspace_endpoint_and_external_keys_are_scoped(self):
        before = dict(os.environ)
        config = self.load()
        self.assertEqual(config.base_url, "https://example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(config.api_key, "synthetic-qwen-key")
        self.assertEqual(dict(os.environ), before)
        self.assertNotIn("synthetic-qwen-key", repr(config))
        self.assertNotIn("example.cn-beijing", repr(config))
        rotated = EmbeddingConfig(config.base_url, "rotated-key")
        self.assertEqual(rotated.provider_hash, config.provider_hash)

    def test_explicit_embedding_url_and_key_reference(self):
        config = self.load(QWEN_EMBEDDING_BASE_URL="'https://embedding.example/v1/'",
                           QWEN_EMBEDDING_API_KEY_ENV="LOCAL_TEST_EMBEDDING_KEY",
                           LOCAL_TEST_EMBEDDING_KEY="'synthetic-separate-key'")
        self.assertEqual(config.base_url, "https://embedding.example/v1")
        self.assertEqual(config.api_key, "synthetic-separate-key")

    def test_key_environment_override_and_empty_value_blocks_fallback(self):
        os.environ["DASHSCOPE_API_KEY"] = "synthetic-process-key"
        self.assertEqual(self.load().api_key, "synthetic-process-key")
        os.environ["DASHSCOPE_API_KEY"] = ""
        with self.assertRaisesRegex(ValueError, "API key"):
            self.load()

    def test_invalid_model_url_and_missing_key_do_not_fall_back_to_chat(self):
        for changes in ({"QWEN_EMBEDDING_MODEL": "other-model"},
                        {"QWEN_RERANK_ENDPOINT": "https://unrelated.example/v1"},
                        {"QWEN_RERANK_ENDPOINT": "https://user:password@dashscope.aliyuncs.com/path"},
                        {"DASHSCOPE_API_KEY": ""}):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    self.load(**changes)

    def test_external_config_path_override_and_missing_file(self):
        expected = self.load()
        with patch.dict(os.environ, {"AML_EMBEDDING_ENV_FILE": str(self.path)}):
            self.assertEqual(embedding_config(), expected)
        with self.assertRaisesRegex(ValueError, "missing"):
            embedding_config(Path(self.folder.name) / "missing.env")

    def test_batches_respect_ten_rows_and_response_indices(self):
        client = Mock()

        def embed(**kwargs):
            self.assertEqual(kwargs["model"], "text-embedding-v4")
            self.assertEqual(kwargs["dimensions"], 1024)
            self.assertEqual(kwargs["encoding_format"], "float")
            return SimpleNamespace(data=[SimpleNamespace(index=i,
                embedding=[float(text)] + [1.0] * (EMBEDDING_DIMENSIONS - 1))
                for i, text in reversed(list(enumerate(kwargs["input"])))])

        client.embeddings.create.side_effect = embed
        vectors = embedding_vectors(client, [str(i) for i in range(23)])
        self.assertEqual([vector[0] for vector in vectors], list(range(23)))
        self.assertEqual([len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list], [10, 10, 3])

    def test_bad_provider_indices_or_dimensions_fail(self):
        client = Mock()
        for rows in ([SimpleNamespace(index=1, embedding=[1.0] * 1024)],
                     [SimpleNamespace(index=0, embedding=[1.0])], []):
            client.embeddings.create.return_value = SimpleNamespace(data=rows)
            with self.assertRaises(ValueError):
                embedding_vectors(client, ["synthetic"])


if __name__ == "__main__":
    unittest.main()

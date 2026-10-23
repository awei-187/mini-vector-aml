import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_vector import MiniVector
from mini_vector.config import setting


class DotEnvTests(unittest.TestCase):
    def test_workspace_root_dotenv_is_found_from_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            project = workspace / "mini-vector"
            package = project / "mini_vector"
            package.mkdir(parents=True)
            (workspace / ".env").write_text("OPENAI_CHAT_MODEL=root-chat\n", encoding="utf-8")
            with patch("mini_vector.config.__file__", str(package / "config.py")), \
                 patch("mini_vector.config.Path.cwd", return_value=project):
                self.assertEqual(setting("OPENAI_CHAT_MODEL"), "root-chat")

    def test_default_client_uses_dotenv_and_explicit_options_win(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env").write_text(
                "OPENAI_API_KEY=file-key\nOPENAI_BASE_URL=https://example.test/v1\n"
                "OPENAI_CHAT_MODEL=chat-file\nOPENAI_EMBEDDING_MODEL=embed-file\n",
                encoding="utf-8",
            )
            with patch("mini_vector.config.Path.cwd", return_value=Path(directory)), \
                 patch("openai.OpenAI") as openai_client, \
                 patch.dict("os.environ", {"OPENAI_API_KEY": "process-key"}):
                memory = MiniVector(db_path=":memory:")
                try:
                    self.assertEqual(memory.chat_model, "chat-file")
                    self.assertEqual(memory.embedding_model, "embed-file")
                    self.assertEqual(openai_client.call_args.kwargs["api_key"], "file-key")
                    self.assertEqual(openai_client.call_args.kwargs["base_url"], "https://example.test/v1")
                finally:
                    memory.close()
                explicit = MiniVector(db_path=":memory:", base_url="https://override.test/v1",
                                      chat_model="explicit")
                try:
                    self.assertEqual(explicit.chat_model, "explicit")
                    self.assertEqual(openai_client.call_args.kwargs["base_url"], "https://override.test/v1")
                finally:
                    explicit.close()

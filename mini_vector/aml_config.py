"""AML-only Qwen embedding configuration; never loads external values into os.environ."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from .config import setting

EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_BATCH_SIZE = 10


def _read_env(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key.strip()] = value
    return values


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)

    @property
    def provider_hash(self):
        return hashlib.sha256(self.base_url.rstrip("/").encode()).hexdigest()


def embedding_config(env_file: str | Path | None = None) -> EmbeddingConfig:
    path = Path(env_file or os.getenv("AML_EMBEDDING_ENV_FILE")
                or setting("AML_EMBEDDING_ENV_FILE") or "E:/.env")
    if not path.is_file():
        raise ValueError("AML embedding configuration file is missing")
    values = _read_env(path)
    if values.get("QWEN_EMBEDDING_MODEL", EMBEDDING_MODEL) != EMBEDDING_MODEL:
        raise ValueError("AML requires text-embedding-v4")
    key_name = values.get("QWEN_EMBEDDING_API_KEY_ENV",
                          values.get("QWEN_RERANK_API_KEY_ENV", "DASHSCOPE_API_KEY"))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_name):
        raise ValueError("Invalid Qwen API key variable name")
    # An explicitly empty process value blocks fallback to another credential.
    api_key = os.environ.get(key_name, values.get(key_name, ""))
    if not api_key.strip():
        raise ValueError("Qwen embedding API key is required")
    explicit_base = values.get("QWEN_EMBEDDING_BASE_URL")
    endpoint = explicit_base or values.get("QWEN_RERANK_ENDPOINT", "")
    parts = urlsplit(endpoint)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment):
        raise ValueError("Invalid Qwen service address")
    if explicit_base:
        base_url = endpoint.rstrip("/")
    else:
        allowed_hosts = {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}
        if parts.hostname not in allowed_hosts and not parts.hostname.endswith(".maas.aliyuncs.com"):
            raise ValueError("Qwen rerank address is not a recognized DashScope host")
        base_url = urlunsplit((parts.scheme, parts.netloc, "/compatible-mode/v1", "", ""))
    return EmbeddingConfig(base_url, api_key)


def create_embedding_client(config: EmbeddingConfig | None = None):
    from openai import OpenAI
    config = config or embedding_config()
    return OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=90.0, max_retries=0)


def embedding_vectors(client, texts: list[str]) -> list[list[float]]:
    """Honor DashScope's ten-row limit and restore each response's input order."""
    vectors = []
    for offset in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[offset:offset + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch,
                                            dimensions=EMBEDDING_DIMENSIONS, encoding_format="float")
        rows = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in rows] != list(range(len(batch))):
            raise ValueError("Embedding provider returned invalid indices")
        if any(len(item.embedding) != EMBEDDING_DIMENSIONS for item in rows):
            raise ValueError("Embedding provider returned an unexpected dimension")
        vectors.extend(item.embedding for item in rows)
    return vectors

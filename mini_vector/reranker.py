"""Independent, instruction-aware DashScope reranker; no chat/embedding client.

Only the query and fact texts are sent. Import and construction never call a
provider. The external credential file is read only when explicitly configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .aml_config import PROVIDER_RETRIES, _read_env
from .config import setting

TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query."
SUPPORT_INSTRUCTION = (
    "Given a question about people and their past conversations, rank memories by how much factual information they provide to answer the question accurately and completely. "
    "Prioritize explicit facts that answer the requested attribute, person, event, time, place, reason, or quantity, with the correct person and relevant conditions. "
    "For multi-part, comparison, counting, or multi-hop questions, also value a memory that supplies a necessary part of the answer or a concrete link needed to combine facts across memories; a memory need not answer the entire question alone. "
    "Respect dates, negation, uncertainty, reported statements, and the distinction between plans and completed events. "
    "Give low relevance to memories that merely share names or topic words without useful evidence for the requested information. Do not invent missing facts."
)


@dataclass(frozen=True)
class RerankerConfig:
    endpoint: str = field(repr=False)
    api_key: str = field(repr=False)
    model: str = "qwen3-rerank"
    instruction: str = DEFAULT_INSTRUCTION
    timeout: float = 120.0

    def __post_init__(self):
        url = urlsplit(self.endpoint)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("Reranker endpoint must be an HTTPS URL without credentials or query")
        if any(not isinstance(v, str) or not v.strip() for v in (self.api_key, self.model, self.instruction)):
            raise ValueError("Reranker key, model and instruction must be nonempty")
        if (isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float))
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("Reranker timeout must be finite and positive")

    def public(self):
        return {"model": self.model, "instruction": self.instruction,
                "instruction_sha256": hashlib.sha256(self.instruction.encode()).hexdigest(),
                "provider_sha256": hashlib.sha256(self.endpoint.encode()).hexdigest(),
                "timeout": self.timeout, "protocol": "dashscope_text_rerank"}


def qwen_reranker_config(*, model=None, instruction=None, endpoint=None, api_key=None,
                         env_file=None, timeout=120.0):
    path = env_file or os.getenv("MINI_VECTOR_RERANKER_ENV_FILE") or setting("MINI_VECTOR_RERANKER_ENV_FILE")
    values = _read_env(Path(path)) if path else {}

    def value(name, default=None):
        return os.environ.get(name, values.get(name, setting(name, default)))

    key_name = value("QWEN_RERANK_API_KEY_ENV", "DASHSCOPE_API_KEY")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_name):
        raise ValueError("Invalid reranker credential variable name")
    return RerankerConfig(endpoint if endpoint is not None else value("QWEN_RERANK_ENDPOINT", ""),
                          api_key if api_key is not None else value(key_name, ""),
                          model if model is not None else value("QWEN_RERANK_MODEL", "qwen3-rerank"),
                          instruction if instruction is not None else value("QWEN_RERANK_INSTRUCTION", DEFAULT_INSTRUCTION),
                          timeout)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


class QwenReranker:
    def __init__(self, config: RerankerConfig, *, transport=None):
        self.config = config
        self.transport = transport

    def payload(self, query, documents):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Reranker query must be nonempty")
        if (not documents or len(documents) > 500
                or any(not isinstance(d, str) or not d.strip() for d in documents)):
            raise ValueError("Reranker requires 1 to 500 nonempty documents")
        return {"model": self.config.model, "input": {"query": query, "documents": list(documents)},
                "parameters": {"top_n": len(documents), "return_documents": False,
                               "instruct": self.config.instruction}}

    def __call__(self, query, documents):
        body = self.payload(query, documents)
        if self.transport:
            return self.transport(body)
        request = urllib.request.Request(self.config.endpoint, data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Authorization": "Bearer " + self.config.api_key, "Content-Type": "application/json"})
        # No redirects. Transient rate-limit/server/network failures are retried a
        # bounded number of times so concurrent load does not silently skip reranking.
        for attempt in range(PROVIDER_RETRIES + 1):
            try:
                with urllib.request.build_opener(_NoRedirect).open(request, timeout=self.config.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in TRANSIENT_STATUS or attempt == PROVIDER_RETRIES:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == PROVIDER_RETRIES:
                    raise
            time.sleep(min(8.0, 1.0 * 2 ** attempt))
        # Provider errors can echo inputs or credentials. Never expose their text.
        return json.loads(json.dumps(raw, ensure_ascii=False).replace(self.config.api_key, "[REDACTED]"))


def parse_scores(response, count):
    if (not isinstance(response, dict) or response.get("code") or response.get("error")
            or not isinstance(response.get("output"), dict)):
        raise ValueError("Invalid reranker response")
    rows = response["output"].get("results")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Reranker must score every candidate")
    scores = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid reranker result")
        index, score = row.get("index"), row.get("relevance_score")
        if (type(index) is not int or not 0 <= index < count or index in scores
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid reranker index or score")
        scores[index] = float(score)
    return [scores[i] for i in range(count)]


def rerank_qwen(reranker, query, candidates, limit):
    """Score a fused prefix, stable ties, retain tail; fail closed to original RRF.

    Candidate objects/text/metadata are never rewritten. Scores remain separate
    from RRF scores; a failed call must not be reported as successful reranking.
    """
    head = candidates[:limit]
    trace = {"status": "skipped", "api_calls": 0, "latency_ms": 0.0,
             "input_ids": [c["id"] for c in head], "scores": []}
    if hasattr(reranker, "config"):
        trace["config"] = reranker.config.public()
    ranked = candidates
    if len(head) >= 2:
        started = time.perf_counter()
        trace["api_calls"] = 1
        try:
            scores = parse_scores(reranker(query, [c["memory"] for c in head]), len(head))
            order = sorted(range(len(head)), key=lambda i: -scores[i])
            ranked = [head[i] for i in order] + candidates[len(head):]
            trace.update(status="succeeded", scores=[{"memory_id": c["id"], "score": s}
                         for c, s in zip(head, scores)])
        except Exception as exc:
            trace.update(status="fallback", error_type=type(exc).__name__)
        trace["latency_ms"] = (time.perf_counter() - started) * 1000
    trace["output_ids"] = [c["id"] for c in ranked]
    return ranked, trace

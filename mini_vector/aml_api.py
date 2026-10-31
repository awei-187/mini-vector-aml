"""Synchronous AML Textual Add/Search service. Run: python -m mini_vector.aml_api."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager, contextmanager
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Annotated, Literal

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, StrictStr
from starlette.concurrency import run_in_threadpool

from .aml_config import (EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, PROVIDER_RETRIES, embedding_config,
                         create_embedding_client, embedding_vectors)
from .aml_store import AMLStore
from .config import setting
from .memory import MiniVector
from .aml_time import date_prefix
from .reranker import INSTRUCTION_PRESETS, SUPPORT_INSTRUCTION, QwenReranker, qwen_reranker_config

CHAT_MODEL = "gpt-4o-mini"
WORKER_THREADS = 64
DEFAULT_DB = Path(__file__).resolve().parents[1] / "runtime" / "aml" / "f3-qwen-support-no-temporal.sqlite3"
logger = logging.getLogger(__name__)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


Text = Annotated[StrictStr, AfterValidator(_nonblank)]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(WireModel):
    role: Literal["user", "assistant"]
    content: Text
    timestamp: StrictInt | None = None


class AddRequest(WireModel):
    request_id: Text
    user_id: Text
    session_id: Text
    messages: list[Message] = Field(min_length=1)


class AddResponse(WireModel):
    success: Literal[True]
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(WireModel):
    query: Text
    user_id: Text
    top_k: Annotated[StrictInt, Field(gt=0)]
    options: list[Text] | None = None


class MemoryRecord(WireModel):
    id: Text
    content: Text
    score: FiniteFloat
    created_at: Text


class SearchResponse(WireModel):
    data: list[MemoryRecord]


class RequestConflict(Exception):
    pass


def _error_status(exc: Exception) -> tuple[int, str]:
    """Inspect wrapped SDK errors without ever serializing their messages."""
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    chain, seen = exc, set()
    while chain is not None and id(chain) not in seen:
        seen.add(id(chain))
        if isinstance(chain, RequestConflict):
            return 409, "request_id_conflict"
        if isinstance(chain, (TimeoutError, APITimeoutError)):
            return 504, "upstream_timeout"
        if isinstance(chain, (ConnectionError, APIConnectionError, APIStatusError)):
            return 503, "upstream_unavailable"
        chain = chain.__cause__ or chain.__context__
    return 500, "internal_error"


class AMLBackend:
    def __init__(self, path: str, *, llm=None, embedder=None, memory_query_enabled=True,
                 rerank_mode="qwen", queries_per_memory=3, rerank_limit=60,
                 reranker_model="qwen3-rerank", reranker_instruction=SUPPORT_INSTRUCTION, reranker_endpoint=None,
                 reranker_api_key=None, reranker_env_file=None, reranker_timeout=120.0,
                 search_time_format="as_is", extraction_prompt="original", reranker=None, query_rewrite_enabled=True, query_rewrite_limit=3):
        if extraction_prompt not in {"original", "time_v2"}:
            raise ValueError("extraction_prompt must be original or time_v2")
        if search_time_format not in {"as_is", "date_prefix"}:
            raise ValueError("search_time_format must be as_is or date_prefix")
        self.search_time_format = search_time_format
        base_url = setting("OPENAI_BASE_URL")
        embedding = embedding_config() if embedder is None else None
        if rerank_mode == "qwen" and reranker is None:
            reranker = QwenReranker(qwen_reranker_config(model=reranker_model,
                instruction=reranker_instruction, endpoint=reranker_endpoint, api_key=reranker_api_key,
                env_file=reranker_env_file, timeout=reranker_timeout))
        reranker_identity = ({"reranker": reranker.config.public()} if rerank_mode == "qwen"
                             and hasattr(reranker, "config") else
                             ({"reranker": "injected"} if rerank_mode == "qwen" else {}))
        self.store = AMLStore(path, {
            "schema": 1, "chat_model": CHAT_MODEL, "embedding_model": EMBEDDING_MODEL,
            "provider_hash": hashlib.sha256((base_url or "https://api.openai.com/v1").encode()).hexdigest(),
            "embedding_provider_hash": embedding.provider_hash if embedding else "injected",
            "embedding_dimensions": EMBEDDING_DIMENSIONS if embedding else None,
            "infer": True, "threshold": 0.1, "extraction_prompt": extraction_prompt,
            "retrieval_version": 3, "memory_query_enabled": memory_query_enabled,
            "queries_per_memory": queries_per_memory, "rerank_mode": rerank_mode,
            "rerank_limit": rerank_limit, "fusion": "rrf_60",
            "query_rewrite_enabled": query_rewrite_enabled, "query_rewrite_limit": query_rewrite_limit,
            **reranker_identity,
        })
        # Adds for one user_id run in arrival order (extraction reads that user's
        # recent context and memories); different users overlap freely.
        self._user_locks: dict[str, list] = {}
        self._user_locks_guard = threading.Lock()
        self.client = None
        self.embedding_client = None
        try:
            if llm is None:
                from openai import OpenAI
                api_key = setting("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required for live AML service")
                self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0,
                                     max_retries=PROVIDER_RETRIES)
            if embedder is None:
                self.embedding_client = create_embedding_client(embedding)

            def complete(messages):
                messages = [dict(message) for message in messages]
                messages[0]["content"] += (" Optional timestamp fields are source-message times in Unix milliseconds; "
                                            "they are not necessarily the time of the described event.")
                response = self.client.chat.completions.create(
                    model=CHAT_MODEL, messages=messages, response_format={"type": "json_object"})
                return response.choices[0].message.content or "{}"

            def embed(texts):
                return embedding_vectors(self.embedding_client, texts)

            source_embed = embedder or embed

            def checked_embed(texts):
                vectors = source_embed(texts)
                if len(vectors) != len(texts) or not vectors:
                    raise ValueError("Invalid embedding batch")
                dimension = len(vectors[0])
                if not dimension or any(len(v) != dimension or not all(math.isfinite(float(x)) for x in v)
                                        or not any(float(x) != 0 for x in v) for v in vectors):
                    raise ValueError("Invalid embedding vectors")
                self.store.check_dimension(dimension)
                return vectors

            self.memory = MiniVector(store=self.store, llm=llm or complete, embedder=checked_embed,
                                     extraction_prompt=extraction_prompt,
                                     chat_model=CHAT_MODEL, embedding_model=EMBEDDING_MODEL,
                                     memory_query_enabled=memory_query_enabled,
                                     queries_per_memory=queries_per_memory,
                                     rerank_mode=rerank_mode, rerank_limit=rerank_limit, reranker=reranker,
                                     query_rewrite_enabled=query_rewrite_enabled,
                                     query_rewrite_limit=query_rewrite_limit)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _user_lock(self, user_id: str):
        with self._user_locks_guard:
            entry = self._user_locks.setdefault(user_id, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._user_locks_guard:
                entry[1] -= 1
                if not entry[1]:
                    del self._user_locks[user_id]

    def _replay(self, request_id: str, digest: str):
        previous = self.store.request(request_id)
        if previous:
            if previous["payload_hash"] != digest:
                raise RequestConflict()
            return json.loads(previous["response"])
        return None

    def add(self, payload: AddRequest):
        canonical = payload.model_dump(exclude_none=True)
        digest = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        with self._user_lock(payload.user_id):
            replay = self._replay(payload.request_id, digest)
            if replay is not None:
                return replay
            # Provider calls run without the storage lock; only the writes below
            # are serialized, in one transaction with the request ledger.
            plan = self.memory.prepare_add(
                canonical["messages"], user_id=payload.user_id,
                metadata={"session_id": payload.session_id, "request_id": payload.request_id,
                          "source_timestamps": [m.timestamp for m in payload.messages]})
            with self.store.atomic():
                # The same request_id may arrive concurrently under another user_id.
                replay = self._replay(payload.request_id, digest)
                if replay is not None:
                    return replay
                self.memory.commit_add(plan)
                response = {"success": True, "request_id": payload.request_id,
                            "user_id": payload.user_id, "session_id": payload.session_id}
                self.store.complete(payload.request_id, digest, response)
                return response

    def search(self, payload: SearchRequest):
        result = self.memory.search(payload.query, filters={"user_id": payload.user_id},
                                    top_k=payload.top_k, threshold=0.1)
        return {"data": [{"id": item["id"], "content": (date_prefix(item.get("metadata"))
                           if self.search_time_format == "date_prefix" else "") + item["memory"],
                          "score": item["score"], "created_at": item["created_at"]}
                         for item in result["results"]]}

    def close(self):
        try:
            if self.client is not None:
                self.client.close()
        finally:
            try:
                if self.embedding_client is not None:
                    self.embedding_client.close()
            finally:
                self.store.close()


def create_app(*, db_path: str | Path | None = None, api_token: str | None = None,
               llm=None, embedder=None, memory_query_enabled: bool = True,
               rerank_mode: str = "qwen", queries_per_memory: int = 3,
               rerank_limit: int = 60, reranker_model="qwen3-rerank", reranker_instruction=SUPPORT_INSTRUCTION,
               reranker_endpoint=None, reranker_api_key=None, reranker_env_file=None,
               reranker_timeout=120.0, search_time_format="as_is", extraction_prompt="original", reranker=None,
               query_rewrite_enabled: bool = True, query_rewrite_limit: int = 3) -> FastAPI:
    if extraction_prompt not in {"original", "time_v2"}:
        raise ValueError("extraction_prompt must be original or time_v2")
    if search_time_format not in {"as_is", "date_prefix"}:
        raise ValueError("search_time_format must be as_is or date_prefix")
    token = api_token if api_token is not None else os.getenv("AML_API_TOKEN") or setting("AML_API_TOKEN")
    if not token or not token.strip():
        raise ValueError("AML_API_TOKEN is required")
    path = str(db_path or os.getenv("AML_DB_PATH") or setting("AML_DB_PATH") or DEFAULT_DB)

    @asynccontextmanager
    async def lifespan(app):
        # Sync routes run in AnyIO's worker threads; leave room for concurrent
        # Add and Search beyond the default 40 while requests wait on providers.
        anyio.to_thread.current_default_thread_limiter().total_tokens = WORKER_THREADS
        try:
            app.state.backend = AMLBackend(path, llm=llm, embedder=embedder,
                memory_query_enabled=memory_query_enabled, rerank_mode=rerank_mode,
                queries_per_memory=queries_per_memory, rerank_limit=rerank_limit,
                reranker_model=reranker_model, reranker_instruction=reranker_instruction,
                reranker_endpoint=reranker_endpoint, reranker_api_key=reranker_api_key,
                reranker_env_file=reranker_env_file, reranker_timeout=reranker_timeout, reranker=reranker,
                search_time_format=search_time_format, extraction_prompt=extraction_prompt,
                query_rewrite_enabled=query_rewrite_enabled, query_rewrite_limit=query_rewrite_limit)
        except Exception:
            # Prevent provider/SQLite exceptions with embedded secrets or paths from reaching server logs.
            raise RuntimeError("AML startup failed: check credentials, database identity and single-process lease") from None
        try:
            yield
        finally:
            await run_in_threadpool(app.state.backend.close)
            app.state.backend = None

    app = FastAPI(title="Mini Vector AML Textual API", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.backend = None

    def authenticate(authorization: Annotated[str | None, Header()] = None):
        if not authorization:
            raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})
        scheme, _, provided = authorization.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(provided.encode(), token.encode()):
            raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Default validation errors echo the input; keep only structural error locations/types.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(item["loc"]), "type": item["type"]} for item in exc.errors()]})

    def dispatch(operation, payload, response_type):
        started = time.perf_counter()
        status = 200
        try:
            backend = app.state.backend
            if backend is None:
                status = 503
                return JSONResponse(status_code=status, content={"detail": {"reason": "not_ready"}})
            result = getattr(backend, operation)(payload)
            return response_type.model_validate(result)
        except Exception as exc:
            status, reason = _error_status(exc)
            return JSONResponse(status_code=status, content={"detail": {"reason": reason}})
        finally:
            logger.info("operation=%s status=%d elapsed_ms=%.0f", operation, status,
                        (time.perf_counter() - started) * 1000)

    @app.get("/health")
    async def health():
        if app.state.backend is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ok"}

    @app.post("/add", response_model=AddResponse, dependencies=[Depends(authenticate)])
    def add(payload: AddRequest):
        return dispatch("add", payload, AddResponse)

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(authenticate)])
    def search(payload: SearchRequest):
        return dispatch("search", payload, SearchResponse)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--extraction-prompt", choices=("original", "time_v2"), default="original")
    parser.add_argument("--search-time-format", choices=("as_is", "date_prefix"), default="as_is")
    parser.add_argument("--disable-memory-queries", action="store_true")
    parser.add_argument("--rerank-mode", choices=("none", "llm", "global", "evidence", "qwen"), default="qwen")
    parser.add_argument("--queries-per-memory", type=int, default=3)
    parser.add_argument("--rerank-limit", type=int, default=60)
    parser.add_argument("--disable-query-rewrite", action="store_true")
    parser.add_argument("--query-rewrite-limit", type=int, default=3)
    parser.add_argument("--reranker-model", default="qwen3-rerank")
    instruction = parser.add_mutually_exclusive_group()
    instruction.add_argument("--reranker-instruction")
    instruction.add_argument("--reranker-instruction-preset", choices=tuple(INSTRUCTION_PRESETS), default="support")
    parser.add_argument("--reranker-env-file")
    parser.add_argument("--reranker-timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.reranker_instruction is None:
        args.reranker_instruction = INSTRUCTION_PRESETS[args.reranker_instruction_preset]
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        app = create_app(db_path=args.db_path, extraction_prompt=args.extraction_prompt,
                         search_time_format=args.search_time_format,
                         memory_query_enabled=not args.disable_memory_queries,
                         rerank_mode=args.rerank_mode, queries_per_memory=args.queries_per_memory,
                         rerank_limit=args.rerank_limit, reranker_model=args.reranker_model,
                         reranker_instruction=args.reranker_instruction,
                         reranker_env_file=args.reranker_env_file, reranker_timeout=args.reranker_timeout,
                         query_rewrite_enabled=not args.disable_query_rewrite,
                         query_rewrite_limit=args.query_rewrite_limit)
    except ValueError:
        parser.exit(2, "AML_API_TOKEN must be configured before starting the service.\n")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()

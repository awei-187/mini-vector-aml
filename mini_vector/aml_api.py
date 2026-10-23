"""Synchronous AML Textual Add/Search service. Run: python -m mini_vector.aml_api."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
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

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, StrictStr
from starlette.concurrency import run_in_threadpool

from .aml_config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, embedding_config, create_embedding_client, embedding_vectors
from .aml_store import AMLStore
from .config import setting
from .memory import MiniVector

CHAT_MODEL = "gpt-4o-mini"
DEFAULT_DB = Path(__file__).resolve().parents[1] / "runtime" / "aml" / "memory.sqlite3"
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
    def __init__(self, path: str, *, llm=None, embedder=None):
        base_url = setting("OPENAI_BASE_URL")
        embedding = embedding_config() if embedder is None else None
        self.store = AMLStore(path, {
            "schema": 1, "chat_model": CHAT_MODEL, "embedding_model": EMBEDDING_MODEL,
            "provider_hash": hashlib.sha256((base_url or "https://api.openai.com/v1").encode()).hexdigest(),
            "embedding_provider_hash": embedding.provider_hash if embedding else "injected",
            "embedding_dimensions": EMBEDDING_DIMENSIONS if embedding else None,
            "infer": True, "threshold": 0.1,
        })
        self.operation_lock = threading.RLock()
        self.client = None
        self.embedding_client = None
        try:
            if llm is None:
                from openai import OpenAI
                api_key = setting("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required for live AML service")
                self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=0)
            if embedder is None:
                self.embedding_client = create_embedding_client(embedding)

            def extract(messages):
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

            self.memory = MiniVector(store=self.store, llm=llm or extract, embedder=checked_embed,
                                     chat_model=CHAT_MODEL, embedding_model=EMBEDDING_MODEL)
        except BaseException:
            self.close()
            raise

    def add(self, payload: AddRequest):
        canonical = payload.model_dump(exclude_none=True)
        digest = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        with self.operation_lock, self.store.atomic():
            previous = self.store.request(payload.request_id)
            if previous:
                if previous["payload_hash"] != digest:
                    raise RequestConflict()
                return json.loads(previous["response"])
            self.memory.add(canonical["messages"], user_id=payload.user_id, infer=True,
                            metadata={"session_id": payload.session_id, "request_id": payload.request_id,
                                      "source_timestamps": [m.timestamp for m in payload.messages]})
            response = {"success": True, "request_id": payload.request_id,
                        "user_id": payload.user_id, "session_id": payload.session_id}
            self.store.complete(payload.request_id, digest, response)
            return response

    def search(self, payload: SearchRequest):
        with self.operation_lock:
            result = self.memory.search(payload.query, filters={"user_id": payload.user_id},
                                        top_k=payload.top_k, threshold=0.1)
            return {"data": [{"id": item["id"], "content": item["memory"],
                              "score": item["score"], "created_at": item["created_at"]}
                             for item in result["results"]]}

    def close(self):
        with self.operation_lock:
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
               llm=None, embedder=None) -> FastAPI:
    token = api_token if api_token is not None else os.getenv("AML_API_TOKEN") or setting("AML_API_TOKEN")
    if not token or not token.strip():
        raise ValueError("AML_API_TOKEN is required")
    path = str(db_path or os.getenv("AML_DB_PATH") or setting("AML_DB_PATH") or DEFAULT_DB)

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.backend = AMLBackend(path, llm=llm, embedder=embedder)
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
    args = parser.parse_args()
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        app = create_app(db_path=args.db_path)
    except ValueError:
        parser.exit(2, "AML_API_TOKEN must be configured before starting the service.\n")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()

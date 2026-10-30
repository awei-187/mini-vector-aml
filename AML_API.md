# AML Textual API: enhanced-v2

## Contract

`GET /health` returns `{"status":"ok"}` after initialization, without auth or
model calls. Health does not prove upstream provider availability.

`POST /add` requires Bearer auth and ordered user/assistant text messages:

```json
{"request_id":"synthetic-1","user_id":"alice","session_id":"session-1","messages":[{"role":"user","content":"My project code is LANTERN-471.","timestamp":1767225600000}]}
```

Timestamp is optional Unix milliseconds and describes the source-message time,
not necessarily the time of a described event. A synchronous HTTP 200 response
is returned only after the full batch, FTS/history/context and request ledger
commit, with the three original IDs:

```json
{"success":true,"request_id":"synthetic-1","user_id":"alice","session_id":"session-1"}
```

The ledger is persistent and request IDs are global to this service database.
Same ID and normalized body replays success; changed content returns 409.
Failed ingestion rolls back atomically and can be retried.

`POST /search` requires Bearer auth:

```json
{"query":"What is my project code?","user_id":"alice","top_k":100,"options":["LANTERN-471","Other"]}
```

Search uses the original query plus up to three model-written subqueries,
ignores options and never generates an answer. User scope is exact (including
surrounding whitespace); sessions are provenance only, so search can retrieve
across sessions for the same user. The response is `{"data":[...]}` with at most
top_k records, each containing `id`, `content`, `score` and `created_at`, in
final rank order. Empty success is `{"data":[]}`.

Routes: fact-vector cosine (threshold 0.1), FTS5 BM25 and memory-side question
vectors for the original query; fact-vector cosine and BM25 for each subquery.
Fusion is equal-weight RRF (k=60); qwen3-rerank orders the first 60 fused
memories. `score` is the normalized RRF score and need not decrease with rank
after reranking. Scores are retrieval scores, not confidence estimates.

## Errors and execution

401: auth failure; 422: invalid schema; 409: conflicting request ID;
503: upstream unavailable; 504: upstream timeout; 500: other internal failure.
Failures are not converted to successful empty search results. Responses avoid
provider exception bodies and credential values. Startup failures are sanitized.

Use one Uvicorn process; it serves up to 64 requests at once in worker threads.
Adds for different users run in parallel; Adds for the same user run one at a
time in arrival order. Searches do not queue behind Adds. Provider calls run
outside the SQLite transaction; only the final writes are serialized. Health
never waits on provider calls. A database lease rejects a second service process
using the same database. A persisted identity rejects incompatible provider,
model or retrieval configuration. Chat and embedding clients use separate
keys/endpoints, 90-second SDK timeouts and two SDK retries for transient errors;
the reranker retries 408/429/5xx and network errors twice. Total request time can
exceed 90 seconds: an Add makes several sequential calls and may wait for the
same user's previous Add. Throughput is bounded by provider rate limits.

## Configuration

Chat settings are `OPENAI_BASE_URL` and `OPENAI_API_KEY`; the model is fixed to
gpt-4o-mini. The historical settings loader checks .env in the working directory,
project root and parent directory before process environment variables.

Set `AML_EMBEDDING_ENV_FILE` explicitly (process env takes precedence) to avoid
the legacy Windows default E:/.env. This private external file is not globally
loaded into the process. It contains QWEN_EMBEDDING_BASE_URL, the fixed
QWEN_EMBEDDING_MODEL=text-embedding-v4, QWEN_EMBEDDING_API_KEY_ENV and its key value.
The existing QWEN_RERANK_ENDPOINT/QWEN_RERANK_API_KEY_ENV format is also supported
for recognized DashScope hosts; the path is mapped to /compatible-mode/v1.
Embedding requests use 1024 dimensions, batches of at most 10 and response-index
validation. Missing or unsupported settings fail rather than fall back to chat.

Reranking reads `QWEN_RERANK_ENDPOINT` (an HTTPS DashScope text-rerank URL) and
the key named by `QWEN_RERANK_API_KEY_ENV` from the file in
`MINI_VECTOR_RERANKER_ENV_FILE` (or `--reranker-env-file`); the model is fixed
to qwen3-rerank with the built-in support instruction.

`AML_API_TOKEN` is required; process env wins over .env. Keep it stable across
restarts. DB path precedence is --db-path, AML_DB_PATH env, .env, then
runtime/aml/f3-qwen-support-no-temporal.sqlite3. Use a new database for an
incompatible configuration; the baseline-v1 database is rejected.

Official contract: https://agentmemoryleaderboard.ai/api-guide.

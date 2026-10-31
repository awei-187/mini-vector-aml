# Mini Vector AML

A compact synchronous memory service for the AML Textual / Open-source Methods
track. Search returns memory evidence, not a generated answer.

This repository contains the **enhanced concurrent version** (`enhanced-v2`,
commit `740cf2b285742a24985a003f7dd8785651de46c8`) and the **final submitted
configuration** built on it (see below). The earlier commit
`a001d583dae81b9cc4565984bd996c1108f0f734` is the frozen baseline
(`baseline-v1`), described by `BASELINE_MANIFEST.json`.

## Final submitted configuration

The AML Full submission runs enhanced-v2 code with three non-default switches:

```bash
.venv/bin/python -m mini_vector.aml_api --host 127.0.0.1 --port 8000 --db-path /var/lib/mini-vector/final-v2list.sqlite3 --extraction-prompt time_v2 --search-time-format date_prefix --reranker-instruction-preset list
```

- `--extraction-prompt time_v2`: the extraction prompt with revised time rules
  (`mini_vector/prompts_time_v2.py`). Only time-related text differs from the
  original prompt: relative times are resolved against the message's observation
  date at the precision the source supports (e.g. "last month" → year and month,
  "last week" → "the week before <date>", no invented day); unresolved relative
  wording keeps an observation anchor such as "(as mentioned on 8 May 2023)"; the
  observation date is never written as the event date unless the source says so.
  Three worked examples with over-precise dates were corrected.
- `--reranker-instruction-preset list`: qwen3-rerank receives `LIST_INSTRUCTION`
  (`mini_vector/reranker.py`), which gives high relevance to every memory that
  contributes a distinct item, instance or necessary link for list, count,
  comparison and multi-part questions, instead of the default support instruction.
- `--search-time-format date_prefix`: each Search `content` starts with the
  observation date of the memory's source messages, e.g.
  `[Stated on Monday, 8 May 2023] ...` (or `[Stated between ... and ...]`). The
  date comes from the Add messages' `timestamp` values; without timestamps no
  prefix is added.

Defaults without these switches remain the enhanced-v2 behavior.

## Method

Add (per AML chunk, synchronous):

1. `gpt-4o-mini` extracts ADD-only, self-contained facts from the ordered new
   messages, given the user's ten most recent messages and the ten most similar
   existing memories (mem0's non-graph extraction workflow). Exact duplicates
   within the user scope are skipped. The final configuration uses the time_v2
   extraction prompt.
2. For each new fact, `gpt-4o-mini` writes 1 to 3 retrieval questions the fact can
   answer (a memory-side question index).
3. Facts and questions are embedded with Qwen `text-embedding-v4` (1024
   dimensions) and committed to SQLite with FTS5, history, recent context and the
   request ledger in one transaction.

Search (per question, top_k up to 100):

1. Three routes for the original question: exact cosine over fact vectors
   (threshold 0.1), FTS5 BM25, and max cosine over each memory's question vectors.
2. `gpt-4o-mini` decomposes the question into up to three subqueries; each adds a
   fact-vector and a BM25 route. A failed rewrite falls back to the three routes.
3. Routes are fused by equal-weight reciprocal rank fusion (k=60), deduplicated by
   memory ID; `qwen3-rerank` with a fixed instruction (default: support; final
   configuration: list) orders the first 60 fused memories. A reranker failure
   keeps the RRF order.
4. With `date_prefix` (final configuration), each returned memory text is
   prefixed with its source observation date.

Options are not used, no answer is generated, and only memory text (plus the
optional observation-date prefix) is returned.

## Concurrency

The service runs one Uvicorn process. Adds for different `user_id` values run in
parallel; Adds for the same `user_id` run one at a time in arrival order, so
extraction sees exactly the context a serial run would. Searches do not queue.
Model and embedding calls run outside the SQLite transaction; only the final
writes are serialized. Per-user vector pools are cached and extended after each
commit. Question generation runs in parallel groups of six facts and embedding
batches of ten run up to eight at a time. Chat and embedding SDK calls retry
transient failures twice; the reranker retries 408/429/5xx and network errors twice.

## Install

Python 3.10+ with SQLite FTS5 is required. The following commands target Linux,
from the repository root; each code block contains one command.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -e '.[api,fast]'
```

Copy `.env.example` to `.env`, fill your chat endpoint/key and a stable random
`AML_API_TOKEN`. Copy `qwen.env.example` to a private configuration file and fill
the embedding and rerank endpoints/key; point both `AML_EMBEDDING_ENV_FILE` and
`MINI_VECTOR_RERANKER_ENV_FILE` to its absolute path. Choose a new writable
`AML_DB_PATH`: the database records its retrieval configuration and a baseline
database is rejected. Never commit populated configs.

```bash
.venv/bin/python -m mini_vector.aml_api --host 127.0.0.1 --port 8000
```

The service exposes only `GET /health`, `POST /add` and `POST /search`.
Add/Search require `Authorization: Bearer <AML_API_TOKEN>`; Health does not.
See [AML_API.md](AML_API.md) for the wire contract and [DEPLOYMENT.md](DEPLOYMENT.md)
for the production layout.

## Verify

Offline tests inject models and use real SQLite; they make no provider calls.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Synthetic live verification calls the configured real providers, starts a
temporary localhost HTTP service and removes its temporary database afterward.

```bash
.venv/bin/python scripts/verify_aml_api.py --live
```

Pre-release checks (2026-10-30, Windows, Python 3.11): 120 offline tests in this
repository passed; 11 synthetic live HTTP checks passed. A synthetic live load
test (about 1,240 words per Add) completed 32 Adds at 16-way concurrency with all
HTTP 200, 30.4 s mean Add latency and 21.5 Adds per minute. These are integration
and throughput checks, not AML Smoke, benchmark scores or evidence of memory quality.

Final-configuration checks (2026-10-31, Windows, Python 3.11): 122 offline tests
passed; a synthetic live HTTP check with the three switches above passed 13 checks
(Add, idempotent replay, 409 conflict, auth, user isolation, prefixed Search
results, persisted time_v2/list identity).

Development data: the prompt and instruction choices were compared offline on
public LoCoMo data (one conversation, conv-26). No benchmark data, questions,
answers or names are included in the code or prompts, and Search never reads
labels.

## Attribution and license

Based on the non-graph memory workflow of mem0: Chhikara, Khant, Aryan, Singh and
Yadav, "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory",
arXiv:2504.19413, 2025; code https://github.com/mem0ai/mem0. See [NOTICE](NOTICE)
for every change and [LICENSE](LICENSE) for Apache-2.0 terms.
AML contract: https://agentmemoryleaderboard.ai/api-guide.

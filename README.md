# Mini Vector Memory

Mini Vector is a compact synchronous agent memory based on the non-graph OSS workflow in mem0. It uses ADD-only fact extraction, memory-side retrieval-question generation, SQLite persistence, independent fact-vector / question-vector / FTS5 BM25 retrieval, per-channel LLM reranking, and RRF fusion. SQLite scans scoped vectors exactly, so this implementation favors a small local deployment over an approximate-nearest-neighbor index. See [RETRIEVAL.md](RETRIEVAL.md) for the enhanced workflow, configuration, migration and local implementation references.

Opt-in fused selection modes and a fixed-corpus LoCoMo experiment are described in
[OPTIMIZATION.md](OPTIMIZATION.md). All three tested modes scored 130/152 on the
first conversation; see [results and costs](RESULTS_20261024.md).

A follow-up using the same cached retrieval scored 137/152 with corrected time
semantics and 140/152 with added source excerpts. See the
[full follow-up results and grading limitations](RESULTS_TEMPORAL_SOURCE_20261024.md).

On the edited147 dataset, the latest policy scored 103/147 with the existing GPT
four-class judge and 111/147 with pinned Jev; see
[edited-data evaluation and judge disagreements](RESULTS_EDITED147_DUAL_20261024.md).

The SQLite scan optimization branch adds bounded decoded-vector caches, optional
float64 NumPy acceleration, scope indexes and batched payload reads. See
[configuration, cache correctness and offline performance replay](PERFORMANCE.md).
Both backends exactly matched all 147 saved retrieval traces and answer contexts;
local warm-cache search measured 505 ms (Python) / 81 ms (NumPy), versus 3,011 ms
for the frozen implementation. See [measurements and limits](RESULTS_VECTOR_SCAN_20261025.md).

## Install and configure

Python 3.10+ and SQLite with FTS5 are required. From PowerShell:

```powershell
cd E:\AML\mini-vector
```

```powershell
D:\conda\envs\agent_aml\python.exe -m pip install -e .
```

Copy `.env.example` to `.env` in the working directory, `mini-vector`, or the AML workspace root and set `OPENAI_API_KEY` for the selected OpenAI-compatible service. Optional API settings include `OPENAI_BASE_URL` and both model names. Explicit constructor arguments take precedence over `.env`. The defaults are `gpt-4o-mini`, `text-embedding-3-small`, and `mini_vector.sqlite3` in the current working directory. Keep `.env` private.

```python
from mini_vector import MiniVector

memory = MiniVector()
memory.add("I live in Berlin and prefer biking to work.", user_id="alice")
found = memory.search("How does Alice commute?", filters={"user_id": "alice"})
print(found["results"])
memory.close()
```

`add(..., infer=True)` retrieves up to ten recent messages and ten similar memories, extracts self-contained facts, deduplicates exact MD5 hashes within scope, generates 1–3 retrieval questions per new fact, embeds facts and individual questions, and records ADD history. Empty extraction still saves context. `infer=False` stores non-system messages directly but also generates retrieval questions by default. At least one of `user_id`, `agent_id`, or `run_id` is required for scoped operations; searches use `filters={...}`.

`search` retrieves each channel independently and fuses ranks using equal-weight RRF before taking global top-k. The backward-compatible default `rerank_mode="llm"` asks the CHAT model to reorder the first30 candidates per channel; it is not a dedicated reranker. `none` skips reranking; `global`/`evidence` use chat-based selection after fusion. `qwen` uses a separate, instruction-aware Qwen reranker after fusion, with its own model, endpoint and credentials. Question hits return original memories. Invalid rerank output falls back to the original order with an explicit trace status. `explain=True` adds component scores and ranks. Text updates rebuild question indexes; metadata updates retain them; deletes remove them. For the original retrieval algorithm use `MiniVector(memory_query_enabled=False, rerank_mode="none")`. Ordinary old databases need explicit `backfill_memory_queries(filters=...)` to index existing facts; opening them never triggers model calls.

### Independent Qwen reranker

```python
from mini_vector import MiniVector
from mini_vector.reranker import SUPPORT_INSTRUCTION

memory = MiniVector(
    "memory.sqlite3",
    chat_model="gpt-4o-mini",          # Extraction/question generation only.
    memory_query_enabled=True,
    rerank_mode="qwen",
    rerank_limit=60,                    # Fused RRF prefix, one scoring request.
    reranker_env_file="E:/.env",        # Existing external Qwen config/credentials.
    reranker_model="qwen3-rerank",       # Independent from chat_model.
    reranker_instruction=SUPPORT_INSTRUCTION,
    reranker_timeout=120.0,
)
try:
    trace = {}
    result = memory.search("Where does Alice live?", filters={"user_id": "alice"},
                           top_k=20, explain=True, research_trace=trace)
finally:
    memory.close()
```

The credential file supplies `QWEN_RERANK_ENDPOINT`, `QWEN_RERANK_MODEL`,
`QWEN_RERANK_API_KEY_ENV` (default `DASHSCOPE_API_KEY`) and that named key.
`QWEN_RERANK_INSTRUCTION` is optional; the standard retrieval instruction is
the fallback. Explicit constructor values override configuration; process
environment values override the external file. Nothing is copied into
`os.environ`. Without an explicit file or `MINI_VECTOR_RERANKER_ENV_FILE`,
configuration uses the normal project/workspace settings. Opening with `none`
does not load or call the reranker.

Qwen receives only query/fact texts and `parameters.instruct`. Score every
candidate in the prefix, sort descending with stable ties, preserve the tail,
then take top-k. `reranker_score` is a separate provider score on scored results;
`score` retains the original RRF value and may no longer descend with output
rank. Neither score is a calibrated probability. Research trace schema4 records
the score mapping, ordering, model/instruction identity and any fallback.
Library calls do not redirect or retry implicitly. The optional AML server also
accepts `--rerank-mode qwen`, `--reranker-model`, `--reranker-instruction`,
`--reranker-env-file` and `--reranker-timeout`; retrieval identity is persisted
with its existing database configuration guard.

The current recommendation remains explicit `rerank_mode="none"`. The completed
147-question paired experiment found GPT/Jev correct counts 101/112 (none),
102/115 (Qwen default instruction), and 97/116 (Qwen support instruction).
Small judge-dependent changes do not establish a stable gain. See the
[results](RESULTS_QWEN_INSTRUCT_20261025.md) and
[predeclared protocol](QWEN_RERANK_EXPERIMENT_20261025.md).

A separate original152 historical-control experiment retained the OLD answer
prompt and generous binary GPT judge: fresh none131/152, Qwen default134/152,
Qwen support136/152 (paired net+3/+5). It used the same frozen RRF60 pool and
top20, without the newer temporal/source policy or Jev. These scores are distinct
from the strict147 results; old global/evidence were not rerun in the same batch.
See [original152 results](RESULTS_QWEN_ORIGINAL152_20261025.md).

## Fact extraction prompt

Since 2026-10-23, `add(infer=True)` uses the user's full Memory Extractor prompt verbatim as its system message, stored in [mini_vector/prompts.py](mini_vector/prompts.py). Runtime JSON supplies new messages, recent messages, existing memories, observation date and current UTC date. Summary and recently extracted memories remain explicitly empty because the library does not produce those inputs. Source dates come from `metadata.observation_date`, message Unix-millisecond timestamps, `metadata.source_timestamps`, or the existing LoCoMo `metadata.session_date`; unknown source dates are marked unknown rather than replaced by the current date. Multiple message dates remain distinct. Numeric source timestamps are interpreted in UTC.

This changes fact extraction only; question generation and per-channel reranking retain their own prompts. The existing parser consumes `memory[].text` and assigns real UUIDs; model-generated `id`, `attributed_to` and `linked_memory_ids` do not automatically become persisted fields or a relationship graph. The AML live client still appends its existing source-timestamp clarification to system messages. Existing memories are not re-extracted; running services must reload the code. New LoCoMo runs freeze and hash the extraction prompt module alongside the other sources; historical results are preserved and do not measure this prompt.

## Deliberate scope

This keeps the main mem0 vector flow and removes provider factories, async duplicates, telemetry, procedural memory, expiration policies, entity-vector linking, linked-memory relationships, hosted services, and advanced metadata operators. In particular, there is no entity extraction/index and no graph. FTS5 BM25 is a real SQLite lexical ranker, not a string-overlap substitute. Unlike a dedicated vector database, SQLite's exact cosine scan is O(number of memories in the selected scope); it is appropriate for compact local memory stores, not large corpora.

Both API operations are injectable for offline tests: `llm(messages) -> JSON string` and `embedder(texts) -> list[list[float]]`. These hooks test orchestration without claiming live provider integration. The default implementation calls the configured OpenAI-compatible chat and embeddings endpoints; it never falls back to fabricated vectors or heuristic fact extraction.

## LoCoMo evaluation

The real conversation-zero evaluation runner is in [evals/README.md](evals/README.md). It ingests all 419 utterances in `conv-26`, evaluates the 152 category 1–4 questions with top-20 retrieval, and saves actual model calls, native retrieval traces, answers, judgments, persistent memory and resumable checkpoints under `eval_outputs/`. Offline UI fixtures are separate from these benchmark runs.

The 2026-10-23 run with the full new extraction prompt and enhanced retrieval scored **135/152 (88.82%)**, with 1,759 memories and 4,920 memory-side questions; independent verification passed. All 456 channel rerank responses failed the exact-permutation contract and fell back, so the score reflects original channel orders fused by RRF. The earlier legacy run scored 126/152 (82.89%); both extraction and retrieval changed, so this is not a prompt-only ablation. See the [run report](E:/AML/mini-vector/eval_outputs/locomo_conv26_newprompt_enhanced_20261023_01/REPORT.md).

## AML Textual API

See [AML_API.md](AML_API.md) for the optional FastAPI service, authentication and atomic idempotent ingestion. It defaults to the enhanced retrieval and uses the existing chat provider for `gpt-4o-mini` and a separate Qwen client for `text-embedding-v4`, with settings read from `E:\.env` (override with `AML_EMBEDDING_ENV_FILE`). Retrieval settings are bound to database identity: use a new database path when upgrading an old AML database. Historical artifacts remain unchanged. The earlier live HTTP verification predates these retrieval changes; current validation uses injected models and is not an official AML Smoke or a benchmark score.

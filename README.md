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

`search` retrieves each channel independently, reranks the first 30 candidates per channel with the LLM, and fuses ranks using equal-weight RRF before taking global top-k. Question hits return original memories. Invalid rerank output falls back to that channel's original order. `explain=True` adds component scores and ranks. Text updates rebuild question indexes; metadata updates retain them; deletes remove them. For the original retrieval algorithm use `MiniVector(memory_query_enabled=False, rerank_mode="none")`. Ordinary old databases need explicit `backfill_memory_queries(filters=...)` to index existing facts; opening them never triggers model calls.

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

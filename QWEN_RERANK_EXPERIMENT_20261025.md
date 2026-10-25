# Qwen instruction-aware reranking: predeclared comparison

Prepared 2026-10-25, before observing new model scores. The user explicitly
authorized the three arms, both judges, public-derived inputs and added costs.

The implementation separates `rerank_mode="qwen"` from the historical chat-LLM
`llm`, `global` and `evidence` modes. Dedicated Qwen requests contain a model,
query, fact-text documents and `parameters.instruct`. They return indexed
relevance scores, never a generated JSON permutation of memory IDs.

## Fixed inputs and comparison

- Reuse the independently verified edited147 run: 419 public LoCoMo turns,
  1,742 extracted facts and 4,813 memory-side questions. No new ingestion,
  extraction, query generation or embedding calls.
- Freeze the SAME top60 facts of each question's three-channel RRF, including
  their text, metadata and order. Verify the original top20 and source context
  exactly. All 147 questions have 60 candidates.
- `none`: first20 of the frozen pool, no reranker.
- `qwen_default`: score all60 using `qwen3-rerank` and the existing default
  instruction; descending relevance, stable original-order ties, take20.
- `qwen_support`: same pool/model/procedure using the existing support-evidence
  instruction. Both exact instructions are constants in `mini_vector/reranker.py`
  and frozen in the run manifest before execution. No score-driven prompt tuning.
- All arms use the same temporal/source answer policy, gpt-4o-mini at temperature0,
  top20, and the same source-window procedure: max4 windows, radius1,
  max12,000 raw source-content characters. Window contents can change because
  their provenance anchors depend on the selected memories. This tests the
  reranker's effect within the complete current answer flow.
- Generate NEW answers for all arms. Do not reuse historical answers/judgments.
  Identical full requests within this new run share one saved result, including
  identical judge states; this avoids paying repeatedly for identical evidence
  or attributing judge variation on identical answers to reranking.
- Apply the same frozen GPT four-class rubric and pinned Jev four-class rubric
  to the SAME saved answer. Jev response must resolve to
  `typesafe/jev-1.13-20260917`. LLM-J means only label `correct` /147;
  report partial/missing/incorrect separately and paired wins/losses.

## Authorization and data boundaries

Up to294 formal Qwen scoring groups,441 distinct answer requests,441 GPT
judgments and441 Jev judgments, plus small synthetic protocol preflights and
bounded retries. Deduplication can reduce actual calls. Chat/answer/GPT judge
uses the configured api.relayrouter.ai service; Qwen uses the existing configured
DashScope dedicated endpoint; Jev uses openrouter.ai/api/v1/systemone.

Qwen receives only the question and candidate fact texts, plus the fixed
instruction. No gold, evidence labels, pseudoqueries, route names, old ranks,
memory IDs or original-source corpus is sent to Qwen. Answers receive the selected
facts and source windows. Gold is sent only to judges, together with the question
and generated answer. Credentials are never logged. Detailed research artifacts
remain ignored local files; public exports contain aggregate scores and hashes.

## Validation and interpretation

Provider/format failure falls back in the library with explicit trace status;
the experiment refuses to count fallback as a successful Qwen result. Independent
verification checks all frozen pools, raw Qwen payloads and scored ordering,
source context, answer requests, both judge payloads, request caches and usage.
Failed returned responses still contribute their reported usage.

This is one already examined conversation, not an unseen conversation or a
generalization claim. Qwen's service alias does not expose an immutable weight
snapshot. Record returned metadata without inventing a fixed model revision.
No new Mem0 run is part of this experiment. Older GPT chat-reranking failures
do not establish whether dedicated Qwen reranking helps.

# Retrieval optimization experiment, 2026-10-24

The preserved source is commit `8955ef5cac9478162df082e5cd38bddf4ae285a9`
on `snapshot/current-20261024`. Work proceeds on
`optimize/evidence-selection-20261024`; the deployed `main` branch is preserved.

The historical latest run scored 135/152 using 1,759 facts and 4,920 generated
questions. Its 456 channel rerank calls all fell back: valid JSON contained
partial selections rather than the required full permutations. This experiment
tests removal of that overhead and useful selection after channel fusion.

## Implementation

The original `none` and `llm` modes and the extraction prompt remain available.
Two opt-in `MiniVector` modes, `global` and `evidence`, fuse the unchanged three
retrieval channels with equal-weight RRF before making a single model call.
`global` selects the strongest evidence; `evidence` additionally asks for
complementary details and avoidance of paraphrase repetition. The corpus is not
deduplicated or rewritten by either selection mode.

The new contract explicitly accepts 1 through `min(top_k, rerank_limit)` distinct
IDs from the fused prefix. Selected facts move to the front; every unselected
fact retains its relative order. Unknown, repeated, empty, malformed or excessive
selections cause complete fallback to RRF. A provider exception records only its
type. Empty and singleton pools make no rerank call. Returned fact text, metadata
and RRF scores are preserved. Consequently RRF scores need not decrease in the
selected result order; they are not model relevance probabilities.

Schema-3 traces record the fused pool, selection input, accepted subset and full
stable tail. The verifier independently recomputes cosine and question pools,
BM25, RRF and selection order; it also checks the actual successful selection
response and candidate payload against stored facts. Selection receives the
question, facts, observation dates and source dialogue IDs, with no gold labels
or reference answers. The shared historical answer and judge prompts remain
unchanged, including their known timestamp wording and generous grading rule.

## Fixed experiment

Build one fresh corpus from original LoCoMo conversation index 0 (`conv-26`):
419 utterances across 19 sessions. Generate up to three retrieval questions per
fact using the current full extraction prompt. Reuse consistent SQLite backups
for three arms, each retrieving 20 facts for the same 152 category 1-4 questions:

| Arm | Selection | Candidate prefix | Model calls per retrieval |
| --- | --- | --- | --- |
| `rrf` | Raw three-channel RRF | Not used | 0 rerank calls |
| `global` | Strongest evidence after RRF | 60 | At most 1 |
| `evidence` | Complementary evidence after RRF | 60 | At most 1 |

All arms use `gpt-4o-mini` for answers and judging and
`text-embedding-3-small` for embeddings through the configured provider.
Category 5 (47 adversarial questions) is excluded under the historical protocol.
Gold answers enter API requests only during judging. No prompt is selected by
looking at gold outcomes. The previous 135/152 run used a different stochastic
corpus and is a historical reference; the new `rrf` arm is the controlled baseline.

The batch refuses configuration/source changes on resume. Corpus reuse requires
the registered dataset and source identities and exactly all 419 ingestion
markers. Successful utterances, retrievals, answers and judgments are committed
independently for restart. Export hashes must match across all three corpora.
Per-arm usage excludes shared ingestion and preflight, which are reported once
under `corpus/api_usage_summary.json`.

Run from this project in a visible Windows PowerShell window:

```powershell
& .\evals\run_optimization.ps1 -RunDir 'E:\AML\mini-vector\eval_outputs\locomo_conv26_optimization_20261024_01'
```

Re-running the same command resumes matching source and configuration. The batch
produces a local `REPORT.md`, `comparison.json`, source hashes, SDK request/response
audits, SQLite snapshots and per-arm independent `verification.json` files.
`eval_outputs/`, runtime databases and private settings remain ignored by Git.

## Validation and interpretation

The pre-experiment offline suite passed 63 tests, covering the legacy API and new
selection/fallback/isolation contracts. A separate injected-client workflow check
exercises the full 419-turn, three-arm checkpoint/reuse/verification path without
live calls; its fixture scores are never benchmark scores. Live preflight checks
both 60-candidate selection contracts before the paid full ingestion.

Live results are pending until the complete experiment passes independent checks.
One conversation with the existing LLM judge cannot establish dataset-wide gains.

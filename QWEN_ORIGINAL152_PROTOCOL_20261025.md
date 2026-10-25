# Qwen on original152: predeclared controls

Prepared before new scores on 2026-10-25. The user explicitly authorized both
Qwen instructions, three152-question answer/judge arms and added costs.

- Reuse the verified original `locomo_conv26_optimization_20261024_01/rrf` corpus:
  419 turns,1,711 facts and4,766 memory-side questions. No ingestion or embedding.
- Freeze the SAME RRF60 facts for every question, select20 for all three arms:
  none, Qwen default instruction, Qwen support instruction. Both exact
  instructions are unchanged from the earlier Qwen experiment and frozen in
  the manifest. Score all60, stable original-order ties. Fallback stops the run.
- Use the ORIGINAL152 context, answer prompt and generous GPT binary judge,
  gpt-4o-mini, temperature0. Preparation checks all152 unchanged-arm contexts
  and answer request bodies against the actual historical requests.
- This deliberately retains the historical event_time wording, including its
  known observation/event-time confusion, to avoid changing answering while
  testing reranking. It is a historical-control evaluation, not the recommended
  current answer policy. Do not interpret the generous binary score as strict
  four-class LLM-J or human correctness.
- Generate new answers for all456 arm/question records. No historical answers
  or judgments are reused. Identical full requests within THIS new run share
  results; preserve their audit links and account usage once.
- No source-window supplement, temporal-policy replacement or JEV calls.
  Gold goes only to the binary judge. Qwen receives only query, fact-text
  documents and instruction, without IDs, gold, source labels or channel ranks.
- Up to304 formal Qwen scoring groups,456 distinct answer requests and456
  binary judgments, plus small synthetic preflights and bounded failure retries.
  Existing DashScope endpoint for Qwen; configured api.relayrouter.ai for GPT.
  No credentials or other project data are sent or logged.
- Freeze input/source/provider fingerprints, preserve returned usage on failed
  protocol attempts and independently verify all456 same-pool records,
  304 scored orders and912 answer/judge request boundaries.
- Old global/evidence130/152 scores remain HISTORICAL references. Those two
  modes are not rerun in this batch; no five-way simultaneous ranking claim.
  Use the fresh none arm for paired Qwen gains. One already examined conversation
  does not establish unseen-conversation generalization or a fixed Qwen weight
  snapshot behind the service alias.

# Research snapshot before optimization

This branch preserves the current local mini-vector implementation before the
2026-10-24 optimization work. Package sources, tests and evaluation launchers are
byte-for-byte copies. License and attribution are retained from the published
baseline. The deployed `main` branch is not changed.

The latest saved original LoCoMo conv-26 run scored 135/152 (88.82%), with 1,759
memories and 4,920 generated retrieval questions. All 456 channel rerank attempts
fell back because their output was not a complete candidate-ID permutation.
This is a combined extraction/retrieval result, not a prompt-only ablation.

`RESEARCH_SNAPSHOT.json` records source hashes and the existing score summary.
Private settings, databases and bulk evaluation data are deliberately excluded.
The original evaluation launcher expects the AML sibling Mem0 reference helper
and public LoCoMo data; its paths are preserved as part of this snapshot.

The original NOTICE describes the deployed baseline; this snapshot additionally
contains the memory-question channel, channel reranking, RRF, and the supplied
full extraction prompt. No experiment raw inputs or API traces are published.

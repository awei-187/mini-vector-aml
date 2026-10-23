# Mini Vector AML baseline

A compact synchronous memory service for the AML Textual / Open-source Methods
track. This repository freezes the deployed baseline, using ADD-only fact
extraction, exact cosine vector retrieval and SQLite FTS5 BM25 reranking.
Search returns memory evidence, not a generated answer. This version does not
contain later retrieval-question generation, per-channel LLM reranking or RRF.

The AML adapter fixes chat to `gpt-4o-mini` and embeddings to Qwen
`text-embedding-v4` (1024 dimensions). Ordinary library embedding defaults are
separate and do not change the competition service configuration.

## Install

Python 3.10+ with SQLite FTS5 is required. The following commands target Linux,
from the repository root; each code block contains one command.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -e '.[api]'
```

Copy `.env.example` to `.env`, fill your chat endpoint/key and a stable random
`AML_API_TOKEN`. Copy `qwen.env.example` to a private configuration file, set its
embedding endpoint/key, and set `AML_EMBEDDING_ENV_FILE` to that absolute path.
Choose a writable `AML_DB_PATH`. The endpoint must support the required models;
the service does not silently substitute them. Never commit populated configs.

```bash
.venv/bin/python -m mini_vector.aml_api --host 127.0.0.1 --port 8000
```

The service exposes only `GET /health`, `POST /add` and `POST /search`.
Add/Search require `Authorization: Bearer <AML_API_TOKEN>`; Health does not.
See [AML_API.md](AML_API.md) for the wire contract and [DEPLOYMENT.md](DEPLOYMENT.md)
for the production layout and HTTPS configuration.

## Verify

Offline tests inject models and use real SQLite; they make no provider calls.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Synthetic live verification calls the configured real model providers, starts a
temporary localhost HTTP service and removes its temporary database afterward.

```bash
.venv/bin/python scripts/verify_aml_api.py --live
```

Reported deployment validation: Ubuntu 22.04 / Python 3.10, 31 offline tests
passed; 11 synthetic live HTTP checks passed, including restart and request
replay. Public HTTPS health and certificate renewal were also checked. These
are deployment/integration checks, not official AML Smoke, benchmark scores or
evidence of memory quality. Official AML Smoke and Full have not been run.

## Version and limits

Use the full Git commit SHA when submitting. `BASELINE_MANIFEST.json` records
SHA-256 hashes of the deployed baseline files. The original archive was used to
prepare this repository; no model logic changes were introduced by packaging.
One worker and one shared operation lock serialize Add/Search: declared active
operation capacity is 1. Health remains independent. Vector scanning is linear
in the selected user's memory count; there is no measured throughput guarantee.
The service is a baseline and has not been validated at full-benchmark scale.

This release deliberately excludes experiment outputs, datasets, databases,
credentials and the later enhanced research implementation.

## Attribution and license

Based on the non-graph memory workflow of [mem0](https://github.com/mem0ai/mem0).
See [NOTICE](NOTICE) for changes and upstream attribution, and [LICENSE](LICENSE)
for Apache-2.0 terms. AML contract: https://agentmemoryleaderboard.ai/api-guide.

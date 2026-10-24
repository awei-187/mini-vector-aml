"""Offline performance/equivalence replay; no model client or network calls.

Use an immutable completed edited147 run. Only database copies under a fresh
output directory are opened by writable stores. Public summary contains metrics
and hashes, never questions, memories, vectors, golds or generated answers.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess
import time
import types

from mini_vector import MiniVector
from mini_vector.answer_context import build_context
from mini_vector.memory import SQLiteVectorStore
from mini_vector.source_context import SourceIndex
from mini_vector.vector_scan import np


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "f49af330c65340c7ae76dae9ae3b5e364a042d66"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+"\n").encode("utf-8"))


def stats(values):
    ordered = sorted(values)
    return {"samples": len(values), "median_ms": statistics.median(values)*1000,
            "p95_ms": ordered[min(len(ordered)-1, int(.95*len(ordered)))]*1000}


def copy_database(source, target):
    with closing(sqlite3.connect(source.as_uri()+"?mode=ro", uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst)


def benchmark(run_dir, output, timing_queries=12, repeats=2):
    if not np:
        raise RuntimeError("Install the optional [fast] extra to benchmark both backends")
    if output.exists():
        raise ValueError("Use a new output directory; existing experiments are never overwritten")
    metadata = read(run_dir/"run_metadata.json")
    baseline_source = subprocess.check_output(
        ["git", "show", BASELINE+":mini_vector/memory.py"], cwd=ROOT).decode("utf-8")
    expected = metadata["identity"]["source_sha256"][str(ROOT/"mini_vector"/"memory.py")]
    assert hashlib.sha256(baseline_source.encode()).hexdigest() == expected
    module = types.ModuleType("mini_vector._baseline_scan")
    exec(compile(baseline_source, "<frozen baseline memory.py>", "exec"), module.__dict__)
    output.mkdir(parents=True)
    database = run_dir/"memory.sqlite3"
    before = digest(database)
    source_index = SourceIndex(read(run_dir/"source_snapshot.json"))
    rows = [read(path) for path in sorted((run_dir/"query_details").glob("*.json"))]
    assert len(rows) == 147
    vectors = []
    for index, row in enumerate(rows):
        calls = [read(path) for path in sorted((run_dir/"api_calls"/f"retrieval_q{index:03d}").glob("*_embedding.json"))]
        call = [call for call in calls if call["status"] == "complete"][-1]
        assert call["request"]["input"] == [row["question"].strip()]
        vectors.append(call["response"]["data"][0]["embedding"])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("No model calls are allowed in this benchmark")

    stores, memories, current = {}, {}, [None]
    replay = {}
    try:
        for backend in ("baseline", "python", "numpy"):
            path = output/(backend+".sqlite3")
            copy_database(database, path)
            stores[backend] = (module.SQLiteVectorStore(str(path)) if backend == "baseline"
                               else SQLiteVectorStore(str(path), vector_backend=backend))
            memory_type = module.MiniVector if backend == "baseline" else MiniVector
            memories[backend] = memory_type(store=stores[backend], llm=forbidden,
                embedder=lambda texts: [current[0]] if len(texts) == 1 else forbidden(), rerank_mode="none")

        def run_query(backend, index, *, check_context=True):
            row = rows[index]
            current[0] = vectors[index]
            trace = {}
            started = time.perf_counter()
            found = memories[backend].search(row["question"], filters=metadata["scope"],
                top_k=20, threshold=.1, explain=True, research_trace=trace)["results"]
            elapsed = time.perf_counter()-started
            assert found == row["retrieval"], (backend, index, "final results changed")
            for channel in ("semantic", "memory_query", "bm25"):
                assert trace["channels"][channel] == row["retrieval_trace"]["channels"][channel], (backend, index, channel)
            assert trace == row["retrieval_trace"], (backend, index, "trace changed")
            if check_context:
                windows, source_trace = source_index.select(row["question"], found)
                assert windows == row["source_windows"] and source_trace == row["source_trace"]
                assert build_context(found, windows) == row["answer_context"]
            return elapsed

        for backend in ("python", "numpy"):
            elapsed = []
            for index in range(147):
                elapsed.append(run_query(backend, index))
                if (index+1) % 25 == 0 or index == 146:
                    print(f"REPLAY {backend} {index+1}/147 exact trace + context", flush=True)
            replay[backend] = {"matching_questions": 147, "first_query_ms": elapsed[0]*1000,
                               "subsequent_queries": stats(elapsed[1:]),
                               "numeric_cache_MiB": sum(p.nbytes for p in stores[backend]._vector_cache.values())/2**20}

        indices = sorted(set(round(i*146/(timing_queries-1)) for i in range(timing_queries)))
        timings = {backend: [] for backend in memories}
        for repeat in range(repeats):
            for j, index in enumerate(indices):
                # Alternate order between rounds. All use the same persisted
                # vectors, explain mode and audit trace; remote embedding is excluded.
                order = tuple(memories) if repeat % 2 == 0 else tuple(reversed(memories))
                for backend in order:
                    timings[backend].append(run_query(backend, index, check_context=False))
                if (j+1) % 4 == 0:
                    print(f"TIMING round {repeat+1}/{repeats} query {j+1}/{len(indices)}", flush=True)
        for backend in ("python", "numpy"):
            stores[backend]._vector_cache.clear()
        cold = {backend: run_query(backend, indices[0], check_context=False)*1000 for backend in ("python", "numpy")}
        summary = {"status": "passed", "baseline_commit": BASELINE,
            "tested_source_sha256": {str(path.relative_to(ROOT)): digest(path)
                for path in (ROOT/"mini_vector"/"memory.py", ROOT/"mini_vector"/"vector_scan.py", Path(__file__))},
            "dataset_sha256": metadata["identity"]["dataset_sha256"], "database_sha256": before,
            "model_calls": 0, "replay": replay,
            "timing": {backend: stats(values) for backend, values in timings.items()},
            "cold_decoded_cache_ms": cold,
            "method": {"timing_queries": len(indices), "repeats": repeats, "questions": 147,
                "scope_count": 1, "fact_count": 1742, "question_vector_count": 4813, "dimension": 1536,
                "timing_includes": "real local search, three channels, payload fetch, RRF, explain and trace",
                "timing_excludes": "model API, source-window lookup, answer generation, judges",
                "cold_means": "empty decoded vector cache; OS page cache is not cleared",
                "precision": "float64 shortlist with guarded original-Python rescoring"},
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "sqlite": sqlite3.sqlite_version},
            "limitations": ["single existing development conversation", "not an end-to-end model latency benchmark",
                "no new answers or judge calls; unchanged contexts do not constitute a new LLM-J score",
                "JSON vectors remain on disk; numeric cache uses additional RAM",
                "ANN, float32/BLOB migration and service concurrency changes not included"]}
        assert digest(database) == before, "Original database was changed"
        baseline_ms = summary["timing"]["baseline"]["median_ms"]
        summary["median_speedup"] = {backend: baseline_ms/summary["timing"][backend]["median_ms"] for backend in ("python", "numpy")}
        write(output/"summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        source_index.close()
        for store in stores.values():
            store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-queries", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 2 <= args.timing_queries <= 147 or args.repeats < 1:
        parser.error("timing-queries must be 2..147 and repeats must be positive")
    benchmark(args.run_dir.resolve(), args.output_dir.resolve(), args.timing_queries, args.repeats)

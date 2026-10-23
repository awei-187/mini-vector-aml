"""Evaluate actual MiniVector on original LoCoMo conversation zero.

Reuses the existing Mem0 experiment's input formatting and answer/judge prompts,
without importing the Mem0 SDK. API requests and responses contain research
data; credentials and HTTP headers are never recorded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from mini_vector import MiniVector, SQLiteVectorStore
from mini_vector.config import setting

HELPER = ROOT / "mem0/locomo_eval/run_locomo.py"
spec = importlib.util.spec_from_file_location("locomo_reference", HELPER)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
read_json = reference.read_json


def write_json(path, payload):
    # Windows readers may briefly prevent os.replace from replacing a file.
    # Retrying the same atomic write leaves the evaluation contents unchanged.
    for attempt in range(10):
        try:
            return reference.write_json(path, payload)
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * 2**attempt, 1.0))


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Audit:
    def __init__(self, root, client):
        self.root, self.client = root, client
        self.local = threading.local()
        self.lock = threading.Lock()
        self.sequence = time.time_ns()

    def scope(self, name):
        self.local.scope = name

    def request(self, kind, kwargs):
        def attempt():
            with self.lock:
                self.sequence += 1
                sequence = self.sequence
            scope = getattr(self.local, "scope", "preflight")
            path = self.root / "api_calls" / scope / f"{sequence}_{kind}.json"
            record = {"operation": kind, "scope": scope, "started_at": now(),
                      "request": kwargs, "status": "running"}
            write_json(path, record)
            started = time.perf_counter()
            try:
                endpoint = self.client.embeddings if kind == "embedding" else self.client.chat.completions
                response = endpoint.create(**kwargs)
                record.update(status="complete", response=response.model_dump(mode="json"))
                return response
            except BaseException as exc:
                record.update(status="failed", error_type=type(exc).__name__,
                              http_status=getattr(exc, "status_code", None))
                raise
            finally:
                record["latency_ms"] = (time.perf_counter() - started) * 1000
                write_json(path, record)
        return reference.call_with_retry(kind, attempt)

    def chat(self, **kwargs):
        return self.request("chat_completion", kwargs)

    def extract(self, messages):
        response = self.chat(model="gpt-4o-mini", messages=messages,
                             response_format={"type": "json_object"})
        return response.choices[0].message.content or "{}"

    def embed(self, texts):
        response = self.request("embedding", {"model": "text-embedding-3-small", "input": texts})
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def answer_client(self):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.chat)))


class DeferredCommit:
    """Defer the store's per-record commits to one outer turn transaction."""
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, key):
        return getattr(self.connection, key)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def commit(self):
        pass


def atomic_ingest(memory, turn, scope):
    """Commit memory, FTS, history, recent context and turn marker together."""
    store = memory.store
    connection = store.db
    with store.lock:
        connection.execute("BEGIN IMMEDIATE")
        store.db = DeferredCommit(connection)
        try:
            started = time.perf_counter()
            result = memory.add(
                [{"role": turn["role"], "content": turn["content"]}], **scope,
                metadata={"benchmark": "locomo10", "session_id": turn["session_key"],
                          "session_key": turn["session_key"], "session_date": turn["session_date"],
                          "dialogue_id": turn["dialogue_id"], "source_dialogue_ids": [turn["dialogue_id"]],
                          "speaker": turn["speaker"]},
            )
            record = {"turn": turn, "result": result, "completed_at": now(),
                      "latency_ms": (time.perf_counter() - started) * 1000}
            connection.execute("INSERT INTO benchmark_ingestion(dialogue_id, record) VALUES (?, ?)",
                               (turn["dialogue_id"], json.dumps(record, ensure_ascii=False)))
            connection.commit()
            return record
        except BaseException:
            connection.rollback()
            raise
        finally:
            store.db = connection


def collect_usage(root):
    usage = {}
    for path in (root / "api_calls").rglob("*.json"):
        record = read_json(path, {})
        key = "preflight" if record.get("scope") == "preflight" else "benchmark"
        group = usage.setdefault(key, {"successful_calls": 0, "failed_calls": 0,
                                       "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        if record.get("status") != "complete":
            group["failed_calls"] += 1
            continue
        group["successful_calls"] += 1
        tokens = record.get("response", {}).get("usage") or {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            group[field] += tokens.get(field, 0) or 0
    return usage


@contextmanager
def run_lock(root):
    import msvcrt
    with (root / "run.lock").open("a+b") as handle:
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def evaluate(args):
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "run_metadata.json"
    previous = read_json(metadata_path, {})
    if previous and not args.resume:
        raise ValueError("Run exists; use --resume")
    if not previous and (root / "memory.sqlite3").exists():
        raise ValueError("Refusing an unregistered pre-existing memory database")
    dataset = args.dataset.resolve()
    entry = json.loads(dataset.read_text(encoding="utf-8-sig"))[0]
    turns = reference.get_turns(entry)
    questions = [(i, qa) for i, qa in enumerate(entry["qa"]) if int(qa["category"]) in {1, 2, 3, 4}]
    if entry["sample_id"] != "conv-26" or len(turns) != 419 or len(questions) != 152:
        raise ValueError("Expected conv-26 with 419 utterances and 152 category 1-4 questions")
    turn_ids = {t["dialogue_id"] for t in turns}
    for _, qa in questions:
        gold, unresolved = reference.gold_dialogue_ids(qa)
        if unresolved or not set(gold).issubset(turn_ids):
            raise ValueError("Gold evidence does not match the input conversation")
    sources = [Path(__file__), PROJECT / "mini_vector/memory.py", PROJECT / "mini_vector/config.py",
               PROJECT / "mini_vector/retrieval.py", PROJECT / "mini_vector/prompts.py", HELPER]
    retrieval_config = {"memory_query_enabled": not args.baseline,
                        "rerank_mode": "none" if args.baseline else "llm"}
    identity = {"dataset_sha256": digest(dataset), "source_sha256": {str(p): digest(p) for p in sources},
                "models": {"extraction": "gpt-4o-mini", "answer": "gpt-4o-mini", "judge": "gpt-4o-mini",
                           "embedding": "text-embedding-3-small"}, "top_k": 20, "threshold": 0.1,
                "retrieval_config": retrieval_config,
                "ingestion_turns": len(turns), "questions": len(questions), "categories_included": [1, 2, 3, 4]}
    if previous and previous["identity"] != identity:
        raise ValueError("Run configuration, dataset or source code differs from the saved run")
    metadata = previous or {
        "engine": "mini-vector", "engine_version": "0.1.0", "benchmark": "LoCoMo-10",
        "sample_id": "conv-26", "conversation_index": 0, "run_id": root.name,
        "created_at": now(), "source_dataset": str(dataset), "identity": identity,
        "python_version": sys.version, "evaluation_mode": "answer_and_judge", "answer_workers": args.workers,
        "memory_reused": False, "scope": {"user_id": "conv-26", "run_id": root.name},
        "storage": ("SQLite cosine + FTS5 BM25; original baseline" if args.baseline else
                    "SQLite memory/query cosine + independent FTS5 BM25; per-channel LLM rerank + RRF"),
        "extraction_temperature": "MiniVector default: omitted from API request",
        "answer_and_judge_temperature": 0,
        "input_format_and_prompts": "Identical helpers to Mem0 locomo_eval/run_locomo.py; frozen source copied",
        "context_timestamp_semantics": "Shared legacy prompt labels session_date as event_time; it is actually the conversation date",
        "provenance_semantics": "Source IDs identify the newly ingested utterance, not an exhaustive fact evidence set",
        "resume_semantics": "One SQLite transaction per utterance including benchmark marker; per-query stage checkpoints",
        "api_audit": "SDK requests/responses and usage; credentials/headers/endpoint omitted; SDK max_retries=0",
    }
    write_json(metadata_path, metadata)
    if not previous:
        for path in sources:
            destination = root / "source_code" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    write_json(root / "input_conversation_0.json", entry)
    write_json(root / "source_snapshot.json", turns)
    progress = {"completed_ingestion_turns": 0, "expected_ingestion_turns": len(turns),
                "completed_questions": 0, "expected_questions": len(questions)}

    def status(state="running", phase="preflight", **extra):
        write_json(root / "run_status.json", {"status": state, "phase": phase, "updated_at": now(),
                                             **progress, **extra})

    from openai import OpenAI
    client = OpenAI(api_key=setting("OPENAI_API_KEY"), base_url=setting("OPENAI_BASE_URL"),
                    timeout=90.0, max_retries=0)
    audit = Audit(root, client)
    memory = None
    try:
        status()
        if not (root / "preflight.json").exists():
            audit.scope("preflight")
            probe = MiniVector(db_path=":memory:", llm=audit.extract, embedder=audit.embed, **retrieval_config)
            try:
                added = probe.add("Alice lives in Berlin.", user_id="probe")
                trace = {}
                found = probe.search("Alice Berlin", filters={"user_id": "probe"}, research_trace=trace)
                if not added["results"] or not found["results"] or not trace["channels"]["bm25"]["candidates"]:
                    raise RuntimeError("Live extraction, embedding or BM25 preflight failed")
                write_json(root / "preflight.json", {"status": "complete", "live_models": True,
                           "sqlite_fts5": True, "embedding_dimension": len(audit.embed(["dimension check"])[0]),
                           "excluded_from_benchmark": True, "completed_at": now()})
            finally:
                probe.close()
        memory = MiniVector(db_path=str(root / "memory.sqlite3"), llm=audit.extract, embedder=audit.embed,
                             **retrieval_config)
        memory.store.db.execute("CREATE TABLE IF NOT EXISTS benchmark_ingestion (dialogue_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        memory.store.db.commit()
        committed = {row[0]: json.loads(row[1]) for row in memory.store.db.execute("SELECT dialogue_id, record FROM benchmark_ingestion")}
        progress["completed_ingestion_turns"] = len(committed)
        for index, turn in enumerate(turns):
            turn_id = turn["dialogue_id"]
            if turn_id not in committed:
                audit.scope(f"ingest_{index:04d}_{turn_id.replace(':', '_')}")
                record = atomic_ingest(memory, turn, metadata["scope"])
                committed[turn_id] = record
                progress["completed_ingestion_turns"] = len(committed)
                print(f"INGEST {len(committed)}/{len(turns)} | {turn_id} | new={len(record['result']['results'])}", flush=True)
            write_json(root / "ingestion" / f"{index:04d}.json", committed[turn_id])
            status(phase="ingestion")
        memories = memory.get_all(filters=metadata["scope"])["results"]
        write_json(root / "memory_snapshot.json", {"count": len(memories), "memories": memories})
        vectors = [dict(row) for row in memory.store.db.execute("SELECT id,data,embedding,metadata FROM memories")]
        for row in vectors:
            row["embedding"] = json.loads(row["embedding"])
            row["metadata"] = json.loads(row["metadata"])
        write_json(root / "memories_embeddings.json", {"memories": vectors})
        query_vectors = [dict(row) for row in memory.store.db.execute(
            "SELECT memory_id,position,query,embedding FROM memory_queries ORDER BY memory_id,position")]
        for row in query_vectors:
            row["embedding"] = json.loads(row["embedding"])
        write_json(root / "memory_queries_embeddings.json", {"queries": query_vectors})
        print(f"INGESTION COMPLETE | memories={len(memories)}", flush=True)

        for processed, (index, qa) in enumerate(questions, 1):
            path = root / "query_details" / f"{index:03d}.json"
            if path.exists():
                continue
            audit.scope(f"retrieval_q{index:03d}")
            trace = {}
            started = time.perf_counter()
            result = memory.search(qa["question"], top_k=20, threshold=0.1, filters=metadata["scope"],
                                   explain=True, research_trace=trace)
            search_ms = (time.perf_counter() - started) * 1000
            if [x["id"] for x in result["results"]] != [x["memory_id"] for x in trace["stages"]["final"]]:
                raise ValueError("Trace final IDs do not match actual MiniVector.search results")
            if trace["channels"]["bm25"]["status"] == "error":
                raise RuntimeError("SQLite BM25 query failed")
            gold, _ = reference.gold_dialogue_ids(qa)
            origins = {source for hit in result["results"] for source in hit.get("metadata", {}).get("source_dialogue_ids", [])}
            record = {"question_id": f"conv0_q{index}", "qa_index": index, "category": int(qa["category"]),
                      "question": qa["question"], "gold_answer": str(qa.get("answer", "")), "evidence": qa.get("evidence", []),
                      "gold_evidence_dialogue_ids": gold,
                      "gold_origin_hit": bool(origins.intersection(gold)) if gold else None,
                      "retrieval": {"query": qa["question"], "top_k": 20, "count": len(result["results"]),
                                    "latency_ms": search_ms, "results": result["results"]},
                      "retrieval_trace": trace, "answer_context": reference.build_context(result["results"])}
            write_json(path, record)
            status(phase="retrieval", completed_retrieval_queries=processed)
            print(f"RETRIEVAL {processed}/{len(questions)} | q{index} | hits={len(result['results'])}", flush=True)
        memory.close()
        memory = None

        def answer_and_judge(index):
            path = root / "query_details" / f"{index:03d}.json"
            record = read_json(path, {})
            answer_client = audit.answer_client()
            if "generation" not in record:
                audit.scope(f"answer_q{index:03d}")
                answer, usage = reference.generate_answer(answer_client, record["question"], record["answer_context"])
                if not answer:
                    raise ValueError("Empty generated answer")
                record["generation"] = {"model": "gpt-4o-mini", "answer": answer, "usage": usage}
                write_json(path, record)
            if "judgment" not in record:
                audit.scope(f"judge_q{index:03d}")
                judgment = reference.judge_answer(answer_client, record["question"], record["gold_answer"], record["generation"]["answer"])
                record["judgment"] = {"model": "gpt-4o-mini", **judgment}
                write_json(path, record)
            return record

        saved = {i: read_json(root / "query_details" / f"{i:03d}.json", {}) for i, _ in questions}
        progress["completed_questions"] = sum("judgment" in r for r in saved.values())
        status(phase="answer_and_judge")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(answer_and_judge, i): i for i, _ in questions if "judgment" not in saved[i]}
            for future in as_completed(futures):
                record = future.result()
                saved[record["qa_index"]] = record
                progress["completed_questions"] += 1
                status(phase="answer_and_judge")
                print(f"EVAL {progress['completed_questions']}/{len(questions)} | {record['question_id']} | cat={record['category']} | {record['judgment']['label']}", flush=True)
        results = [saved[i] for i, _ in questions]
        summary = reference.summarize(results)
        summary.update(memory_count=len(memories), ingestion_turns=len(turns),
                       mean_retrieval_latency_ms=sum(r["retrieval"]["latency_ms"] for r in results) / len(results))
        write_json(root / "summary.json", summary)
        write_json(root / "final_results.json", {"metadata": metadata, "summary": summary, "results": results})
        write_json(root / "api_usage_summary.json", collect_usage(root))
        status("complete", "complete", memory_count=len(memories))
        print(f"COMPLETE | {summary['correct']}/{summary['total']} | accuracy={summary['accuracy']:.2%}", flush=True)
    except BaseException as exc:
        status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "exception",
               error_type=type(exc).__name__)
        print(f"STOPPED | error_type={type(exc).__name__}; inspect run_status.json and API audit status", flush=True)
        raise
    finally:
        if memory:
            memory.close()
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "graphiti/locomo_eval/data/locomo.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline", action="store_true",
                        help="Disable memory questions and LLM reranking; use original retrieval")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(args.run_dir):
        evaluate(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        # Avoid printing provider exception messages, URLs, headers or credentials.
        print(f"FAILED: {type(exc).__name__}", flush=True)
        sys.exit(1)

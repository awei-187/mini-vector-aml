"""Two optional paid answer-policy experiments on already frozen retrieval.

Default mode prepares contexts offline. --run-models explicitly starts answer
and judge calls, and must be used only within the user's current authorization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys

import run_locomo as runner
from mini_vector.answer_context import answer_messages, build_context
from mini_vector.config import setting
from mini_vector.source_context import SourceIndex

ARMS = ("temporal", "temporal_source")


def identity(baseline):
    sources = [Path(__file__), Path(__file__).with_name("verify_temporal_source.py"),
               Path(__file__).with_name("run_locomo.py"), runner.HELPER,
               runner.PROJECT / "mini_vector/answer_context.py", runner.PROJECT / "mini_vector/source_context.py",
               runner.PROJECT / "mini_vector/config.py"]
    return {"baseline": str(baseline), "baseline_sha256": runner.digest(baseline / "final_results.json"),
            "turns_sha256": runner.digest(baseline / "source_snapshot.json"),
            "source_sha256": {str(p): runner.digest(p) for p in sources}, "arms": list(ARMS),
            "model": "gpt-4o-mini", "temperature": 0, "memory_rebuilt": False,
            "new_embeddings": False, "retrieval": "cached actual RRF top-20; no ranking changes",
            "source_windows": {"max_windows": 4, "radius": 1, "max_chars": 12000}}


def prepare(root, baseline):
    old = runner.read_json(baseline / "final_results.json", {})
    verified = runner.read_json(baseline / "verification.json", {})
    if verified.get("status") != "passed" or verified.get("questions") != 152:
        raise ValueError("Expected independently verified full RRF baseline")
    if old["metadata"]["identity"]["retrieval_config"]["rerank_mode"] != "none":
        raise ValueError("Expected raw RRF retrieval")
    expected = identity(baseline)
    manifest = root / "experiment_plan.json"
    if manifest.exists() and runner.read_json(manifest, {}) != expected:
        raise ValueError("Saved source, baseline or experiment configuration changed")
    runner.write_json(manifest, expected)
    for original in expected["source_sha256"]:
        destination = root / "source_code" / Path(original).relative_to(runner.ROOT)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
    turns = runner.read_json(baseline / "source_snapshot.json", [])
    if len(turns) != 419 or len(old["results"]) != 152:
        raise ValueError("Unexpected benchmark scope")
    index = SourceIndex(turns)
    try:
        for row in old["results"]:
            facts = row["retrieval"]["results"]
            windows, trace = index.select(row["question"], facts)
            for arm in ARMS:
                context = build_context(facts, windows if arm == "temporal_source" else ())
                path = root / arm / "query_details" / f"{row['qa_index']:03d}.json"
                item = {key: row[key] for key in ("question_id", "qa_index", "category", "question", "gold_answer")}
                item.update(arm=arm, retrieval=facts, answer_context=context,
                            source_windows=windows if arm == "temporal_source" else [],
                            source_trace=trace if arm == "temporal_source" else None)
                saved = runner.read_json(path, {})
                if saved:
                    if any(saved.get(key) != value for key, value in item.items()):
                        raise ValueError("Saved question input changed")
                else:
                    runner.write_json(path, item)
    finally:
        index.close()
    return old


def run_arm(root, arm, old, workers):
    from openai import OpenAI
    child = root / arm
    client = OpenAI(api_key=setting("OPENAI_API_KEY"), base_url=setting("OPENAI_BASE_URL"),
                    timeout=90, max_retries=0)
    audit = runner.Audit(child, client)
    indices = [row["qa_index"] for row in old["results"]]
    def work(index):
        path = child / "query_details" / f"{index:03d}.json"
        row = runner.read_json(path, {})
        if "generation" not in row:
            audit.scope(f"answer_q{index:03d}")
            response = audit.chat(model="gpt-4o-mini", temperature=0,
                                  messages=answer_messages(row["question"], row["answer_context"]))
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise ValueError("Empty model answer")
            row["generation"] = {"answer": answer, "usage": response.usage.model_dump() if response.usage else {}}
            runner.write_json(path, row)
        if "judgment" not in row:
            audit.scope(f"judge_q{index:03d}")
            row["judgment"] = runner.reference.judge_answer(audit.answer_client(), row["question"],
                                                            row["gold_answer"], row["generation"]["answer"])
            runner.write_json(path, row)
        return row
    completed = sum("judgment" in runner.read_json(child / "query_details" / f"{i:03d}.json", {}) for i in indices)
    try:
        runner.write_json(child / "run_status.json", {"status": "running", "completed_questions": completed})
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i) for i in indices
                       if "judgment" not in runner.read_json(child / "query_details" / f"{i:03d}.json", {})]
            for future in as_completed(futures):
                row = future.result()
                completed += 1
                runner.write_json(child / "run_status.json", {"status": "running", "completed_questions": completed})
                print(f"{arm} {completed}/152 | q{row['qa_index']} | {row['judgment']['label']}", flush=True)
        rows = [runner.read_json(child / "query_details" / f"{i:03d}.json", {}) for i in indices]
        summary = runner.reference.summarize(rows)
        runner.write_json(child / "summary.json", summary)
        runner.write_json(child / "final_results.json", {"summary": summary, "results": rows})
        runner.write_json(child / "api_usage_summary.json", runner.collect_usage(child))
        runner.write_json(child / "run_status.json", {"status": "complete", "completed_questions": 152})
    except BaseException as exc:
        runner.write_json(child / "run_status.json", {"status": "failed", "completed_questions": completed,
                                                     "error_type": type(exc).__name__})
        raise
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-models", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with runner.run_lock(root):
        old = prepare(root, args.baseline.resolve())
        if not args.run_models:
            print("PREPARED: two arms x 152 questions; no model or embedding calls", flush=True)
            return
        from verify_temporal_source import verify
        for arm in ARMS:
            run_arm(root, arm, old, args.workers)
            verify(root, arm)
        runner.write_json(root / "comparison.json", {"status": "complete", "completed_at": runner.now(),
            "baseline": old["summary"], "arms": {arm: runner.read_json(root / arm / "summary.json", {}) for arm in ARMS}})
        print("COMPLETE: "+json.dumps(runner.read_json(root / "comparison.json", {})), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}; inspect local audit and status", flush=True)
        sys.exit(1)

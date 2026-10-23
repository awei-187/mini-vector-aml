"""One fresh LoCoMo corpus and three fixed-corpus retrieval experiments.

Run in the agent_aml environment. Resumption never repeats a completed stage.
This runner makes paid calls only via run_locomo.py and its audited clients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import run_locomo as runner

ARMS = (("rrf", "none"), ("global", "global"), ("evidence", "evidence"))


def verify_corpus_exports(corpus, child):
    """Check values rather than insignificant object-key order in fact snapshots."""
    for filename in ("memory_snapshot.json", "memories_embeddings.json", "memory_queries_embeddings.json"):
        assert (corpus / filename).is_file() and (child / filename).is_file(), filename
    assert runner.read_json(child / "memory_snapshot.json", {}) == runner.read_json(corpus / "memory_snapshot.json", {})
    for filename in ("memories_embeddings.json", "memory_queries_embeddings.json"):
        assert runner.digest(child / filename) == runner.digest(corpus / filename), filename


def stage(root, extra):
    root.mkdir(parents=True, exist_ok=True)
    status = runner.read_json(root / "run_status.json", {})
    exit_path = root / "process_exit_code.txt"
    if status.get("status") != "complete" or not exit_path.exists() or exit_path.read_text().strip() != "0":
        command = [sys.executable, "-u", str(Path(__file__).with_name("run_locomo.py")),
                   "--run-dir", str(root), "--rerank-limit", "60", *extra]
        if (root / "run_metadata.json").exists():
            command.append("--resume")
        result = subprocess.run(command, check=False)
        exit_path.write_text(str(result.returncode), encoding="ascii")
        if result.returncode:
            raise RuntimeError(f"Stage failed: {root.name}; inspect local audit and status")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name("run_locomo.py"),
               Path(__file__).with_name("verify_results.py"),
               *sorted((runner.PROJECT / "mini_vector").glob("*.py")), runner.HELPER]
    plan = {"dataset": str(runner.ROOT / "graphiti/locomo_eval/data/locomo.json"),
            "source_sha256": {str(p): runner.digest(p) for p in sources},
            "ingestion": "one fresh corpus, full extraction prompt, three memory-side questions",
            "arms": [{"name": name, "rerank_mode": mode, "rerank_limit": 60, "top_k": 20}
                     for name, mode in ARMS],
            "controls": "same corpus, models, answer/judge prompts, 152 category 1-4 questions",
            "exclusions": "47 category-5 adversarial questions follow the historical protocol"}
    previous = runner.read_json(root / "experiment_plan.json", {})
    if previous and previous != plan:
        raise ValueError("Experiment source or configuration changed; start a new registered run")
    runner.write_json(root / "experiment_plan.json", plan)
    corpus = root / "corpus"
    stage(corpus, ["--ingest-only", "--rerank-mode", "none"])
    receipt = runner.read_json(corpus / "corpus_receipt.json", {})
    assert receipt.get("status") == "complete" and receipt["ingestion_turns"] == 419
    groups = {}
    for name, mode in ARMS:
        child = root / name
        stage(child, ["--reuse-corpus", str(corpus), "--rerank-mode", mode])
        if not (child / "verification.json").exists():
            subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("verify_results.py")),
                            str(child)], check=True)
        verification = runner.read_json(child / "verification.json", {})
        assert verification["status"] == "passed" and verification["questions"] == 152
        verify_corpus_exports(corpus, child)
        groups[name] = {"summary": runner.read_json(child / "summary.json", {}),
                        "usage": runner.read_json(child / "api_usage_summary.json", {}),
                        "verification": verification}
    output = {"status": "complete", "completed_at": runner.now(), "corpus": receipt,
              "ingestion_usage": runner.read_json(corpus / "api_usage_summary.json", {}), "arms": groups}
    runner.write_json(root / "comparison.json", output)
    lines = ["# LoCoMo first-conversation optimization experiment", "",
             "One newly ingested corpus; all three arms use identical stored facts and vectors.", "",
             "| Arm | Correct / 152 | Accuracy | Mean retrieval ms | Global selection success |",
             "| --- | --- | --- | --- | --- |"]
    for name, group in groups.items():
        summary = group["summary"]
        states = group["verification"].get("rerank_counts", {}).get("global", {})
        lines.append(f"| {name} | {summary['correct']}/152 | {summary['accuracy']:.2%} | "
                     f"{summary['mean_retrieval_latency_ms']:.1f} | {states.get('succeeded', 0)} |")
    lines += ["", "Scores use the existing gpt-4o-mini answer judge and are from one conversation. "
              "They are not independently adjudicated or evidence of dataset-wide generalization.", "",
              "The historical 135/152 run used a different generated corpus and is a reference only. "
              "The RRF arm is the causal comparison within this experiment.", "",
              "Category 5 remains excluded under the historical protocol. API usage and source hashes "
              "are retained locally; private settings and raw benchmark traces are not published."]
    (root / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({name: row["summary"] for name, row in groups.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"OPTIMIZATION FAILED: {type(exc).__name__}; inspect local stage status", flush=True)
        sys.exit(1)

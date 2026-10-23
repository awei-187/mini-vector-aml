"""Verify cached facts, verbatim source windows, API boundaries and scoring."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import run_locomo as runner
from mini_vector.answer_context import answer_messages, build_context


def verify(root, arm):
    plan = runner.read_json(root / "experiment_plan.json", {})
    for filename, expected in plan["source_sha256"].items():
        frozen = root / "source_code" / Path(filename).relative_to(runner.ROOT)
        assert runner.digest(frozen) == expected
    baseline = Path(plan["baseline"])
    assert runner.digest(baseline / "final_results.json") == plan["baseline_sha256"]
    assert runner.digest(baseline / "source_snapshot.json") == plan["turns_sha256"]
    old = runner.read_json(baseline / "final_results.json", {})
    turns = runner.read_json(baseline / "source_snapshot.json", [])
    by_id = {turn["dialogue_id"]: turn for turn in turns}
    positions = {turn["dialogue_id"]: i for i, turn in enumerate(turns)}
    child = root / arm
    final = runner.read_json(child / "final_results.json", {})
    assert runner.read_json(child / "run_status.json", {}) == {"status": "complete", "completed_questions": 152}
    assert len(final["results"]) == len(old["results"]) == 152
    answer_calls = judge_calls = 0
    for row, original in zip(final["results"], old["results"]):
        index = row["qa_index"]
        assert row == runner.read_json(child / "query_details" / f"{index:03d}.json", {})
        assert all(row[key] == original[key] for key in ("question_id", "qa_index", "category", "question", "gold_answer"))
        assert row["retrieval"] == original["retrieval"]["results"]
        windows = row["source_windows"]
        assert len(windows) <= plan["source_windows"]["max_windows"]
        used, chars = set(), 0
        for window in windows:
            anchor = window["anchor"]
            assert anchor in by_id
            assert any(message["source_id"] == anchor for message in window["messages"])
            for message in window["messages"]:
                sid = message["source_id"]
                assert sid in by_id and sid not in used
                used.add(sid)
                turn = by_id[sid]
                assert turn["session_key"] == by_id[anchor]["session_key"]
                assert abs(positions[sid]-positions[anchor]) <= plan["source_windows"]["radius"]
                assert message == {"source_id": sid, "speaker": turn["speaker"],
                                   "observation_time": turn["session_date"], "text": turn["content"]}
                chars += len(message["text"])
        assert chars <= plan["source_windows"]["max_chars"]
        if arm == "temporal":
            assert not windows and row["source_trace"] is None
        else:
            assert row["source_trace"]["source_text_chars"] == chars
            assert set(row["source_trace"]["selected_source_ids"]) == used
        assert row["answer_context"] == build_context(row["retrieval"], windows)
        calls = {}
        for phase in ("answer", "judge"):
            records = [runner.read_json(p, {}) for p in sorted(
                (child / "api_calls" / f"{phase}_q{index:03d}").glob("*_chat_completion.json"))]
            successful = [record for record in records if record["status"] == "complete"]
            assert successful
            calls[phase] = successful[-1]
            assert calls[phase]["request"]["model"] == "gpt-4o-mini"
            assert calls[phase]["request"]["temperature"] == 0
        assert calls["answer"]["request"]["messages"] == answer_messages(row["question"], row["answer_context"])
        assert row["generation"]["answer"] == calls["answer"]["response"]["choices"][0]["message"]["content"].strip()
        judge_text = calls["judge"]["request"]["messages"][-1]["content"]
        assert f"Gold answer: {row['gold_answer']}" in judge_text
        assert f"Generated answer: {row['generation']['answer']}" in judge_text
        judgment = row["judgment"]
        assert judgment["label"] in {"CORRECT", "WRONG"}
        assert judgment["correct"] == (judgment["label"] == "CORRECT")
        response = json.loads(calls["judge"]["response"]["choices"][0]["message"]["content"])
        observed = str(response["is_correct"]).strip().upper()
        if observed not in {"CORRECT", "WRONG"}:
            labels = set(re.findall(r"\b(CORRECT|WRONG)\b", observed))
            assert len(labels) == 1
            observed = labels.pop()
        assert judgment["label"] == observed
        answer_calls += 1
        judge_calls += 1
    summary = runner.reference.summarize(final["results"])
    assert final["summary"] == runner.read_json(child / "summary.json", {}) == summary
    assert runner.read_json(child / "api_usage_summary.json", {}) == runner.collect_usage(child)
    assert not list((child / "api_calls").rglob("*_embedding.json"))
    report = {"status": "passed", "verified_at": runner.now(), "questions": 152, "arm": arm,
              "correct": summary["correct"], "accuracy": summary["accuracy"],
              "answer_api_records_verified": answer_calls, "judge_api_records_verified": judge_calls,
              "checks": ["frozen source and baseline identities", "exact cached top-20 facts",
                         "verbatim source text and per-message observation dates", "bounded same-session neighboring turns",
                         "no additional embedding calls", "exact answer request messages exclude gold fields",
                         "saved answers match actual API responses", "judge inputs, labels, summaries and usage"]}
    runner.write_json(child / "verification.json", report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("arm", choices=["temporal", "temporal_source"])
    args = parser.parse_args()
    verify(args.run_dir.resolve(), args.arm)

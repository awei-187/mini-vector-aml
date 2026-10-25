"""Independent audit of frozen pools, score sorting, context and both judges."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import run_locomo as runner
import run_qwen_ablation as experiment
import dual_judge_contract as judges
from mini_vector.answer_context import answer_messages, build_context
from mini_vector.source_context import SourceIndex


def successful(root,folder):
    rows = [runner.read_json(p,{}) for p in sorted((root/folder).glob("*.json"))]
    complete = [r for r in rows if r.get("status") == "complete"]
    assert complete, folder
    return complete[-1]


def verify(root):
    read = lambda name:runner.read_json(root/name,{})
    manifest = read("manifest.json");identity = manifest["identity"]
    assert read("run_status.json")["status"] == "complete"
    assert read("run_status.json")["completed"] == 441
    for name, expected in identity["input_sha256"].items():
        assert runner.digest(Path(name)) == expected
    for name, expected in manifest["frozen_sha256"].items():
        assert runner.digest(root/name) == expected
    for name, expected in identity["source_sha256"].items():
        assert runner.digest(root/"source_code"/name) == expected
    baseline = Path(identity["baseline"])
    facts = {r["id"]:r for r in runner.read_json(baseline/"memory_snapshot.json",{})["memories"]}
    frozen = read("frozen_inputs.json");assert len(frozen) == 147
    turns = read("source_snapshot.json");assert len(turns) == 419
    contract = read("judge_contract.json");assert contract == judges.load_contract()
    index = SourceIndex(turns)
    arms = {arm:[] for arm in experiment.ARMS}
    try:
        for i,item in enumerate(frozen):
            old = runner.read_json(baseline/"query_details"/f"{i:03d}.json",{})
            channel_ids = [c["eligible_ids"] for c in old["retrieval_trace"]["channels"].values()]
            scores = {}
            for ids in channel_ids:
                for rank,mid in enumerate(ids,1):scores[mid] = scores.get(mid,0.) + 1/(60+rank)
            full = sorted(scores,key=lambda mid:(-scores[mid],mid))
            assert full == old["retrieval_trace"]["stages"]["combined_order"]
            candidates = item["candidates"];ids = [c["id"] for c in candidates]
            assert ids == full[:60] and len(ids) == len(set(ids))
            assert item["pool_sha256"] == experiment.fingerprint(candidates)
            assert item["question"] == old["question"] and item["gold_answer"] == old["gold_answer"]
            for c in candidates:
                assert {k:c[k] for k in facts[c["id"]]} == facts[c["id"]]
            for arm in experiment.ARMS:
                row = read(f"{arm}/query_details/{i:03d}.json")
                arms[arm].append(row)
                assert all(row[k] == item[k] for k in ("qa_index","question","gold_answer","category","pool_sha256"))
                expected_ids = ids
                if arm != "none":
                    api = successful(root,f"qwen_api/{arm}/q{i:03d}")
                    expected_body = {"model":"qwen3-rerank","input":{
                        "query":item["question"],"documents":[c["memory"] for c in candidates]},
                        "parameters":{"top_n":len(ids),"return_documents":False,
                                      "instruct":identity["instructions"][arm]}}
                    assert api["request"] == expected_body
                    raw = api["response"];assert not raw.get("code") and not raw.get("error")
                    results = raw["output"]["results"]
                    assert len(results) == len(ids)
                    values = {r["index"]:r["relevance_score"] for r in results}
                    assert set(values) == set(range(len(ids)))
                    assert all(type(r["index"]) is int for r in results)
                    assert all(type(s) in (int,float) and 0 <= s <= 1 for s in values.values())
                    expected_ids = [ids[j] for j in sorted(range(len(ids)),key=lambda j:-values[j])]
                    trace = row["rerank_trace"]
                    assert trace["status"] == "succeeded" and trace["input_ids"] == ids
                    assert trace["output_ids"] == expected_ids
                    assert trace["scores"] == [{"memory_id":mid,"score":values[j]} for j,mid in enumerate(ids)]
                else:
                    assert row["rerank_trace"] == {"status":"disabled","api_calls":0}
                assert row["retrieval"] == [candidates[ids.index(mid)] for mid in expected_ids[:20]]
                windows,trace = index.select(row["question"],row["retrieval"],**identity["source_windows"])
                assert windows == row["source_windows"] and trace == row["source_trace"]
                assert row["answer_context"] == build_context(row["retrieval"],windows)
                if arm == "none":assert row["answer_context"] == old["answer_context"]
                body = {"model":"gpt-4o-mini","temperature":0,"messages":answer_messages(row["question"],row["answer_context"])}
                key = experiment.fingerprint(body);assert row["answer_request_sha256"] == key
                cache = read(f"request_cache/answer/{key}.json")
                assert cache == {"request":body,"result":row["generation"]}
                api = successful(root,"api_calls/answer_"+key)
                assert api["request"] == body
                assert api["response"]["choices"][0]["message"]["content"].strip() == row["generation"]["answer"]
                state = judges.judge_state(row["question"],row["gold_answer"],row["generation"]["answer"])
                gpt = judges.gpt_messages(state,contract)
                key = experiment.fingerprint(gpt);assert row["gpt_request_sha256"] == key
                assert read(f"request_cache/gpt/{key}.json") == {"request":gpt,"result":row["gpt_judgment"]}
                api = successful(root,"api_calls/gpt_judge_q"+key)
                assert api["request"] == {"model":"gpt-4o-mini","temperature":0,"response_format":{"type":"json_object"},"messages":gpt}
                parsed = judges.parse_gpt(api["response"]["choices"][0]["message"]["content"])
                assert all(row["gpt_judgment"][k] == v for k,v in parsed.items())
                body = judges.jev_payload(state,contract)
                key = experiment.fingerprint(body);assert row["jev_request_sha256"] == key
                assert read(f"request_cache/jev/{key}.json") == {"request":body,"result":row["jev_judgment"]}
                api = successful(root,"jev_api/"+key)
                assert api["request"] == body and judges.validate_jev(api["response"]) == row["jev_judgment"]
    finally:index.close()
    comparison = read("comparison.json")
    for arm,rows in arms.items():
        for judge in ("gpt","jev"):
            counts = Counter(r[judge+"_judgment"]["label"] for r in rows)
            saved = comparison["arms"][arm]["judges"][judge]
            assert saved["counts"] == {label:counts[label] for label in judges.LABELS}
            assert saved["llm_j"] == counts["correct"]/147
    assert comparison == experiment.compare(arms)
    assert read("api_usage_summary.json") == experiment.collect_all_usage(root)
    # No historical answer cache and no embedding calls are allowed.
    assert not list((root/"api_calls").rglob("*_embedding.json"))
    report = {"status":"passed","verified_at":runner.now(),"questions":147,"arms":3,
              "same_pool_verified":441,"fresh_context_and_api_verified":441,
              "qwen_score_order_verified":294,"dual_judge_payloads_verified":882,
              "frozen_sources_verified":len(identity["source_sha256"]),"no_embedding_or_ingestion":True}
    runner.write_json(root/"verification.json",report)
    print(json.dumps(report),flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path)
    verify(p.parse_args().root.resolve())

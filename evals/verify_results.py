"""Verify finished run identities, transactional ingestion, rankings and scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_locomo as runner


def normalized_bm25(candidates):
    scores = {}
    if candidates:
        best = max(hit["raw_score"] for hit in candidates)
        assert best > 0
        for hit in candidates:
            value = 1 / (1 + math.exp(-8 * (hit["raw_score"] / best - 0.5)))
            assert abs(value - hit["normalized_score"]) < 1e-10
            scores[hit["memory_id"]] = value
    return scores


def verify_enhanced_ranking(trace, retrieved, query_vector, query_rows, exported, question_norms=None):
    """Recompute schema-2 question pools, channel permutations and RRF."""
    channels = trace["channels"]
    assert trace["schema_version"] in {2, 3}
    assert set(channels) == {"semantic", "bm25", "memory_query"}
    lexical = channels["bm25"]
    assert lexical["status"] == "recorded"
    bm25_scores = normalized_bm25(lexical["lexical_details"]["candidates"])
    assert [hit["memory_id"] for hit in lexical["candidates"]] == list(bm25_scores)
    for hit in lexical["candidates"]:
        assert abs(hit["score"] - bm25_scores[hit["memory_id"]]) < 1e-10

    query_norm = sum(x*x for x in query_vector)
    grouped = {}
    for row in query_rows:
        assert row["memory_id"] in exported
        grouped.setdefault(row["memory_id"], []).append(row)
    expected = []
    if trace["memory_query_enabled"]:
        # Preserve memories' insertion order for stable equal-score ties.
        for memory_id in exported:
            scores = []
            for row in sorted(grouped.get(memory_id, []), key=lambda item: item["position"]):
                vector = row["embedding"]
                assert len(vector) == len(query_vector)
                squared_norm = (question_norms[(memory_id, row["position"])] if question_norms is not None
                                else sum(x*x for x in vector))
                denominator = math.sqrt(query_norm * squared_norm)
                value = max(0., min(1., sum(x*y for x, y in zip(query_vector, vector)) / denominator)) if denominator else 0.
                scores.append(value)
            if scores:
                expected.append((memory_id, max(scores)))
        expected.sort(key=lambda pair: pair[1], reverse=True)
    observed = channels["memory_query"]["candidates"]
    expected = expected[:trace["candidate_limit"]]
    assert [mid for mid, _ in expected] == [hit["memory_id"] for hit in observed]
    assert all(abs(score - hit["score"]) < 1e-10 for (_, score), hit in zip(expected, observed))

    merged, ranks, active = {}, {}, 0
    for name, channel in channels.items():
        candidates = channel["candidates"]
        assert len(candidates) <= trace["candidate_limit"]
        assert len({hit["memory_id"] for hit in candidates}) == len(candidates)
        assert all(hit["memory_id"] in exported and hit["text"] == exported[hit["memory_id"]]["data"]
                   for hit in candidates)
        eligible = [hit["memory_id"] for hit in candidates if name == "bm25" or hit["score"] >= trace["threshold"]]
        assert channel["eligible_ids"] == eligible
        info = channel["rerank"]
        head = eligible[:trace["rerank_limit"]]
        assert info["input_ids"] == head
        output = info["output_ids"]
        assert len(output) == len(eligible) and set(output) == set(eligible)
        assert output[len(head):] == eligible[len(head):]
        if trace["rerank_mode"] != "llm":
            assert info["status"] == "disabled" and info["api_calls"] == 0
        elif len(head) < 2:
            assert info["status"] == "skipped" and info["api_calls"] == 0
        else:
            assert info["status"] in {"succeeded", "fallback"} and info["api_calls"] == 1
        if info["status"] != "succeeded":
            assert output == eligible
        active += bool(output)
        for rank, memory_id in enumerate(output, 1):
            merged[memory_id] = merged.get(memory_id, 0.) + 1 / (60 + rank)
            ranks.setdefault(memory_id, {})[name] = rank
    ordered = sorted(merged, key=lambda mid: (-merged[mid], mid))
    assert trace["stages"]["combined_order"] == ordered
    states = {name: channel["rerank"]["status"] for name, channel in channels.items()}
    if trace["schema_version"] == 3:
        assert trace["rerank_mode"] in {"global", "evidence"}
        info = trace["global_rerank"]
        head = ordered[:trace["rerank_limit"]]
        assert info["input_ids"] == head
        assert info["max_select"] == min(trace["top_k"], len(head))
        assert info["diverse"] == (trace["rerank_mode"] == "evidence")
        selected = info["selected_ids"]
        if len(head) < 2:
            assert info["status"] == "skipped" and info["api_calls"] == 0 and not selected
        elif info["status"] == "succeeded":
            assert info["api_calls"] == 1
            assert 1 <= len(selected) <= info["max_select"]
            assert len(set(selected)) == len(selected) and set(selected) <= set(head)
        else:
            assert info["status"] == "fallback" and info["api_calls"] == 1 and not selected
        ordered = selected + [mid for mid in ordered if mid not in set(selected)]
        assert info["output_ids"] == trace["stages"]["selected_order"] == ordered
        states["global"] = info["status"]
    assert [hit["id"] for hit in retrieved] == ordered[:trace["top_k"]]
    assert [hit["memory_id"] for hit in trace["stages"]["final"]] == ordered[:trace["top_k"]]
    for item, final_item in zip(retrieved, trace["stages"]["final"]):
        mid = item["id"]
        score = min(1., merged[mid] / (active / 61))
        assert abs(score - item["score"]) < 1e-10
        assert abs(score - final_item["score"]) < 1e-10
        details = item["score_details"]
        assert details == final_item["score_details"]
        assert details["channel_ranks"] == ranks[mid]
        assert abs(details["rrf_score"] - merged[mid]) < 1e-10
    return states


def verify(root):
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8-sig"))
    status = read("run_status.json")
    assert status["status"] == "complete", status
    assert status["completed_ingestion_turns"] == status["expected_ingestion_turns"] == 419
    assert status["completed_questions"] == status["expected_questions"] == 152
    assert (root / "process_exit_code.txt").read_text().strip() == "0"
    metadata = read("run_metadata.json")
    entry = read("input_conversation_0.json")
    assert entry["sample_id"] == "conv-26"
    source = Path(metadata["source_dataset"])
    assert runner.digest(source) == metadata["identity"]["dataset_sha256"]
    assert json.loads(source.read_text(encoding="utf-8-sig"))[0] == entry
    for original, expected in metadata["identity"]["source_sha256"].items():
        frozen = root / "source_code" / Path(original).relative_to(runner.ROOT)
        assert runner.digest(frozen) == expected
    actual_turns = runner.reference.get_turns(entry)
    assert read("source_snapshot.json") == actual_turns
    connection = sqlite3.connect(f"file:{(root / 'memory.sqlite3').as_posix()}?mode=ro", uri=True)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    committed = {row[0]: json.loads(row[1]) for row in connection.execute("SELECT dialogue_id,record FROM benchmark_ingestion")}
    assert len(committed) == 419
    all_ids = []
    for index, turn in enumerate(actual_turns):
        record = read(f"ingestion/{index:04d}.json")
        assert committed[turn["dialogue_id"]] == record
        assert record["turn"] == turn
        all_ids.extend(item["id"] for item in record["result"]["results"])
    stored_ids = {row[0] for row in connection.execute("SELECT id FROM memories")}
    assert len(all_ids) == len(set(all_ids)) and set(all_ids) == stored_ids
    assert connection.execute("SELECT count(*) FROM memory_fts").fetchone()[0] == len(stored_ids)
    assert connection.execute("SELECT count(*) FROM history WHERE event='ADD'").fetchone()[0] == len(stored_ids)
    exported = {m["id"]: m for m in read("memories_embeddings.json")["memories"]}
    stored_norms = {mid: sum(x*x for x in m["embedding"]) for mid, m in exported.items()}
    for mid, text, vector, meta in connection.execute("SELECT id,data,embedding,metadata FROM memories"):
        assert exported[mid] == {"id": mid, "data": text, "embedding": json.loads(vector), "metadata": json.loads(meta)}
    query_path = root / "memory_queries_embeddings.json"
    query_rows = read("memory_queries_embeddings.json")["queries"] if query_path.exists() else []
    question_norms = {(row["memory_id"], row["position"]): sum(x*x for x in row["embedding"])
                      for row in query_rows}
    if query_path.exists():
        stored_queries = [{"memory_id": mid, "position": position, "query": text, "embedding": json.loads(vector)}
                          for mid, position, text, vector in connection.execute(
                              "SELECT memory_id,position,query,embedding FROM memory_queries ORDER BY memory_id,position")]
        assert stored_queries == query_rows
    frozen_memories = {m["id"]: m for m in read("memory_snapshot.json")["memories"]}
    assert set(frozen_memories) == stored_ids
    final = read("final_results.json")
    results = final["results"]
    expected_qas = [(i, qa) for i, qa in enumerate(entry["qa"]) if int(qa["category"]) in {1, 2, 3, 4}]
    assert len(results) == len(expected_qas) == 152
    enhanced_queries, rerank_counts = 0, {}
    for record, (index, qa) in zip(results, expected_qas):
        assert record == read(f"query_details/{index:03d}.json")
        assert record["question_id"] == f"conv0_q{index}" and record["qa_index"] == index
        assert record["question"] == qa["question"] and record["gold_answer"] == str(qa["answer"])
        assert record["category"] == int(qa["category"])
        assert record["gold_evidence_dialogue_ids"] == runner.reference.gold_dialogue_ids(qa)[0]
        retrieved = record["retrieval"]["results"]
        trace = record["retrieval_trace"]
        semantic = trace["channels"]["semantic"]["candidates"]
        embedding_calls = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(
                           (root / "api_calls" / f"retrieval_q{index:03d}").glob("*_embedding.json"))]
        successful = [call for call in embedding_calls if call["status"] == "complete"]
        assert successful
        # A resumed query may have an earlier successful embedding but no saved
        # result. The last successful attempt precedes the committed checkpoint.
        call = successful[-1]
        assert call["request"]["input"] == [qa["question"].strip()]
        query_vector = call["response"]["data"][0]["embedding"]
        query_norm = sum(x*x for x in query_vector)
        exact = []
        for memory_id, exported_memory in exported.items():
            vector = exported_memory["embedding"]
            assert len(vector) == len(query_vector) == 1536
            denominator = math.sqrt(stored_norms[memory_id] * query_norm)
            cosine = max(0., min(1., sum(x*y for x, y in zip(query_vector, vector)) / denominator)) if denominator else 0.
            exact.append((memory_id, cosine))
        exact.sort(key=lambda pair: pair[1], reverse=True)
        assert [mid for mid, _ in exact[:80]] == [hit["memory_id"] for hit in semantic]
        assert all(abs(score - hit["score"]) < 1e-10 for (_, score), hit in zip(exact[:80], semantic))
        if trace["schema_version"] in {2, 3}:
            statuses = verify_enhanced_ranking(trace, retrieved, query_vector, query_rows, exported, question_norms)
            enhanced_queries += 1
            for name, state in statuses.items():
                rerank_counts.setdefault(name, {})[state] = rerank_counts.get(name, {}).get(state, 0) + 1
            if trace["schema_version"] == 3 and trace["global_rerank"]["status"] == "succeeded":
                calls = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(
                         (root / "api_calls" / f"retrieval_q{index:03d}").glob("*_chat_completion.json"))]
                call = [c for c in calls if c["status"] == "complete"][-1]
                payload = json.loads(call["request"]["messages"][-1]["content"])
                assert set(payload) == {"query", "max_select", "candidates"}
                assert payload["query"] == qa["question"]
                info = trace["global_rerank"]
                assert payload["max_select"] == info["max_select"]
                rows = payload["candidates"]
                assert len(rows) == len(info["input_ids"])
                mapped = {}
                for position, (row, mid) in enumerate(zip(rows, info["input_ids"]), 1):
                    assert row == {"id": f"m{position:02d}", "text": exported[mid]["data"],
                                   "observation_date": exported[mid]["metadata"].get("session_date"),
                                   "source_dialogue_ids": exported[mid]["metadata"].get("source_dialogue_ids", [])}
                    mapped[row["id"]] = mid
                response = json.loads(call["response"]["choices"][0]["message"]["content"])
                assert [mapped[key] for key in response["ids"]] == info["selected_ids"]
        else:
            assert trace["schema_version"] == 1
            bm25 = trace["channels"]["bm25"]["candidates"]
            assert trace["channels"]["bm25"]["status"] == "recorded"
            normalized = normalized_bm25(bm25)
            ranked = [(hit["memory_id"], (hit["score"] + normalized.get(hit["memory_id"], 0)) / (2 if bm25 else 1))
                      for hit in semantic if hit["score"] >= 0.1]
            ranked.sort(key=lambda pair: pair[1], reverse=True)
            assert [mid for mid, _ in ranked] == trace["stages"]["combined_order"]
            assert [mid for mid, _ in ranked[:20]] == [m["id"] for m in retrieved]
            assert [m["id"] for m in retrieved] == [m["memory_id"] for m in trace["stages"]["final"]]
            assert all(abs(score - hit["score"]) < 1e-10 for (_, score), hit in zip(ranked[:20], retrieved))
        assert all(hit["memory"] == frozen_memories[hit["id"]]["memory"] for hit in retrieved)
        assert record["answer_context"] == runner.reference.build_context(retrieved)
        assert record["generation"]["answer"].strip()
        judgment = record["judgment"]
        assert judgment["label"] in {"CORRECT", "WRONG"}
        assert judgment["correct"] == (judgment["label"] == "CORRECT")
    expected_summary = runner.reference.summarize(results)
    summary = read("summary.json")
    assert all(summary[key] == value for key, value in expected_summary.items())
    assert final["summary"] == summary and final["metadata"] == metadata
    assert summary["memory_count"] == len(stored_ids)
    usage = read("api_usage_summary.json")
    assert usage == runner.collect_usage(root)
    connection.close()
    report = {"status": "passed", "verified_at": runner.now(), "ingestion_turns": 419,
              "questions": 152, "memories": len(stored_ids), "correct": summary["correct"],
              "accuracy": summary["accuracy"], "checks": ["dataset and frozen source identities", "SQLite integrity",
              "atomic ingestion records and snapshots", "all stored embedding exports",
              "all 152 semantic pools recomputed from real vectors", "all 152 native final rankings",
              "BM25 normalization and combined scores", "answer contexts and judge labels", "summary and API usage"]}
    if enhanced_queries:
        assert enhanced_queries == 152
        report.update(enhanced_queries=enhanced_queries, memory_query_count=len(query_rows), rerank_counts=rerank_counts)
        report["checks"].extend(["all stored question vectors", "all 152 memory-question pools",
                                 "channel rerank permutations and tails", "normalized three-channel RRF"])
    runner.write_json(root / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    verify(parser.parse_args().run_dir.resolve())

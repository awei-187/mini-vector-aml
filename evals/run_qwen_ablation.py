"""Frozen edited147 candidates: no reranker vs Qwen default/support instruct.

Preparation is offline. --run-models requires explicit authorization for this
new paid run. Reuses no historical answers/judgments and performs no ingestion
or embeddings. Identical NEW answer/judge inputs share a within-run result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request

import run_locomo as runner
import run_edited_dual as dual
import dual_judge_contract as judges
from mini_vector.answer_context import answer_messages, build_context
from mini_vector.config import setting
from mini_vector.reranker import (DEFAULT_INSTRUCTION, SUPPORT_INSTRUCTION, QwenReranker,
    _NoRedirect, parse_scores, qwen_reranker_config, rerank_qwen)
from mini_vector.source_context import SourceIndex

ARMS = ("none", "qwen_default", "qwen_support")
INSTRUCTIONS = {"qwen_default": DEFAULT_INSTRUCTION, "qwen_support": SUPPORT_INSTRUCTION}
BASELINE = runner.PROJECT / "eval_outputs/edited147_dual_20261024_01"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def freeze_candidates(row, facts, limit=60):
    trace = row["retrieval_trace"]
    if trace["rerank_mode"] != "none" or trace["memory_query_enabled"] is not True:
        raise ValueError("Expected three-channel RRF without reranking")
    ranks, active = {}, 0
    for channel in trace["channels"].values():
        ids = channel["eligible_ids"]
        if channel["rerank"]["output_ids"] != ids:
            raise ValueError("Historical channel ordering was changed")
        active += bool(ids)
        for rank, mid in enumerate(ids, 1):
            ranks.setdefault(mid, []).append(rank)
    rrf = {mid: sum(1 / (60 + rank) for rank in positions) for mid, positions in ranks.items()}
    order = sorted(rrf, key=lambda mid: (-rrf[mid], mid))
    if order != trace["stages"]["combined_order"]:
        raise ValueError("Frozen RRF order cannot be reproduced")
    if order[:20] != [r["id"] for r in row["retrieval"]]:
        raise ValueError("Frozen top20 mismatch")
    items = [{**facts[mid], "score": min(1., rrf[mid] / (active / 61))} for mid in order[:limit]]
    for new, old in zip(items, row["retrieval"]):
        if new["memory"] != old["memory"] or new["metadata"] != old["metadata"] or new["score"] != old["score"]:
            raise ValueError("Frozen fact or score mismatch")
    if len(items) < 20:
        raise ValueError("Not enough candidates")
    return items


def prepare(root, baseline=BASELINE):
    if runner.read_json(baseline / "verification.json", {}).get("status") != "passed":
        raise ValueError("Baseline is not independently verified")
    config = qwen_reranker_config(env_file="E:/.env")
    if config.model != "qwen3-rerank":
        raise ValueError("This experiment is predeclared for qwen3-rerank")
    jev, _ = judges.jev_settings()
    source_paths = [*sorted((runner.PROJECT / "mini_vector").glob("*.py")), Path(__file__),
                    Path(__file__).with_name("verify_qwen_ablation.py"), Path(runner.__file__),
                    Path(dual.__file__), Path(judges.__file__)]
    input_paths = [baseline / name for name in ("verification.json", "memory_snapshot.json", "source_snapshot.json",
                                               "input_conversation_0.json", "judge_contract.json")]
    input_paths += sorted((baseline / "query_details").glob("*.json"))
    identity = {"baseline": str(baseline), "input_sha256": {str(p): runner.digest(p) for p in input_paths},
                "source_sha256": {str(p.relative_to(runner.PROJECT)): runner.digest(p) for p in source_paths},
                "arms": ARMS, "instructions": INSTRUCTIONS, "qwen": config.public(), "jev": jev,
                "answer_model": "gpt-4o-mini", "gpt_judge_model": "gpt-4o-mini", "temperature": 0,
                "chat_provider_sha256": fingerprint(setting("OPENAI_BASE_URL")),
                "candidate_limit": 60, "top_k": 20, "questions": 147,
                "source_windows": {"max_windows":4,"radius":1,"max_chars":12000},
                "scope": "Same frozen RRF60 pool, fresh answers, temporal_source, GPT and pinned Jev",
                "no_ingestion_or_embeddings": True, "within_run_input_cache_only": True}
    # JSON normalizes tuples on disk.
    identity = json.loads(json.dumps(identity))
    previous = runner.read_json(root / "manifest.json", {})
    if previous and previous["identity"] != identity:
        raise ValueError("Inputs, source or provider configuration changed; use a new run")
    if previous:
        for name, expected in previous["frozen_sha256"].items():
            if runner.digest(root / name) != expected:
                raise ValueError("Frozen experiment inputs changed")
        return previous
    for path in source_paths:
        target = root / "source_code" / path.relative_to(runner.PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    facts = {r["id"]: r for r in runner.read_json(baseline / "memory_snapshot.json", {})["memories"]}
    turns = runner.read_json(baseline / "source_snapshot.json", [])
    entry = runner.read_json(baseline / "input_conversation_0.json", {})
    contract = runner.read_json(baseline / "judge_contract.json", {})
    if len(facts) != 1742 or len(turns) != 419 or len(entry["qa"]) != 147 or contract != judges.load_contract():
        raise ValueError("Unexpected edited147 baseline or judge contract")
    rows = []
    index = SourceIndex(turns)
    try:
        for i, qa in enumerate(entry["qa"]):
            old = runner.read_json(baseline / "query_details" / f"{i:03d}.json", {})
            if old["question"] != qa["question"] or old["gold_answer"] != qa.get("answer", ""):
                raise ValueError("Question/gold identity mismatch")
            candidates = freeze_candidates(old, facts)
            windows, _ = index.select(old["question"], candidates[:20])
            if windows != old["source_windows"] or build_context(candidates[:20], windows) != old["answer_context"]:
                raise ValueError("Unchanged-arm source/context replay mismatch")
            rows.append({"qa_index":i, "question":old["question"], "gold_answer":old["gold_answer"],
                         "category":old["category"], "candidates":candidates,
                         "pool_sha256":fingerprint(candidates)})
    finally:
        index.close()
    runner.write_json(root / "frozen_inputs.json", rows)
    runner.write_json(root / "source_snapshot.json", turns)
    runner.write_json(root / "judge_contract.json", contract)
    manifest = {"created_at":runner.now(), "identity":identity,
                "frozen_sha256":{n:runner.digest(root/n) for n in ("frozen_inputs.json","source_snapshot.json","judge_contract.json")}}
    runner.write_json(root / "manifest.json", manifest)
    runner.write_json(root / "run_status.json", {"status":"prepared", "model_calls":0})
    return manifest


def audited_qwen(root, name, config, body):
    """Bounded, restartable attempts; preserve usage on protocol failures too."""
    folder = root / "qwen_api" / name
    previous = sorted(folder.glob("*.json"))
    for path in previous:
        row = runner.read_json(path,{})
        if row["request"] != body:
            raise ValueError("Cached Qwen payload mismatch")
        if row.get("status") == "complete":
            parse_scores(row["response"], len(body["input"]["documents"]))
            return row["response"]
    if len(previous) >= 3:
        raise RuntimeError("Qwen attempt budget exhausted")
    for attempt in range(len(previous)+1,4):
        record = {"status":"running", "started_at":runner.now(), "request":body}
        path = folder / f"{attempt:02d}.json"
        runner.write_json(path, record)
        started, fatal = time.perf_counter(), False
        try:
            request = urllib.request.Request(config.endpoint, data=json.dumps(body,ensure_ascii=False).encode(),
                headers={"Authorization":"Bearer "+config.api_key,"Content-Type":"application/json"})
            with urllib.request.build_opener(_NoRedirect).open(request, timeout=config.timeout) as response:
                raw = json.loads(response.read().decode())
            raw = json.loads(json.dumps(raw,ensure_ascii=False).replace(config.api_key,"[REDACTED]"))
            record["response"] = raw
            parse_scores(raw,len(body["input"]["documents"]))
            record["status"] = "complete"
            return raw
        except urllib.error.HTTPError as exc:
            record.update(status="failed",error_type="HTTPError",http_status=exc.code)
            fatal = exc.code < 500 and exc.code != 429
            exc.close()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record.update(status="failed",error_type=type(exc).__name__)
        finally:
            record.update(finished_at=runner.now(),latency_ms=(time.perf_counter()-started)*1000)
            runner.write_json(path,record)
        if fatal or attempt == 3:
            raise RuntimeError("Qwen API/protocol failure; inspect sanitized audit")
        time.sleep(2**(attempt-1))


def collect_all_usage(root):
    result = {"gpt":runner.collect_usage(root)}
    for provider, folder in (("qwen","qwen_api"),("jev","jev_api")):
        groups = {}
        for path in (root/folder).rglob("*.json"):
            r = runner.read_json(path,{})
            key = "preflight" if "preflight" in path.parts else "benchmark"
            group = groups.setdefault(key,{"attempts":0,"completed":0,"failed":0,"returned_usage":[]})
            group["attempts"] += 1
            group["completed"] += r.get("status") == "complete"
            group["failed"] += r.get("status") != "complete"
            usage = r.get("response",{}).get("usage")
            if usage is not None:
                group["returned_usage"].append(usage)
        result[provider] = groups
    return result


def compare(arms):
    summary = {arm:dual.summarize(rows) for arm,rows in arms.items()}
    pairs = {}
    for base, target in (("none","qwen_default"),("none","qwen_support"),("qwen_default","qwen_support")):
        key = base+"_to_"+target
        pairs[key] = {}
        for judge in ("gpt","jev"):
            wins, losses = [], []
            for a,b in zip(arms[base],arms[target]):
                x,y = a[judge+"_judgment"]["label"]=="correct", b[judge+"_judgment"]["label"]=="correct"
                if y and not x:wins.append(a["qa_index"])
                if x and not y:losses.append(a["qa_index"])
            pairs[key][judge] = {"wins":wins,"losses":losses,"net":len(wins)-len(losses)}
        pairs[key]["same_top20_order"] = sum([x["id"] for x in a["retrieval"]]==[x["id"] for x in b["retrieval"]]
                                            for a,b in zip(arms[base],arms[target]))
        pairs[key]["same_top20_set"] = sum({x["id"] for x in a["retrieval"]}=={x["id"] for x in b["retrieval"]}
                                          for a,b in zip(arms[base],arms[target]))
    return {"arms":summary,"paired":pairs}


def run(root, workers):
    manifest = prepare(root)
    frozen = runner.read_json(root/"frozen_inputs.json",[])
    contract = runner.read_json(root/"judge_contract.json",{})
    turns = runner.read_json(root/"source_snapshot.json",[])
    from openai import OpenAI
    client = OpenAI(api_key=setting("OPENAI_API_KEY"),base_url=setting("OPENAI_BASE_URL"),timeout=90,max_retries=0)
    audit = runner.Audit(root,client)
    status = {"status":"running","phase":"preflight","completed":0,"target":441}
    def progress(**kwargs):
        status.update(updated_at=runner.now(),**kwargs);runner.write_json(root/"run_status.json",status)
    try:
        if not (root/"preflight.json").exists():
            for arm,instruction in INSTRUCTIONS.items():
                cfg = qwen_reranker_config(env_file="E:/.env",instruction=instruction)
                reranker = QwenReranker(cfg,transport=lambda body,cfg=cfg,arm=arm:audited_qwen(root,"preflight/"+arm,cfg,body))
                if len(parse_scores(reranker("Where does Ada live?",["Ada lives in Oslo.","Ben likes tennis."]),2)) != 2:
                    raise ValueError("Qwen preflight failed")
            audit.scope("preflight")
            prompt = answer_messages("Where does Ada live?","Ada lives in Oslo.")
            audit.chat(model="gpt-4o-mini",temperature=0,messages=prompt)
            state = judges.judge_state("Name the city.","Oslo","Oslo")
            if dual.gpt_judge(audit,state,contract)["label"] != "correct":raise ValueError("GPT preflight")
            if judges.jev_call(root,"preflight",state,contract,runner.write_json,runner.read_json,runner.now)["label"] != "correct":
                raise ValueError("Jev preflight")
            runner.write_json(root/"preflight.json",{"status":"complete","excluded_from_benchmark":True})
        # Each worker owns its SQLite source index. No shared SQLite connection.
        def retrieval_work(i):
            item = frozen[i]
            source = SourceIndex(turns)
            try:
                for arm in ARMS:
                    path = root/arm/"query_details"/f"{i:03d}.json"
                    if path.exists():continue
                    candidates = item["candidates"]
                    trace = {"status":"disabled","api_calls":0}
                    ordered = candidates
                    if arm != "none":
                        cfg = qwen_reranker_config(env_file="E:/.env",instruction=INSTRUCTIONS[arm])
                        reranker = QwenReranker(cfg,transport=lambda body,cfg=cfg,arm=arm:audited_qwen(root,f"{arm}/q{i:03d}",cfg,body))
                        ordered,trace = rerank_qwen(reranker,item["question"],candidates,60)
                        if trace["status"] != "succeeded":
                            raise RuntimeError("A fallback is not a successful Qwen experiment")
                    found = ordered[:20]
                    windows,source_trace = source.select(item["question"],found,**manifest["identity"]["source_windows"])
                    row = {k:item[k] for k in ("qa_index","question","gold_answer","category","pool_sha256")}
                    row.update(arm=arm,retrieval=found,rerank_trace=trace,source_windows=windows,
                               source_trace=source_trace,answer_context=build_context(found,windows))
                    runner.write_json(path,row)
            finally:source.close()
        progress(phase="qwen_rerank")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for completed,task in enumerate(as_completed([pool.submit(retrieval_work,i) for i in range(147)]),1):
                task.result();progress(completed_retrieval_questions=completed)
                print(f"QWEN {completed}/147 (two instructions)",flush=True)
        # Shared within-run caches; locks ensure identical payloads call once.
        locks, locks_lock = {}, threading.Lock()
        def cached(kind,body,compute):
            key = fingerprint(body)
            with locks_lock:lock = locks.setdefault(kind+key,threading.Lock())
            with lock:
                path = root/"request_cache"/kind/f"{key}.json"
                if path.exists():
                    data = runner.read_json(path,{})
                    if data["request"] != body:raise ValueError("Request hash collision")
                    return data["result"],key
                result = compute(key)
                runner.write_json(path,{"request":body,"result":result})
                return result,key
        def evaluate(arm,i):
            path = root/arm/"query_details"/f"{i:03d}.json"
            row = runner.read_json(path,{})
            if "generation" not in row:
                body = {"model":"gpt-4o-mini","temperature":0,"messages":answer_messages(row["question"],row["answer_context"])}
                def answer(key):
                    audit.scope("answer_"+key)
                    response = audit.chat(**body)
                    text = (response.choices[0].message.content or "").strip()
                    if not text:raise ValueError("Empty answer")
                    return {"answer":text,"usage":response.usage.model_dump() if response.usage else {}}
                row["generation"],row["answer_request_sha256"] = cached("answer",body,answer)
                runner.write_json(path,row)
            state = judges.judge_state(row["question"],row["gold_answer"],row["generation"]["answer"])
            if "gpt_judgment" not in row:
                body = judges.gpt_messages(state,contract)
                def gpt(key):
                    audit.scope("gpt_judge_q"+key)
                    return dual.gpt_judge(audit,state,contract)
                row["gpt_judgment"],row["gpt_request_sha256"] = cached("gpt",body,gpt)
                runner.write_json(path,row)
            if "jev_judgment" not in row:
                body = judges.jev_payload(state,contract)
                def jev(key):return judges.jev_call(root,key,state,contract,runner.write_json,runner.read_json,runner.now)
                row["jev_judgment"],row["jev_request_sha256"] = cached("jev",body,jev)
                runner.write_json(path,row)
            return row
        progress(phase="answer_and_dual_judge")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Rotate arm order per question to avoid an entire arm always running first.
            tasks = [pool.submit(evaluate,ARMS[(j+i)%3],i) for i in range(147) for j in range(3)]
            for completed,task in enumerate(as_completed(tasks),1):
                row = task.result();progress(completed=completed)
                print(f"DUAL {completed}/441 | {row['arm']} q{row['qa_index']} GPT={row['gpt_judgment']['label']} JEV={row['jev_judgment']['label']}",flush=True)
        arms = {arm:[runner.read_json(root/arm/"query_details"/f"{i:03d}.json",{}) for i in range(147)] for arm in ARMS}
        runner.write_json(root/"comparison.json",compare(arms))
        runner.write_json(root/"api_usage_summary.json",collect_all_usage(root))
        progress(status="complete",phase="complete")
    except BaseException as exc:
        progress(status="failed",error_type=type(exc).__name__)
        runner.write_json(root/"api_usage_summary.json",collect_all_usage(root))
        raise
    finally:client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--run-models",action="store_true")
    p.add_argument("--workers",type=int,default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:p.error("workers must be 1 to 8")
    root = args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    with runner.run_lock(root):
        prepare(root)
        if args.run_models:run(root,args.workers)
        else:print("PREPARED: 147 frozen RRF60 pools, 3 arms, no model calls",flush=True)


if __name__ == "__main__":
    try:main()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}; inspect sanitized local audit",flush=True)
        raise SystemExit(1)

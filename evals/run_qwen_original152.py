"""Original152, frozen RRF60: none vs dedicated Qwen default/support.

Historical answer/context/binary-judge prompts are retained for comparability
with the original global/evidence experiment. No temporal/source policy changes,
no fresh corpus or embeddings. Preparation never calls any provider.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import run_locomo as runner
import run_qwen_ablation as common
from mini_vector.config import setting
from mini_vector.reranker import QwenReranker, qwen_reranker_config, rerank_qwen

ARMS = common.ARMS
BASELINE = runner.PROJECT / "eval_outputs/locomo_conv26_optimization_20261024_01/rrf"


def historical_request(kind, question, *, context="", gold="", answer="", retry=False):
    """Capture the existing helper's exact request without a network client."""
    captured = []
    def create(**body):
        captured.append(body)
        text = ("{}" if retry and len(captured)==1 else '{"is_correct":"CORRECT","reasoning":"capture"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    if kind == "answer":runner.reference.generate_answer(client,question,context)
    elif kind == "judge":runner.reference.judge_answer(client,question,gold,answer)
    else:raise ValueError("Unknown historical request kind")
    return captured[-1]


def prepare(root):
    verified = runner.read_json(BASELINE/"verification.json",{})
    if verified.get("status")!="passed" or verified.get("questions")!=152:
        raise ValueError("Expected verified original152 RRF baseline")
    cfg = qwen_reranker_config(env_file="E:/.env")
    if cfg.model!="qwen3-rerank":raise ValueError("Expected configured qwen3-rerank")
    sources={str(p.relative_to(runner.PROJECT)):p for p in (
        *sorted((runner.PROJECT/"mini_vector").glob("*.py")),Path(__file__),
        Path(__file__).with_name("verify_qwen_original152.py"),Path(common.__file__),Path(runner.__file__))}
    sources["external/mem0_locomo_reference.py"]=runner.HELPER
    paths=[BASELINE/n for n in ("verification.json","final_results.json","memory_snapshot.json",
                                "source_snapshot.json","input_conversation_0.json")]
    identity={"baseline":str(BASELINE),"baseline_inputs":{str(p):runner.digest(p) for p in paths},
        "source_sha256":{name:runner.digest(path) for name,path in sources.items()},
        "questions":152,"candidate_limit":60,"top_k":20,"arms":list(ARMS),
        "instructions":common.INSTRUCTIONS,"qwen":cfg.public(),"model":"gpt-4o-mini","temperature":0,
        "chat_provider_sha256":common.fingerprint(setting("OPENAI_BASE_URL")),
        "answer_policy":"historical_original152_event_time_context",
        "judge_policy":"historical_generous_GPT_CORRECT_WRONG",
        "fresh_answers":True,"within_run_exact_request_cache_only":True,
        "no_ingestion_or_embeddings":True,"source_windows":False,
        "comparison_limit":"Old global/evidence are historical references; not rerun in this new batch"}
    previous=runner.read_json(root/"manifest.json",{})
    if previous:
        if previous["identity"]!=identity:raise ValueError("Inputs/source/config changed; create a new run")
        for name,expected in previous["frozen_sha256"].items():
            if runner.digest(root/name)!=expected:raise ValueError("Frozen input changed")
        return previous
    old=runner.read_json(BASELINE/"final_results.json",{})
    entry=runner.read_json(BASELINE/"input_conversation_0.json",{})
    facts={r["id"]:r for r in runner.read_json(BASELINE/"memory_snapshot.json",{})["memories"]}
    if len(facts)!=1711 or len(old["results"])!=152 or len(runner.reference.get_turns(entry))!=419:
        raise ValueError("Unexpected original152 corpus")
    rows=[]
    for original in old["results"]:
        i=original["qa_index"];qa=entry["qa"][i]
        if (original["question"]!=qa["question"] or original["gold_answer"]!=str(qa.get("answer",""))
                or int(qa["category"]) not in (1,2,3,4)):
            raise ValueError("Question identity mismatch")
        normalized={**original,"retrieval":original["retrieval"]["results"]}
        candidates=common.freeze_candidates(normalized,facts)
        context=runner.reference.build_context(candidates[:20])
        if context!=original["answer_context"]:raise ValueError("Historical baseline context changed")
        actual=[runner.read_json(p,{}) for p in sorted((BASELINE/"api_calls"/f"answer_q{i:03d}").glob("*.json"))]
        request=[r["request"] for r in actual if r.get("status")=="complete"][-1]
        if request!=historical_request("answer",original["question"],context=context):
            raise ValueError("Historical answer prompt changed")
        rows.append({k:original[k] for k in ("question_id","qa_index","category","question","gold_answer")})
        rows[-1].update(candidates=candidates,pool_sha256=common.fingerprint(candidates))
    for name,path in sources.items():
        target=root/"source_code"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    runner.write_json(root/"frozen_inputs.json",rows)
    manifest={"created_at":runner.now(),"identity":identity,
              "frozen_sha256":{"frozen_inputs.json":runner.digest(root/"frozen_inputs.json")}}
    runner.write_json(root/"manifest.json",manifest)
    runner.write_json(root/"run_status.json",{"status":"prepared","model_calls":0})
    return manifest


def summarize(arms):
    groups={}
    for arm,rows in arms.items():
        count=sum(r["judgment"]["correct"] for r in rows)
        groups[arm]={"total":len(rows),"correct":count,"wrong":len(rows)-count,
                     "accuracy":count/len(rows),"by_category":{}}
        for category in (1,2,3,4):
            subset=[r for r in rows if r["category"]==category]
            groups[arm]["by_category"][str(category)]={"total":len(subset),
                "correct":sum(r["judgment"]["correct"] for r in subset)}
    pairs={}
    for a,b in (("none","qwen_default"),("none","qwen_support"),("qwen_default","qwen_support")):
        wins=[];losses=[]
        for x,y in zip(arms[a],arms[b]):
            assert x["qa_index"]==y["qa_index"]
            if not x["judgment"]["correct"] and y["judgment"]["correct"]:wins.append(x["qa_index"])
            if x["judgment"]["correct"] and not y["judgment"]["correct"]:losses.append(x["qa_index"])
        pairs[a+"_to_"+b]={"wins":wins,"losses":losses,"net":len(wins)-len(losses),
            "same_top20_set":sum({r['id'] for r in x['retrieval']}=={r['id'] for r in y['retrieval']}
                                  for x,y in zip(arms[a],arms[b]))}
    return {"arms":groups,"paired":pairs}


def run(root,workers):
    prepare(root)
    frozen=runner.read_json(root/"frozen_inputs.json",[])
    from openai import OpenAI
    client=OpenAI(api_key=setting("OPENAI_API_KEY"),base_url=setting("OPENAI_BASE_URL"),timeout=90,max_retries=0)
    audit=runner.Audit(root,client)
    status={"status":"running","phase":"preflight","completed":0,"target":456}
    def progress(**updates):
        status.update(updated_at=runner.now(),**updates);runner.write_json(root/"run_status.json",status)
    locks={};guard=threading.Lock()
    def cached(kind,body,compute):
        key=common.fingerprint(body)
        with guard:lock=locks.setdefault(kind+key,threading.Lock())
        with lock:
            path=root/"request_cache"/kind/f"{key}.json"
            if path.exists():
                saved=runner.read_json(path,{})
                if saved["request"]!=body:raise ValueError("Request cache mismatch")
                return saved["result"],key
            value=compute(key);runner.write_json(path,{"request":body,"result":value});return value,key
    try:
        if not (root/"preflight.json").exists():
            for arm,instruction in common.INSTRUCTIONS.items():
                cfg=qwen_reranker_config(env_file="E:/.env",instruction=instruction)
                reranker=QwenReranker(cfg,transport=lambda body,cfg=cfg,arm=arm:
                    common.audited_qwen(root,"preflight/"+arm,cfg,body))
                ranked,trace=rerank_qwen(reranker,"Where does Ada live?",
                    [{"id":"a","memory":"Ada lives in Oslo."},{"id":"b","memory":"Ben likes tennis."}],60)
                if trace["status"]!="succeeded":raise ValueError("Qwen preflight failed")
            audit.scope("preflight")
            runner.reference.generate_answer(audit.answer_client(),"Where does Ada live?",
                                             runner.reference.build_context([{"memory":"Ada lives in Oslo."}]))
            if not runner.reference.judge_answer(audit.answer_client(),"Name the city.","Oslo","Oslo")["correct"]:
                raise ValueError("Binary judge preflight failed")
            runner.write_json(root/"preflight.json",{"status":"complete","excluded_from_benchmark":True})
        def retrieval(item):
            i=item["qa_index"]
            for arm in ARMS:
                path=root/arm/"query_details"/f"{i:03d}.json"
                if path.exists():continue
                candidates=item["candidates"];ordered=candidates;trace={"status":"disabled","api_calls":0}
                if arm!="none":
                    cfg=qwen_reranker_config(env_file="E:/.env",instruction=common.INSTRUCTIONS[arm])
                    scorer=QwenReranker(cfg,transport=lambda body,cfg=cfg,arm=arm:
                        common.audited_qwen(root,f"{arm}/q{i:03d}",cfg,body))
                    ordered,trace=rerank_qwen(scorer,item["question"],candidates,60)
                    if trace["status"]!="succeeded":raise RuntimeError("Do not count fallback as Qwen success")
                row={k:item[k] for k in ("question_id","qa_index","category","question","gold_answer","pool_sha256")}
                row.update(arm=arm,retrieval=ordered[:20],rerank_trace=trace,
                           answer_context=runner.reference.build_context(ordered[:20]))
                runner.write_json(path,row)
        progress(phase="qwen_rerank")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done,task in enumerate(as_completed([pool.submit(retrieval,r) for r in frozen]),1):
                task.result();progress(completed_retrieval_questions=done);print(f"QWEN {done}/152",flush=True)
        def evaluate(arm,item):
            i=item["qa_index"];path=root/arm/"query_details"/f"{i:03d}.json";row=runner.read_json(path,{})
            if "generation" not in row:
                body=historical_request("answer",row["question"],context=row["answer_context"])
                def answer(key):
                    audit.scope("answer_"+key)
                    text,usage=runner.reference.generate_answer(audit.answer_client(),row["question"],row["answer_context"])
                    if not text:raise ValueError("Empty answer")
                    return {"answer":text,"usage":usage}
                row["generation"],row["answer_request_sha256"]=cached("answer",body,answer);runner.write_json(path,row)
            if "judgment" not in row:
                body=historical_request("judge",row["question"],gold=row["gold_answer"],answer=row["generation"]["answer"])
                def judge(key):
                    audit.scope("judge_"+key)
                    return runner.reference.judge_answer(audit.answer_client(),row["question"],
                                                         row["gold_answer"],row["generation"]["answer"])
                row["judgment"],row["judge_request_sha256"]=cached("judge",body,judge);runner.write_json(path,row)
            return row
        progress(phase="answer_and_binary_judge")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks=[pool.submit(evaluate,ARMS[(i+j)%3],item) for i,item in enumerate(frozen) for j in range(3)]
            for done,task in enumerate(as_completed(tasks),1):
                row=task.result();progress(completed=done)
                print(f"EVAL {done}/456 | {row['arm']} q{row['qa_index']} {row['judgment']['label']}",flush=True)
        arms={a:[runner.read_json(root/a/"query_details"/f"{r['qa_index']:03d}.json",{}) for r in frozen] for a in ARMS}
        runner.write_json(root/"comparison.json",summarize(arms))
        runner.write_json(root/"api_usage_summary.json",common.collect_all_usage(root))
        progress(status="complete",phase="complete")
    except BaseException as exc:
        progress(status="failed",error_type=type(exc).__name__)
        runner.write_json(root/"api_usage_summary.json",common.collect_all_usage(root));raise
    finally:client.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--run-models",action="store_true")
    p.add_argument("--workers",type=int,default=4);args=p.parse_args()
    if not 1<=args.workers<=8:p.error("workers must be 1..8")
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    with runner.run_lock(root):
        prepare(root)
        if args.run_models:run(root,args.workers)
        else:print("PREPARED original152: frozen60, 3 arms, historical answer/binary judge, no model calls",flush=True)


if __name__=="__main__":
    try:main()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}; inspect sanitized local audit",flush=True);raise SystemExit(1)

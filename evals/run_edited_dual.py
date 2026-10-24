"""Latest temporal+source MiniVector on edited LoCoMo, GPT and Jev four classes."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

import run_locomo as runner
import dual_judge_contract as judges
from mini_vector import MiniVector
from mini_vector.answer_context import answer_messages, build_context
from mini_vector.config import setting
from mini_vector.source_context import SourceIndex

DATASET = Path('E:/LLM judge/locomo_first_conversation_edited.json')


def prepare(root, dataset):
    data = runner.read_json(dataset, [])
    entry = data[0]
    turns = runner.reference.get_turns(entry)
    if entry['sample_id'] != 'conv-26' or len(turns) != 419 or len(entry['qa']) != 147:
        raise ValueError('Expected edited conv-26, 419 turns and exactly 147 questions')
    for qa in entry['qa']:
        ids, unresolved = runner.reference.gold_dialogue_ids(qa)
        if int(qa['category']) not in {1,2,3,4} or unresolved or not set(ids) <= {t['dialogue_id'] for t in turns}:
            raise ValueError('Invalid category or dataset evidence identity')
    sources = [Path(__file__), Path(__file__).with_name('dual_judge_contract.py'),
               Path(__file__).with_name('verify_edited_dual.py'), Path(runner.__file__),
               Path(__file__).with_name('verify_results.py'), runner.HELPER,
               judges.GPT_PROMPT, judges.MODELS, judges.JEV_RUBRIC,
               *sorted((runner.PROJECT/'mini_vector').glob('*.py'))]
    settings, _ = judges.jev_settings()
    identity = {'dataset': str(dataset), 'dataset_sha256': runner.digest(dataset),
                'source_sha256': {str(p): runner.digest(p) for p in sources},
                'chat_model': 'gpt-4o-mini', 'embedding_model': 'text-embedding-3-small',
                'chat_provider_fingerprint': hashlib.sha256(str(setting('OPENAI_BASE_URL')).encode()).hexdigest(),
                'jev': settings, 'temperature': 0, 'top_k':20,'threshold':.1,
                'memory_query_enabled':True,'rerank_mode':'none','queries_per_memory':3,
                'latest_answer_policy':'temporal_source','source_windows':{'max_windows':4,'radius':1,'max_chars':12000},
                'ingestion_turns':419,'questions':147,'judges':['gpt_four_class','jev_four_class']}
    previous = runner.read_json(root/'run_metadata.json',{})
    if previous and previous['identity'] != identity:
        raise ValueError('Dataset, source or configured services changed; use a new run directory')
    if not previous and (root/'memory.sqlite3').exists():
        raise ValueError('Unregistered database')
    metadata = previous or {'created_at':runner.now(),'scope':{'user_id':'edited-conv-26','run_id':root.name},
                            'identity':identity,'history_reused':False,
                            'authorization':'User explicitly approved fresh edited147 build, answers, GPT+Jev judges and costs'}
    runner.write_json(root/'run_metadata.json',metadata)
    for path in sources:
        # Keep workspace and external review sources in separate relative trees.
        name = path.relative_to(runner.ROOT) if path.is_relative_to(runner.ROOT) else Path('external')/path.relative_to(Path('E:/LLM judge'))
        target=root/'source_code'/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    runner.write_json(root/'input_conversation_0.json',entry)
    runner.write_json(root/'source_snapshot.json',turns)
    contract=judges.load_contract()
    runner.write_json(root/'judge_contract.json',contract)
    return entry,turns,metadata,contract


def export_memory(root,memory,scope):
    facts=memory.get_all(filters=scope)['results']
    runner.write_json(root/'memory_snapshot.json',{'count':len(facts),'memories':facts})
    rows=[dict(r) for r in memory.store.db.execute('SELECT id,data,embedding,metadata FROM memories')]
    for row in rows:
        row['embedding']=json.loads(row['embedding']);row['metadata']=json.loads(row['metadata'])
    runner.write_json(root/'memories_embeddings.json',{'memories':rows})
    queries=[dict(r) for r in memory.store.db.execute('SELECT memory_id,position,query,embedding FROM memory_queries ORDER BY memory_id,position')]
    for row in queries: row['embedding']=json.loads(row['embedding'])
    runner.write_json(root/'memory_queries_embeddings.json',{'queries':queries})
    return len(facts),len(queries)


def gpt_judge(audit,state,contract):
    messages=judges.gpt_messages(state,contract)
    scope=getattr(audit.local,'scope','')
    completed=[]
    if scope.startswith('gpt_judge_q'):
        for path in sorted((audit.root/'api_calls'/scope).glob('*_chat_completion.json')):
            record=runner.read_json(path,{})
            if record.get('status')=='complete':
                if record['request']['messages']!=messages:raise ValueError('GPT saved input changed')
                completed.append(record)
        for record in reversed(completed):
            try:
                result=judges.parse_gpt(record['response']['choices'][0]['message']['content'])
                result['usage']=record['response'].get('usage') or {}
                return result
            except (ValueError,TypeError,KeyError):pass
    for _ in range(len(completed),3):
        response=audit.chat(model='gpt-4o-mini',temperature=0,response_format={'type':'json_object'},
                            messages=messages)
        try:
            result=judges.parse_gpt(response.choices[0].message.content or '{}')
            result['usage']=response.usage.model_dump() if response.usage else {}
            return result
        except (ValueError,TypeError,KeyError):
            pass
    raise ValueError('GPT judge protocol failed after three attempts')


def summarize(rows):
    result={'total':len(rows),'judges':{},'by_category':{}}
    for name in ('gpt','jev'):
        counts=Counter(row[name+'_judgment']['label'] for row in rows)
        result['judges'][name]={'counts':{label:counts[label] for label in judges.LABELS},
                               'llm_j':counts['correct']/len(rows) if rows else None}
    matrix={a:{b:sum(r['gpt_judgment']['label']==a and r['jev_judgment']['label']==b for r in rows)
               for b in judges.LABELS} for a in judges.LABELS}
    result['confusion_matrix_gpt_to_jev']=matrix
    result['agreement_count']=sum(matrix[a][a] for a in judges.LABELS)
    result['agreement_rate']=result['agreement_count']/len(rows) if rows else None
    for category in sorted({r['category'] for r in rows}):
        subset=[r for r in rows if r['category']==category]
        result['by_category'][str(category)]={'total':len(subset),**{n:dict(Counter(r[n+'_judgment']['label'] for r in subset)) for n in ('gpt','jev')}}
    return result


def evaluate(root,dataset,run_models,workers):
    entry,turns,metadata,contract=prepare(root,dataset)
    if not run_models:
        print('PREPARED: edited147, fresh419 ingestion, latest policy, GPT+Jev; no API calls',flush=True);return
    from openai import OpenAI
    client=OpenAI(api_key=setting('OPENAI_API_KEY'),base_url=setting('OPENAI_BASE_URL'),timeout=90,max_retries=0)
    audit=runner.Audit(root,client);memory=None
    progress={'status':'running','phase':'preflight','completed_ingestion_turns':0,'completed_retrieval_queries':0,'completed_questions':0}
    def status(phase,**values):
        progress.update(phase=phase,updated_at=runner.now(),**values);runner.write_json(root/'run_status.json',progress)
    try:
        status('preflight')
        if not (root/'preflight.json').exists():
            audit.scope('preflight')
            probe=MiniVector(db_path=':memory:',llm=audit.extract,embedder=audit.embed,rerank_mode='none')
            try:
                added=probe.add('Ada lives in Oslo.',user_id='probe')
                if not added['results'] or not probe.search('Ada Oslo',filters={'user_id':'probe'})['results']:
                    raise ValueError('Live memory preflight failed')
            finally: probe.close()
            synthetic=judges.judge_state('Name the city.','Oslo','Oslo')
            if gpt_judge(audit,synthetic,contract)['label']!='correct': raise ValueError('GPT protocol preflight failed')
            if judges.jev_call(root,'preflight',synthetic,contract,runner.write_json,runner.read_json,runner.now)['label']!='correct':
                raise ValueError('Jev protocol preflight failed')
            runner.write_json(root/'preflight.json',{'status':'complete','completed_at':runner.now(),'excluded_from_benchmark':True})
        memory=MiniVector(db_path=str(root/'memory.sqlite3'),llm=audit.extract,embedder=audit.embed,rerank_mode='none')
        memory.store.db.execute('CREATE TABLE IF NOT EXISTS benchmark_ingestion (dialogue_id TEXT PRIMARY KEY,record TEXT NOT NULL)')
        memory.store.db.commit()
        committed={row[0]:json.loads(row[1]) for row in memory.store.db.execute('SELECT dialogue_id,record FROM benchmark_ingestion')}
        if not set(committed)<={t['dialogue_id'] for t in turns}: raise ValueError('Unknown committed source')
        for index,turn in enumerate(turns):
            sid=turn['dialogue_id']
            if sid not in committed:
                audit.scope(f'ingest_{index:04d}_{sid.replace(":","_")}')
                committed[sid]=runner.atomic_ingest(memory,turn,metadata['scope'])
                print(f'INGEST {len(committed)}/419 | {sid}',flush=True)
            if committed[sid]['turn']!=turn: raise ValueError('Committed turn changed')
            runner.write_json(root/'ingestion'/f'{index:04d}.json',committed[sid])
            status('ingestion',completed_ingestion_turns=len(committed))
        count,qcount=export_memory(root,memory,metadata['scope'])
        sources=SourceIndex(turns)
        try:
            for index,qa in enumerate(entry['qa']):
                path=root/'query_details'/f'{index:03d}.json'
                if not path.exists():
                    audit.scope(f'retrieval_q{index:03d}');trace={}
                    found=memory.search(qa['question'],top_k=20,threshold=.1,filters=metadata['scope'],explain=True,research_trace=trace)['results']
                    if trace['channels']['bm25']['status']=='error':raise RuntimeError('BM25 failed')
                    windows,source_trace=sources.select(qa['question'],found)
                    runner.write_json(path,{'question_id':f'current_q{index}','qa_index':index,'category':int(qa['category']),
                        'question':qa['question'],'gold_answer':qa.get('answer',''),'evidence':qa.get('evidence',[]),
                        'retrieval':found,'retrieval_trace':trace,'source_windows':windows,'source_trace':source_trace,
                        'answer_context':build_context(found,windows)})
                    print(f'RETRIEVAL {index+1}/147',flush=True)
                status('retrieval',completed_retrieval_queries=index+1)
        finally:sources.close()
        memory.close();memory=None
        def work(index):
            path=root/'query_details'/f'{index:03d}.json';row=runner.read_json(path,{})
            if 'generation' not in row:
                audit.scope(f'answer_q{index:03d}')
                response=audit.chat(model='gpt-4o-mini',temperature=0,messages=answer_messages(row['question'],row['answer_context']))
                answer=(response.choices[0].message.content or '').strip()
                if not answer:raise ValueError('Empty answer')
                row['generation']={'answer':answer,'usage':response.usage.model_dump() if response.usage else {}}
                runner.write_json(path,row)
            state=judges.judge_state(row['question'],row['gold_answer'],row['generation']['answer'])
            if 'gpt_judgment' not in row:
                audit.scope(f'gpt_judge_q{index:03d}');row['gpt_judgment']=gpt_judge(audit,state,contract);runner.write_json(path,row)
            if 'jev_judgment' not in row:
                row['jev_judgment']=judges.jev_call(root,f'q{index:03d}',state,contract,runner.write_json,runner.read_json,runner.now)
                runner.write_json(path,row)
            return row
        saved={i:runner.read_json(root/'query_details'/f'{i:03d}.json',{}) for i in range(147)}
        done=sum('gpt_judgment' in r and 'jev_judgment' in r for r in saved.values())
        status('answer_and_dual_judge',completed_questions=done)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks=[pool.submit(work,i) for i,row in saved.items() if not ('gpt_judgment' in row and 'jev_judgment' in row)]
            for task in as_completed(tasks):
                row=task.result();saved[row['qa_index']]=row;done+=1
                status('answer_and_dual_judge',completed_questions=done)
                print(f'DUAL EVAL {done}/147 | {row["question_id"]} | GPT={row["gpt_judgment"]["label"]} JEV={row["jev_judgment"]["label"]}',flush=True)
        rows=[saved[i] for i in range(147)];summary=summarize(rows)
        summary.update(memory_count=count,memory_query_count=qcount,ingestion_turns=419)
        runner.write_json(root/'summary.json',summary)
        runner.write_json(root/'final_results.json',{'metadata':metadata,'summary':summary,'results':rows})
        runner.write_json(root/'api_usage_summary.json',runner.collect_usage(root))
        status('complete',status='complete',memory_count=count,memory_query_count=qcount)
    except BaseException as exc:
        status('exception',status='failed',error_type=type(exc).__name__);raise
    finally:
        if memory:memory.close()
        client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--dataset',type=Path,default=DATASET)
    parser.add_argument('--run-models',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('workers must be between 1 and 8')
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    with runner.run_lock(root):evaluate(root,args.dataset.resolve(),args.run_models,args.workers)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'FAILED: {type(exc).__name__}; inspect sanitized local audit',flush=True);raise SystemExit(1)

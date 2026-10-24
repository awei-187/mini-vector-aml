"""Independent edited147 ingestion, numerical retrieval and dual-judge audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import run_locomo as runner
import run_edited_dual as experiment
import dual_judge_contract as judges
from verify_results import verify_enhanced_ranking
from mini_vector.answer_context import answer_messages,build_context
from mini_vector.source_context import SourceIndex


def verify(root):
    read=lambda name:runner.read_json(root/name,{})
    metadata=read('run_metadata.json');identity=metadata['identity']
    state=read('run_status.json')
    assert state['status']=='complete' and state['completed_ingestion_turns']==419 and state['completed_questions']==147
    assert (root/'process_exit_code.txt').read_text().strip()=='0'
    assert runner.digest(Path(identity['dataset']))==identity['dataset_sha256']
    entry=read('input_conversation_0.json')
    assert runner.read_json(Path(identity['dataset']),[])[0]==entry
    for filename,expected in identity['source_sha256'].items():
        path=Path(filename)
        relative=path.relative_to(runner.ROOT) if path.is_relative_to(runner.ROOT) else Path('external')/path.relative_to(Path('E:/LLM judge'))
        assert runner.digest(root/'source_code'/relative)==expected
    turns=runner.reference.get_turns(entry)
    assert read('source_snapshot.json')==turns and len(turns)==419
    source_by_id={t['dialogue_id']:t for t in turns}
    positions={t['dialogue_id']:i for i,t in enumerate(turns)}
    db=sqlite3.connect(f"file:{(root/'memory.sqlite3').as_posix()}?mode=ro",uri=True)
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    committed={sid:json.loads(value) for sid,value in db.execute('SELECT dialogue_id,record FROM benchmark_ingestion')}
    ids=[]
    for index,turn in enumerate(turns):
        record=read(f'ingestion/{index:04d}.json')
        assert record==committed[turn['dialogue_id']] and record['turn']==turn
        ids.extend(item['id'] for item in record['result']['results'])
    stored=[{'id':sid,'data':text,'embedding':json.loads(vector),'metadata':json.loads(meta)}
            for sid,text,vector,meta in db.execute('SELECT id,data,embedding,metadata FROM memories')]
    exported=read('memories_embeddings.json')['memories']
    assert stored==exported
    facts={item['id']:item for item in exported}
    frozen_facts={item['id']:item for item in read('memory_snapshot.json')['memories']}
    assert set(frozen_facts)==set(facts)
    promoted={'user_id','agent_id','run_id','role','actor_id','attributed_to'}
    reserved=promoted|{'id','memory','hash','data','created_at','updated_at','metadata'}
    for mid,raw in facts.items():
        item=frozen_facts[mid];meta=raw['metadata']
        assert item['memory']==raw['data'] and item['hash']==hashlib.md5(raw['data'].encode()).hexdigest()
        assert item['metadata']=={k:v for k,v in meta.items() if k not in reserved}
        assert all(item[k]==v for k,v in meta.items() if k in promoted|{'created_at','updated_at'})
    assert len(ids)==len(set(ids))==len(facts) and set(ids)==set(facts)
    assert db.execute('SELECT count(*) FROM history WHERE event="ADD"').fetchone()[0]==len(facts)
    assert db.execute('SELECT count(*) FROM memory_fts').fetchone()[0]==len(facts)
    queries=[{'memory_id':mid,'position':pos,'query':text,'embedding':json.loads(vector)}
             for mid,pos,text,vector in db.execute('SELECT memory_id,position,query,embedding FROM memory_queries ORDER BY memory_id,position')]
    assert queries==read('memory_queries_embeddings.json')['queries']
    norms={mid:sum(x*x for x in item['embedding']) for mid,item in facts.items()}
    qnorms={(q['memory_id'],q['position']):sum(x*x for x in q['embedding']) for q in queries}
    contract=read('judge_contract.json');final=read('final_results.json')
    assert contract==judges.load_contract()
    rows=final['results'];assert len(rows)==147
    source_index=SourceIndex(turns)
    try:
        for index,(qa,row) in enumerate(zip(entry['qa'],rows)):
            assert row==read(f'query_details/{index:03d}.json')
            assert row['qa_index']==index and row['question_id']==f'current_q{index}'
            assert row['question']==qa['question'] and row['gold_answer']==qa.get('answer','') and row['category']==int(qa['category'])
            trace=row['retrieval_trace'];retrieved=row['retrieval']
            assert identity['rerank_mode']==trace['rerank_mode']=='none'
            def successful(scope,kind):
                calls=[runner.read_json(p,{}) for p in sorted((root/'api_calls'/scope).glob('*_'+kind+'.json'))]
                valid=[c for c in calls if c['status']=='complete'];assert valid
                return valid[-1]
            emb=successful(f'retrieval_q{index:03d}','embedding')
            assert emb['request']['input']==[qa['question'].strip()]
            vector=emb['response']['data'][0]['embedding'];assert len(vector)==1536
            vnorm=sum(x*x for x in vector)
            exact=[]
            for mid,item in facts.items():
                data=item['embedding'];assert len(data)==1536
                denom=math.sqrt(norms[mid]*vnorm)
                score=max(0.,min(1.,sum(x*y for x,y in zip(data,vector))/denom)) if denom else 0.
                exact.append((mid,score))
            exact.sort(key=lambda x:x[1],reverse=True)
            semantic=trace['channels']['semantic']['candidates']
            assert [mid for mid,_ in exact[:80]]==[r['memory_id'] for r in semantic]
            assert all(abs(score-hit['score'])<1e-10 for (_,score),hit in zip(exact[:80],semantic))
            verify_enhanced_ranking(trace,retrieved,vector,queries,facts,qnorms)
            assert all({key:item[key] for key in frozen_facts[item['id']]}==frozen_facts[item['id']] for item in retrieved)
            windows,source_trace=source_index.select(row['question'],retrieved)
            assert windows==row['source_windows'] and source_trace==row['source_trace']
            used=set()
            for window in windows:
                anchor=window['anchor']
                for message in window['messages']:
                    sid=message['source_id'];assert sid not in used;used.add(sid)
                    original=source_by_id[sid]
                    assert abs(positions[sid]-positions[anchor])<=1 and original['session_key']==source_by_id[anchor]['session_key']
                    assert message=={'source_id':sid,'speaker':original['speaker'],'observation_time':original['session_date'],'text':original['content']}
            assert row['answer_context']==build_context(retrieved,windows)
            answer=successful(f'answer_q{index:03d}','chat_completion')
            assert answer['request']=={'model':'gpt-4o-mini','temperature':0,'messages':answer_messages(row['question'],row['answer_context'])}
            assert answer['response']['choices'][0]['message']['content'].strip()==row['generation']['answer']
            state=judges.judge_state(row['question'],row['gold_answer'],row['generation']['answer'])
            gpt=successful(f'gpt_judge_q{index:03d}','chat_completion')
            assert gpt['request']['messages']==judges.gpt_messages(state,contract)
            expected=judges.parse_gpt(gpt['response']['choices'][0]['message']['content'])
            assert all(row['gpt_judgment'][key]==value for key,value in expected.items())
            jcalls=[runner.read_json(p,{}) for p in sorted((root/'jev_api'/f'q{index:03d}').glob('*.json'))]
            jev=[call for call in jcalls if call['status']=='complete'][-1]
            assert jev['request']==judges.jev_payload(state,contract)
            assert row['jev_judgment']==judges.validate_jev(jev['response'])
    finally:source_index.close();db.close()
    summary=experiment.summarize(rows)
    saved=read('summary.json')
    assert all(saved[key]==value for key,value in summary.items()) and saved==final['summary'] and final['metadata']==metadata
    assert saved['memory_count']==len(facts) and saved['memory_query_count']==len(queries)
    assert read('api_usage_summary.json')==runner.collect_usage(root)
    report={'status':'passed','verified_at':runner.now(),'ingestion_turns':419,'questions':147,
            'memory_count':len(facts),'memory_query_count':len(queries),
            'gpt_counts':summary['judges']['gpt']['counts'],'jev_counts':summary['judges']['jev']['counts'],
            'checks':['dataset and frozen source hashes','419 atomic ingestion markers and all fact IDs',
                      'SQLite integrity and fact/question vector exports','all147 semantic and question pools recomputed',
                      'BM25 normalization, RRF, exact final20 facts','same-session verbatim bounded original windows',
                      'exact gold-free answer requests and actual returned answers','same answer in GPT and Jev payloads',
                      'original four-class rubric, GPT schema and pinned Jev probability responses','summaries and API usage']}
    runner.write_json(root/'verification.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path)
    verify(p.parse_args().run_dir.resolve())

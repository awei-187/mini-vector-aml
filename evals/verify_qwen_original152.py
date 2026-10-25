"""Independent original152 frozen-pool, historical-prompt and API verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import run_locomo as runner
import run_qwen_original152 as run
import run_qwen_ablation as common


def completed(root,folder):
    rows=[runner.read_json(p,{}) for p in sorted((root/folder).glob('*.json'))]
    valid=[r for r in rows if r.get('status')=='complete']
    assert valid,folder
    return valid[-1]


def verify(root):
    read=lambda n:runner.read_json(root/n,{})
    manifest=read('manifest.json');identity=manifest['identity']
    assert read('run_status.json')['status']=='complete' and read('run_status.json')['completed']==456
    for name,expected in identity['baseline_inputs'].items():assert runner.digest(Path(name))==expected
    for name,expected in identity['source_sha256'].items():assert runner.digest(root/'source_code'/name)==expected
    for name,expected in manifest['frozen_sha256'].items():assert runner.digest(root/name)==expected
    base=Path(identity['baseline']);old=runner.read_json(base/'final_results.json',{})
    facts={r['id']:r for r in runner.read_json(base/'memory_snapshot.json',{})['memories']}
    frozen=read('frozen_inputs.json');assert len(frozen)==152 and len(facts)==1711
    arms={a:[] for a in run.ARMS}
    for item,previous in zip(frozen,old['results']):
        i=item['qa_index'];assert i==previous['qa_index']
        assert all(item[k]==previous[k] for k in ('question_id','question','gold_answer','category'))
        scores={}
        for channel in previous['retrieval_trace']['channels'].values():
            assert channel['rerank']['output_ids']==channel['eligible_ids']
            for rank,mid in enumerate(channel['eligible_ids'],1):scores[mid]=scores.get(mid,0.)+1/(60+rank)
        order=sorted(scores,key=lambda mid:(-scores[mid],mid))
        assert order==previous['retrieval_trace']['stages']['combined_order']
        candidates=item['candidates'];ids=[c['id'] for c in candidates]
        assert ids==order[:60] and len(ids)==60 and len(set(ids))==60
        assert item['pool_sha256']==common.fingerprint(candidates)
        for c in candidates:assert {k:c[k] for k in facts[c['id']]}==facts[c['id']]
        for arm in run.ARMS:
            row=read(f'{arm}/query_details/{i:03d}.json');arms[arm].append(row)
            assert all(row[k]==item[k] for k in ('question_id','qa_index','question','gold_answer','category','pool_sha256'))
            expected=ids
            if arm!='none':
                api=completed(root,f'qwen_api/{arm}/q{i:03d}')
                assert api['request']=={'model':'qwen3-rerank','input':{
                    'query':item['question'],'documents':[c['memory'] for c in candidates]},'parameters':{
                    'top_n':60,'return_documents':False,'instruct':identity['instructions'][arm]}}
                raw=api['response'];assert not raw.get('error') and not raw.get('code')
                values=raw['output']['results'];assert len(values)==60
                mapping={r['index']:r['relevance_score'] for r in values}
                assert set(mapping)==set(range(60)) and all(type(r['index']) is int for r in values)
                assert all(type(v) in (int,float) and 0<=v<=1 for v in mapping.values())
                expected=[ids[j] for j in sorted(range(60),key=lambda j:-mapping[j])]
                trace=row['rerank_trace'];assert trace['status']=='succeeded'
                assert trace['input_ids']==ids and trace['output_ids']==expected
                assert trace['scores']==[{'memory_id':mid,'score':mapping[j]} for j,mid in enumerate(ids)]
            else:assert row['rerank_trace']=={'status':'disabled','api_calls':0}
            assert row['retrieval']==[candidates[ids.index(mid)] for mid in expected[:20]]
            context=runner.reference.build_context(row['retrieval'])
            assert row['answer_context']==context
            if arm=='none':assert context==previous['answer_context']
            request=run.historical_request('answer',item['question'],context=context)
            key=common.fingerprint(request);assert row['answer_request_sha256']==key
            assert read(f'request_cache/answer/{key}.json')=={'request':request,'result':row['generation']}
            api=completed(root,'api_calls/answer_'+key);assert api['request']==request
            assert api['response']['choices'][0]['message']['content'].strip()==row['generation']['answer']
            first=run.historical_request('judge',item['question'],gold=item['gold_answer'],answer=row['generation']['answer'])
            retry=run.historical_request('judge',item['question'],gold=item['gold_answer'],answer=row['generation']['answer'],retry=True)
            key=common.fingerprint(first);assert row['judge_request_sha256']==key
            assert read(f'request_cache/judge/{key}.json')=={'request':first,'result':row['judgment']}
            api=completed(root,'api_calls/judge_'+key);assert api['request'] in (first,retry)
            response=json.loads(api['response']['choices'][0]['message']['content'])
            label=str(response['is_correct']).strip().upper()
            if label not in ('CORRECT','WRONG'):
                tokens=set(re.findall(r'\b(CORRECT|WRONG)\b',label));assert len(tokens)==1;label=tokens.pop()
            assert row['judgment']['label']==label and row['judgment']['correct']==(label=='CORRECT')
    comparison=read('comparison.json');assert comparison==run.summarize(arms)
    for arm,rows in arms.items():
        assert comparison['arms'][arm]['correct']==sum(r['judgment']['label']=='CORRECT' for r in rows)
    assert read('api_usage_summary.json')==common.collect_all_usage(root)
    assert not list((root/'api_calls').rglob('*_embedding.json'))
    assert not (root/'jev_api').exists()
    report={'status':'passed','verified_at':runner.now(),'questions':152,'arms':3,'same_pool_verified':456,
            'qwen_scored_order_verified':304,'historical_answer_and_judge_requests_verified':912,
            'baseline_contexts_exact':152,'frozen_source_files':len(identity['source_sha256']),
            'no_ingestion_embedding_or_jev':True}
    runner.write_json(root/'verification.json',report);print(json.dumps(report),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);verify(p.parse_args().root.resolve())

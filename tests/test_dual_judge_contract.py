"""Judge boundaries and typed protocols; no external calls."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

spec=importlib.util.spec_from_file_location('edited_dual_contract',Path(__file__).resolve().parents[1]/'evals/dual_judge_contract.py')
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)


class DualJudgeContractTests(unittest.TestCase):
    def response(self):
        return {'model':'typesafe/jev-1.13-20260917','answers':{'answer_quality':{
            'type':'choice','choice':'3','probabilities':{'0':0.,'1':0.,'2':0.,'3':1.},'confidence':1.}},'usage':{}}

    def test_four_classes_are_categories_not_rounded_scores(self):
        for label in contract.LABELS:
            self.assertEqual(contract.parse_gpt(json.dumps({'label':label,'reason':'说明'}))['label'],label)
        for wrong in ({'label':'CORRECT','reason':'x'},{'label':'correct','reason':''},
                      {'label':'correct','reason':'x','score':1},{'score':.7}):
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):contract.parse_gpt(json.dumps(wrong))

    def test_both_judges_get_only_same_original_question_gold_and_answer(self):
        rules=contract.load_contract();state=contract.judge_state('query',2022,'candidate')
        messages=contract.gpt_messages(state,rules)
        self.assertEqual(json.loads(messages[-1]['content'])['input'],state)
        self.assertEqual(contract.jev_payload(state,rules)['state'],state)
        self.assertEqual(set(state),{'query','gold_answer','candidate_answer'})
        for function in (contract.gpt_messages,contract.jev_payload):
            with self.assertRaises(ValueError):function({**state,'retrieved_memories':['private']},rules)

    def test_jev_pin_distribution_and_confidence_are_strict(self):
        self.assertEqual(contract.validate_jev(self.response())['label'],'correct')
        bad=self.response();bad['model']='changed-model'
        with self.assertRaises(RuntimeError):contract.validate_jev(bad)
        changes=[('choice','m03'),('confidence',float('nan')),('probabilities',{'0':.5,'1':.5,'2':.5,'3':.5})]
        for key,value in changes:
            bad=copy.deepcopy(self.response());bad['answers']['answer_quality'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):contract.validate_jev(bad)

    def test_confusion_summary_uses_same_147_rows_without_label_weighting(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evals'))
        import run_edited_dual as run
        rows=[{'category':1,'gpt_judgment':{'label':'correct'},'jev_judgment':{'label':'partial'}},
              {'category':2,'gpt_judgment':{'label':'incorrect'},'jev_judgment':{'label':'incorrect'}}]
        summary=run.summarize(rows)
        self.assertEqual(summary['judges']['gpt']['llm_j'],.5)
        self.assertEqual(summary['judges']['jev']['llm_j'],0.)
        self.assertEqual(summary['agreement_rate'],.5)

    def test_saved_valid_gpt_response_recovers_without_a_paid_repeat(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evals'))
        import run_edited_dual as run
        rules=contract.load_contract();state=contract.judge_state('query','gold','answer')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'api_calls/gpt_judge_q000/1_chat_completion.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'status':'complete','request':{'messages':contract.gpt_messages(state,rules)},
                'response':{'choices':[{'message':{'content':'{"label":"correct","reason":"cached"}'}}],
                            'usage':{'total_tokens':10}}}))
            audit=SimpleNamespace(root=root,local=SimpleNamespace(scope='gpt_judge_q000'),chat=Mock())
            self.assertEqual(run.gpt_judge(audit,state,rules)['label'],'correct')
            audit.chat.assert_not_called()

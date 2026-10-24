"""Existing four-class answer rubric and strict Jev Choice transport.

No provider requests occur on import. Judge inputs are exactly query, original
gold_answer and the same saved candidate_answer; memory/context is excluded.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

WORK = Path('E:/LLM judge/mem0_locomo_dual_eval')
GPT_PROMPT = WORK / 'prompts/answer_direct.txt'
MODELS = WORK / 'src/dual_judge/models.py'
JEV_RUBRIC = WORK / 'jev_answer_eval/prepared/rubric.json'
JEV_ENV = Path('E:/AML/jev-reranker-study/.env')
LABELS = ('correct', 'partial', 'missing', 'incorrect')
CODE_TO_LABEL = {'0': 'incorrect', '1': 'missing', '2': 'partial', '3': 'correct'}


def load_contract():
    spec = importlib.util.spec_from_file_location('frozen_dual_judge_models', MODELS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {'gpt_system': GPT_PROMPT.read_text(encoding='utf-8-sig'),
            'gpt_schema': module.DirectAnswerResult.model_json_schema(),
            'jev_rubric': json.loads(JEV_RUBRIC.read_text(encoding='utf-8-sig'))}


def judge_state(question, gold, answer):
    return {'query': question, 'gold_answer': gold, 'candidate_answer': answer}


def gpt_messages(state, contract):
    if set(state) != {'query', 'gold_answer', 'candidate_answer'}:
        raise ValueError('Judge state isolation violation')
    return [{'role': 'system', 'content': contract['gpt_system']},
            {'role': 'user', 'content': json.dumps({'input': state, 'output_schema': contract['gpt_schema']}, ensure_ascii=False)}]


def parse_gpt(text):
    row = json.loads(text)
    if (not isinstance(row, dict) or set(row) != {'label', 'reason'} or row['label'] not in LABELS
            or not isinstance(row['reason'], str) or not row['reason'].strip()):
        raise ValueError('Invalid four-class GPT judgment')
    return row


def jev_settings():
    values = {}
    for line in JEV_ENV.read_text(encoding='utf-8-sig').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('\"\'')
    endpoint = values.get('JEV_ENDPOINT', '')
    url = urlsplit(endpoint)
    credential = values.get('JEV_API_KEY_ENV') or 'OPENROUTER_API_KEY'
    if (url.scheme != 'https' or url.hostname != 'openrouter.ai' or url.path != '/api/v1/systemone'
            or url.username or url.password or url.query or url.fragment):
        raise ValueError('Unexpected Jev provider')
    if (values.get('JEV_MODEL') != 'typesafe/jev-1.13'
            or values.get('JEV_EXPECTED_RESOLVED_MODEL') != 'typesafe/jev-1.13-20260917'
            or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', credential) or not values.get(credential)):
        raise ValueError('Missing Jev credential or wrong model pin')
    return {'endpoint': endpoint, 'model': 'typesafe/jev-1.13',
            'expected_model': 'typesafe/jev-1.13-20260917'}, values[credential]


def jev_payload(state, contract):
    if set(state) != {'query', 'gold_answer', 'candidate_answer'}:
        raise ValueError('Judge state isolation violation')
    rubric = contract['jev_rubric']
    return {'model': 'typesafe/jev-1.13', 'state': state, 'questions': {'answer_quality': {
        'type': 'choice', 'instructions': rubric['instructions'], 'criteria': rubric['criteria']}}}


def validate_jev(response):
    if not isinstance(response, dict) or response.get('error'):
        raise ValueError('Invalid Jev response')
    if response.get('model') != 'typesafe/jev-1.13-20260917':
        raise RuntimeError('Jev model pin mismatch')
    if set(response.get('answers', {})) != {'answer_quality'}:
        raise ValueError('Invalid Jev question identity')
    decision = response['answers']['answer_quality']
    code, probabilities = decision.get('choice'), decision.get('probabilities')
    if (decision.get('type') != 'choice' or not isinstance(code,str) or code not in CODE_TO_LABEL
            or not isinstance(probabilities, dict) or set(probabilities) != set(CODE_TO_LABEL)):
        raise ValueError('Invalid Jev categorical response')
    for value in [*probabilities.values(), decision.get('confidence')]:
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid Jev probability')
    if abs(sum(probabilities.values())-1) > .001 or probabilities[code]+1e-6 < max(probabilities.values()):
        raise ValueError('Invalid Jev distribution')
    return {'label': CODE_TO_LABEL[code], 'numeric_code': int(code), 'probabilities': probabilities,
            'confidence': decision['confidence'], 'resolved_model': response['model'],
            'usage': response.get('usage') or {}}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def jev_call(root, name, state, contract, write, read, now):
    settings, key = jev_settings()
    body = jev_payload(state, contract)
    if len(json.dumps(body,ensure_ascii=False).encode('utf-8'))>150000:
        raise ValueError('Jev request byte budget exceeded; never truncate')
    paths = sorted((root / 'jev_api' / name).glob('*.json'))
    for path in reversed(paths):
        previous = read(path, {})
        if previous.get('status') == 'complete':
            if previous['request'] != body:
                raise ValueError('Jev cached input mismatch')
            return validate_jev(previous['response'])
    if len(paths) >= 3:
        raise RuntimeError('Jev attempt budget exhausted')
    for attempt in range(len(paths)+1, 4):
        record = {'started_at': now(), 'request': body, 'status': 'running'}
        path = root / 'jev_api' / name / f'{attempt:02d}.json'
        write(path, record)
        fatal = False
        started = time.perf_counter()
        try:
            request = urllib.request.Request(settings['endpoint'], data=json.dumps(body,ensure_ascii=False).encode('utf-8'),
                headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
            with urllib.request.build_opener(NoRedirect).open(request, timeout=120) as response:
                raw = json.loads(response.read().decode('utf-8'))
            raw = json.loads(json.dumps(raw,ensure_ascii=False).replace(key,'[REDACTED]'))
            record['response'] = raw
            result = validate_jev(raw)
            record.update(status='complete', response=raw)
            return result
        except urllib.error.HTTPError as exc:
            record.update(status='failed', error_type='HTTPError', http_status=exc.code)
            fatal = exc.code < 500 and exc.code != 429
            exc.close()
        except RuntimeError:
            record.update(status='failed', error_type='ModelPinMismatch')
            fatal = True
        except (urllib.error.URLError, OSError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            record.update(status='failed', error_type=type(exc).__name__)
        finally:
            record.update(finished_at=now(), latency_ms=(time.perf_counter()-started)*1000)
            write(path, record)
        if fatal or attempt == 3:
            raise RuntimeError('Jev provider or protocol failed; inspect sanitized local audit')
        time.sleep(2**(attempt-1))

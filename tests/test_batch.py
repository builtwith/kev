"""Batch API contract and equivalence, with deterministic logits and no weights."""
import pytest
import torch
from fastapi.testclient import TestClient
from kev import serve


class Model:
    def __init__(self): self.batches = []

    def encode(self, tok, rec, **kwargs):
        if rec['state'] == 'invalid': raise ValueError('invalid input')
        return {'ids': list(range(len(rec['state']) + 1)), 'seg': [0], 'rec': rec}

    def forward_batch(self, encs):
        assert not torch.is_grad_enabled()
        self.batches.append(len(encs))
        return [[torch.arange(len(q['options']), dtype=torch.float32) * len(e['ids']) / 10
                 for q in e['rec']['questions']] for e in encs]

    def probs(self, enc):
        with torch.no_grad(): return [z.softmax(-1) for z in self.forward_batch([enc])[0]]


@pytest.fixture
def client(monkeypatch):
    model = Model()
    monkeypatch.setitem(serve.STATE, 'model', model)
    monkeypatch.setitem(serve.STATE, 'dev', 'cpu')
    monkeypatch.setattr(serve, 'output_tokens', lambda tok, answers: 7)
    with TestClient(serve.app) as client:
        yield client, model


QUESTIONS = {
    'category': {'type': 'choice', 'instructions': 'Category?', 'criteria': {'a': None, 'b': None}},
    'yes': {'type': 'noul', 'instructions': 'Yes?'},
    'score': {'type': 'score', 'instructions': 'Level?', 'criteria': ['low', 'mid', 'high']},
}


@pytest.mark.parametrize('temperature', [1.0, 2.0])
def test_batch_matches_single_and_chunks(client, monkeypatch, temperature):
    c, model = client
    monkeypatch.setattr(serve, 'TEMPERATURE', temperature)
    states = ['a', 'longer state', 'third', 'x', 'last']
    r = c.post('/v1/systemone/batch', json={'states': states, 'questions': QUESTIONS, 'batch_size': 2})
    assert r.status_code == 200
    assert model.batches == [2, 2, 1]
    assert r.json()['latency_ms'] >= 0
    for state, result in zip(states, r.json()['results'], strict=True):
        single = c.post('/v1/systemone', json={'state': state, 'questions': QUESTIONS}).json()
        assert result == {k: v for k, v in single.items() if k != 'latency_ms'}


@pytest.mark.parametrize('changes', [{'states': []}, {'states': ['s'] * 65}, {'batch_size': 0},
                                     {'batch_size': 33}, {'batch_size': 1.5}, {'questions': {}}, {'states': ['invalid']}])
def test_invalid_batch_does_not_run_model(client, changes):
    c, model = client
    r = c.post('/v1/systemone/batch', json={'states': ['s'], 'questions': QUESTIONS, **changes})
    assert r.status_code == 422
    assert not model.batches


def test_date_facts(client, monkeypatch):
    c, model = client
    monkeypatch.setattr(serve, 'DATE_FACTS', True)
    monkeypatch.setattr(serve, 'with_date_facts', lambda state: state + ' facts')
    r = c.post('/v1/systemone/batch', json={'states': ['s'], 'questions': QUESTIONS})
    assert r.json()['results'][0]['usage']['input_tokens'] == len('s facts') + 1


def test_oom_explains_how_to_retry(client, monkeypatch):
    c, model = client
    def oom(encs): raise torch.OutOfMemoryError('oom')
    monkeypatch.setattr(model, 'forward_batch', oom)
    r = c.post('/v1/systemone/batch', json={'states': ['s'], 'questions': QUESTIONS})
    assert r.status_code == 503
    assert 'reduce batch_size' in r.json()['detail']


@pytest.mark.parametrize("endpoint,payload", [
    ("/v1/systemone", {"state": "a", "questions": QUESTIONS}),
    ("/v1/systemone/batch", {"states": ["a", "longer state"], "questions": QUESTIONS}),
])
def test_brotli_matches_plain_json(client, endpoint, payload):
    import brotli
    import json
    c, model = client
    plain = c.post(endpoint, json=payload)
    compressed = c.post(endpoint, content=brotli.compress(json.dumps(payload).encode()),
                        headers={"Content-Type": "application/json", "Content-Encoding": "br"})
    assert compressed.status_code == plain.status_code == 200
    a, b = plain.json(), compressed.json()
    a.pop("latency_ms"); b.pop("latency_ms")
    assert a == b

@pytest.mark.parametrize('endpoint,payload', [
    ('/v1/systemone', {'state': 'a'}),
    ('/v1/systemone/batch', {'states': ['a', 'b']}),
    ('/v1/systemone/separate', {'state': 'a'}),
    ('/v1/systemone/permute', {'request': {'state': 'a'}, 'question': 'category', 'n_perm': 2}),
])
def test_question_url_matches_inline(client, monkeypatch, endpoint, payload):
    import io
    import json
    from unittest.mock import Mock
    from kev import question_urls
    c, _ = client
    question_urls._cache.clear()
    opener = Mock()
    opener.open.side_effect = lambda *a, **k: io.BytesIO(json.dumps(QUESTIONS).encode())
    monkeypatch.setattr(question_urls, 'build_opener', lambda *a: opener)
    def body(questions):
        if 'request' in payload:
            return {**payload, 'request': {**payload['request'], 'questions': questions}}
        return {**payload, 'questions': questions}
    def without_timings(value):
        if isinstance(value, dict):
            return {k: without_timings(v) for k, v in value.items() if k != 'latency_ms'}
        if isinstance(value, list): return [without_timings(v) for v in value]
        return value
    inline = c.post(endpoint, json=body(QUESTIONS))
    assert inline.status_code == 200
    for _ in range(2):
        linked = c.post(endpoint, json=body('https://cloud.builtwith.jp/raw/kev/questions3.json'))
        assert linked.status_code == 200, linked.text
        assert without_timings(linked.json()) == without_timings(inline.json())
    assert opener.open.call_count == 1
    question_urls._cache.clear()

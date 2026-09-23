from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import pytest
from kev.model import encode, user_tokens
from kev.question_cache import QuestionTokenCache

class Tokenizer:
    def __init__(self): self.calls=[]
    def __call__(self,text,**kw): self.calls.append(text);return SimpleNamespace(input_ids=list(text.encode()))
    def convert_tokens_to_ids(self,t): return 1000+sum(map(ord,t))

def record(state='some page'):
    return {'state':state,'questions':[{'instr':'Choose <|fim_prefix|> safely','options':['shop','agency','other'],'label':0}]}

@pytest.mark.parametrize('isolate',[False,True])
def test_cached_encoding_is_identical_and_never_caches_state(isolate):
    tok=Tokenizer();cache=QuestionTokenCache(user_tokens)
    for state in ['page one','page two','page one']:
        r=record(state);expected=encode(tok,r,option_isolation=isolate)
        assert encode(tok,r,option_isolation=isolate,question_cache=cache)==expected
    assert cache.misses==1 and cache.hits==2
    assert len(cache.entries)==1
    assert tok.calls.count('page one')==4

def test_question_order_and_tokenizer_identity_are_in_cache_key():
    cache=QuestionTokenCache(user_tokens);a=Tokenizer();b=Tokenizer()
    x=cache.get(a,'Question',['one','two']);y=cache.get(a,'Question',['two','one'])
    assert x[1]==tuple(reversed(y[1]))
    cache.get(b,'Question',['one','two']);assert cache.misses==3

def test_cache_is_bounded_and_does_not_retain_oversized_questions():
    cache=QuestionTokenCache(user_tokens,max_entries=2,max_tokens=20,max_chars=30);tok=Tokenizer()
    for s in ['one','two','three']:cache.get(tok,s,['yes','no'])
    assert len(cache.entries)==2
    cache.get(tok,'x'*31,['yes']);cache.get(tok,'x'*19,['yes']);assert len(cache.entries)==2
    assert all(k[1] in ['two','three'] for k in cache.entries)

def test_branch_limit_and_special_token_sanitization_still_apply():
    tok=Tokenizer();cache=QuestionTokenCache(user_tokens)
    with pytest.raises(ValueError,match='branch too long'):encode(tok,record(),max_branch=10,question_cache=cache)
    assert all('<|fim_prefix|>' not in text for text in tok.calls)

def test_concurrent_requests_cannot_corrupt_cached_values():
    tok=Tokenizer();cache=QuestionTokenCache(user_tokens)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(lambda _:cache.get(tok,'same',['a','b']),range(50)))
    assert all(v==values[0] for v in values);assert len(cache.entries)==1
    with pytest.raises(TypeError):values[0][0][0]=99


@pytest.mark.parametrize('cached', [False, True])
def test_inference_reserves_full_branch_before_truncating_page(cached):
    tok = Tokenizer()
    cache = QuestionTokenCache(user_tokens) if cached else None
    rec = {'state': 'x' * 9000, 'questions': [
        {'instr': 'q', 'options': ['a' * 1436], 'label': 0}]}
    # q marker + instruction + option delimiters/text + decide = 1441.
    with pytest.raises(ValueError, match='branch too long: 1441'):
        encode(tok, rec, max_state=8192, max_branch=8192, question_cache=cache)
    actual = encode(tok, rec, max_state=8192, max_branch=8192,
                    question_cache=cache, fit_state_to_branch=True)
    assert len(actual['ids']) == 8192
    assert actual['seg'].count(0) == 6751
    assert actual['state_truncated']
    expected = encode(tok, {**rec, 'state': 'x' * 6750}, max_state=8192,
                      max_branch=8192, question_cache=cache)
    assert {k:v for k,v in actual.items() if k != 'state_truncated'} == {
        k:v for k,v in expected.items() if k != 'state_truncated'}


def test_inference_budget_uses_longest_question_and_preserves_short_inputs():
    tok = Tokenizer()
    rec = record('short')
    assert encode(tok, rec) == encode(tok, rec, fit_state_to_branch=True)
    rec = {'state': 'x' * 100, 'questions': [
        {'instr': 'q', 'options': ['a'], 'label': 0},
        {'instr': 'q', 'options': ['b' * 20], 'label': 0}]}
    enc = encode(tok, rec, max_state=100, max_branch=50, fit_state_to_branch=True)
    assert enc['seg'].count(0) == 25
    assert max(enc['pos']) == 49
    assert enc['state_truncated']
    with pytest.raises(ValueError, match='state exceeds'):
        encode(tok, rec, max_state=100, max_branch=50, strict=True, fit_state_to_branch=True)
    with pytest.raises(ValueError, match='branch too long'):
        encode(tok, rec, max_branch=25, fit_state_to_branch=True)

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

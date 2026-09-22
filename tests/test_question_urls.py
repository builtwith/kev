import io
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
import brotli
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kev.api import SystemOneRequest, SystemOneBatchRequest
from kev.compression import BrotliRoute
from kev import question_urls as urls

URL = 'https://cloud.builtwith.jp/raw/kev/questions3.json'
QUESTIONS = {'health': {'type': 'noul', 'instructions': 'Is the sky blue?'}}

@pytest.fixture(autouse=True)
def reset_cache():
    urls._cache.clear()
    yield
    urls._cache.clear()

@pytest.fixture
def download(monkeypatch):
    opener = Mock()
    opener.open.side_effect = lambda *a, **k: io.BytesIO(json.dumps(QUESTIONS).encode())
    monkeypatch.setattr(urls, 'build_opener', lambda *a: opener)
    return opener.open

def test_concurrent_download_once_and_independent_results(download):
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(urls.resolve_questions, [URL] * 16))
    assert download.call_count == 1
    values[0]['health'].instructions = 'changed'
    assert urls.resolve_questions(URL)['health'].instructions == 'Is the sky blue?'

@pytest.mark.parametrize('url', ['file:///tmp/q.json', 'http://cloud.builtwith.jp/q.json',
    'https://169.254.169.254/q.json', 'https://cloud.builtwith.jp.evil.test/q.json',
    'https://user:pass@cloud.builtwith.jp/q.json', 'https://cloud.builtwith.jp:444/q.json'])
def test_disallowed_urls(url, download):
    with pytest.raises(urls.HTTPException) as exc:
        urls.resolve_questions(url)
    assert exc.value.status_code == 422
    download.assert_not_called()

@pytest.mark.parametrize('raw,status', [(b'no json', 422), (b'{}', 422),
    (b'{"bad":{"type":"choice","instructions":"x","criteria":{}}}', 422),
    (b'x' * 1025, 413)])
def test_invalid_documents_not_cached(raw, status, download, monkeypatch):
    monkeypatch.setattr(urls, 'MAX_BYTES', 1024)
    download.side_effect = lambda *a, **k: io.BytesIO(raw)
    with pytest.raises(urls.HTTPException) as exc:
        urls.resolve_questions(URL)
    assert exc.value.status_code == status
    assert not urls._cache

def test_download_failure_retries(download):
    download.side_effect = urls.URLError('offline')
    with pytest.raises(urls.HTTPException) as exc:
        urls.resolve_questions(URL)
    assert exc.value.status_code == 502
    download.side_effect = lambda *a, **k: io.BytesIO(json.dumps(QUESTIONS).encode())
    assert urls.resolve_questions(URL)
    assert download.call_count == 2

def test_lru_eviction(download, monkeypatch):
    monkeypatch.setattr(urls, 'CACHE_SIZE', 2)
    for suffix in ['?v=1', '?v=2', '?v=1', '?v=3']:
        urls.resolve_questions(URL + suffix)
    assert list(urls._cache) == [URL + '?v=1', URL + '?v=3']
    assert download.call_count == 3

def test_redirects_rejected():
    assert urls.NoRedirects().redirect_request(None, None, 302, '', {}, 'http://127.0.0.1') is None

def test_http_inline_url_and_brotli(download):
    app = FastAPI()
    app.router.route_class = BrotliRoute
    @app.post('/single')
    def single(req: SystemOneRequest):
        return urls.resolve_questions(req.questions)
    @app.post('/batch')
    def batch(req: SystemOneBatchRequest):
        return urls.resolve_questions(req.questions)
    client = TestClient(app)
    for route, state in [('/single', {'state': 'blue sky'}), ('/batch', {'states': ['blue sky']})]:
        for questions in [QUESTIONS, URL]:
            raw = json.dumps({**state, 'questions': questions}).encode()
            response = client.post(route, content=brotli.compress(raw), headers={
                'Content-Type': 'application/json', 'Content-Encoding': 'br'})
            assert response.status_code == 200, response.text
            assert response.json()['health']['instructions'] == QUESTIONS['health']['instructions']
    assert download.call_count == 1

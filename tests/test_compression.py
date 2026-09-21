"""Request decoding tests require no model weights."""
import json
import subprocess
import brotli
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from kev.compression import BrotliRoute

app = FastAPI()
app.router.route_class = BrotliRoute

@app.post("/echo")
async def echo(payload: dict, request: Request):
    assert await request.body() == await request.body()
    return payload

client = TestClient(app)
PAYLOAD = {"states": [{"description": "caf\u00e9 \U0001f600 " * 200}], "questions": {}}
RAW = json.dumps(PAYLOAD, ensure_ascii=False).encode()

def post(body, encoding="br"):
    return client.post("/echo", content=body, headers={"Content-Type": "application/json", "Content-Encoding": encoding})

def test_plain_and_brotli():
    for body, encoding in [(RAW, "identity"), (brotli.compress(RAW), "br"), (brotli.compress(RAW), "BR")]:
        response = post(body, encoding)
        assert response.status_code == 200
        assert response.json() == PAYLOAD
    assert client.post("/echo", json=PAYLOAD).json() == PAYLOAD

@pytest.mark.parametrize("body", [b"invalid", b"", brotli.compress(RAW)[:-1], brotli.compress(RAW) + b"junk"])
def test_invalid_streams(body):
    assert post(body).status_code == 400

@pytest.mark.parametrize("encoding", ["gzip", "br, br", "br, gzip"])
def test_unsupported(encoding):
    assert post(RAW, encoding).status_code == 415

def test_limits(monkeypatch):
    monkeypatch.setattr("kev.compression.MAX_BODY_BYTES", 128)
    assert post(brotli.compress(b"a" * 10000)).status_code == 413
    assert post(bytes(range(256))).status_code == 413

def test_invalid_json():
    assert post(brotli.compress(b"not json")).status_code == 422

def test_node_brotli_interoperability():
    result = subprocess.run(["node", "-e", "const z=require('node:zlib'); process.stdout.write(z.brotliCompressSync(Buffer.from(process.argv[1]), {params:{[z.constants.BROTLI_PARAM_QUALITY]:4}}));", RAW.decode()], capture_output=True, check=True)
    assert len(result.stdout) < len(RAW)
    response = post(result.stdout)
    assert response.status_code == 200
    assert response.json() == PAYLOAD

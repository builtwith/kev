import types
import pytest
import torch
from kev.inference import enable_pointwise_compile

class Qwen3_5RMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.ones(4))
    def forward(self, x): return x * self.weight

def model(device='cuda', training=False, dtype=torch.bfloat16):
    lm = torch.nn.Sequential(Qwen3_5RMSNorm(), torch.nn.Linear(4, 4))
    lm.to(dtype); lm.train(training)
    return types.SimpleNamespace(lm=lm, device=device, training=training)

def test_compile_is_opt_in_cuda_inference_only(monkeypatch):
    monkeypatch.delenv('KEV_COMPILE_POINTWISE', raising=False)
    def forbidden(*a, **kw): pytest.fail('should not compile')
    assert enable_pointwise_compile(model(), compiler=forbidden) == 0
    assert enable_pointwise_compile(model(dtype=torch.float32), enabled=True, compiler=forbidden) == 0
    assert enable_pointwise_compile(model('cpu'), enabled=True, compiler=forbidden) == 0
    assert enable_pointwise_compile(model(training=True), enabled=True, compiler=forbidden) == 0

def test_scoped_compilation_preserves_precision_weights_and_autograd():
    m = model(); compiled_calls = []; builds = []; before = {k: v.clone() for k,v in m.lm.state_dict().items()}
    def compiler(eager, **kwargs):
        builds.append(kwargs)
        def run(*args, **kw): compiled_calls.append(True); return eager(*args, **kw)
        return run
    assert enable_pointwise_compile(m, enabled=True, compiler=compiler) == 1
    assert builds == [{'dynamic':True, 'fullgraph':True, 'options':{'emulate_precision_casts':True}}]
    assert enable_pointwise_compile(m, enabled=True, compiler=compiler) == 1
    assert len(builds) == 1
    with torch.inference_mode(): m.lm(torch.ones(2,4,dtype=torch.bfloat16))
    assert len(compiled_calls) == 1
    m.lm(torch.ones(2,4,dtype=torch.bfloat16,requires_grad=True)).sum().backward()
    assert len(compiled_calls) == 1
    assert all(torch.equal(v,m.lm.state_dict()[k]) for k,v in before.items())

@pytest.mark.parametrize('error_name', ['Unsupported', 'FailOnRecompileLimitHit'])
def test_compiler_failure_falls_back_once(error_name):
    from torch._dynamo import exc
    error_type = getattr(exc, error_name)
    m = model(); calls=[]
    def compiler(eager, **kwargs):
        def run(*args, **kw): calls.append(True); raise error_type('unavailable compiler')
        return run
    enable_pointwise_compile(m, enabled=True, compiler=compiler)
    with torch.inference_mode():
        for _ in range(2): assert torch.equal(m.lm[0](torch.ones(2,4,dtype=torch.bfloat16)),torch.ones(2,4,dtype=torch.bfloat16))
    assert len(calls)==1

def test_oom_is_not_swallowed_as_a_compiler_failure():
    m = model()
    def compiler(eager, **kwargs):
        def run(*args, **kw): raise torch.OutOfMemoryError('oom')
        return run
    enable_pointwise_compile(m, enabled=True, compiler=compiler)
    with torch.inference_mode(), pytest.raises(torch.OutOfMemoryError): m.lm[0](torch.ones(2,4,dtype=torch.bfloat16))

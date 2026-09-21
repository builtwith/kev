"""CLI device selection without downloading model weights or datasets."""
import sys

import pytest

from kev import evaluate


def test_peak_rss_is_available_on_this_platform():
    from kev.train import peak_rss_bytes
    assert peak_rss_bytes() > 0


@pytest.mark.parametrize("cuda,mps,override,expected", [
    (True, True, None, "cuda"),
    (False, True, None, "mps"),
    (False, False, None, "cpu"),
    (True, False, "cpu", "cpu"),
])
def test_evaluate_selects_device(monkeypatch, cuda, mps, override, expected):
    monkeypatch.setattr(sys, "argv", ["kev.evaluate"] + (["--device", override] if override else []))
    monkeypatch.setattr(evaluate.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(evaluate.torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(evaluate.torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(evaluate, "build", lambda *args: [])
    monkeypatch.setattr(evaluate.torch, "load", lambda *args, **kwargs: {})

    class ReachedModelLoad(Exception):
        pass

    def load(run, device):
        assert device == expected
        raise ReachedModelLoad

    monkeypatch.setattr(evaluate, "load", load)
    with pytest.raises(ReachedModelLoad):
        evaluate.main()


@pytest.mark.parametrize("module_name", ["evaluate", "serve"])
def test_explicit_cuda_rejects_unavailable_backend(monkeypatch, module_name):
    from kev import serve
    module = evaluate if module_name == "evaluate" else serve
    monkeypatch.setattr(sys, "argv", [f"kev.{module_name}", "--device", "cuda"])
    monkeypatch.setattr(evaluate.torch.cuda, "is_available", lambda: False)
    # Serving resolves its checkpoint before loading the model.
    monkeypatch.setattr(evaluate, "resolve_run", lambda run: run)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2

"""Optional BF16 CUDA serving fusion without changing weights, tokens, or precision.

Compile only Qwen3.5's small normalization/MLP functions. The recurrent model,
attention masks, pointer head, and prefix cache stay on their existing paths.
"""
import functools
import logging
import os

import torch

_LOG = logging.getLogger(__name__)
_TARGETS = frozenset({"Qwen3_5RMSNorm", "Qwen3_5RMSNormGated", "Qwen3_5MLP"})


def _guarded_forward(module, eager, compiled):
    from torch._dynamo.exc import BackendCompilerFailed, Unsupported, FailOnRecompileLimitHit
    failed = False

    @functools.wraps(eager)
    def forward(*args, **kwargs):
        nonlocal failed
        # This optimization is for serving. Retain eager autograd/training.
        if failed or module.training or torch.is_grad_enabled():
            return eager(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except (BackendCompilerFailed, Unsupported, FailOnRecompileLimitHit) as error:
            failed = True
            _LOG.warning("Pointwise compilation unavailable for %s; using eager: %s", type(module).__name__, error)
            return eager(*args, **kwargs)
    return forward


def enable_pointwise_compile(model, *, enabled=None, compiler=None):
    """Opt in with KEV_COMPILE_POINTWISE=1; return the number of wrapped modules.

    Dynamic shapes avoid one compilation for each input sequence length.
    Preserve BF16 rounding between fused operations, rather than silently
    removing the eager downcast/upcast steps. No CUDA graphs or GPU-stack upgrade.
    """
    if enabled is None:
        enabled = os.environ.get("KEV_COMPILE_POINTWISE", "0") == "1"
    if not enabled or model.training or not str(model.device).startswith("cuda"):
        return 0
    # The exact FP32 evaluation path must stay eager. This optimization was
    # validated for the BF16 CUDA serving path only.
    if next(model.lm.parameters()).dtype != torch.bfloat16:
        return 0
    # Norms share Python code across several head widths and 1/2-row shapes.
    # Health probes also use no_grad rather than inference_mode. Eight cached
    # graphs is insufficient for these legitimate variants; keep a finite cap.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    compiler = compiler or torch.compile
    count = 0
    for module in model.lm.modules():
        if type(module).__name__ not in _TARGETS:
            continue
        count += 1
        if getattr(module, "_kev_pointwise_compiled", False):
            continue
        eager = module.forward
        compiled = compiler(eager, dynamic=True, fullgraph=True,
                            options={"emulate_precision_casts": True})
        module.forward = _guarded_forward(module, eager, compiled)
        module._kev_pointwise_compiled = True
    _LOG.info("CUDA pointwise compilation enabled for %d modules (eager precision casts preserved)", count)
    return count

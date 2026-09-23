# Upstream Compatibility

Reviewed `jaredpalmer/kev` main through `84f23b3` on September 23, 2026.
This fork selectively incorporates compatible fixes; it is not a full merge of that revision.

## Incorporated Fixes

- `a2087dc` and `5e94a28`: explicit setuptools packaging and an editable lockfile entry so installation does not fail on multiple top-level directories. The fork retains its Brotli dependency and Windows CUDA package source.
- `1ba5ca9`: serialize probabilities to four decimal places so distributions with many options stay within TypeSafe's sum tolerance. Response fields stay the same; numeric precision increases from two to four decimals.

## Deferred Changes

The remaining upstream changes include a connected serving, checkpoint-loading, model, and research refactor. A wholesale merge would remove `/api/*` endpoints, replace the `id` and `aliases` fields in `/v1/models` with `name`, change serving precision and backend defaults, and load calibration temperatures from checkpoints. These affect existing clients or inference results and require a separate migration and model-level validation.

The new serving implementation also replaces the state and loading interfaces used by this fork's batch endpoint. Its batch path would need adaptation and parity testing against the new torch and MLX implementations. Research and benchmark results from that newer implementation are not imported as claims about this fork.

Batch requests, Brotli request decoding, and shared question URLs with per-worker caching retain their existing implementation. Host allowlists remain deployment configuration, with no allowed URL hosts by default. Existing installer commit pins continue to select their previously tested revision until explicitly updated.

### Optional CUDA serving optimizations

`KEV_COMPILE_POINTWISE=1` enables selective compilation of Qwen3.5 normalization
and MLP functions during BF16 CUDA inference. It preserves explicit precision
casts and leaves weights, calibration, attention, and API behavior unchanged.
CPU, FP32, training, and autograd use eager execution. Supported compiler failures
fall back to eager execution; out-of-memory errors remain visible to callers.
This is a fork-local optimization, not an imported upstream change.

Serving also caches shared question tokens in a bounded, thread-safe LRU. State
tokens are always recomputed. Cached encoding retains identical token IDs,
positions, option isolation, and branch-limit validation.

On a fixed 96-input 4B/L4 sample, eager microbatch four reached 164.8 inputs/min
and compiled microbatch two reached 214.6 inputs/min. These configurations are
not numerically identical: one top choice changed and the maximum probability
difference was 0.0278. This is a throughput measurement, not an accuracy claim.
A trial compiling the FP32 unmerged reference differed by up to 0.000582 with
no top-choice changes; FP32 compilation is therefore excluded from this feature.
Legacy smoke-checkpoint parity tests require a checkpoint not available during
this validation. The exact FP32 reference path remains eager.

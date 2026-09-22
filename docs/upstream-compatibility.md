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

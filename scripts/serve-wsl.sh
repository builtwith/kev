#!/usr/bin/env bash
# Run from PowerShell: wsl -d Debian -- bash scripts/serve-wsl.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python="${KEV_WSL_ENV:-$HOME/.local/share/kev/venv}/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Kev's WSL environment is missing: $python" >&2
  exit 1
fi
export KEV_DTYPE="${KEV_DTYPE:-bf16}"
export PYTHONUNBUFFERED=1
# Reuse this workstation's downloaded weights when no cache override is set.
if [[ -z "${HF_HUB_CACHE:-}" && -d /mnt/c/Users/gary_w553/.cache/huggingface/hub ]]; then
  export HF_HUB_CACHE=/mnt/c/Users/gary_w553/.cache/huggingface/hub
fi
# Use Python directly: uv sync would replace the required Triton override.
exec "$python" -m kev.serve --run jaredpalmer/kev-4b --port 8009 --device cuda "$@"

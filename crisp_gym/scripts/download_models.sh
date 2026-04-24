#!/usr/bin/env bash
set -euo pipefail

# Downloads policy checkpoints into the default HF cache (HF_HOME).
# Already-downloaded files are skipped — safe to re-run.
# Requires hf_xet for fast transfer:
#   pixi run -e kilted-lerobot pip install hf_xet

REPOS=(
    "OliverHausdoerfer/ditflow_lego_simple_filtered_new_state_fixed_stats_only_wrist_v1"
    "OliverHausdoerfer/diffusion_lego_simple_filtered_new_state_fixed_stats_only_wrist_v1"
    "OliverHausdoerfer/pi05_stack_lego_simple_20260423"
)

for repo in "${REPOS[@]}"; do
    echo "=== Downloading ${repo} ==="
    pixi run -e kilted-lerobot hf download "${repo}"
done

echo "Done. Cache: ${HF_HOME:-$HOME/.cache/huggingface}"

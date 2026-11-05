#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

ARGS=(--data_root "$DATA_ROOT" --device "$DEVICE")
if [[ -n "${CKPT_PATH:-}" ]]; then
  ARGS+=(--ckpt "$CKPT_PATH")
fi

"$PYTHON_BIN" "$REPO_ROOT/OneRec_seq_infer.py" \
  --topk 10 \
  --valid_ratio 0.2 \
  --split_seed 42 \
  --eval_candidate_size 1000 \
  --eval_per_part 200 \
  --eval_parts "PartI,PartII,PartIII" \
  --seed 1234 \
  --ckpt_prefix "OneRec_seq" \
  "${ARGS[@]}" "$@"

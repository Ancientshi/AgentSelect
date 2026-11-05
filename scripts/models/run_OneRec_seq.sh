#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

ARGS=(--data_root "$DATA_ROOT" --device "$DEVICE")
if [[ -n "${BGE_FEATURE_DIR:-}" ]]; then
  ARGS+=(--bge_feature_dir "$BGE_FEATURE_DIR")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/../../OneRec_seq.py" \
  --mode sft \
  --topk 10 \
  --epochs 5 \
  --batch_size 1024 \
  --candidate_size 200 \
  --train_mask 1 \
  --cand_extra 64 \
  --rand_neg_ratio 0.25 \
  --use_nce 1 \
  --nce_lambda 0.1 \
  --nce_temp 0.07 \
  --init_tok_from_bge 1 \
  --freeze_tok_emb 1 \
  --amp 1 \
  --eval_candidate_size 1000 \
  --eval_per_part 200 \
  --eval_parts "PartI,PartII,PartIII" \
  --skip_eval 0 \
  "${ARGS[@]}" "$@"

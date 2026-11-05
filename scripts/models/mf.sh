#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

EPOCHS=11
BATCH_SIZE=4096

python "$SCRIPT_DIR/../../run_mf.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --factors 128 \
  --neg_per_pos "$NEG_PER_POS" \
  --knn_N "$KNN_N" \
  --eval_cand_size "$EVAL_CAND_SIZE" \
  --score_mode dot \
  --use_llm_id_emb 1 \
  --use_tool_id_emb 1 \
  --use_agent_id_emb 1 \
  --alpha_llm 1.0 \
  --alpha_tool 1.0 \
  --exp_name "mf_test_cleaned_dataset_use_agent_id_emb"

# ebdf315d_d6cb5ebf cleaned_pairs
# 01915b90_e6e90caa old_pairs
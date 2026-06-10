#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="ignore::FutureWarning"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

# DNN settings, aligned with the TF-IDF DNN script.
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda:0}"

TEXT_HIDDEN="${TEXT_HIDDEN:-256}"
ID_DIM="${ID_DIM:-64}"

# Feature/ID switches, aligned with DNN TF-IDF and TwoTower BGE.
USE_TOOL_ID_EMB="${USE_TOOL_ID_EMB:-${USE_TOOL_EMB:-1}}"
USE_LLM_ID_EMB="${USE_LLM_ID_EMB:-${USE_AGENT_ID_EMB:-1}}"
USE_MODEL_CONTENT_VECTOR="${USE_MODEL_CONTENT_VECTOR:-1}"
USE_TOOL_CONTENT_VECTOR="${USE_TOOL_CONTENT_VECTOR:-1}"
USE_QUERY_ID_EMB="${USE_QUERY_ID_EMB:-0}"
USE_AGENT_ID_EMB="${USE_AGENT_ID_EMB:-0}"

TOPK="${TOPK:-10}"
EVAL_CAND_SIZE="${EVAL_CAND_SIZE:-1000}"

# BGE settings, aligned with TwoTower BGE.
EMBED_BACKEND="${EMBED_BACKEND:-local}"
BGE_MODEL="${BGE_MODEL:-path_to/models/BAAI/bge-m3}"
BGE_DEVICE="${BGE_DEVICE:-$DEVICE}"
BGE_FP16="${BGE_FP16:-1}"

EMBED_URL="${EMBED_URL:-http://127.0.0.1:8500/get_embedding}"
EMBED_BATCH="${EMBED_BATCH:-64}"

# Optional cache controls.
REBUILD_FEATURE_CACHE="${REBUILD_FEATURE_CACHE:-0}"
REBUILD_TRAINING_CACHE="${REBUILD_TRAINING_CACHE:-0}"

# Optional explicit BGE feature cache dir.
# Leave empty to use data_sig + BGE settings to build a deterministic shared cache path.
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-path_to/AgentSelect/dataset/.cache/shared/features/dnn_bge_94c91bc8_01915b90}"

CMD=(
  python "$SCRIPT_DIR/../../run_dnn_bge.py"
  --data_root "$DATA_ROOT"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --text_hidden "$TEXT_HIDDEN"
  --id_dim "$ID_DIM"
  --embed_backend "$EMBED_BACKEND"
  --bge_model "$BGE_MODEL"
  --bge_device "$BGE_DEVICE"
  --bge_fp16 "$BGE_FP16"
  --embed_url "$EMBED_URL"
  --embed_batch "$EMBED_BATCH"
  --topk "$TOPK"
  --eval_cand_size "$EVAL_CAND_SIZE"
  --rebuild_feature_cache "$REBUILD_FEATURE_CACHE"
  --rebuild_training_cache "$REBUILD_TRAINING_CACHE"
  --use_query_id_emb "$USE_QUERY_ID_EMB"
  --use_llm_id_emb "$USE_LLM_ID_EMB"
  --use_tool_id_emb "$USE_TOOL_ID_EMB"
  --use_model_content_vector "$USE_MODEL_CONTENT_VECTOR"
  --use_tool_content_vector "$USE_TOOL_CONTENT_VECTOR"
  --use_agent_id_emb "$USE_AGENT_ID_EMB"
  --exp_name "bpr_dnn_bge"
)

if [[ -n "$FEATURE_CACHE_DIR" ]]; then
  CMD+=(--feature_cache_dir "$FEATURE_CACHE_DIR")
fi

"${CMD[@]}"

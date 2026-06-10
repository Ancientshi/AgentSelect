#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="ignore::FutureWarning"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

EPOCHS=4
BATCH_SIZE=512
DEVICE="cuda:0"

MAX_FEATURES="${MAX_FEATURES:-5000}"
HID="${HID:-256}"
TEMPERATURE="${TEMPERATURE:-0.07}"
EVAL_CHUNK="${EVAL_CHUNK:-8192}"
USE_TOOL_ID_EMB="${USE_TOOL_ID_EMB:-${USE_TOOL_EMB:-1}}"
USE_LLM_ID_EMB="${USE_LLM_ID_EMB:-${USE_AGENT_ID_EMB:-1}}"
USE_MODEL_CONTENT_VECTOR="${USE_MODEL_CONTENT_VECTOR:-1}"
USE_TOOL_CONTENT_VECTOR="${USE_TOOL_CONTENT_VECTOR:-1}"
USE_QUERY_ID_EMB="${USE_QUERY_ID_EMB:-0}"
TOPK="${TOPK:-10}"

EMBED_BACKEND="${EMBED_BACKEND:-local}"
BGE_MODEL="${BGE_MODEL:-path_to/models/BAAI/bge-m3}"
BGE_DEVICE="${BGE_DEVICE:-$DEVICE}"
BGE_FP16="${BGE_FP16:-1}"

EMBED_URL="${EMBED_URL:-http://127.0.0.1:8500/get_embedding}"
EMBED_BATCH="${EMBED_BATCH:-64}"

EVAL_CAND_SIZE="${EVAL_CAND_SIZE:-1000}"
AMP="${AMP:-1}"

USE_TOOL_ID_EMB="${USE_TOOL_ID_EMB:-${USE_TOOL_EMB:-1}}"
USE_LLM_ID_EMB="${USE_LLM_ID_EMB:-${USE_AGENT_ID_EMB:-1}}"
USE_MODEL_CONTENT_VECTOR="${USE_MODEL_CONTENT_VECTOR:-1}"
USE_TOOL_CONTENT_VECTOR="${USE_TOOL_CONTENT_VECTOR:-1}"
USE_QUERY_ID_EMB="${USE_QUERY_ID_EMB:-0}"
TOPK="${TOPK:-10}"

python "$SCRIPT_DIR/../../run_twotower_bge.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --embed_backend "$EMBED_BACKEND" \
  --bge_model "$BGE_MODEL" \
  --bge_device "$BGE_DEVICE" \
  --bge_fp16 "$BGE_FP16" \
  --embed_url "$EMBED_URL" \
  --embed_batch "$EMBED_BATCH" \
  --hid "$HID" \
  --temperature "$TEMPERATURE" \
  --topk "$TOPK" \
  --eval_chunk "$EVAL_CHUNK" \
  --eval_cand_size "$EVAL_CAND_SIZE" \
  --amp "$AMP" \
  --use_tool_id_emb "$USE_TOOL_ID_EMB" \
  --use_llm_id_emb "$USE_LLM_ID_EMB" \
  --use_model_content_vector "$USE_MODEL_CONTENT_VECTOR" \
  --use_tool_content_vector "$USE_TOOL_CONTENT_VECTOR" \
  --use_query_id_emb "$USE_QUERY_ID_EMB" \
  --use_agent_id_emb 0 \
  --feature_cache_dir path_to/AgentSelect/dataset/.cache/shared/features/twotower_bge_58631b56_01915b90 \
  --exp_name "two_tower_bge_use_agent_id_emb"
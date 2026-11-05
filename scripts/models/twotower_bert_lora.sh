#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

# TwoTower-BERT LoRA setting:
# - Training remains TwoTower + in-batch InfoNCE.
# - BERT encoder is updated online through lightweight LoRA adapters.
DEVICE="${DEVICE:-cuda:4}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"

TUNE_MODE="${TUNE_MODE:-lora}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-path_to/models/distilbert-base-uncased}"
POOLING="${POOLING:-cls}"
MAX_LEN="${MAX_LEN:-128}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-256}"

HID="${HID:-256}"
TEMPERATURE="${TEMPERATURE:-0.07}"
EVAL_CHUNK="${EVAL_CHUNK:-8192}"
TOPK="${TOPK:-10}"
AMP="${AMP:-0}"

UNFREEZE_LAST_N="${UNFREEZE_LAST_N:-2}"
UNFREEZE_EMB="${UNFREEZE_EMB:-1}"
GRAD_CKPT="${GRAD_CKPT:-0}"
ENCODER_LR="${ENCODER_LR:-5e-5}"
ENCODER_WEIGHT_DECAY="${ENCODER_WEIGHT_DECAY:-0.01}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGETS="${LORA_TARGETS:-q_lin,k_lin,v_lin,out_lin}"

USE_TOOL_ID_EMB="${USE_TOOL_ID_EMB:-${USE_TOOL_EMB:-1}}"
USE_LLM_ID_EMB="${USE_LLM_ID_EMB:-${USE_AGENT_ID_EMB:-1}}"
USE_MODEL_CONTENT_VECTOR="${USE_MODEL_CONTENT_VECTOR:-1}"
USE_TOOL_CONTENT_VECTOR="${USE_TOOL_CONTENT_VECTOR:-1}"
USE_QUERY_ID_EMB="${USE_QUERY_ID_EMB:-0}"
USE_AGENT_ID_EMB="${USE_AGENT_ID_EMB:-0}"
SOFT_EVAL="${SOFT_EVAL:-0}"

python "$SCRIPT_DIR/../../run_twotower_bert.py" \
  --eval_cand_size "$EVAL_CAND_SIZE" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --pretrained_model "$PRETRAINED_MODEL" \
  --max_len "$MAX_LEN" \
  --pooling "$POOLING" \
  --encode_batch_size "$ENCODE_BATCH_SIZE" \
  --hid "$HID" \
  --temperature "$TEMPERATURE" \
  --topk "$TOPK" \
  --eval_chunk "$EVAL_CHUNK" \
  --amp "$AMP" \
  --tune_mode "$TUNE_MODE" \
  --unfreeze_last_n "$UNFREEZE_LAST_N" \
  --unfreeze_emb "$UNFREEZE_EMB" \
  --grad_ckpt "$GRAD_CKPT" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_targets "$LORA_TARGETS" \
  --encoder_lr "$ENCODER_LR" \
  --encoder_weight_decay "$ENCODER_WEIGHT_DECAY" \
  --use_tool_id_emb "$USE_TOOL_ID_EMB" \
  --use_llm_id_emb "$USE_LLM_ID_EMB" \
  --use_model_content_vector "$USE_MODEL_CONTENT_VECTOR" \
  --use_tool_content_vector "$USE_TOOL_CONTENT_VECTOR" \
  --use_query_id_emb "$USE_QUERY_ID_EMB" \
  --use_agent_id_emb "$USE_AGENT_ID_EMB" \
  --exp_name "twotower_bert_${TUNE_MODE}" \
  --soft_eval "$SOFT_EVAL"

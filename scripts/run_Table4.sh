#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

N="${N:-5}"
RUNS="${RUNS:-5}"
BASE_SEED="${BASE_SEED:-20260001}"
SCORING_URL="${SCORING_URL:-http://127.0.0.1:8501/compute_scores}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/table4_$(date +%Y%m%d_%H%M%S)}"

ARGS=(--data_root "$DATA_ROOT" --scoring_url "$SCORING_URL" --device "$DEVICE" --N "$N")
if [[ -n "${MODEL_PATH:-}" ]]; then
  ARGS+=(--model_path "$MODEL_PATH")
fi
if [[ -n "${QUESTIONS_JSONL:-}" ]]; then
  ARGS+=(--questions_jsonl "$QUESTIONS_JSONL")
fi
if [[ "${USE_GPT:-1}" == "1" ]]; then
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY or use USE_GPT=0 for heuristic agent selection.}"
  ARGS+=(--use_gpt --use_gpt_tool_query)
fi

mkdir -p "$OUT_DIR"
SUMMARY_FILE="$OUT_DIR/summary.txt"
: > "$SUMMARY_FILE"

for ((i = 1; i <= RUNS; i++)); do
  SEED=$((BASE_SEED + i))
  LOG_FILE="$OUT_DIR/run_${i}_seed_${SEED}.log"
  echo "Run ${i}/${RUNS}, seed ${SEED}: ${LOG_FILE}" | tee -a "$SUMMARY_FILE"
  "$PYTHON_BIN" "$REPO_ROOT/run_Table4.py" \
    --seed "$SEED" \
    "${ARGS[@]}" "$@" 2>&1 | tee "$LOG_FILE"
done

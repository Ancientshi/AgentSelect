#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DEVICE=cpu
source "$SCRIPT_DIR/common_env.sh"

ARGS=(--data_root "$DATA_ROOT")
if [[ -n "${MODEL_PATH:-}" ]]; then
  ARGS+=(--model_path "$MODEL_PATH")
fi
if [[ -n "${MAX_FEATURES:-}" ]]; then
  ARGS+=(--max_features "$MAX_FEATURES")
fi

"$PYTHON_BIN" "$REPO_ROOT/infer_twotower_tfidf.py" \
  --device "$DEVICE" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --topk "${TOPK:-10}" \
  "${ARGS[@]}" "$@"

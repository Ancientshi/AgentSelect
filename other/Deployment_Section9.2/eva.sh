#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this evaluation.}"

"${PYTHON_BIN:-python}" "$SCRIPT_DIR/evaluator.py" \
  --run_result_path "$SCRIPT_DIR/OneAgent_results.run1.500.jsonl" \
  --out_path "$SCRIPT_DIR/gpt_rerank_alignment1.jsonl" \
  --model gpt-5.4-mini \
  --resume "$@"

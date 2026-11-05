#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON_BIN:-python}" "$SCRIPT_DIR/viz_alignment_metrics.py" \
  --inputs "$SCRIPT_DIR/gpt_rerank_alignment1.jsonl" \
  --out_dir "$SCRIPT_DIR/viz_gpt_alignment" \
  --dedup "$@"

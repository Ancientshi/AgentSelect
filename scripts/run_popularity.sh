#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
: "${ANALYSIS_SCRIPT:=${REPO_ROOT}/analysis/popularity_analysis.py}"
: "${OUTPUT_DIR:=${REPO_ROOT}/analysis/artifacts_content_pop}"

if [[ ! -f "$ANALYSIS_SCRIPT" ]]; then
  echo "Popularity analysis source is not included. Set ANALYSIS_SCRIPT to your copy of popularity_analysis.py." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

# Ensure the repo is on PYTHONPATH so agent_rec can be imported.
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"

"$PYTHON_BIN" "$ANALYSIS_SCRIPT" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

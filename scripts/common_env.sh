#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/AgentSelect}"
DEVICE="${DEVICE:-${DEFAULT_DEVICE:-cuda:0}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXP_SUFFIX="${EXP_SUFFIX:-}"

NEG_PER_POS="${NEG_PER_POS:-1}"
KNN_N="${KNN_N:-3}"
EVAL_CAND_SIZE="${EVAL_CAND_SIZE:-1000}"
ID_DIM="${ID_DIM:-32}"

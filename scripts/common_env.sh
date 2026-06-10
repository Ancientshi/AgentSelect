#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-path_to/AgentSelect/dataset}"
DEVICE="${DEVICE:-cuda:5}"
EXP_SUFFIX="${EXP_SUFFIX:-}"

NEG_PER_POS="${NEG_PER_POS:-1}"
KNN_N="${KNN_N:-3}"
EVAL_CAND_SIZE="${EVAL_CAND_SIZE:-1000}"
ID_DIM="${ID_DIM:-32}"
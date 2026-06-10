#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
MODEL_NAME="hkuds/easyrec-roberta-base"
SAVE_DIR="$SCRIPT_DIR/runs/easyrec_benchmark2_lora_run"
TRIPLES_CACHE="${TRIPLES_CACHE:-$SAVE_DIR/triples_cache.npy.gz}"

if [[ -z "$BENCHMARK_ROOT" ]]; then
  echo "Please set BENCHMARK_ROOT to the benchmark 2 dataset root before running." >&2
  echo "Example: BENCHMARK_ROOT=/path/to/benchmark\\ 2 bash run_easyrec_benchmark_eval_mulerun.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" "$SCRIPT_DIR/train_easyrec_benchmark_eval_mulerun.py" \
  --benchmark_root "$BENCHMARK_ROOT" \
  --save_dir "$SAVE_DIR" \
  --model_name "$MODEL_NAME" \
  --epochs 3 \
  --batch_size 640 \
  --accum_steps 1 \
  --lr 1e-5 \
  --max_len 192 \
  --num_workers 2 \
  --amp auto \
  --pooler_type cls \
  --temp 0.05 \
  --train_pos_topk 5 \
  --rand_neg_per_pos 1 \
  --triples_cache "$TRIPLES_CACHE"

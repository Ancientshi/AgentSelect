export OPENAI_API_KEY='xxx'

python evaluator.py \
  --run_result_path OneAgent_results.run1.500.jsonl \
  --out_path gpt_rerank_alignment1.jsonl \
  --model gpt-5.4-mini \
  --resume

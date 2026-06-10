# MuleRun Real-World Validation

This README describes the implementation details for the **MuleRun Agent Marketplace validation** used in **Section 9.1: Validation on MuleRun Agent Marketplace** of the paper.

The purpose of this experiment is to examine whether the supervision learned from **AGENTSELECT** can transfer to a realistic and unseen external agent marketplace. We use **MuleRun**, a public marketplace containing more than 100 task-oriented agents, as the validation environment.


## Validation Goal

The MuleRun validation is designed to test whether models tuned with AGENTSELECT supervision can better match open-ended user requests with real-world agent capabilities.

The main comparison reported in this section is between **EasyRec** and **EasyRec\***, where **EasyRec\*** denotes EasyRec fine-tuned with AGENTSELECT supervision.


## Repository Structure

A typical directory structure is:

```text
.
├── MuleRun_Dataset/
│   ├── ...
├── infer_BGE_MuleRun.py
├── infer_EasyRec_MuleRun.py
├── run_easyrec_benchmark_eval_mulerun.sh
└── README.md
```



## EasyRec LoRA Fine-Tuning and Evaluation

This part fine-tunes EasyRec with AGENTSELECT supervision and evaluates the tuned model on MuleRun.

### Training

Set the benchmark root path and run:

```bash
BENCHMARK_ROOT=/path/to/benchmark
bash run_easyrec_benchmark_eval_mulerun.sh
```

Here, `BENCHMARK_ROOT` should point to the root directory that contains the benchmark data and training resources required by the EasyRec fine-tuning pipeline.

After training, merge the LoRA weights into the base EasyRec model before running evaluation.

### Evaluation

After fine-tuning and merging the tuned EasyRec model, run:

```bash
python infer_EasyRec_MuleRun.py \
  --data_root ./MuleRun_Dataset \
  --model_dir /path/to/merged_tuned_EasyRec_model \
  --device cuda:0 \
  --max_len 512 \
  --encode_batch 256
```

## About MuleRun Dataset

The raw agent information crawled from the MuleRun platform is stored in:

```text
./MuleRun_Dataset/mulerun_agents_original.jsonl
```

Based on the collected MuleRun agents, we use GPT-based data synthesis to generate diverse user queries with different difficulty levels. The prompt is available in `prompt.txt`. We also apply additional data construction procedures to reformat the original agent descriptions, resulting in the final MuleRun validation files, including agents, questions, rankings, and tools.


## Notes

- MuleRun is treated as an unseen external marketplace.
- The validation focuses on capability matching between user requests and agent descriptions.
- Tool primitives are inferred from each MuleRun agent description to align the data format with the Toolkit-only setting.
- The main paper result uses this setting to support the claim that AGENTSELECT provides transferable supervision for agent recommendation.



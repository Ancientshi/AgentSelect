# AgentSelect

AgentSelect is a benchmark for recommending agent configurations from natural-language queries. Agents are represented by a backbone language model and a set of tools. The benchmark combines model-only, tool-only, and compositional agents.

[Paper](https://openreview.net/pdf?id=4M5Kj2UqaM) · [Dataset](https://drive.google.com/drive/folders/1p3VprcU3ziEqL360lixmAvSj-lWHOgNC)

## Dataset

Download the dataset folder from Google Drive. Each of `PartI/`, `PartII/`, and `PartIII/` must contain `agents.jsonl`, `queries.jsonl`, and `rankings.jsonl`. Keep `tools.jsonl` and `llms.jsonl` at the dataset root. `Merged/` provides a combined view for independent analysis.

## Quick start

Python 3.10 or later is required. Clone the repository and prepare a separate dataset directory for the training and inference scripts:

```bash
git clone https://github.com/Ancientshi/AgentSelect.git
cd AgentSelect
```

```bash
python scripts/prepare_optimized_dataset.py \
  --input /path/to/downloaded/Dataset \
  --output ./data/AgentSelect
```

The output directory must not already exist. To run the TF-IDF retrieval example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_generative.py \
  --data_root ./data/AgentSelect \
  --query "How can I analyze a CSV file?" \
  --top_k 3
```

This command retrieves agents by text similarity and formats their configurations; it does not train a model. Other baseline and evaluation entry points are in the repository root and `scripts/`.

## Evaluation

The benchmark uses the top 10, 1, and 5 positive agents for Parts I, II, and III, respectively. The reported ranking cutoff is top 10; results are reported for each part and overall. See the training and evaluation scripts for the sampling and split configuration.

## Citation

```bibtex
@inproceedings{shi2026agentselect,
  title     = {AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation},
  author    = {Shi, Yunxiao and Xu, Wujiang and Chen, Tingwei and Shang, Haoning and Yang, Ling and Wan, Yunfeng and Cao, Zhuo and Zi, Xing and Metaxas, Dimitris N. and Xu, Min},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=4M5Kj2UqaM}
}
```

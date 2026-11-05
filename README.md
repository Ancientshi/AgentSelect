# AgentSelect

AgentSelect is a benchmark for recommending agent configurations from natural-language queries. Agents are represented by a backbone language model and a set of tools. The benchmark combines model-only, tool-only, and compositional agents.

[Paper](https://openreview.net/pdf?id=4M5Kj2UqaM) · [Dataset](https://drive.google.com/file/d/1YyuE6GLp_p73PGE5emAmPrnpSrRVlpxi/view?usp=drive_link)

## Dataset

Download [Dataset.zip](https://drive.google.com/file/d/1YyuE6GLp_p73PGE5emAmPrnpSrRVlpxi/view?usp=drive_link). Each of `PartI/`, `PartII/`, and `PartIII/` contains queries, agents, and rankings in JSONL format. `Merged/` provides a combined view.

## Quick start

Python 3.10 or later is required. Clone the repository and place the downloaded `Dataset.zip` in the project directory:

```bash
git clone https://github.com/Ancientshi/AgentSelect.git
cd AgentSelect
```

```bash
unzip Dataset.zip -d ./data
python scripts/prepare_optimized_dataset.py \
  --input ./data/Dataset \
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

This command retrieves agents by text similarity and formats their configurations; it does not train a model. Other baseline and evaluation entry points are in the repository root and `scripts/` and may require additional dependencies.

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

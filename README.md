--------


# AgentSelectBench 🌟
[![Benchmark Introduction](https://img.shields.io/badge/Docs-Getting%20Started-black)](https://v0-agent-recommendation-website.vercel.app)
[![ICML 2026](https://img.shields.io/badge/ICML%202026-yellow)](#citation)



**AgentSelectBench (AGENTSELECT)** is a unified-supervision benchmark for **narrative query-to-agent recommendation**: given a free-form natural-language request, rank **deployable agent configurations** represented as capability profiles **(backbone LLM, toolkit)**. It systematically converts heterogeneous evaluation artifacts (LLM leaderboards, tool-use benchmarks, etc.) into **query-conditioned, positive-only** interactions for training and evaluating agent recommenders at scale.

🔗 **Getting Started / Documentation:** https://v0-agent-recommendation-website.vercel.app/#getting-started  
🧪 **Online Demo (Agent Recommendation API, WIP):** https://api.achieva-ai.com/OneAgent/

<div align="center">
  <img src="images/Figure1.png" alt="Figure 1" width="800"/>
</div>

> **Status:** this repository is **under active refinement**. We are progressively cleaning code, adding missing scripts/docs, and improving reproducibility. If, during review, you notice incomplete parts or rough edges, please treat them as ongoing engineering work—we are actively consolidating everything.

---



## Why AgentSelectBench ✨

Modern agent ecosystems offer an exploding space of configurations, but existing benchmarks evaluate **components in isolation** (models or tools). AgentSelectBench instead supports the end task:

* **Input:** a narrative query (no persistent user ID; intent is fully expressed in the query)
* **Output:** a ranked list of **deployable agents** as capability profiles **(M, T)**
* **Supervision:** **positive-only** query–agent interactions unified across sources 

---

## Capability Profile Format 🧾

Each agent is represented as a **capability profile**:

* **Backbone LLM**: `M`
* **Toolkit**: `T` (a set of tools with name + description)
* Stored as a **YAML configuration** to keep agents *deployable* (while we benchmark the stable capability core `(M, T)`). When deployed with agent framework, some additional configurations `C` may also required. 

<div align="center">
  <img src="images/Table1.png" alt="Table 1" width="300"/>
</div>

---


## Benchmark Overview 📦


<div align="center">
  <img src="images/Figure3.png" alt="Figure 3" width="800"/>
</div>

AgentSelectBench comprises three complementary dataset parts:

### 🧩 Part I — LLM-only Agents

Query-conditioned supervision derived from LLM evaluations/leaderboards (tools absent). Positives are typically constructed as **top-k** preferred backbones per query. 

### 🧰 Part II — Toolkit-only Agents

Tool-use benchmarks provide the **required/reference toolkit** for each query; we treat each query’s toolkit as the positive target (backbone fixed to a placeholder). 

### 🔗 Part III — Compositional Agents

We synthesize realistic **(M, T)** configurations by retrieving relevant components and composing them into candidate agents, yielding **pseudo-positive** interactions designed to reflect capability-consistent supervision. 

**Scale (current release):** 111,179 queries, 107,721 agents, 251,103 interactions aggregated from 40+ sources. 

---

## Project Code Structure 🏗️

A typical structure (may evolve as we refactor):

```
AgentSelectBench/
├── agent_rec/                      # Research scaffold for agent recommendation
│   ├── data/                        # Dataset loaders / parsing
│   ├── features/                    # Unified feature interfaces (text + IDs)
│   ├── models/                      # Baselines (MF/LightFM/TwoTower/etc.)
│   ├── eval/                        # Metrics + evaluation harness
│   └── utils.py                     # Shared utilities (metrics printing, etc.)
├── scripts/                         # Helper scripts (training / eval wrappers)
├── run_bpr_mf_knn.py                # MF baseline with KNN query-vector surrogate
├── run_lightfm_handwritten.py       # LightFM baseline
├── run_generative.py                # Inference-only structured/generative baseline
├── run_generative_train.py          # Optional: seq2seq finetuning from exported pairs
├── requirements.txt
└── README.md
```

---

## Getting Started 🚀

### 1) Clone the repository

```bash
git clone https://github.com/<your-org-or-anon-link>/AgentSelectBench.git
cd AgentSelectBench
```

### 2) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3) Prepare dataset

AgentSelectBench is constructed from **publicly available** leaderboards / benchmarks.
Depending on upstream redistribution constraints, we provide **derived annotations/statistics** and scripts to reconstruct raw sources when required. 

---

## Evaluation Protocol 📊

* **Positives:** Part I (top-10), Part II (top-1), Part III (top-5)
* **Ranking cutoff:** fixed **Top-10** evaluation
* **Reporting:** metrics are reported for **Part I / Part II / Part III / Overall** 

---

## Quick Runs 🛠️

### Run (BPR-MF + KNN q-vector)

```bash
python run_bpr_mf_knn.py \
  --data_root /path/to/dataset_root \
  --device cuda:0 \
  --epochs 5 --batch_size 4096 --factors 128 --neg_per_pos 1 \
  --knn_N 3 --eval_cand_size 100 --score_mode dot
```

### Run (LightFM)

```bash
python run_lightfm_handwritten.py \
  --data_root /path/to/dataset_root \
  --device cuda:0 \
  --epochs 5 --batch_size 4096 --factors 128 --neg_per_pos 1 \
  --alpha_id 1.0 --alpha_feat 1.0 --max_features 5000 \
  --knn_N 3 --eval_cand_size 100 \
  --use_tool_id_emb 1
```

**Note:** this scaffold assumes you have `utils.py` in the same folder as the entry scripts, providing `print_metrics_table(...)` (consistent with the current research scaffold).

---

## Generative / Structured Baseline (Inference-Only) ✍️

This entrypoint is **inference-only** (no training). It fits a TF-IDF retriever each run, then formats structured token outputs for a query. You can also export supervised pairs for training an external seq2seq model.

### Generate structured token outputs (LLM + tools) for a query

```bash
python run_generative.py \
  --data_root /path/to/dataset_root \
  --query "How do I write a web scraper?" \
  --top_k 3 \
  --with_metadata 1
```

### Export supervised pairs for seq2seq finetuning (JSONL)

```bash
python run_generative.py \
  --data_root /path/to/dataset_root \
  --export_pairs /tmp/generative_pairs.jsonl \
  --max_examples 5000
```

### Fine-tune a seq2seq model (e.g., T5) on exported pairs

```bash
python run_generative_train.py \
  --data_root /path/to/dataset_root \
  --output_dir /tmp/generative_t5_ckpt \
  --model_name t5-small \
  --epochs 3 \
  --batch_size 16
```

### Shell helper (env-style)

```bash
DATA_ROOT=/path/to/dataset_root \
QUERY="How do I write a web scraper?" \
TOP_K=3 \
WITH_METADATA=1 \
./scripts/run_generative.sh
```

---

## Reproducibility Notes 🧪

* We will keep adding: dataset build scripts, caching, and deterministic evaluation harnesses.
* If you are reviewing this work and find a missing script or unclear step, it likely reflects ongoing repository cleanup rather than missing methodology; please feel free to flag it—we are actively addressing gaps.

---

## Citation 📚

@inproceedings{
anonymous2026agentselect,
title={AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation},
author={Yunxiao Shi, Wujiang Xu, Tingwei Chen, Haoning Shang, Ling Yang, Yunfeng Wan, Zhuo Cao, Xing Zi, Dimitris N. Metaxas, Min Xu},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=4M5Kj2UqaM}
}

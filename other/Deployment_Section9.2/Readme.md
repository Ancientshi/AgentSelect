
# End-to-End Agent Performance Evaluation

This directory contains the experimental artifacts for the end-to-end agent performance evaluation used in **Appendix B**, **Table 7**, and **Section 9.2** of the paper.

## Overview

This experiment samples **200 / 500 queries from Part III** of the benchmark and performs a deployment-level evaluation of recommended agents. The goal is to examine whether the ranking produced by the recommendation model is aligned, at a high level, with the agents' actual end-to-end performance after instantiation and execution.

More specifically, the experiment evaluates whether agents ranked higher by the recommender tend to achieve better downstream performance when they are constructed on demand and executed in a realistic serving environment.

## Experimental Pipeline

The end-to-end pipeline consists of the following components:

- **Recommendation model:** `TwoTower (TF-IDF)`
- **Agent construction framework:** `Agno`
- **LLM service provider:** `SiliconFlow`
- **Tool simulation service:** `MirrorAPI`, driven by `GPT-5-Nano`
- **Evaluation model:** GPT-based evaluator in `evaluator.py`

For each query:

1. The recommendation model retrieves the top candidate agents.
2. Each selected agent is instantiated on demand through the **Agno** framework.
3. The instantiated agent is driven by its backbone LLM and may invoke the adaptive MCP tool service when needed.
4. Tool execution is simulated using the **MirrorAPI** method, implemented with **GPT-5-Nano**, which provides adaptive MCP-style services.
5. The resulting outputs of the selected agents are collected and compared.

## LLM Mapping Issue and Real Rank

A practical issue arises because the **expected core LLM** of some selected agents is **not fully supported by SiliconFlow**.

To address this, we apply a **similarity-based mapping algorithm** with a threshold of **0.6**. For an unsupported expected core LLM, the algorithm selects the most similar available LLM from the SiliconFlow service as its substitute.

As a result, the originally selected 5 agents may not exactly match their intended backbone LLMs at execution time. Therefore, the selected agents should, in principle, be **re-ranked again by the recommendation model after LLM substitution**, so as to obtain their **actual recommendation rank under the executed configuration**.

This re-ranked result is the more appropriate reference when comparing recommendation ranking against true end-to-end performance.

## Result Files

The main experimental outputs are:

- `OneAgent_results.run1.500.jsonl`
- `OneAgent_results.run2.500.jsonl`
- `OneAgent_results.run3.500.jsonl`

These files contain the execution results for the 500-query settings, respectively. In this folder we provide some example excution results. For get full excultion results, can run scripts/run_Table5.sh

## Evaluation

After agent execution, the outputs of the 5 selected agents for each query are evaluated and ranked using a GPT-based evaluator.

Run:
```bash
bash eva.sh
```

This will invoke `evaluator.py`, which uses GPT to rank the execution results of the 5 agents for each query.

## Visualization

To visualize the relationship between:

* the recommendation ranking, and
* the observed end-to-end performance ranking,

run:

```bash
bash visual.sh
```

This script generates the visualizations used to compare recommender ordering with deployment-time performance.

## Relevance to the Paper

These experimental results provide the empirical support for:

* **Appendix B**
* **Table 7**
* **Section 9.2**

In particular, they serve as deployment-oriented evidence that the recommendation model's ranking is positively aligned, at a coarse-grained level, with actual end-to-end agent performance.

## Notes

* This evaluation is intended as a **high-level deployment validation**, rather than a fully controlled benchmark of absolute agent capability.
* Because backbone substitution may occur due to infrastructure constraints, the **executed agent configuration** can differ slightly from the originally expected agent specification.
* For this reason, the **post-mapping real rank** is an important reference when interpreting the alignment between recommendation scores and execution performance.


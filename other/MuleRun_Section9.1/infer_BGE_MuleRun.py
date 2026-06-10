#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_BGE_fullrank_MuleRun.py
-----------------------------
Evaluate a BGE cross-encoder / reranker on the MuleRun_Dataset setting.

Overview:
- Uses MuleRun_Dataset queries and the shared tool pool.
- Ranks the full candidate agent set for each query.
- Reports P/R/F1/Hit/nDCG/MRR @ K.

Notes:
- Evaluation items are prepared serially for reproducibility.
- The reranker only scores query-document pairs.
"""

import os
import math
import json
import random
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


all_agents, all_questions, all_rankings, tools, agent_text_cache = None, None, None, None, None


# -------------------- Data loading --------------------
def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_data(data_root: str):
    """Load MuleRun_Dataset from the given data root."""
    all_agents, all_questions, all_rankings = {}, {}, {}

    agents = load_json(os.path.join(data_root, "agents", "merge.json"))
    questions = load_json(os.path.join(data_root, "questions", "merge.json"))
    rankings = load_json(os.path.join(data_root, "rankings", "merge.json"))

    all_agents.update(agents)
    all_questions.update(questions)
    all_rankings.update(rankings["rankings"])

    tools = load_json(os.path.join(data_root, "Tools", "merge.json"))
    return all_agents, all_questions, all_rankings, tools


# -------------------- Text construction --------------------
def tool_text(tools: Dict[str, dict], tn: str) -> str:
    t = tools.get(tn, {}) or {}
    return f"{tn} {t.get('description', '')}".strip()


def agent_text(all_agents: Dict[str, dict], tools: Dict[str, dict], aid: str) -> str:
    a = all_agents.get(aid, {}) or {}
    mname = a.get("M", {}).get("name", "") or ""
    tlst = a.get("T", {}).get("tools", []) or []
    txt = (mname + " || " + " | ".join([tool_text(tools, t) for t in tlst])).strip(" |")
    return txt or mname


# -------------------- BGE scoring --------------------
def batched_tokenize_and_score(
    *,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    qtext: str,
    doc_texts: List[str],
    max_len: int,
    rerank_batch: int,
    use_amp: bool,
) -> np.ndarray:
    if not doc_texts:
        return np.zeros((0,), dtype=np.float32)

    scores_all: List[np.ndarray] = []

    for i in range(0, len(doc_texts), rerank_batch):
        batch_docs = doc_texts[i:i + rerank_batch]
        enc = tokenizer(
            [qtext] * len(batch_docs),
            batch_docs,
            truncation=True,
            padding="longest",
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**enc)
        else:
            out = model(**enc)

        s = out.logits.squeeze(-1).float().detach().cpu().numpy()
        scores_all.append(s)

    if not scores_all:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(scores_all, axis=0)


# -------------------- Metrics --------------------
def dcg_at_k(hits: List[int]) -> float:
    dcg = 0.0
    for i, h in enumerate(hits):
        if h:
            dcg += 1.0 / math.log2(i + 2.0)
    return dcg


# -------------------- Evaluation item preparation --------------------
def prepare_eval_items(
    qids_in_rank: List[str],
    a_ids: List[str],
    *,
    pos_topk: int,
) -> List[Dict[str, Any]]:
    """
    Prepare deterministic full-ranking evaluation items.

    High-level behavior:
    - removes negative sampling and candidate-size logic
    - ranks against the full agent pool for each query
    - reuses cached agent text to avoid repeated string assembly
    """
    global all_agents, all_questions, all_rankings, tools, agent_text_cache

    items = []

    # Full candidate pool, preserving the existing order.
    a_ids_list = [aid for aid in a_ids if aid in all_agents]

    for qid in qids_in_rank:
        # Positive set, truncated to pos_topk.
        gt_all = [aid for aid in (all_rankings.get(qid, []) or []) if aid in all_agents]
        gt = gt_all[:pos_topk]
        if not gt:
            continue
        rel_set = set(gt)

        # Full-ranking setup over the complete candidate pool.
        cand_ids = a_ids_list

        # Query text.
        qtext = (all_questions.get(qid, {}) or {}).get("input", "") or ""
        if ", specifically" in qtext:
            qtext = qtext.split(", specifically", 1)[1].strip()
        else:
            qtext = qtext.strip()

        # Candidate text from cache.
        doc_texts = [agent_text_cache.get(aid, "") for aid in cand_ids]

        items.append({
            "qid": qid,
            "qtext": qtext,
            "cand_ids": cand_ids,
            "doc_texts": doc_texts,
            "rel_set": rel_set,
        })

    return items


# -------------------- Evaluation --------------------
def evaluate_bge(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    items: List[Dict[str, Any]],
    *,
    ks: Tuple[int, ...],
    max_len: int,
    rerank_batch: int,
    use_amp: bool,
    max_eval: int = 0,
) -> Dict[int, Dict[str, float]]:
    if max_eval and len(items) > max_eval:
        items = items[:max_eval]

    agg = {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}
    ref_k = 10 if 10 in ks else max(ks)
    done = 0

    pbar = tqdm(items, desc="Evaluating (BGE full-rank)", dynamic_ncols=True)
    for it in pbar:
        scores = batched_tokenize_and_score(
            model=model,
            tokenizer=tokenizer,
            device=device,
            qtext=it["qtext"],
            doc_texts=it["doc_texts"],
            max_len=max_len,
            rerank_batch=rerank_batch,
            use_amp=use_amp,
        )

        max_k = max(ks)
        order = np.argsort(-scores)[:max_k]
        pred_ids = [it["cand_ids"][i] for i in order]
        bin_hits = [1 if aid in it["rel_set"] else 0 for aid in pred_ids]

        for k in ks:
            topk_hits = bin_hits[:k]
            Hk = sum(topk_hits)
            P = Hk / float(k)
            R = Hk / float(len(it["rel_set"]))
            F1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
            Hit = 1.0 if Hk > 0 else 0.0

            dcg = dcg_at_k(topk_hits)
            ideal = min(len(it["rel_set"]), k)
            idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal > 0 else 0.0
            nDCG = (dcg / idcg) if idcg > 0 else 0.0

            rr = 0.0
            for i, h in enumerate(topk_hits):
                if h:
                    rr = 1.0 / float(i + 1)
                    break

            agg[k]["P"] += P
            agg[k]["R"] += R
            agg[k]["F1"] += F1
            agg[k]["Hit"] += Hit
            agg[k]["nDCG"] += nDCG
            agg[k]["MRR"] += rr

        done += 1
        ref = agg[ref_k]
        pbar.set_postfix({
            "done": done,
            f"P@{ref_k}": f"{(ref['P'] / done):.4f}",
            f"nDCG@{ref_k}": f"{(ref['nDCG'] / done):.4f}",
            f"MRR@{ref_k}": f"{(ref['MRR'] / done):.4f}",
        })

    if done == 0:
        return {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}

    for k in ks:
        for m in agg[k]:
            agg[k][m] /= done
    return agg


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    default_data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MuleRun_Dataset")
    ap.add_argument(
        "--data_root",
        type=str,
        default=default_data_root,
        help="Path to MuleRun_Dataset. Default: MuleRun_Dataset under the same directory as this Python file.",
    )
    ap.add_argument("--model_dir", type=str, required=True, help="Path to the reranker checkpoint, either a HF model or a PEFT adapter.")
    ap.add_argument("--model_name", type=str, default=None, help="HF base model name when --peft=0.")
    ap.add_argument("--peft", type=int, default=0, help="Set to 1 to load a PEFT adapter; otherwise load a standard HF checkpoint.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--pos_topk", type=int, default=10)
    ap.add_argument("--ks", type=str, default="1,5,10")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--rerank_batch", type=int, default=256)
    ap.add_argument("--use_amp", type=int, default=1, help="Enable CUDA autocast with float16 when available.")
    ap.add_argument("--max_eval", type=int, default=1080, help="Maximum number of queries to evaluate; 0 means no limit.")
    args = ap.parse_args()

    # Basic environment setup.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Load data.
    global all_agents, all_questions, all_rankings, tools, agent_text_cache
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)

    q_ids = list(all_questions.keys())
    a_ids = list(all_agents.keys())
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]

    # Cache agent text.
    agent_text_cache = {}
    for aid, a in all_agents.items():
        mname = (a.get("M", {}) or {}).get("name", "") or ""
        tlst = (a.get("T", {}) or {}).get("tools", []) or []

        parts = [mname]
        if tlst:
            tool_parts = []
            for tn in tlst:
                t = (tools.get(tn, {}) or {})
                desc = t.get("description", "")
                tool_parts.append(f"{tn} {desc}".strip())
            parts.append(" || " + " | ".join(tool_parts))
        agent_text_cache[aid] = "".join(parts).strip(" |")

    rng = random.Random(args.seed)
    eval_qids = qids_in_rank[:]
    rng.shuffle(eval_qids)

    # Build evaluation items serially for reproducibility.
    items = prepare_eval_items(
        qids_in_rank=eval_qids,
        a_ids=a_ids,
        pos_topk=args.pos_topk,
    )

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())

    # Load model and tokenizer.
    use_fast = os.environ.get("HF_NO_FAST_TOKENIZER", "0") != "1"
    tok_src = args.model_name if (args.peft == 0 and args.model_name) else args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=use_fast)

    if args.peft == 0:
        mdl_src = args.model_name if args.model_name else args.model_dir
        model = AutoModelForSequenceClassification.from_pretrained(mdl_src, num_labels=1)
    else:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=1)
        except Exception:
            from peft import PeftConfig, PeftModel
            peft_cfg = PeftConfig.from_pretrained(args.model_dir)
            base = AutoModelForSequenceClassification.from_pretrained(
                peft_cfg.base_model_name_or_path,
                num_labels=1,
            )
            model = PeftModel.from_pretrained(base, args.model_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[warn] CUDA is not available, running on CPU instead of {args.device}.")

    model.to(device)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run evaluation.
    metrics = evaluate_bge(
        model=model,
        tokenizer=tokenizer,
        device=device,
        items=items,
        ks=ks,
        max_len=args.max_len,
        rerank_batch=args.rerank_batch,
        use_amp=bool(args.use_amp),
        max_eval=args.max_eval,
    )

    for k in ks:
        m = metrics[k]
        print(
            f"@{k}: "
            f"P={m['P']:.4f} "
            f"R={m['R']:.4f} "
            f"F1={m['F1']:.4f} "
            f"Hit={m['Hit']:.4f} "
            f"nDCG={m['nDCG']:.4f} "
            f"MRR={m['MRR']:.4f}"
        )


if __name__ == "__main__":
    """
    Expected directory layout:

    infer_BGE_fullrank_MuleRun.py
    MuleRun_Dataset/
      agents/merge.json
      questions/merge.json
      rankings/merge.json
      Tools/merge.json

    Example for a standard HF reranker checkpoint:

    python infer_BGE_fullrank_MuleRun.py \
      --model_name BAAI/bge-reranker-base \
      --model_dir /path/to/reranker \
      --peft 0 \
      --device cuda:0 \
      --pos_topk 10 \
      --ks 1,5,10 \
      --max_len 256 \
      --rerank_batch 256 \
      --max_eval 1080

    Example for a PEFT adapter:

    python infer_BGE_fullrank_MuleRun.py \
      --model_dir /path/to/reranker \
      --peft 1 \
      --device cuda:0 \
      --pos_topk 10 \
      --ks 1,5,10 \
      --max_len 256 \
      --rerank_batch 256 \
      --max_eval 1080
    """
    main()

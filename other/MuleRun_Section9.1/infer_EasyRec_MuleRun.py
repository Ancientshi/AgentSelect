#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a full EasyRec LoRA checkpoint on MuleRun.

This Section9.1 copy is intentionally minimal and targets the delivered merged
`best/` checkpoint produced by the local LoRA training script.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from EasyRec_LoRA_model import Easyrec, default_model_args


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def pick_first_existing(paths: List[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    tried = "\n".join(str(p) for p in paths)
    raise FileNotFoundError(f"Could not find any of the expected files:\n{tried}")


def relax_legacy_torch_load_guard():
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils

    modeling_utils.check_torch_load_is_safe = lambda: None
    import_utils.check_torch_load_is_safe = lambda: None


def collect_data(data_root: str):
    root = Path(data_root).resolve()
    part_root = root / "PartV"

    if (part_root / "agents" / "merge.json").exists():
        data_dir = part_root
    elif (root / "agents" / "merge.json").exists():
        data_dir = root
    else:
        raise FileNotFoundError(
            f"Could not locate MuleRun data under {root}. Expected either "
            f"{part_root / 'agents' / 'merge.json'} or {root / 'agents' / 'merge.json'}."
        )

    tools_path = pick_first_existing(
        [
            root / "Tools" / "merge2.json",
            root / "Tools" / "merge.json",
            data_dir / "Tools" / "merge2.json",
            data_dir / "Tools" / "merge.json",
        ]
    )

    agents = load_json(data_dir / "agents" / "merge.json")
    questions = load_json(data_dir / "questions" / "merge.json")
    rankings_obj = load_json(data_dir / "rankings" / "merge.json")
    rankings = rankings_obj["rankings"] if isinstance(rankings_obj, dict) and "rankings" in rankings_obj else rankings_obj
    all_tools = load_json(tools_path)
    return agents, questions, rankings, all_tools


def tool_text(tool_map: Dict[str, dict], tool_name: str) -> str:
    tool = tool_map.get(tool_name, {}) or {}
    return f"{tool_name} {tool.get('description', '')}".strip()


def agent_text(agent_map: Dict[str, dict], tool_map: Dict[str, dict], aid: str) -> str:
    agent = agent_map.get(aid, {}) or {}
    model_name = agent.get("M", {}).get("name", "") or ""
    tool_list = agent.get("T", {}).get("tools", []) or []
    text = (model_name + " || " + " | ".join(tool_text(tool_map, t) for t in tool_list)).strip(" |")
    return text or model_name


def build_eval_items(
    *,
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    all_rankings: Dict[str, List[str]],
    tools: Dict[str, dict],
    pos_topk: int,
):
    agent_ids = [aid for aid in all_agents]
    agent_texts = [agent_text(all_agents, tools, aid) for aid in agent_ids]

    items = []
    for qid in all_questions:
        if qid not in all_rankings:
            continue
        rel_all = [aid for aid in (all_rankings.get(qid, []) or []) if aid in all_agents]
        rel_ids = rel_all[:pos_topk]
        if not rel_ids:
            continue

        qtext = (all_questions.get(qid, {}) or {}).get("input", "") or ""
        if ", specifically" in qtext:
            qtext = qtext.split(", specifically", 1)[1].strip()
        else:
            qtext = qtext.strip()

        items.append(
            {
                "qid": qid,
                "qtext": qtext,
                "cand_ids": agent_ids,
                "doc_texts": agent_texts,
                "rel_set": set(rel_ids),
            }
        )
    return items


def dcg_at_k(hits: List[int]) -> float:
    dcg = 0.0
    for idx, hit in enumerate(hits):
        if hit:
            dcg += 1.0 / math.log2(idx + 2.0)
    return dcg


def update_aggregates(agg, rel_set: set, cand_ids: List[str], scores: np.ndarray, ks: Tuple[int, ...]):
    max_k = max(ks)
    order = np.argsort(-scores)[:max_k]
    pred_ids = [cand_ids[i] for i in order]
    bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]

    for k in ks:
        topk_hits = bin_hits[:k]
        num_hits = sum(topk_hits)
        precision = num_hits / float(k)
        recall = num_hits / float(len(rel_set))
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        hit = 1.0 if num_hits > 0 else 0.0

        dcg = dcg_at_k(topk_hits)
        ideal = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal > 0 else 0.0
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        rr = 0.0
        for idx, is_hit in enumerate(topk_hits):
            if is_hit:
                rr = 1.0 / float(idx + 1)
                break

        agg[k]["P"] += precision
        agg[k]["R"] += recall
        agg[k]["F1"] += f1
        agg[k]["Hit"] += hit
        agg[k]["nDCG"] += ndcg
        agg[k]["MRR"] += rr


def tokenize_texts(tokenizer, texts: List[str], max_len: int, device: torch.device):
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        return_token_type_ids=False,
    )
    return {k: v.to(device, non_blocking=True) for k, v in tokenized.items() if k in ("input_ids", "attention_mask")}


@torch.inference_mode()
def encode_easyrec_texts(
    model,
    tokenizer,
    texts: List[str],
    *,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not texts:
        hidden_size = getattr(model.config, "hidden_size", 768)
        return torch.empty((0, hidden_size), dtype=torch.float32)

    encoded = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encode EasyRec", dynamic_ncols=True):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenize_texts(tokenizer, batch_texts, max_len, device)
        outputs = model.encode(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        embeddings = F.normalize(outputs.pooler_output.detach().float(), p=2, dim=-1).cpu()
        encoded.append(embeddings)
    return torch.cat(encoded, dim=0)


def load_easyrec_checkpoint(model_dir: str, device: torch.device):
    relax_legacy_torch_load_guard()
    cfg = AutoConfig.from_pretrained(model_dir)
    model = Easyrec.from_pretrained(
        model_dir,
        config=cfg,
        model_args=default_model_args(cfg),
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_easyrec(
    model,
    tokenizer,
    device: torch.device,
    items: List[Dict[str, Any]],
    *,
    ks: Tuple[int, ...],
    max_len: int,
    encode_batch: int,
    max_eval: int,
) -> Dict[int, Dict[str, float]]:
    if max_eval and len(items) > max_eval:
        items = items[:max_eval]

    agg = {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}
    if not items:
        return agg

    ref_k = 10 if 10 in ks else max(ks)
    full_cand_ids = items[0]["cand_ids"]
    aid_to_idx = {aid: idx for idx, aid in enumerate(full_cand_ids)}
    doc_embeddings = encode_easyrec_texts(
        model,
        tokenizer,
        items[0]["doc_texts"],
        max_len=max_len,
        batch_size=encode_batch,
        device=device,
    ).numpy().astype(np.float32)
    query_embeddings = encode_easyrec_texts(
        model,
        tokenizer,
        [item["qtext"] for item in items],
        max_len=max_len,
        batch_size=encode_batch,
        device=device,
    ).numpy().astype(np.float32)

    pbar = tqdm(enumerate(items), total=len(items), desc="Evaluating MuleRun (EasyRec)", dynamic_ncols=True)
    done = 0
    for idx, item in pbar:
        cand_idx = [aid_to_idx[aid] for aid in item["cand_ids"]]
        scores = doc_embeddings[cand_idx] @ query_embeddings[idx]
        update_aggregates(agg, item["rel_set"], item["cand_ids"], scores, ks)
        done += 1

        ref = agg[ref_k]
        pbar.set_postfix(
            {
                "done": done,
                f"P@{ref_k}": f"{(ref['P'] / done):.4f}",
                f"nDCG@{ref_k}": f"{(ref['nDCG'] / done):.4f}",
                f"MRR@{ref_k}": f"{(ref['MRR'] / done):.4f}",
            }
        )

    for k in ks:
        for metric_name in agg[k]:
            agg[k][metric_name] /= done
    return agg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Direct MuleRun dataset root or benchmark root containing PartV.")
    ap.add_argument("--model_dir", type=str, required=True, help="Merged EasyRec checkpoint path, e.g. best/.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--pos_topk", type=int, default=10)
    ap.add_argument("--ks", type=str, default="1,5,10")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--encode_batch", type=int, default=256)
    ap.add_argument("--max_eval", type=int, default=1080)
    ap.add_argument("--save_path", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    items = build_eval_items(
        all_agents=all_agents,
        all_questions=all_questions,
        all_rankings=all_rankings,
        tools=tools,
        pos_topk=args.pos_topk,
    )
    print(f"Loaded MuleRun data: {len(all_agents)} agents, {len(all_questions)} questions, {len(items)} eval items.")

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[warn] CUDA not available, running on CPU instead of {args.device}.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, tokenizer = load_easyrec_checkpoint(args.model_dir, device)
    metrics = evaluate_easyrec(
        model=model,
        tokenizer=tokenizer,
        device=device,
        items=items,
        ks=ks,
        max_len=args.max_len,
        encode_batch=args.encode_batch,
        max_eval=args.max_eval,
    )

    for k in ks:
        metric = metrics[k]
        print(
            f"EASYREC MuleRun @{k}: "
            f"P={metric['P']:.4f} "
            f"R={metric['R']:.4f} "
            f"F1={metric['F1']:.4f} "
            f"Hit={metric['Hit']:.4f} "
            f"nDCG={metric['nDCG']:.4f} "
            f"MRR={metric['MRR']:.4f}"
        )

    if args.save_path:
        save_json(
            {
                "data_root": str(Path(args.data_root).resolve()),
                "model_dir": args.model_dir,
                "pos_topk": args.pos_topk,
                "ks": list(ks),
                "max_len": args.max_len,
                "encode_batch": args.encode_batch,
                "max_eval": args.max_eval,
                "metrics": metrics,
            },
            args.save_path,
        )
        print(f"Saved metrics to {args.save_path}")


if __name__ == "__main__":
    main()

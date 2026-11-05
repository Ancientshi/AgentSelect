#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNN-BGE Agent Recommender (BPR) — aligned with run_twotower_bge.py.

This script keeps the BGE data/feature/evaluation flow from run_twotower_bge.py,
but replaces the TwoTower/InfoNCE learner with SimpleBPRDNN + BPR loss.

Main alignment points with run_twotower_bge.py:
1) BGE feature cache construction/loading is unchanged.
2) --train_parts controls the split used to build training pairs.
3) --eval_parts controls final reporting independently from train valid_qids.
4) BGE feature matrices stay on CPU; only current train/eval batches move to GPU.
5) Training cache key includes train_parts and pair_type=q_pos_neg_posTopK.

Example:
python run_dnn_bge.py \
  --data_root path_to/AgentSelect/dataset \
  --device cuda:4 \
  --epochs 5 \
  --batch_size 512 \
  --embed_backend local \
  --bge_model path_to/models/BAAI/bge-m3 \
  --bge_device cuda:4 \
  --bge_fp16 1 \
  --embed_batch 64 \
  --amp 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

import agent_rec.features as features_mod
from agent_rec.cli_common import add_shared_training_args
from agent_rec.config import POS_TOPK, POS_TOPK_BY_PART
from agent_rec.data import build_training_pairs, stratified_train_valid_split
from agent_rec.features import (
    build_agent_content_view,
    build_twotower_bge_feature_cache,
    load_feature_cache,
    save_feature_cache,
)
from agent_rec.models.dnn import SimpleBPRDNN, bpr_loss
from agent_rec.run_common import (
    bootstrap_run,
    cache_key_from_meta,
    cache_key_from_text,
    load_or_build_training_cache,
    shared_cache_dir,
)

# Reuse the same local eval utilities as run_twotower_bge.py when available.
# The full DNN scorer is pairwise rather than dot-product two-tower, so the
# actual sampled scoring loop below is DNN-specific, but output formatting and
# soft-eval LLM mapping stay aligned with eval.py.
try:
    from eval import build_aid_to_llm, print_sample_recommendations  # type: ignore
except Exception:  # pragma: no cover - only for environments without local eval.py
    build_aid_to_llm = None
    print_sample_recommendations = None

try:
    from utils import print_metrics_table
except Exception:  # pragma: no cover
    print_metrics_table = None


# ===== BGE-M3 embedding backend: copied from run_twotower_bge.py =====
_BGE_MODEL = None


def _get_bge_model(model_path: str, device: str = "cuda", use_fp16: bool = True):
    """Load BGE-M3 once. Prefer FlagEmbedding BGEM3FlagModel."""
    global _BGE_MODEL
    if _BGE_MODEL is not None:
        return _BGE_MODEL

    print(f"[bge] loading model from: {model_path}")
    print(f"[bge] device={device}, use_fp16={use_fp16}")

    try:
        from FlagEmbedding import BGEM3FlagModel

        _BGE_MODEL = ("flag", BGEM3FlagModel(model_path, use_fp16=use_fp16, device=device))
        print("[bge] loaded with FlagEmbedding.BGEM3FlagModel")
    except Exception as e:
        print(f"[bge] FlagEmbedding load failed: {repr(e)}")
        print("[bge] fallback to sentence_transformers.SentenceTransformer")
        from sentence_transformers import SentenceTransformer

        _BGE_MODEL = ("st", SentenceTransformer(model_path, device=device))
        print("[bge] loaded with sentence-transformers")

    return _BGE_MODEL


def release_bge_model() -> None:
    """Free BGE model after feature cache is built, so DNN training can use GPU memory."""
    global _BGE_MODEL
    if _BGE_MODEL is not None:
        print("[bge] releasing local BGE model from memory")
    _BGE_MODEL = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def bge_batch_embed(
    texts: List[str],
    embed_url: str = "",
    batch_size: int = 64,
    desc: str = "Embedding",
    *,
    model_path: str = "path_to/models/BAAI/bge-m3",
    device: str = "cuda",
    use_fp16: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Drop-in replacement for agent_rec.features.batch_embed()."""
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    backend, model = _get_bge_model(model_path, device=device, use_fp16=use_fp16)

    chunks: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, dynamic_ncols=True):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        if backend == "flag":
            out = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            emb = out["dense_vecs"]
        else:
            emb = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

        emb = np.asarray(emb, dtype=np.float32)
        if normalize:
            denom = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(denom, 1e-8)
        chunks.append(emb.astype(np.float32, copy=False))

    return np.vstack(chunks).astype(np.float32, copy=False)


def feature_cache_exists(cache_dir: str) -> bool:
    """BGE feature cache completeness check."""
    required = [
        "Q.npy",
        "A_model_content.npy",
        "A_tool_content.npy",
        "A_text_full.npy",
        "agent_tool_idx_padded.npy",
        "agent_tool_mask.npy",
        "agent_llm_idx.npy",
        "q_ids.json",
        "a_ids.json",
        "tool_id_vocab.json",
        "llm_vocab.json",
        "tool_names.json",
    ]
    return os.path.isdir(cache_dir) and all(os.path.exists(os.path.join(cache_dir, f)) for f in required)


# ===== DNN-specific sampled evaluation on BGE vectors =====
def _pos_k_for_part(qid: str, qid_to_part: Dict[str, str], pos_topk_by_part: Dict[str, int], default: int) -> int:
    return int(pos_topk_by_part.get(qid_to_part.get(qid, ""), default))


def _metric_at_k(ranked_is_pos: Sequence[bool], num_pos: int, k: int) -> Dict[str, float]:
    top = list(ranked_is_pos[:k])
    hit_count = int(sum(top))
    precision = hit_count / float(k) if k > 0 else 0.0
    recall = hit_count / float(num_pos) if num_pos > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    hit = 1.0 if hit_count > 0 else 0.0

    dcg = 0.0
    for rank, ok in enumerate(top, start=1):
        if ok:
            dcg += 1.0 / math.log2(rank + 1.0)
    ideal_hits = min(num_pos, k)
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    mrr = 0.0
    for rank, ok in enumerate(top, start=1):
        if ok:
            mrr = 1.0 / float(rank)
            break

    return {"P": precision, "R": recall, "F1": f1, "Hit": hit, "nDCG": ndcg, "MRR": mrr}


def _print_metrics(title: str, metrics: Dict[int, Dict[str, float]], exp_name: str, topk: int) -> None:
    if print_metrics_table is not None:
        print_metrics_table(title, metrics, ks=(topk,), filename=exp_name)
        return
    print(f"== {title} ==")
    print("  @K |       P       R      F1     Hit    nDCG     MRR")
    print("-" * 54)
    m = metrics[topk]
    print(
        f"{topk:4d} | {m['P']:.4f} {m['R']:.4f} {m['F1']:.4f} "
        f"{m['Hit']:.4f} {m['nDCG']:.4f} {m['MRR']:.4f}"
    )


def score_dnn_candidates(
    *,
    model: SimpleBPRDNN,
    q_vec_np: np.ndarray,
    q_idx: int,
    cand_idx_np: np.ndarray,
    A_cpu: np.ndarray,
    device: torch.device,
    chunk: int,
    amp: bool,
) -> np.ndarray:
    """Score one query against candidate agents with the DNN pairwise scorer."""
    scores: List[np.ndarray] = []
    model.eval()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp and device.type == "cuda" else nullcontext()
    with torch.no_grad():
        for start in range(0, len(cand_idx_np), chunk):
            idx = cand_idx_np[start:start + chunk]
            q_batch = np.repeat(q_vec_np[None, :], len(idx), axis=0)
            q_t = torch.from_numpy(q_batch.astype(np.float32, copy=False)).to(device, non_blocking=True)
            a_t = torch.from_numpy(A_cpu[idx].astype(np.float32, copy=False)).to(device, non_blocking=True)
            idx_t = torch.from_numpy(idx.astype(np.int64, copy=False)).to(device, non_blocking=True)
            q_idx_t = torch.full((len(idx),), int(q_idx), dtype=torch.long, device=device)

            # SimpleBPRDNN.forward returns (pos_score, neg_score). To get a
            # single candidate score, pass the same candidate as pos and neg and
            # take pos_score. This matches the original DNN evaluation style.
            with autocast_ctx:
                pos_score, _ = model(q_t, a_t, a_t, idx_t, idx_t, q_idx=q_idx_t)
            scores.append(pos_score.detach().float().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def evaluate_parts_dnn_bge(
    *,
    model: SimpleBPRDNN,
    Q_cpu: np.ndarray,
    A_cpu: np.ndarray,
    qid2idx: Dict[str, int],
    a_ids: Sequence[str],
    aid2idx: Dict[str, int],
    all_rankings: Dict[str, Sequence],
    qids_in_rank: Sequence[str],
    qid_to_part: Dict[str, str],
    eval_parts: Sequence[str],
    device: torch.device,
    topk: int,
    cand_size: int,
    rng_seed: int,
    eval_chunk: int,
    amp: bool,
    exp_name: str,
    soft_eval: bool,
    aid_to_llm: Optional[Dict[str, str]],
    pos_topk_by_part: Dict[str, int],
    pos_topk_default: int,
) -> None:
    """Sampled Top-K evaluation for DNN over precomputed BGE vectors."""
    all_agent_indices = np.arange(len(a_ids), dtype=np.int64)
    eval_part_set = set(eval_parts)
    part_to_qids: Dict[str, List[str]] = {p: [] for p in eval_parts}
    for qid in qids_in_rank:
        part = qid_to_part.get(qid, "")
        if part in eval_part_set:
            part_to_qids.setdefault(part, []).append(qid)

    print("[eval] " + " | ".join(f"{p}={len(part_to_qids.get(p, []))}" for p in eval_parts))

    for part in eval_parts:
        qids_part = part_to_qids.get(part, [])
        if not qids_part:
            print(f"[eval] skip {part}: no qids")
            continue

        sums = {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0}
        done = 0
        skipped = 0
        rng = np.random.default_rng(int(rng_seed) + abs(hash(part)) % 1000003)
        pbar = tqdm(qids_part, desc=f"Evaluating {part} (sampled DNN-BGE)", dynamic_ncols=True)

        for qid in pbar:
            ranking = all_rankings.get(qid)
            if not ranking or qid not in qid2idx:
                skipped += 1
                continue

            pos_k = _pos_k_for_part(qid, qid_to_part, pos_topk_by_part, pos_topk_default)
            pos_aids = [x[0] if isinstance(x, (list, tuple)) else x for x in list(ranking)[:pos_k]]
            pos_idx = [aid2idx[a] for a in pos_aids if a in aid2idx]
            if not pos_idx:
                skipped += 1
                continue

            pos_idx_set = set(pos_idx)
            neg_pool = np.array([i for i in all_agent_indices if int(i) not in pos_idx_set], dtype=np.int64)
            n_neg = max(0, int(cand_size) - len(pos_idx))
            if n_neg > 0 and len(neg_pool) > 0:
                if len(neg_pool) > n_neg:
                    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).astype(np.int64)
                else:
                    neg_idx = neg_pool
                cand_idx = np.concatenate([np.asarray(pos_idx, dtype=np.int64), neg_idx], axis=0)
            else:
                cand_idx = np.asarray(pos_idx, dtype=np.int64)

            # Remove accidental duplicates while preserving order.
            _, unique_pos = np.unique(cand_idx, return_index=True)
            cand_idx = cand_idx[np.sort(unique_pos)]

            q_idx = int(qid2idx[qid])
            scores = score_dnn_candidates(
                model=model,
                q_vec_np=Q_cpu[q_idx],
                q_idx=q_idx,
                cand_idx_np=cand_idx,
                A_cpu=A_cpu,
                device=device,
                chunk=int(eval_chunk),
                amp=amp,
            )
            order = np.argsort(-scores)
            ranked_idx = cand_idx[order]

            if soft_eval and aid_to_llm is not None:
                gt_llms = {aid_to_llm.get(a_ids[i], "") for i in pos_idx}
                gt_llms.discard("")
                ranked_is_pos = [aid_to_llm.get(a_ids[int(i)], "") in gt_llms for i in ranked_idx]
                num_pos_eval = max(1, len(gt_llms))
            else:
                ranked_is_pos = [int(i) in pos_idx_set for i in ranked_idx]
                num_pos_eval = len(pos_idx_set)

            m = _metric_at_k(ranked_is_pos, num_pos_eval, int(topk))
            for key in sums:
                sums[key] += m[key]
            done += 1
            if done > 0:
                pbar.set_postfix({
                    "done": done,
                    "skipped": skipped,
                    f"P@{topk}": f"{sums['P'] / done:.4f}",
                    f"nDCG@{topk}": f"{sums['nDCG'] / done:.4f}",
                    f"MRR@{topk}": f"{sums['MRR'] / done:.4f}",
                    "Ncand": len(cand_idx),
                })

        if done == 0:
            print(f"[eval] skip {part}: no valid examples after filtering")
            continue
        avg = {key: val / done for key, val in sums.items()}
        _print_metrics(f"Validation {part} (sampled DNN-BGE)", {int(topk): avg}, exp_name, int(topk))


def main() -> None:
    parser = argparse.ArgumentParser()
    add_shared_training_args(
        parser,
        exp_name_default="dnn_bge",
        device_default="cpu",
        epochs_default=5,
        batch_size_default=512,
        lr_default=1e-3,
        include_eval_cand=True,
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="",
        help="If set, directly load this BGE feature cache directory and ignore computed feature cache key.",
    )
    parser.add_argument("--embed_url", type=str, default="http://127.0.0.1:8502/get_embedding")
    parser.add_argument("--embed_batch", type=int, default=64)
    parser.add_argument("--embed_backend", type=str, default="local", choices=["local", "api"])
    parser.add_argument("--bge_model", type=str, default="path_to/models/BAAI/bge-m3")
    parser.add_argument("--bge_device", type=str, default="", help="empty means use --device")
    parser.add_argument("--bge_fp16", type=int, default=1)
    parser.add_argument("--text_hidden", type=int, default=256)
    parser.add_argument("--id_dim", type=int, default=64)
    parser.add_argument("--rebuild_feature_cache", type=int, default=0)
    parser.add_argument("--eval_chunk", type=int, default=8192, help="agent scoring chunk size for DNN eval")
    parser.add_argument("--amp", type=int, default=0, help="1 to enable autocast on CUDA (bfloat16)")
    parser.add_argument("--use_tool_id_emb", type=int, default=1)
    parser.add_argument("--use_llm_id_emb", type=int, default=1)
    parser.add_argument("--use_tool_emb", type=int, default=None, help="Deprecated alias for --use_tool_id_emb")
    parser.add_argument("--use_query_id_emb", type=int, default=0)
    parser.add_argument("--use_agent_id_emb", type=int, default=0)
    parser.add_argument("--use_model_content_vector", type=int, default=1)
    parser.add_argument("--use_tool_content_vector", type=int, default=1)
    parser.add_argument(
        "--soft_eval", "--soft_eva",
        dest="soft_eval",
        type=int,
        default=0,
        help="1 to evaluate by backbone-LLM match instead of exact agent id (default: 0)",
    )
    args = parser.parse_args()

    use_tool_id_emb = bool(args.use_tool_id_emb if args.use_tool_emb is None else args.use_tool_emb)
    use_llm_id_emb = bool(args.use_llm_id_emb)
    use_agent_id_emb = bool(args.use_agent_id_emb)
    use_model_content_vector = bool(args.use_model_content_vector)
    use_tool_content_vector = bool(args.use_tool_content_vector)

    boot = bootstrap_run(
        data_root=args.data_root,
        exp_name=args.exp_name,
        topk=args.topk,
        with_tools=True,
    )

    bundle = boot.bundle
    tools = boot.tools
    all_agents = bundle.all_agents
    all_questions = bundle.all_questions
    all_rankings = bundle.all_rankings
    qid_to_part = bundle.qid_to_part

    q_ids = boot.q_ids
    a_ids = boot.a_ids
    qid2idx = boot.qid2idx
    aid2idx = boot.aid2idx
    qids_in_rank = boot.qids_in_rank
    data_sig = boot.data_sig
    exp_cache_dir = boot.exp_cache_dir

    if args.embed_backend == "local":
        bge_device_for_cache = args.bge_device if args.bge_device else args.device
        embed_sig = cache_key_from_text(
            f"local|{args.bge_model}|{args.embed_batch}|{bge_device_for_cache}|fp16={args.bge_fp16}"
        )
    else:
        embed_sig = cache_key_from_text(f"api|{args.embed_url}|{args.embed_batch}")

    if args.feature_cache_dir:
        feature_cache_dir = args.feature_cache_dir
        print(f"[cache] using user-specified feature cache dir: {feature_cache_dir}")
    else:
        feature_cache_dir = shared_cache_dir(
            args.data_root,
            "features",
            f"twotower_bge_{embed_sig}_{data_sig}",
        )

    print(f"[cache] feature_cache_dir = {feature_cache_dir}")
    print(f"[cache] feature_cache_exists = {feature_cache_exists(feature_cache_dir)}")
    print(f"[cache] rebuild_feature_cache = {args.rebuild_feature_cache}")

    if feature_cache_exists(feature_cache_dir) and args.rebuild_feature_cache == 0:
        feature_cache = load_feature_cache(feature_cache_dir)
    else:
        if args.embed_backend == "local":
            bge_device = args.bge_device if args.bge_device else args.device

            def _patched_batch_embed(texts, embed_url="", batch_size=64, desc="Embedding", *unused_args, **unused_kwargs):
                return bge_batch_embed(
                    texts,
                    embed_url=embed_url,
                    batch_size=batch_size,
                    desc=desc,
                    model_path=args.bge_model,
                    device=bge_device,
                    use_fp16=bool(args.bge_fp16),
                    normalize=True,
                )

            features_mod.batch_embed = _patched_batch_embed
            print(f"[embed] using BGE-M3: {args.bge_model}")
            print(f"[embed] bge_device={bge_device}")
        else:
            print(f"[embed] using API embedding service: {args.embed_url}")

        feature_cache = build_twotower_bge_feature_cache(
            all_agents,
            all_questions,
            tools,
            embed_url=args.embed_url,
            embed_batch=args.embed_batch,
        )
        save_feature_cache(feature_cache_dir, feature_cache)
        print(f"[cache] saved features to {feature_cache_dir}")

        if args.embed_backend == "local":
            release_bge_model()

    if list(feature_cache.q_ids) != list(q_ids):
        raise RuntimeError(
            "[cache] q_ids in feature_cache do not match bootstrap q_ids. "
            "Please rebuild with --rebuild_feature_cache 1."
        )
    if list(feature_cache.a_ids) != list(a_ids):
        raise RuntimeError(
            "[cache] a_ids in feature_cache do not match bootstrap a_ids. "
            "Please rebuild with --rebuild_feature_cache 1."
        )

    Q_cpu = feature_cache.Q.astype(np.float32)
    A_cpu = build_agent_content_view(
        cache=feature_cache,
        use_model_content_vector=use_model_content_vector,
        use_tool_content_vector=use_tool_content_vector,
    ).astype(np.float32)

    aid_to_llm = None
    if build_aid_to_llm is not None:
        aid_to_llm = build_aid_to_llm(
            a_ids=a_ids,
            llm_vocab=list(feature_cache.llm_vocab),
            agent_llm_idx=np.asarray(feature_cache.agent_llm_idx, dtype=np.int64),
        )

    tool_ids_np = feature_cache.agent_tool_idx_padded
    tool_mask_np = feature_cache.agent_tool_mask

    train_parts = list(args.train_parts)
    eval_parts = list(args.eval_parts)
    qids_for_training = [qid for qid in qids_in_rank if qid_to_part.get(qid, "") in set(train_parts)]
    if not qids_for_training:
        raise RuntimeError(
            f"No qids found for train_parts={train_parts}. "
            f"Available parts include: {sorted(set(qid_to_part.values()))}"
        )
    print(f"[parts] train_parts={train_parts} -> qids={len(qids_for_training)}; eval_parts={eval_parts}")

    want_meta = {
        "data_sig": data_sig,
        "pos_topk_by_part": POS_TOPK_BY_PART,
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
        "pair_type": "q_pos_neg_posTopK",
        "train_parts": train_parts,
    }
    training_cache_dir = shared_cache_dir(args.data_root, "training", f"{data_sig}_{cache_key_from_meta(want_meta)}")

    def build_cache():
        train_qids, valid_qids = stratified_train_valid_split(
            qids_for_training,
            qid_to_part=qid_to_part,
            valid_ratio=args.valid_ratio,
            seed=args.split_seed,
        )
        print(f"[split] train_parts={train_parts} train={len(train_qids)}  valid={len(valid_qids)}")
        pairs = build_training_pairs(
            {qid: all_rankings[qid] for qid in train_qids},
            a_ids,
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            neg_per_pos=args.neg_per_pos,
            rng_seed=args.rng_seed_pairs,
        )
        pairs_idx = [(qid2idx[q], aid2idx[p], aid2idx[n]) for (q, p, n) in pairs]
        return train_qids, valid_qids, np.asarray(pairs_idx, dtype=np.int64)

    train_qids, valid_qids, pairs_idx_np = load_or_build_training_cache(
        training_cache_dir,
        args.rebuild_training_cache,
        want_meta,
        build_cache,
    )

    device = torch.device(args.device)
    model = SimpleBPRDNN(
        d_q=int(Q_cpu.shape[1]),
        d_a=int(A_cpu.shape[1]),
        num_tools=int(len(feature_cache.tool_id_vocab)),
        num_llm_ids=int(len(feature_cache.llm_vocab)),
        agent_tool_indices_padded=torch.tensor(tool_ids_np, dtype=torch.long, device=device),
        agent_tool_mask=torch.tensor(tool_mask_np, dtype=torch.float32, device=device),
        agent_llm_idx=torch.tensor(feature_cache.agent_llm_idx, dtype=torch.long, device=device),
        text_hidden=int(args.text_hidden),
        id_dim=int(args.id_dim),
        num_queries=len(q_ids),
        num_agents=len(a_ids),
        use_query_id_emb=bool(args.use_query_id_emb),
        use_agent_id_emb=use_agent_id_emb,
        use_tool_id_emb=use_tool_id_emb,
        use_llm_id_emb=use_llm_id_emb,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_pairs = int(pairs_idx_np.shape[0])
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    use_amp = args.amp == 1 and device.type == "cuda"
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(pairs_idx_np)
        total_loss = 0.0
        model.train()
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for b in pbar:
            sl = slice(b * args.batch_size, min((b + 1) * args.batch_size, num_pairs))
            batch = pairs_idx_np[sl]
            if batch.size == 0:
                continue

            q_idx = batch[:, 0]
            pos_idx = batch[:, 1]
            neg_idx = batch[:, 2]

            q_vec = torch.from_numpy(Q_cpu[q_idx]).to(device, non_blocking=True)
            pos_vec = torch.from_numpy(A_cpu[pos_idx]).to(device, non_blocking=True)
            neg_vec = torch.from_numpy(A_cpu[neg_idx]).to(device, non_blocking=True)
            q_idx_t = torch.from_numpy(q_idx).to(device, non_blocking=True)
            pos_idx_t = torch.from_numpy(pos_idx).to(device, non_blocking=True)
            neg_idx_t = torch.from_numpy(neg_idx).to(device, non_blocking=True)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with autocast_ctx:
                pos_score, neg_score = model(q_vec, pos_vec, neg_vec, pos_idx_t, neg_idx_t, q_idx=q_idx_t)
                loss = bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss / (b + 1):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total_loss / max(1, num_batches)):.4f}")

    model_dir = os.path.join(exp_cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{args.exp_name}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{args.exp_name}_{data_sig}.json")

    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {
            "d_q": int(Q_cpu.shape[1]),
            "d_a": int(A_cpu.shape[1]),
            "text_hidden": int(args.text_hidden),
            "id_dim": int(args.id_dim),
            "num_tools": int(len(feature_cache.tool_id_vocab)),
            "num_llm_ids": int(len(feature_cache.llm_vocab)),
            "num_agents": int(len(a_ids)),
        },
        "parts": {"train_parts": train_parts, "eval_parts": eval_parts},
        "flags": {
            "use_tool_id_emb": use_tool_id_emb,
            "use_llm_id_emb": use_llm_id_emb,
            "use_model_content_vector": use_model_content_vector,
            "use_tool_content_vector": use_tool_content_vector,
            "use_query_id_emb": bool(args.use_query_id_emb),
            "use_agent_id_emb": use_agent_id_emb,
        },
        "mappings": {"q_ids": q_ids, "a_ids": a_ids, "tool_names": feature_cache.tool_names},
    }
    torch.save(ckpt, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_sig": data_sig,
                "q_ids": q_ids,
                "a_ids": a_ids,
                "tool_names": feature_cache.tool_names,
                "train_parts": train_parts,
                "eval_parts": eval_parts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[save] model -> {model_path}")
    print(f"[save] meta  -> {meta_path}")

    evaluate_parts_dnn_bge(
        model=model,
        Q_cpu=Q_cpu,
        A_cpu=A_cpu,
        qid2idx=qid2idx,
        a_ids=a_ids,
        aid2idx=aid2idx,
        all_rankings=all_rankings,
        qids_in_rank=qids_in_rank,
        qid_to_part=qid_to_part,
        eval_parts=eval_parts,
        device=device,
        topk=int(args.topk),
        cand_size=int(args.eval_cand_size),
        rng_seed=int(args.rng_seed_pairs),
        eval_chunk=int(args.eval_chunk),
        amp=use_amp,
        exp_name=args.exp_name,
        soft_eval=bool(args.soft_eval),
        aid_to_llm=aid_to_llm,
        pos_topk_by_part=POS_TOPK_BY_PART,
        pos_topk_default=POS_TOPK,
    )

    # The original print_sample_recommendations in eval.py expects a two-tower
    # encoder with encode_q/encode_a. SimpleBPRDNN is a pair scorer, so we do
    # not call it here to avoid misleading output or interface errors.


if __name__ == "__main__":
    main()

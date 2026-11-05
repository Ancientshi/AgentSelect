#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-Tower BGE Agent Recommender (InfoNCE) — OOM-safe + BGE-M3 version.

Key points:
1) Keep embedding matrices on CPU; move ONLY current batch to GPU.
2) Evaluation & inference are chunked/sampled over agents.
3) --eval_chunk controls agent-encoding batch size for optional inference.
4) --amp optionally enables autocast(bfloat16) on CUDA.
5) Feature cache can be built with BGE-M3 instead of HTTP API.

Default BGE model path:
  path_to/models/BAAI/bge-m3

Example:
python run_twotower_bge.py \
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
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

import agent_rec.features as features_mod
from agent_rec.cli_common import add_shared_training_args
from agent_rec.config import POS_TOPK, POS_TOPK_BY_PART
from agent_rec.data import stratified_train_valid_split
from agent_rec.features import (
    build_agent_content_view,
    build_twotower_bge_feature_cache,
    feature_cache_exists,
    load_feature_cache,
    save_feature_cache,
)
from agent_rec.models.two_tower import TwoTowerTFIDF
from agent_rec.run_common import (
    bootstrap_run,
    cache_key_from_meta,
    cache_key_from_text,
    build_pos_pairs,
    load_or_build_training_cache,
    shared_cache_dir,
)
from eval import build_aid_to_llm, evaluate_parts, print_sample_recommendations


# ===== BGE-M3 embedding backend =====
_BGE_MODEL = None


def _get_bge_model(model_path: str, device: str = "cuda", use_fp16: bool = True):
    """
    Load BGE-M3 once. Prefer FlagEmbedding BGEM3FlagModel.
    Fallback to sentence-transformers if FlagEmbedding is unavailable.
    """
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

        model = SentenceTransformer(model_path, device=device)
        _BGE_MODEL = ("st", model)
        print("[bge] loaded with sentence-transformers")

    return _BGE_MODEL


def release_bge_model() -> None:
    """Free BGE model after feature cache is built, so training can use GPU memory."""
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
    """
    Drop-in replacement for agent_rec.features.batch_embed().
    It keeps the same leading arguments so build_twotower_bge_feature_cache() can call it unchanged.
    """
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
            denom = np.maximum(denom, 1e-8)
            emb = emb / denom

        chunks.append(emb.astype(np.float32, copy=False))

    return np.vstack(chunks).astype(np.float32, copy=False)


def info_nce_loss(qe: torch.Tensor, ae: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = qe @ ae.t()
    labels = torch.arange(qe.size(0), device=qe.device)
    return torch.nn.functional.cross_entropy(logits / temperature, labels)




def feature_cache_exists(cache_dir: str) -> bool:
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
    return os.path.isdir(cache_dir) and all(
        os.path.exists(os.path.join(cache_dir, f)) for f in required
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    add_shared_training_args(
        parser,
        exp_name_default="two_tower_bge",
        device_default="cpu",
        epochs_default=3,
        batch_size_default=512,
        lr_default=3e-4,
        include_neg_per_pos=False,
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
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--rebuild_feature_cache", type=int, default=0)
    parser.add_argument("--eval_chunk", type=int, default=8192, help="batch size over agents for inference")
    parser.add_argument("--amp", type=int, default=0, help="1 to enable autocast on CUDA (bfloat16)")
    parser.add_argument("--use_tool_id_emb", type=int, default=1)
    parser.add_argument("--use_llm_id_emb", type=int, default=0, help="1 to add LLM-ID embedding into agent tower")
    parser.add_argument(
        "--use_model_content_vector", type=int, default=1, help="1 to include V_model(A) in agent content view"
    )
    parser.add_argument(
        "--use_tool_content_vector", type=int, default=1, help="1 to include V_tool_content(A) in agent content view"
    )
    parser.add_argument("--use_tool_emb", type=int, default=None, help="Deprecated alias for --use_tool_id_emb")
    parser.add_argument("--use_agent_id_emb", type=int, default=0, help="1 to add learnable per-agent ID embedding into agent tower")
    parser.add_argument("--use_query_id_emb", type=int, default=0, help="1 to add query-ID embedding into query tower")
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

            def _patched_batch_embed(
                texts,
                embed_url="",
                batch_size=64,
                desc="Embedding",
                *unused_args,
                **unused_kwargs,
            ):
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

    # Sanity check: feature rows must match bootstrap id order used by training/eval.
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
    )

    # agent_id -> backbone LLM string (used only when --soft_eval 1)
    aid_to_llm = build_aid_to_llm(
        a_ids=a_ids,
        llm_vocab=list(feature_cache.llm_vocab),
        agent_llm_idx=np.asarray(feature_cache.agent_llm_idx, dtype=np.int64),
    )

    tool_ids_np = feature_cache.agent_tool_idx_padded
    tool_mask_np = feature_cache.agent_tool_mask

    # Make --train_parts control which question parts are used for training.
    # This mirrors run_twotower_tfidf.py. Keep train_parts in the cache key so a
    # stale PartI+II+III training cache is never reused for PartIII-only runs.
    train_parts = list(args.train_parts)
    eval_parts = list(args.eval_parts)
    qids_for_training = [
        qid for qid in qids_in_rank
        if qid_to_part.get(qid, "") in set(train_parts)
    ]
    if not qids_for_training:
        raise RuntimeError(
            f"No qids found for train_parts={train_parts}. "
            f"Available parts include: {sorted(set(qid_to_part.values()))}"
        )
    print(
        f"[parts] train_parts={train_parts} -> qids={len(qids_for_training)}; "
        f"eval_parts={eval_parts}"
    )

    want_meta = {
        "data_sig": data_sig,
        "pos_topk_by_part": POS_TOPK_BY_PART,
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
        "pair_type": "q_pos_only_posTopK",
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
        print(
            f"[split] train_parts={train_parts} "
            f"train={len(train_qids)}  valid={len(valid_qids)}"
        )
        pairs = build_pos_pairs(
            {qid: all_rankings[qid] for qid in train_qids},
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            rng_seed=args.rng_seed_pairs,
        )
        pairs_idx = [(qid2idx[q], aid2idx[a]) for (q, a) in pairs]
        pairs_idx_np = np.array(pairs_idx, dtype=np.int64)
        return train_qids, valid_qids, pairs_idx_np

    train_qids, valid_qids, pairs_idx_np = load_or_build_training_cache(
        training_cache_dir,
        args.rebuild_training_cache,
        want_meta,
        build_cache,
    )

    device = torch.device(args.device)
    encoder = TwoTowerTFIDF(
        d_q=int(Q_cpu.shape[1]),
        d_a=int(A_cpu.shape[1]),
        num_tools=int(len(feature_cache.tool_id_vocab)),
        num_llm_ids=int(len(feature_cache.llm_vocab)),
        agent_tool_idx_padded=torch.tensor(tool_ids_np, dtype=torch.long, device=device),
        agent_tool_mask=torch.tensor(tool_mask_np, dtype=torch.float32, device=device),
        agent_llm_idx=torch.tensor(feature_cache.agent_llm_idx, dtype=torch.long, device=device),
        hid=args.hid,
        use_tool_id_emb=use_tool_id_emb,
        use_llm_id_emb=use_llm_id_emb,
        num_agents=len(a_ids),
        num_queries=len(q_ids),
        use_query_id_emb=bool(args.use_query_id_emb),
        use_agent_id_emb=use_agent_id_emb,
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    num_pairs = pairs_idx_np.shape[0]
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    use_amp = args.amp == 1 and device.type == "cuda"
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(pairs_idx_np)
        total = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        encoder.train()
        for b in pbar:
            sl = slice(b * args.batch_size, min((b + 1) * args.batch_size, num_pairs))
            batch = pairs_idx_np[sl]
            if batch.size == 0:
                continue
            q_idx = batch[:, 0]
            a_idx = batch[:, 1]

            q_vec = torch.from_numpy(Q_cpu[q_idx]).to(device, non_blocking=True)
            q_idx_t = torch.from_numpy(q_idx).to(device, non_blocking=True)
            a_pos = torch.from_numpy(A_cpu[a_idx]).to(device, non_blocking=True)
            a_idx_t = torch.from_numpy(a_idx).to(device, non_blocking=True)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with autocast_ctx:
                qe = encoder.encode_q(q_vec, q_idx=q_idx_t)
                ae = encoder.encode_a(a_pos, a_idx_t)
                loss = info_nce_loss(qe, ae, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total / (b + 1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - InfoNCE: {(total / max(1, num_batches)):.4f}")

    model_dir = os.path.join(exp_cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{args.exp_name}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{args.exp_name}_{data_sig}.json")

    ckpt = {
        "state_dict": encoder.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {
            "d_q": int(Q_cpu.shape[1]),
            "d_a": int(A_cpu.shape[1]),
            "hid": int(args.hid),
            "num_tools": int(len(feature_cache.tool_id_vocab)),
            "num_agents": int(len(a_ids)),
        },
        "parts": {
            "train_parts": train_parts,
            "eval_parts": eval_parts,
        },
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

    topk = int(args.topk)

    # Final reporting is controlled by --eval_parts, not by valid_qids.
    # The actual sampled-eval implementation is shared in eval.py so BGE and
    # TF-IDF can stay aligned except for feature/cache construction.
    evaluate_parts(
        encoder=encoder,
        Q_cpu=Q_cpu,
        A_cpu=A_cpu,
        qid2idx=qid2idx,
        a_ids=a_ids,
        all_rankings=all_rankings,
        qids_in_rank=qids_in_rank,
        qid_to_part=qid_to_part,
        eval_parts=eval_parts,
        device=device,
        topk=topk,
        cand_size=args.eval_cand_size,
        rng_seed=args.rng_seed_pairs,
        amp=use_amp,
        exp_name=args.exp_name,
        soft_eval=bool(args.soft_eval),
        aid_to_llm=aid_to_llm,
        pos_topk_by_part=POS_TOPK_BY_PART,
        pos_topk_default=POS_TOPK,
    )

    print_sample_recommendations(
        encoder=encoder,
        Q_cpu=Q_cpu,
        A_cpu=A_cpu,
        qid2idx=qid2idx,
        a_ids=a_ids,
        q_ids=q_ids,
        all_questions=all_questions,
        device=device,
        topk=args.topk,
        chunk=args.eval_chunk,
        num_samples=5,
    )


if __name__ == "__main__":
    main()

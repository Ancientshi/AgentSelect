#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime
import numpy as np
import torch
from tqdm.auto import tqdm

from agent_rec.cli_common import add_shared_training_args
from agent_rec.config import POS_TOPK, POS_TOPK_BY_PART, TFIDF_MAX_FEATURES
from agent_rec.data import stratified_train_valid_split
from agent_rec.features import (
    build_agent_content_view,
    build_feature_cache,
    feature_cache_exists,
    load_feature_cache,
    save_feature_cache,
    save_vectorizers,
    load_vectorizers,
)
from agent_rec.models.two_tower import TwoTowerTFIDF
from agent_rec.run_common import (
    bootstrap_run,
    cache_key_from_meta,
    build_pos_pairs,
    load_or_build_training_cache,
    shared_cache_dir,
)

from eval import build_aid_to_llm, evaluate_parts, print_sample_recommendations


def info_nce_loss(qe: torch.Tensor, ae: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = qe @ ae.t()
    labels = torch.arange(qe.size(0), device=qe.device)
    return torch.nn.functional.cross_entropy(logits / temperature, labels)




def main() -> None:
    parser = argparse.ArgumentParser()
    add_shared_training_args(
        parser,
        exp_name_default="two_tower_tfidf",
        device_default="cpu",
        epochs_default=3,
        batch_size_default=512,
        lr_default=3e-4,
        include_neg_per_pos=False,
    )
    parser.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)
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
    feature_cache_dir = shared_cache_dir(
        args.data_root,
        "features",
        f"tfidf_{args.max_features}_{data_sig}",
    )

    if feature_cache_exists(feature_cache_dir) and args.rebuild_feature_cache == 0:
        feature_cache = load_feature_cache(feature_cache_dir)
        vecs = load_vectorizers(feature_cache_dir)
        if vecs is None:
            raise RuntimeError(
                f"[cache] feature cache exists but vectorizers are missing in {feature_cache_dir}. "
                f"Please rebuild with --rebuild_feature_cache 1."
            )
        print(f"[cache] loaded features from {feature_cache_dir}")
    else:
        feature_cache, vecs = build_feature_cache(
            all_agents, all_questions, tools, max_features=args.max_features
        )
        os.makedirs(feature_cache_dir, exist_ok=True)
        save_feature_cache(feature_cache_dir, feature_cache)
        save_vectorizers(feature_cache_dir, vecs)
        print(f"[cache] rebuilt & saved features to {feature_cache_dir}")

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

    # Make --train_parts actually control which question parts are used for training.
    # IMPORTANT: include train_parts in the training cache key, otherwise an old
    # PartI+II+III cache can be silently reused when running --train_parts PartIII.
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

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
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
    data_sig = want_meta["data_sig"]
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
    # The actual sampled-eval implementation is shared in eval.py so TF-IDF
    # and BGE stay aligned except for feature/cache construction.
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

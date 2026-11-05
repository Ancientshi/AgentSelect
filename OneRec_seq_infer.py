#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer/Eval script for OneRec++ one-step model (Query->Agent).

What it does:
  - Rebuilds TF-IDF features exactly like training (fit on all questions).
  - Rebuilds vocab (sorted agent ids) to match ckpt token table.
  - Uses the same split logic (stratified by part).
  - Samples 200 queries per part from train + valid and prints metric tables.
  - Adds OOD diagnostics:
      * How much of predicted TopK are "train-seen GT agents"
      * Unseen-valid-GT coverage and Hit@K on queries whose GT is unseen.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm

from agent_rec.config import DEFAULT_DATA_ROOT, TFIDF_MAX_FEATURES, EVAL_TOPK, pos_topk_for_qid
from agent_rec.data import collect_data, dataset_content_signature, load_tools, qids_with_rankings, stratified_train_valid_split
from agent_rec.eval import sample_eval_candidates
from agent_rec.run_common import set_global_seed, warn_if_topk_diff

from OneRec_seq import (
    AgentVocab,
    OneStepGenerator,
    QueryEncoder,
    build_q_vectorizer,
    dataset_signature,
    evaluate_sampled,
    sample_qids_by_part,
)
from utils import print_metrics_table


def build_train_seen_agent_set(
    qids: List[str],
    all_rankings: Dict[str, List[str]],
    qid_to_part: Dict[str, str],
) -> set:
    s = set()
    for qid in qids:
        k_pos = pos_topk_for_qid(qid, qid_to_part)
        gt = all_rankings.get(qid, [])[:k_pos]
        s.update(gt)
    return s

# ---------------------- eval with diagnostics ----------------------

@torch.no_grad()
def eval_with_diag(
    gen: OneStepGenerator,
    q_enc: QueryEncoder,
    Q_csr,
    qid2idx: Dict[str,int],
    a_ids: List[str],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    qid_to_part: Dict[str, str],
    train_seen_agents: set,
    device: torch.device,
    topk: int,
    cand_size: int,
    rng_seed: int,
):
    vocab = AgentVocab(a_ids)
    all_agent_set = set(a_ids)


    ks = (topk,)
    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt = 0
    skipped = 0

    # diagnostics
    frac_pred_in_train_seen_sum = 0.0
    all_pred_in_train_seen_cnt = 0
    unique_pred = set()

    # unseen-valid-GT diagnostics (query-level)
    unseen_gt_query_cnt = 0
    unseen_gt_hit_cnt = 0
    unseen_gt_pred_cover_sum = 0.0  # fraction of preds in unseen_gt_set, averaged over those queries

    # precompute query encodings on the fly (batch size 1, cheap enough)
    q_enc.eval()
    gen.eval()

    pbar = tqdm(eval_qids, desc="Eval", total=len(eval_qids))
    for qid in pbar:
        k_pos = pos_topk_for_qid(qid, qid_to_part)
        gt_list = [aid for aid in all_rankings.get(qid, [])[:k_pos] if aid in all_agent_set]
        if not gt_list:
            skipped += 1
            continue

        rel_set = set(gt_list)

        # build sampled candidate pool (ensure GT included)
        cand_ids = sample_eval_candidates(a_ids, gt_list, cand_size, qid=qid, seed=rng_seed)

        cand_tokens = [vocab.aid_to_token(a) for a in cand_ids]
        cand_tok = torch.tensor([cand_tokens], dtype=torch.long, device=device)

        # encode q (densify only this row)
        qi = qid2idx[qid]
        q_dense = Q_csr[qi:qi+1].toarray().astype(np.float32)
        q_x = torch.from_numpy(q_dense).to(device, non_blocking=True)

        enc = q_enc(q_x)
        pred_tok = gen.generate(enc, topk=topk, cand_tok=cand_tok, temperature=0.0)

        pred_ids = []
        for t in pred_tok[0].tolist():
            aid = vocab.token_to_aid(t)
            if aid is not None:
                pred_ids.append(aid)

        for a in pred_ids:
            unique_pred.add(a)

        # metrics
        md = evaluate_sampled(pred_ids, rel_set, ks)
        for k in ks:
            for m in md[k]:
                agg[k][m] += md[k][m]
        cnt += 1

        # diag: train-seen coverage
        if len(pred_ids) > 0:
            in_train_seen = sum(1 for a in pred_ids if a in train_seen_agents)
            frac = in_train_seen / float(len(pred_ids))
            frac_pred_in_train_seen_sum += frac
            if in_train_seen == len(pred_ids):
                all_pred_in_train_seen_cnt += 1

        # diag: unseen-valid-GT
        unseen_gt_set = rel_set - train_seen_agents
        if len(unseen_gt_set) > 0:
            unseen_gt_query_cnt += 1
            hit_unseen = 1 if any(a in unseen_gt_set for a in pred_ids) else 0
            unseen_gt_hit_cnt += hit_unseen
            if pred_ids:
                unseen_gt_pred_cover_sum += sum(a in unseen_gt_set for a in pred_ids) / len(pred_ids)

        if cnt > 0 and (cnt % 50 == 0):
            pbar.set_postfix({
                "done": cnt,
                "skip": skipped,
                "P@K": f"{agg[topk]['P']/cnt:.4f}",
                "nDCG@K": f"{agg[topk]['nDCG']/cnt:.4f}",
                "trainSeenFrac": f"{(frac_pred_in_train_seen_sum/cnt):.3f}",
            })

    if cnt == 0:
        metrics = {topk:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]}}
        diag = {}
        return metrics, diag

    for m in agg[topk]:
        agg[topk][m] /= cnt

    diag = {
        "n_eval": cnt,
        "n_skipped": skipped,
        "unique_pred_agents": len(unique_pred),
        "avg_frac_pred_in_train_seen": frac_pred_in_train_seen_sum / cnt,
        "pct_queries_all_pred_in_train_seen": all_pred_in_train_seen_cnt / cnt,
        "unseen_gt_query_cnt": unseen_gt_query_cnt,
        "unseen_gt_hit_rate": (unseen_gt_hit_cnt / unseen_gt_query_cnt) if unseen_gt_query_cnt > 0 else 0.0,
        "unseen_gt_avg_frac_pred": (unseen_gt_pred_cover_sum / unseen_gt_query_cnt) if unseen_gt_query_cnt > 0 else 0.0,
    }
    return agg, diag

def pretty_print_diag(title: str, diag: dict):
    print(f"\n---- {title} (OOD diagnostics) ----")
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"{k:>32s}: {v:.6f}")
        else:
            print(f"{k:>32s}: {v}")

# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--device", type=str, default="cpu")

    # TF-IDF / shape must match ckpt
    ap.add_argument("--max_features", type=int, help="Use the checkpoint value when omitted.")

    # split
    ap.add_argument("--valid_ratio", type=float, help="Use the checkpoint value when omitted.")
    ap.add_argument("--split_seed", type=int, help="Use the checkpoint value when omitted.")

    # eval
    ap.add_argument("--topk", type=int, default=EVAL_TOPK)
    ap.add_argument("--eval_candidate_size", type=int, help="Use the checkpoint value when omitted.")
    ap.add_argument("--eval_per_part", type=int, help="Use the checkpoint value when omitted.")
    ap.add_argument("--eval_parts", type=str, help="Use the checkpoint scope when omitted.")
    ap.add_argument("--seed", type=int, help="Seed for sampling evaluation queries.")
    ap.add_argument("--eval_seed", type=int, help="Seed for sampled evaluation candidates.")

    # ckpt
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to ckpt .pt. If empty, auto-locate via data_sig under .cache/OneRec_seq/models/")
    ap.add_argument("--ckpt_prefix", type=str, default="OneRec_seq",
                    help="Cache prefix used during training (default: OneRec_seq).")

    args = ap.parse_args()
    device = torch.device(args.device)

    # ---------- load data ----------
    bundle = collect_data(args.data_root)
    all_questions = bundle.all_questions
    all_rankings = bundle.all_rankings
    qid_to_part = bundle.qid_to_part
    q_ids = sorted(all_questions)
    q_texts = [all_questions[qid].get("input", "") for qid in q_ids]
    a_ids = sorted(bundle.all_agents)
    qid2idx = {qid:i for i,qid in enumerate(q_ids)}
    vocab = AgentVocab(a_ids)

    # ---------- locate ckpt ----------
    data_sig = dataset_signature(a_ids, all_rankings)

    if args.ckpt is None:
        ckpt = os.path.join(
            args.data_root,
            f".cache/{args.ckpt_prefix}",
            "models",
            f"{args.ckpt_prefix}_{data_sig}.pt",
        )
        args.ckpt = ckpt

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"OneRec checkpoint not found: {args.ckpt}")
    print(f"[ckpt] load: {args.ckpt}")
    ck = torch.load(args.ckpt, map_location=device)
    if not isinstance(ck, dict) or "q_enc" not in ck or "gen" not in ck:
        raise ValueError(f"Invalid OneRec checkpoint: {args.ckpt}")
    if ck.get("data_sig") != data_sig:
        raise ValueError("Checkpoint does not match the selected dataset's agent IDs and rankings.")
    if ck.get("content_sig") is not None:
        current_sig = dataset_content_signature(bundle, load_tools(args.data_root))
        if ck["content_sig"] != current_sig:
            raise ValueError("Checkpoint data has changed. Use the original dataset or retrain OneRec.")

    settings = ck.get("args", {})
    for name, default in (("max_features", TFIDF_MAX_FEATURES), ("valid_ratio", 0.2), ("split_seed", 42)):
        requested = getattr(args, name)
        saved = settings.get(name)
        if requested is not None and saved is not None and requested != saved:
            raise ValueError(f"--{name}={requested} does not match checkpoint value {saved}.")
        if requested is None:
            setattr(args, name, saved if saved is not None else default)
    for name, default in (("eval_candidate_size", 1000), ("eval_per_part", 200), ("seed", 1234), ("eval_seed", 42)):
        if getattr(args, name) is None:
            setattr(args, name, settings.get(name, default))
    if args.eval_parts is None:
        scope = ck.get("dpo_part_scope", "ALL")
        args.eval_parts = scope if scope != "ALL" else settings.get("eval_parts", "PartI,PartII,PartIII")
    eval_parts = [part.strip() for part in args.eval_parts.split(",") if part.strip()]
    if args.topk <= 0 or args.eval_candidate_size < args.topk:
        raise ValueError("--eval_candidate_size must be at least --topk, and --topk must be positive.")
    warn_if_topk_diff(args.topk, expected=EVAL_TOPK)
    set_global_seed(args.seed)

    # ---------- rebuild TF-IDF ----------
    _, Q_csr = build_q_vectorizer(q_texts, args.max_features)
    print(f"[tfidf] Q={Q_csr.shape}")

    # ---------- split ----------
    qids_in_rank = qids_with_rankings(q_ids, all_rankings)
    train_qids, valid_qids = stratified_train_valid_split(
        qids_in_rank,
        qid_to_part=qid_to_part,
        valid_ratio=args.valid_ratio,
        seed=args.split_seed
    )
    print(f"[split] train={len(train_qids)} valid={len(valid_qids)} (valid_ratio={args.valid_ratio}, seed={args.split_seed})")

    # ---------- build train-seen GT agent set ----------
    train_seen_agents = build_train_seen_agent_set(
        train_qids, all_rankings, qid_to_part
    )
    valid_seen_agents = build_train_seen_agent_set(
        valid_qids, all_rankings, qid_to_part
    )
    unseen_valid_agents = valid_seen_agents - train_seen_agents
    print(f"[agents] train_seen_gt={len(train_seen_agents)} valid_gt={len(valid_seen_agents)} unseen_valid_gt={len(unseen_valid_agents)}")

    # ---------- build models and load weights ----------
    q_sd = ck["q_enc"]
    g_sd = ck["gen"]

    # infer dims from ckpt tensors (DO NOT GUESS)
    d_q = Q_csr.shape[1]
    if q_sd["net.0.weight"].shape[1] != d_q or g_sd["tok_emb.weight"].shape[0] != vocab.vocab_size:
        raise ValueError("Checkpoint dimensions do not match the rebuilt query features and agent vocabulary.")

    # QueryEncoder hidden (=enc_dim) from first Linear: net.0.weight is (enc_dim, d_q)
    enc_dim = int(q_sd["net.0.weight"].shape[0])

    # Generator dims from q_proj.0.weight: (gen_hidden, enc_dim)
    gen_hidden = int(g_sd["q_proj.0.weight"].shape[0])
    tok_dim = int(g_sd["tok_emb.weight"].shape[1])

    q_enc = QueryEncoder(d_q=d_q, hidden=enc_dim, dropout=0.0).to(device)
    gen = OneStepGenerator(
        enc_dim=enc_dim,
        vocab_size=vocab.vocab_size,
        tok_dim=tok_dim,
        hidden=gen_hidden,
        dropout=0.0
    ).to(device)


    q_enc.load_state_dict(q_sd, strict=True)
    gen.load_state_dict(g_sd, strict=True)
    q_enc.eval(); gen.eval()

    print(f"[model] enc_dim={enc_dim} tok_dim={tok_dim} vocab={vocab.vocab_size} ck_mode={ck.get('mode','?')}")

    # ---------- sample qids ----------
    parts = eval_parts
    train_eval_qids = sample_qids_by_part(train_qids, qid_to_part, per_part=args.eval_per_part, seed=args.seed, parts=parts)
    valid_eval_qids = sample_qids_by_part(valid_qids, qid_to_part, per_part=args.eval_per_part, seed=args.seed, parts=parts)
    print(f"[eval] train_sampled={len(train_eval_qids)} valid_sampled={len(valid_eval_qids)} per_part={args.eval_per_part} parts={parts}")

    # ---------- run eval: train ----------
    m_train, d_train = eval_with_diag(
        gen=gen, q_enc=q_enc, Q_csr=Q_csr, qid2idx=qid2idx,
        a_ids=a_ids, all_rankings=all_rankings, eval_qids=train_eval_qids,
        qid_to_part=qid_to_part, train_seen_agents=train_seen_agents,
        device=device, topk=args.topk, cand_size=args.eval_candidate_size,
        rng_seed=args.eval_seed
    )
    print_metrics_table("TRAIN (sampled per-part)", m_train, ks=(args.topk,), filename="OneRec_infer")
    pretty_print_diag("TRAIN", d_train)

    # per-part train
    for p in parts:
        q_part = [q for q in train_eval_qids if qid_to_part.get(q, "Unknown") == p]
        if not q_part:
            continue
        mp, dp = eval_with_diag(
            gen=gen, q_enc=q_enc, Q_csr=Q_csr, qid2idx=qid2idx,
            a_ids=a_ids, all_rankings=all_rankings, eval_qids=q_part,
            qid_to_part=qid_to_part, train_seen_agents=train_seen_agents,
            device=device, topk=args.topk, cand_size=args.eval_candidate_size,
            rng_seed=args.eval_seed
        )
        print_metrics_table(f"TRAIN {p} (n={len(q_part)})", mp, ks=(args.topk,), filename="OneRec_infer")
        pretty_print_diag(f"TRAIN {p}", dp)

    # ---------- run eval: valid ----------
    m_valid, d_valid = eval_with_diag(
        gen=gen, q_enc=q_enc, Q_csr=Q_csr, qid2idx=qid2idx,
        a_ids=a_ids, all_rankings=all_rankings, eval_qids=valid_eval_qids,
        qid_to_part=qid_to_part, train_seen_agents=train_seen_agents,
        device=device, topk=args.topk, cand_size=args.eval_candidate_size,
        rng_seed=args.eval_seed
    )
    print_metrics_table("VALID (sampled per-part)", m_valid, ks=(args.topk,), filename="OneRec_infer")
    pretty_print_diag("VALID", d_valid)

    # per-part valid
    for p in parts:
        q_part = [q for q in valid_eval_qids if qid_to_part.get(q, "Unknown") == p]
        if not q_part:
            continue
        mp, dp = eval_with_diag(
            gen=gen, q_enc=q_enc, Q_csr=Q_csr, qid2idx=qid2idx,
            a_ids=a_ids, all_rankings=all_rankings, eval_qids=q_part,
            qid_to_part=qid_to_part, train_seen_agents=train_seen_agents,
            device=device, topk=args.topk, cand_size=args.eval_candidate_size,
            rng_seed=args.eval_seed
        )
        print_metrics_table(f"VALID {p} (n={len(q_part)})", mp, ks=(args.topk,), filename="OneRec_infer")
        pretty_print_diag(f"VALID {p}", dp)

    # extra: global unseen-valid-GT summary
    print("\n==== Global unseen-valid-GT summary (set-level) ====")
    print(f"unseen_valid_gt_agents={len(unseen_valid_agents)} (valid_gt - train_gt)")
    # 这里不做更重的统计了（你上面 query-level unseen_hit 已经够判断“只背训练 agent 还是能命中 valid GT”）

if __name__ == "__main__":
    main()

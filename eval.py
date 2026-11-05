#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared Two-Tower evaluation utilities.

This file is intentionally feature-agnostic: TF-IDF/BGE/DNN-style two-tower
scripts can reuse the same sampled evaluation and inference reporting logic,
while keeping their own feature/cache construction separate.
"""

from __future__ import annotations

import hashlib
import math
import random
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from agent_rec.config import POS_TOPK, POS_TOPK_BY_PART
from agent_rec.eval import split_eval_qids_by_part
from utils import print_metrics_table


def stable_seed_from_qid(qid: str, rng_seed: int) -> int:
    """Stable per-query seed; do not use Python hash(), which changes across processes."""
    h = hashlib.blake2b(str(qid).encode("utf-8"), digest_size=8).digest()
    q_seed = int.from_bytes(h, byteorder="little", signed=False)
    return (q_seed ^ (int(rng_seed) * 16777619)) & 0xFFFFFFFF


def build_aid_to_llm(a_ids: List[str], llm_vocab: List[str], agent_llm_idx: np.ndarray) -> Dict[str, str]:
    """Build agent_id -> backbone LLM string mapping for optional soft_eval."""
    agent_llm_idx = np.asarray(agent_llm_idx, dtype=np.int64)
    aid_to_llm: Dict[str, str] = {}
    for i, aid in enumerate(a_ids):
        idx = int(agent_llm_idx[i]) if i < agent_llm_idx.shape[0] else -1
        aid_to_llm[aid] = llm_vocab[idx] if 0 <= idx < len(llm_vocab) else ""
    return aid_to_llm


@torch.no_grad()
def evaluate_sampled_twotower(
    encoder,
    Q_cpu: np.ndarray,
    A_cpu: np.ndarray,
    qid2idx: Dict[str, int],
    a_ids: List[str],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    device: torch.device,
    ks: Tuple[int, ...] = (5, 10, 50),
    cand_size: int = 100,
    rng_seed: int = 0,
    amp: bool = False,
    qid_to_part: Dict[str, str] | None = None,
    pos_topk_by_part: Dict[str, int] = POS_TOPK_BY_PART,
    pos_topk_default: int = POS_TOPK,
    soft_eval: bool = False,
    aid_to_llm: Dict[str, str] | None = None,
) -> Dict[int, Dict[str, float]]:
    """
    Sampled q-vector evaluation shared by TF-IDF/BGE two-tower models.

    Candidate set = ground-truth positives + sampled negatives.
    strict mode: relevance is exact agent id match.
    soft_eval mode: relevance is backbone-LLM match, counted at most once.
    """
    max_k = max(ks)
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}
    if soft_eval and aid_to_llm is None:
        raise ValueError("soft_eval=True requires aid_to_llm mapping (agent_id -> backbone_llm).")

    def _binary_hits_soft(pred_ids: List[str], rel_llm_set: set[str], k: int) -> List[int]:
        hits = []
        seen = set()
        for aid in pred_ids[:k]:
            llm = aid_to_llm.get(aid, "")  # type: ignore[union-attr]
            if llm in rel_llm_set and llm not in seen:
                hits.append(1)
                seen.add(llm)
            else:
                hits.append(0)
        return hits

    agg = {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}
    cnt = 0
    skipped = 0
    ref_k = 10 if 10 in ks else max_k

    pbar = tqdm(eval_qids, desc="Evaluating (sampled)", leave=True, dynamic_ncols=True)
    for qid in pbar:
        pos_k = pos_topk_by_part.get(qid_to_part.get(qid, ""), pos_topk_default) if qid_to_part else pos_topk_default
        gt_list = [aid for aid in all_rankings.get(qid, [])[:pos_k] if aid in aid2idx]
        if not gt_list:
            skipped += 1
            pbar.set_postfix({"done": cnt, "skipped": skipped})
            continue

        rel_set = set(gt_list)
        rel_llm_set: set[str] = set()
        if soft_eval:
            rel_llm_set = {aid_to_llm.get(aid, "") for aid in gt_list}  # type: ignore[union-attr]
            rel_llm_set.discard("")
            if not rel_llm_set:
                skipped += 1
                pbar.set_postfix({"done": cnt, "skipped": skipped})
                continue

        rnd = random.Random(stable_seed_from_qid(qid, rng_seed))
        need_neg = max(0, cand_size - len(gt_list))

        neg_ids: List[str] = []
        if need_neg > 0:
            attempts = 0
            while len(neg_ids) < need_neg and attempts < need_neg * 80:
                aid = rnd.choice(a_ids)
                attempts += 1
                if aid in rel_set:
                    continue
                if soft_eval:
                    llm = aid_to_llm.get(aid, "")  # type: ignore[union-attr]
                    if llm in rel_llm_set:
                        continue
                if aid in neg_ids:
                    continue
                neg_ids.append(aid)

        cand_ids = gt_list + neg_ids
        cand_idx = [aid2idx[a] for a in cand_ids]

        qi = qid2idx[qid]
        qv = torch.from_numpy(Q_cpu[qi : qi + 1]).to(device)
        av = torch.from_numpy(A_cpu[cand_idx]).to(device)
        q_idx_t = torch.tensor([qi], dtype=torch.long, device=device)
        a_idx_t = torch.tensor(cand_idx, dtype=torch.long, device=device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (amp and device.type == "cuda")
            else nullcontext()
        )
        with autocast_ctx:
            qe = encoder.encode_q(qv, q_idx=q_idx_t)
            ae = encoder.encode_a(av, a_idx_t)
            scores = (qe @ ae.t()).float().squeeze(0).cpu().numpy()

        order = np.argsort(-scores)[:max_k]
        pred_ids = [cand_ids[i] for i in order]

        if not soft_eval:
            bin_hits_full = [1 if aid in rel_set else 0 for aid in pred_ids]
            num_rel = len(rel_set)
            get_top_hits = lambda kk: bin_hits_full[:kk]
        else:
            num_rel = len(rel_llm_set)
            get_top_hits = lambda kk: _binary_hits_soft(pred_ids, rel_llm_set, kk)

        for k in ks:
            top = get_top_hits(k)
            Hk = sum(top)
            P = Hk / float(k)
            R = Hk / float(num_rel) if num_rel else 0.0
            F1 = (2 * P * R) / (P + R) if (P + R) else 0.0
            Hit = 1.0 if Hk > 0 else 0.0
            dcg = sum(1.0 / math.log2(i + 2.0) for i, h in enumerate(top) if h)
            ideal = min(k, num_rel)
            idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal else 0.0
            nDCG = (dcg / idcg) if idcg > 0 else 0.0
            rr = 0.0
            for i, h in enumerate(top):
                if h:
                    rr = 1.0 / float(i + 1)
                    break

            agg[k]["P"] += P
            agg[k]["R"] += R
            agg[k]["F1"] += F1
            agg[k]["Hit"] += Hit
            agg[k]["nDCG"] += nDCG
            agg[k]["MRR"] += rr

        cnt += 1
        ref = agg[ref_k]
        pbar.set_postfix(
            {
                "done": cnt,
                "skipped": skipped,
                f"P@{ref_k}": f"{(ref['P'] / cnt):.4f}",
                f"nDCG@{ref_k}": f"{(ref['nDCG'] / cnt):.4f}",
                f"MRR@{ref_k}": f"{(ref['MRR'] / cnt):.4f}",
                "Ncand": len(cand_ids),
            }
        )

    if cnt == 0:
        return {k: {m: 0.0 for m in ["P", "R", "F1", "Hit", "nDCG", "MRR"]} for k in ks}

    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg


def evaluate_parts(
    *,
    encoder,
    Q_cpu: np.ndarray,
    A_cpu: np.ndarray,
    qid2idx: Dict[str, int],
    a_ids: List[str],
    all_rankings: Dict[str, List[str]],
    qids_in_rank: List[str],
    qid_to_part: Dict[str, str],
    eval_parts: List[str],
    device: torch.device,
    topk: int,
    cand_size: int,
    rng_seed: int,
    amp: bool,
    exp_name: str,
    soft_eval: bool = False,
    aid_to_llm: Dict[str, str] | None = None,
    pos_topk_by_part: Dict[str, int] = POS_TOPK_BY_PART,
    pos_topk_default: int = POS_TOPK,
) -> None:
    """Evaluate and print one metrics table per requested part."""
    eval_part_set = set(eval_parts)
    eval_qids = [qid for qid in qids_in_rank if qid_to_part.get(qid, "") in eval_part_set]
    part_splits = split_eval_qids_by_part(eval_qids, qid_to_part=qid_to_part)

    print("[eval] " + " | ".join(f"{part}={len(part_splits.get(part, []))}" for part in eval_parts))

    for part in eval_parts:
        qids_part = part_splits.get(part, [])
        if not qids_part:
            print(f"[eval] skip {part}: no qids")
            continue
        m_part = evaluate_sampled_twotower(
            encoder,
            Q_cpu,
            A_cpu,
            qid2idx,
            a_ids,
            all_rankings,
            qids_part,
            device=device,
            ks=(topk,),
            cand_size=cand_size,
            rng_seed=rng_seed,
            amp=amp,
            qid_to_part=qid_to_part,
            pos_topk_by_part=pos_topk_by_part,
            pos_topk_default=pos_topk_default,
            soft_eval=soft_eval,
            aid_to_llm=aid_to_llm,
        )
        print_metrics_table(f"Validation {part} (sampled q-vector)", m_part, ks=(topk,), filename=exp_name)


@torch.no_grad()
def recommend_topk_for_qid(
    *,
    encoder,
    Q_cpu: np.ndarray,
    A_cpu: np.ndarray,
    qid2idx: Dict[str, int],
    a_ids: List[str],
    qid: str,
    device: torch.device,
    topk: int = 10,
    chunk: int = 8192,
) -> List[Tuple[str, float]]:
    """Chunked full-candidate top-k inference for one query id."""
    qi = qid2idx[qid]
    qv = torch.from_numpy(Q_cpu[qi : qi + 1]).to(device)
    q_idx_t = torch.tensor([qi], dtype=torch.long, device=device)
    qe = encoder.encode_q(qv, q_idx=q_idx_t)

    best_scores: List[float] = []
    best_ids: List[int] = []
    num_agents = len(a_ids)
    for i in range(0, num_agents, chunk):
        j = min(i + chunk, num_agents)
        a_idx = torch.arange(i, j, dtype=torch.long, device=device)
        av = torch.from_numpy(A_cpu[i:j]).to(device)
        ae = encoder.encode_a(av, a_idx)
        scores = (qe @ ae.t()).squeeze(0)
        k = min(topk, j - i)
        top_scores, top_local_idx = torch.topk(scores, k)
        best_scores.extend(top_scores.cpu().tolist())
        best_ids.extend([i + int(t) for t in top_local_idx.cpu().tolist()])

    best_scores_t = torch.tensor(best_scores)
    best_ids_t = torch.tensor(best_ids)
    k = min(topk, best_scores_t.numel())
    final_scores, final_idx = torch.topk(best_scores_t, k)
    return [(a_ids[int(best_ids_t[idx])], float(final_scores[n].item())) for n, idx in enumerate(final_idx)]


def print_sample_recommendations(
    *,
    encoder,
    Q_cpu: np.ndarray,
    A_cpu: np.ndarray,
    qid2idx: Dict[str, int],
    a_ids: List[str],
    q_ids: List[str],
    all_questions: Dict[str, dict],
    device: torch.device,
    topk: int,
    chunk: int,
    num_samples: int = 5,
) -> None:
    """Print a small sanity-check recommendation list for the first few queries."""
    for qid in q_ids[: min(num_samples, len(q_ids))]:
        recs = recommend_topk_for_qid(
            encoder=encoder,
            Q_cpu=Q_cpu,
            A_cpu=A_cpu,
            qid2idx=qid2idx,
            a_ids=a_ids,
            qid=qid,
            device=device,
            topk=topk,
            chunk=chunk,
        )
        qtext = all_questions[qid]["input"][:80].replace("\n", " ")
        print(f"\nQuestion: {qid}  |  {qtext}")
        for r, (aid, s) in enumerate(recs, 1):
            print(f"  {r:2d}. {aid:>20s}  score={s:.4f}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import hashlib
import math
import random
from typing import Collection, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import EPS, POS_TOPK, POS_TOPK_BY_PART, EVAL_TOPK
from .knn import knn_qvec_for_question_text
from .scoring import score_candidates, ScoreMode
from .models.base import RecommenderBase


def stable_seed_from_qid(qid: str, seed: int) -> int:
    digest = hashlib.blake2b(qid.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") ^ (seed * 16777619)) & 0xFFFFFFFF


def sample_eval_candidates(
    a_ids: Sequence[str],
    positives: Sequence[str],
    cand_size: int,
    *,
    qid: str,
    seed: int,
    exclude: Collection[str] | None = None,
) -> List[str]:
    """Sample without replacement from an ordered, unique agent catalog."""
    if cand_size <= 0:
        raise ValueError("Candidate size must be positive")
    rng = random.Random(stable_seed_from_qid(qid, seed))
    candidates = list(dict.fromkeys(positives))
    need = max(0, cand_size - len(candidates))
    if need:
        blocked = set(candidates)
        blocked.update(exclude or ())
        if len(blocked) + need > len(a_ids) // 2:
            pool = [aid for aid in a_ids if aid not in blocked]
            candidates.extend(rng.sample(pool, min(need, len(pool))))
        else:
            while need:
                aid = rng.choice(a_ids)
                if aid not in blocked:
                    candidates.append(aid)
                    blocked.add(aid)
                    need -= 1
    # Avoid systematically ranking the injected positives first on score ties.
    rng.shuffle(candidates)
    return candidates


def _ideal_dcg(k: int, num_rel: int) -> float:
    ideal = min(k, num_rel)
    return sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal else 0.0


def _metrics_at_k(pred: List[str], rel_set: set, k: int) -> Dict[str, float]:
    top = pred[:k]
    hits = [1 if a in rel_set else 0 for a in top]
    Hk = int(sum(hits))
    num_rel = max(1, len(rel_set))

    P = Hk / float(k)
    R = Hk / float(num_rel)
    F1 = (2 * P * R) / (P + R) if (P + R) else 0.0
    Hit = 1.0 if Hk > 0 else 0.0

    dcg = sum(1.0 / math.log2(i + 2.0) for i, h in enumerate(hits) if h)
    idcg = _ideal_dcg(k, len(rel_set))
    nDCG = (dcg / idcg) if idcg > 0 else 0.0

    rr = 0.0
    for i, h in enumerate(hits):
        if h:
            rr = 1.0 / float(i + 1)
            break

    return {"P": P, "R": R, "F1": F1, "Hit": Hit, "nDCG": nDCG, "MRR": rr}


@np.errstate(all="ignore")
def evaluate_sampled_knn_top10(
    model: RecommenderBase,
    aid2idx: Dict[str, int],
    all_rankings: Dict[str, List[str]],
    all_questions: Dict[str, dict],
    eval_qids: List[str],
    knn_cache: dict,
    cand_size: int = 100,
    knn_N: int = 3,
    qid_to_part: Optional[Dict[str, str]] = None,
    pos_topk_by_part: Dict[str, int] = POS_TOPK_BY_PART,
    pos_topk_default: int = POS_TOPK,
    topk: int = EVAL_TOPK,
    score_mode: ScoreMode = "dot",
    seed: int = 123,
    desc: str = "Evaluating",
    use_torch: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    Sampled eval with fixed top10:
      - candidates = positives ∪ random negatives to cand_size
      - qv via TF-IDF KNN weighted avg of train Q latent vectors
      - score on candidate subset
      - optionally score via torch on the model device (GPU if available)
    Return format aligned with print_metrics_table: {10: {...}}
    """
    A = model.export_agent_embeddings()                  # (Na,F)
    bias_a = model.export_agent_bias()                   # (Na,) or None
    device = next(model.parameters()).device

    use_torch_backend = bool(use_torch and torch.cuda.is_available() and device.type == "cuda")
    if use_torch_backend:
        A_t = torch.from_numpy(A).to(device=device, non_blocking=True)
        bias_t = torch.from_numpy(bias_a).to(device=device, non_blocking=True) if bias_a is not None else None
    else:
        bias_t = None

    all_agents = sorted(aid2idx)

    agg = {m: 0.0 for m in ["P", "R", "F1", "Hit", "nDCG", "MRR"]}
    cnt, skipped = 0, 0

    pbar = tqdm(eval_qids, desc=desc, leave=True, dynamic_ncols=True)
    for qid in pbar:
        k = pos_topk_by_part.get(qid_to_part.get(qid, ""), pos_topk_default) if qid_to_part else pos_topk_default
        gt = [aid for aid in all_rankings.get(qid, [])[:k] if aid in aid2idx]
        if not gt:
            skipped += 1
            pbar.set_postfix({"done": cnt, "skipped": skipped})
            continue

        rel_set = set(gt)
        cand = sample_eval_candidates(all_agents, gt, cand_size, qid=qid, seed=seed)

        qtext = all_questions.get(qid, {}).get("input", "")
        qv = knn_qvec_for_question_text(qtext, knn_cache, N=knn_N)  # (F,)

        ai_idx = np.array([aid2idx[a] for a in cand], dtype=np.int64)

        if use_torch_backend:
            ai_idx_t = torch.from_numpy(ai_idx).to(device=device, non_blocking=True)
            cand_emb = A_t.index_select(0, ai_idx_t)
            qv_t = torch.from_numpy(qv).to(device=device, non_blocking=True)

            if score_mode == "dot":
                s_t = cand_emb @ qv_t
            elif score_mode == "cosine":
                cand_norm = torch.linalg.norm(cand_emb, dim=1, keepdim=True) + EPS
                q_norm = torch.linalg.norm(qv_t) + EPS
                s_t = (cand_emb / cand_norm) @ (qv_t / q_norm)
            else:
                raise ValueError(f"Unknown score mode: {score_mode}")

            if bias_t is not None:
                s_t = s_t + bias_t.index_select(0, ai_idx_t)

            k_top = min(topk, s_t.numel())
            order = torch.argsort(s_t, descending=True, stable=True)[:k_top].cpu().tolist()
        else:
            s = score_candidates(A, qv, ai_idx, bias_a=bias_a, mode=score_mode)
            order = np.argsort(-s, kind="stable")[:topk].tolist()

        pred = [cand[i] for i in order]

        met = _metrics_at_k(pred, rel_set, topk)
        for m in agg:
            agg[m] += met[m]
        cnt += 1

        pbar.set_postfix({
            "done": cnt,
            "skipped": skipped,
            f"P@{topk}": f"{(agg['P']/cnt):.4f}",
            f"nDCG@{topk}": f"{(agg['nDCG']/cnt):.4f}",
            f"MRR@{topk}": f"{(agg['MRR']/cnt):.4f}",
            "Ncand": len(cand),
        })

    if cnt == 0:
        return {topk: {m: 0.0 for m in agg}}

    for m in agg:
        agg[m] /= cnt

    return {topk: agg}


def split_eval_qids_by_part(eval_qids: List[str], qid_to_part: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"PartI": [], "PartII": [], "PartIII": [], "Unknown": []}
    for qid in eval_qids:
        out.setdefault(qid_to_part.get(qid, "Unknown"), []).append(qid)
    return out


@np.errstate(all="ignore")
def evaluate_sampled_direct_top10(
    model,
    aid2idx: Dict[str, int],
    qid2idx: Dict[str, int],
    all_rankings: Dict[str, List[str]],
    all_questions: Dict[str, dict],
    eval_qids: List[str],
    q_vectorizer,
    A_text_full: np.ndarray,
    cand_size: int = 100,
    qid_to_part: Optional[Dict[str, str]] = None,
    pos_topk_by_part: Dict[str, int] = POS_TOPK_BY_PART,
    pos_topk_default: int = POS_TOPK,
    topk: int = EVAL_TOPK,
    seed: int = 123,
    desc: str = "Evaluating",
) -> Dict[int, Dict[str, float]]:
    """
    Sampled eval with fixed top10:
      - candidates = positives ∪ random negatives to cand_size
      - qv via TF-IDF vectorizer on question text
      - model scores directly with query + agent text features (no KNN)
    Return format aligned with print_metrics_table: {10: {...}}
    """
    device = next(model.parameters()).device
    A_t = torch.tensor(A_text_full, dtype=torch.float32, device=device)

    all_agents = sorted(aid2idx)

    agg = {m: 0.0 for m in ["P", "R", "F1", "Hit", "nDCG", "MRR"]}
    cnt, skipped = 0, 0

    pbar = tqdm(eval_qids, desc=desc, leave=True, dynamic_ncols=True)
    for qid in pbar:
        k = pos_topk_by_part.get(qid_to_part.get(qid, ""), pos_topk_default) if qid_to_part else pos_topk_default
        gt = [aid for aid in all_rankings.get(qid, [])[:k] if aid in aid2idx]
        if not gt:
            skipped += 1
            pbar.set_postfix({"done": cnt, "skipped": skipped})
            continue

        rel_set = set(gt)
        cand = sample_eval_candidates(all_agents, gt, cand_size, qid=qid, seed=seed)

        qtext = all_questions.get(qid, {}).get("input", "")
        qv_np = q_vectorizer.transform([qtext]).toarray().astype(np.float32)[0]
        qv = torch.tensor(qv_np, dtype=torch.float32, device=device).unsqueeze(0)

        ai_idx = torch.tensor([aid2idx[a] for a in cand], dtype=torch.long, device=device)
        q_idx = torch.full((len(cand),), qid2idx[qid], dtype=torch.long, device=device)
        qv_rep = qv.repeat(len(cand), 1)
        with torch.no_grad():
            scores = model.forward_score(qv_rep, A_t[ai_idx], ai_idx, q_idx=q_idx).detach().cpu().numpy()
        order = np.argsort(-scores, kind="stable")[:topk]
        pred = [cand[i] for i in order]

        met = _metrics_at_k(pred, rel_set, topk)
        for m in agg:
            agg[m] += met[m]
        cnt += 1

        pbar.set_postfix({
            "done": cnt,
            "skipped": skipped,
            f"P@{topk}": f"{(agg['P']/cnt):.4f}",
            f"nDCG@{topk}": f"{(agg['nDCG']/cnt):.4f}",
            f"MRR@{topk}": f"{(agg['MRR']/cnt):.4f}",
            "Ncand": len(cand),
        })

    if cnt == 0:
        return {topk: {m: 0.0 for m in agg}}

    for m in agg:
        agg[m] /= cnt

    return {topk: agg}


@np.errstate(all="ignore")
def evaluate_sampled_embedding_topk(
    model,
    qid2idx: Dict[str, int],
    aid2idx: Dict[str, int],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    Q_t: torch.Tensor,
    A_t: torch.Tensor,
    cand_size: int = 100,
    qid_to_part: Optional[Dict[str, str]] = None,
    pos_topk_by_part: Dict[str, int] = POS_TOPK_BY_PART,
    pos_topk_default: int = POS_TOPK,
    topk: int = EVAL_TOPK,
    seed: int = 123,
    desc: str = "Evaluating",
) -> Dict[int, Dict[str, float]]:
    """Sampled eval using precomputed query/agent embeddings."""
    device = next(model.parameters()).device
    Q_t = Q_t.to(device)
    A_t = A_t.to(device)

    all_agents = sorted(aid2idx)

    agg = {m: 0.0 for m in ["P", "R", "F1", "Hit", "nDCG", "MRR"]}
    cnt, skipped = 0, 0

    pbar = tqdm(eval_qids, desc=desc, leave=True, dynamic_ncols=True)
    for qid in pbar:
        k = pos_topk_by_part.get(qid_to_part.get(qid, ""), pos_topk_default) if qid_to_part else pos_topk_default
        gt = [aid for aid in all_rankings.get(qid, [])[:k] if aid in aid2idx]
        if not gt:
            skipped += 1
            pbar.set_postfix({"done": cnt, "skipped": skipped})
            continue

        rel_set = set(gt)
        cand = sample_eval_candidates(all_agents, gt, cand_size, qid=qid, seed=seed)

        qi = qid2idx[qid]
        qv = Q_t[qi : qi + 1].repeat(len(cand), 1)
        ai_idx = torch.tensor([aid2idx[a] for a in cand], dtype=torch.long, device=device)
        q_idx = torch.full((len(cand),), qi, dtype=torch.long, device=device)

        with torch.no_grad():
            scores = model.forward_score(qv, A_t[ai_idx], ai_idx, q_idx=q_idx).detach().cpu().numpy()
        order = np.argsort(-scores, kind="stable")[:topk]
        pred = [cand[i] for i in order]

        met = _metrics_at_k(pred, rel_set, topk)
        for m in agg:
            agg[m] += met[m]
        cnt += 1

        pbar.set_postfix({
            "done": cnt,
            "skipped": skipped,
            f"P@{topk}": f"{(agg['P']/cnt):.4f}",
            f"nDCG@{topk}": f"{(agg['nDCG']/cnt):.4f}",
            f"MRR@{topk}": f"{(agg['MRR']/cnt):.4f}",
            "Ncand": len(cand),
        })

    if cnt == 0:
        return {topk: {m: 0.0 for m in agg}}

    for m in agg:
        agg[m] /= cnt

    return {topk: agg}
